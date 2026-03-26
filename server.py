"""
server.py — Streaming Reasoning Server for Aletheia-Core
==========================================================
Exposes a single FastAPI endpoint that accepts a math problem and streams
the model's step-by-step algebraic reasoning back to the caller using
**Server-Sent Events (SSE)**.

Endpoint
--------
``POST /v1/solve``

Request body (JSON)
~~~~~~~~~~~~~~~~~~~
::

    {
        "problem": "Solve: 3*x + 7 = 16",
        "max_new_tokens": 512,      // optional, default 512
        "temperature": 0.8,         // optional, default 0.8
        "top_k": 50                 // optional, default 50 (0 = disable)
    }

SSE event stream
~~~~~~~~~~~~~~~~
Each event has ``event: step`` or ``event: answer`` and a JSON data payload:

Step event (one per reasoning step inside ``<think>…</think>``)::

    event: step
    data: {
        "step_number": 1,
        "expression": "3*x = 16 - 7",
        "action": "Transpose",
        "action_index": 4
    }

Answer event (final result)::

    event: answer
    data: {
        "answer": "x = 3",
        "done": true
    }

Error event (if something goes wrong during generation)::

    event: error
    data: {"detail": "<error message>"}

Usage
-----
    # Start the server (development):
    python server.py --checkpoint deploy/aletheia_core_v1.pt --port 8000

    # Or via uvicorn directly:
    uvicorn server:app --host 0.0.0.0 --port 8000

    # Query with curl:
    curl -N -X POST http://localhost:8000/v1/solve \\
         -H "Content-Type: application/json" \\
         -d '{"problem": "Solve: 3*x + 7 = 16"}'

Health check
------------
``GET /healthz`` → ``{"status": "ok", "model_loaded": true}``
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import torch
import torch.nn.functional as F

# FastAPI + SSE
try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    from fastapi.responses import JSONResponse  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    from sse_starlette.sse import EventSourceResponse  # type: ignore
    _SSE_AVAILABLE = True
except ImportError:
    _SSE_AVAILABLE = False

from model import AletheiaCore, build_aletheia_core, ACTION_VOCAB, NUM_ACTIONS
from trainer import (
    SimpleTokenizer,
    build_tokenizer,
    _extract_think_and_answer,
    _parse_steps,
)

# ------------------------------------------------------------------
# Check that server dependencies are available
# ------------------------------------------------------------------
if not _FASTAPI_AVAILABLE:
    raise ImportError(
        "FastAPI is required to run server.py.\n"
        "Install with:  pip install fastapi uvicorn[standard] sse-starlette"
    )
if not _SSE_AVAILABLE:
    raise ImportError(
        "sse-starlette is required for Server-Sent Events.\n"
        "Install with:  pip install sse-starlette"
    )


# =============================================================================
# Globals (set at startup)
# =============================================================================

_model: Optional[AletheiaCore] = None
_tokenizer: Optional[SimpleTokenizer] = None
_device: torch.device = torch.device("cpu")
_model_config: Optional[Dict] = None   # loaded from model_config.json if present


# =============================================================================
# FastAPI application
# =============================================================================

app = FastAPI(
    title="Aletheia-Core Reasoning API",
    description=(
        "Streaming algebraic reasoning using the Aletheia-Core transformer. "
        "Responses are delivered via Server-Sent Events (SSE)."
    ),
    version="1.0.0",
)


# =============================================================================
# Request / response schemas
# =============================================================================

class SolveRequest(BaseModel):
    """Request body for the /v1/solve endpoint."""
    problem: str = Field(..., description="The math problem to solve (plain text)")
    max_new_tokens: int = Field(512, ge=1, le=2048, description="Max tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=0, description="Top-k sampling (0 = disabled)")


# =============================================================================
# Inference helpers
# =============================================================================

_EXPR_RE = re.compile(r"(?:x\s*=|=\s*)([^=\n,]+)")


def _format_prompt(problem: str) -> str:
    return f"Problem: {problem}\n<think>\n"


@torch.no_grad()
def _generate_trace(request: SolveRequest) -> str:
    """Generate a full reasoning trace for *request.problem*.

    Runs synchronously (safe to call from a thread-pool executor).
    """
    assert _model is not None and _tokenizer is not None

    prompt = _format_prompt(request.problem)
    ids = torch.tensor(
        [_tokenizer.encode(prompt)], dtype=torch.long, device=_device
    )

    top_k = request.top_k if request.top_k > 0 else None
    temp = max(request.temperature, 1e-6)

    gen_ids = _model.generate(
        ids,
        max_new_tokens=request.max_new_tokens,
        temperature=temp,
        top_k=top_k,
        eos_token_id=_tokenizer.EOS_ID,
    )
    return _tokenizer.decode(gen_ids[0].tolist())


@torch.no_grad()
def _predict_action(prompt: str) -> tuple[str, int]:
    """Query the Reasoning Head for the most likely next action.

    Returns
    -------
    (action_name, action_index)
    """
    assert _model is not None and _tokenizer is not None

    ids = torch.tensor(
        [_tokenizer.encode(prompt)[-512:]], dtype=torch.long, device=_device
    )
    out = _model(ids)
    action_logits = out["action_logits"][:, -1, :]           # (1, NUM_ACTIONS)
    probs = F.softmax(action_logits, dim=-1).squeeze(0)      # (NUM_ACTIONS,)
    action_idx = int(probs.argmax().item())
    return ACTION_VOCAB[action_idx], action_idx


# =============================================================================
# SSE event generator
# =============================================================================

async def _solve_event_generator(request: SolveRequest) -> AsyncGenerator[Dict, None]:
    """Async generator that yields SSE events for a solve request.

    Steps
    -----
    1. Run model generation in a thread-pool executor so the event loop
       is not blocked.
    2. Parse ``<think>…</think>`` blocks into individual steps.
    3. For each step, predict its action label via the Reasoning Head.
    4. Yield one ``step`` SSE event per reasoning step.
    5. Yield one ``answer`` SSE event at the end.
    """
    try:
        loop = asyncio.get_event_loop()

        # Run synchronous inference in a thread pool so we don't block
        trace: str = await loop.run_in_executor(None, _generate_trace, request)

        think_body, answer_body = _extract_think_and_answer(trace)

        if think_body is not None:
            steps = _parse_steps(think_body)
            prompt_so_far = _format_prompt(request.problem)

            for i, step_expr in enumerate(steps, start=1):
                # Predict action label for this step
                action_name, action_idx = await loop.run_in_executor(
                    None, _predict_action, prompt_so_far
                )

                payload = json.dumps({
                    "step_number": i,
                    "expression": step_expr,
                    "action": action_name,
                    "action_index": action_idx,
                })
                yield {"event": "step", "data": payload}

                # Advance the running prompt for the next action prediction
                prompt_so_far += f"Step {i}: {step_expr}\n"

                # Tiny yield to let the event loop breathe
                await asyncio.sleep(0)

        # Final answer event
        answer_payload = json.dumps({
            "answer": answer_body if answer_body is not None else "",
            "done": True,
        })
        yield {"event": "answer", "data": answer_payload}

    except (RuntimeError, ValueError, TypeError) as exc:
        import traceback
        traceback.print_exc()
        error_payload = json.dumps({"detail": str(exc)})
        yield {"event": "error", "data": error_payload}
    except Exception as exc:  # noqa: BLE001 — catch-all so the SSE stream always closes cleanly
        import traceback
        traceback.print_exc()
        error_payload = json.dumps({"detail": f"Unexpected error: {exc}"})
        yield {"event": "error", "data": error_payload}


# =============================================================================
# Routes
# =============================================================================

@app.get("/healthz", response_class=JSONResponse)
async def health_check():
    """Simple health-check endpoint."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/v1/solve")
async def solve(request: SolveRequest):
    """Stream step-by-step algebraic reasoning via SSE.

    Returns an ``EventSourceResponse`` whose body is a sequence of
    ``step`` events (one per reasoning step) followed by a single
    ``answer`` event.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return EventSourceResponse(_solve_event_generator(request))


# =============================================================================
# Server startup / model loading
# =============================================================================

def load_model(
    checkpoint: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "cpu",
    dtype: str = "float32",
) -> None:
    """Load the model and tokeniser into the module-level globals.

    Parameters
    ----------
    checkpoint:
        Path to a ``.pt`` or ``.safetensors`` state-dict.  If ``None``,
        a random-weight model is used (useful for testing the server).
    config_path:
        Optional path to ``model_config.json`` (produced by export.py).
        Used to log model information at startup.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, …).
    dtype:
        Model floating-point dtype (``"float32"``, ``"bfloat16"``, …).
    """
    global _model, _tokenizer, _device, _model_config

    _device = torch.device(device)
    torch_dtype = getattr(torch, dtype, torch.float32)

    _tokenizer = build_tokenizer()
    _model = build_aletheia_core(vocab_size=_tokenizer.vocab_size)

    if checkpoint:
        # Support both .pt and .safetensors formats
        if checkpoint.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file  # type: ignore
                state_dict = load_file(checkpoint, device=device)
            except ImportError:
                raise RuntimeError(
                    "safetensors package is required to load .safetensors files.\n"
                    "Install with:  pip install safetensors"
                )
        else:
            ckpt = torch.load(checkpoint, map_location=device)
            state_dict = ckpt.get("model_state", ckpt)

        _model.load_state_dict(state_dict)
        print(f"[server] Loaded model weights from: {checkpoint}")
    else:
        print("[server] No checkpoint provided — running with random weights.")

    _model = _model.to(_device).to(torch_dtype)
    _model.eval()

    # Optionally load model_config.json for display / metadata
    _cfg_path = Path(config_path) if config_path else None
    if _cfg_path is None and checkpoint:
        # Try sibling config file: same directory as the checkpoint
        _cfg_path = Path(checkpoint).parent / "model_config.json"
    if _cfg_path and _cfg_path.exists():
        with open(_cfg_path, encoding="utf-8") as f:
            _model_config = json.load(f)
        print(f"[server] Loaded model config from: {_cfg_path}")

    print(
        f"[server] Model ready on {device} ({dtype}) — "
        f"{_model.num_parameters():,} trainable parameters."
    )


# =============================================================================
# CLI entry point (runs the uvicorn server directly)
# =============================================================================

def _parse_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Start the Aletheia-Core streaming reasoning server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=None,
                   help=".pt or .safetensors model weights file")
    p.add_argument("--config", default=None,
                   help="model_config.json path (optional)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--reload", action="store_true",
                   help="Enable uvicorn auto-reload (development mode)")
    return p.parse_args()


if __name__ == "__main__":
    import uvicorn  # type: ignore

    args = _parse_cli()

    # Load model before starting the server so it's ready for the first request
    load_model(
        checkpoint=args.checkpoint,
        config_path=args.config,
        device=args.device,
        dtype=args.dtype,
    )

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
