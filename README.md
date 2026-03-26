<div align="center">

# Aletheia-Core — The Reasoning Brain 🧠

Streaming algebraic reasoning with a lightweight transformer and a reactive UI. This repository bundles:

- A PyTorch implementation of the **Aletheia-Core** model plus training/export utilities.
- A FastAPI **streaming SSE server** that emits every reasoning step as it is generated.
- A Vite + React **frontend** that renders the live trace and final answer with KaTeX.

</div>

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Model Specifications](#model-specifications)
- [Repository Layout](#repository-layout)
- [Backend (FastAPI SSE server)](#backend-fastapi-sse-server)
- [Frontend (Vite + React)](#frontend-vite--react)
- [API Reference](#api-reference)
- [Development & Testing](#development--testing)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1) Backend – create env and install deps
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Start streaming reasoning server (expects a checkpoint)
python server.py --checkpoint /path/to/aletheia_core_v1.pt --port 8000

# 3) Frontend – install and run dev server
npm install
VITE_API_ENDPOINT=http://localhost:8000/v1/solve npm run dev
```

Open the printed Vite URL (default `http://localhost:5173`) and ask Aletheia to solve a math problem. The UI will stream each algebraic step and render the final answer with KaTeX.

## Prerequisites

- **Python** ≥ 3.10
- **Node.js** ≥ 18 and **npm**
- (Optional) CUDA-capable GPU for fast inference/training
- A trained Aletheia-Core checkpoint (`.pt` or safetensors) for inference

## Model Specifications

- **Architecture:** Decoder-only transformer with 12 layers, 12 attention heads, 768 hidden dimensions, SwiGLU feed-forward blocks (4× width), and RMSNorm. Uses RoPE for positional encoding and falls back to PyTorch SDPA when FlashAttention is unavailable.
- **Context length:** 2048 tokens (configurable via `max_seq_len`).
- **Action reasoning head:** Predicts one of eight algebraic actions at each position: `Expand`, `Factor`, `Simplify`, `Substitute`, `Transpose`, `Combine`, `Evaluate`, `Done`.
- **Vocabulary:** Defaults to 50,257 tokens to match the training tokenizer (`SimpleTokenizer`/HF-compatible). Pass a different `vocab_size` when constructing the model if you use another tokenizer.
- **Checkpoint format:** Standard PyTorch state-dict (`.pt`) or `safetensors`. `export.py` merges optional LoRA adapters, then writes `aletheia_core_v1.safetensors` and a `model_config.json` (includes architecture hyperparams and the reasoning head labels) for serving.
- **Generation defaults:** `max_new_tokens=512`, `temperature=0.8`, `top_k=50` (all configurable on the API/server).
- **Hardware:** Runs on CPU or GPU; prefers bfloat16/float16 on CUDA and automatically uses FlashAttention when installed.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `server.py` | FastAPI + SSE server that streams reasoning traces (`POST /v1/solve`). |
| `model.py`, `trainer.py`, `rlvr_trainer.py`, `alignment.py` | Core model, supervised/RL training utilities, tokenizer helpers. |
| `data_gen.py`, `eval.py`, `export.py`, `verify.py` | Data generation, evaluation, export, and algebraic verification helpers. |
| `src/` | Vite + React frontend (TypeScript) with streaming UI and KaTeX rendering. |
| `requirements.txt` | Python dependencies for training + serving. |
| `package.json` | Frontend scripts and npm dependencies. |

## Backend (FastAPI SSE server)

The server exposes a single SSE endpoint that streams reasoning steps and the final answer.

### Install & Run

```bash
pip install -r requirements.txt
python server.py --checkpoint /path/to/aletheia_core_v1.pt --port 8000
# or: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

**Health check:** `GET /healthz` → `{"status": "ok", "model_loaded": true}`

### Example cURL

```bash
curl -N -X POST http://localhost:8000/v1/solve \
  -H "Content-Type: application/json" \
  -d '{"problem": "Solve: 3*x + 7 = 16"}'
```

## Frontend (Vite + React)

The UI is in `src/` and consumes the SSE stream from `useAletheia`.

### Local development

```bash
npm install
# Point the UI at your API server
echo "VITE_API_ENDPOINT=http://localhost:8000/v1/solve" > .env.local
npm run dev
```

Visit the printed URL (default `http://localhost:5173`). The text box sends problems to the API and renders the live trace.

### Production build

```bash
npm run build
npm run preview   # serve the built assets locally
```

## API Reference

- **Endpoint:** `POST /v1/solve`
- **Content-Type:** `application/json`
- **Accept:** `text/event-stream`

**Request body:**

```json
{
  "problem": "Solve: 3*x + 7 = 16",
  "max_new_tokens": 512,
  "temperature": 0.8,
  "top_k": 50
}
```

- `max_new_tokens` (optional) — defaults to `512`
- `temperature` (optional) — defaults to `0.8`
- `top_k` (optional) — defaults to `50` (`0` disables top-k)

**SSE events:**

- `event: step`  
  `data: {"step_number": 1, "expression": "3*x = 16 - 7", "action": "Transpose", "action_index": 4}`
- `event: answer`  
  `data: {"answer": "x = 3", "done": true}`
- `event: error`  
  `data: {"detail": "<message>"}`

The frontend sanitizes and renders `expression`/`answer` with KaTeX.

## Development & Testing

- **Frontend type-check + build:** `npm run build`
- **Dev server:** `npm run dev`
- **Backend server (dev):** `python server.py --checkpoint <path> --port 8000`

> Tip: Run the backend first, then start `npm run dev` with `VITE_API_ENDPOINT` pointing at the API to see streamed steps live.

## Troubleshooting

- **Missing frontend dependency:** Run `npm install` to ensure `node_modules` is present.
- **CORS or connection errors:** Verify `VITE_API_ENDPOINT` matches your backend URL (scheme/port) and that the backend is reachable.
- **Slow or stuck generation:** Ensure your checkpoint is loaded on a GPU (if available) and adjust `max_new_tokens`/`temperature` as needed.

---

Happy reasoning! If you run into issues, please open an issue or start a discussion.
