"""
export.py — Production Export for Aletheia-Core
=================================================
Prepares a trained Aletheia-Core checkpoint for production deployment:

1. **LoRA adapter merge** — If the checkpoint includes LoRA adapter weights
   (saved under the ``"lora_state"`` key by :mod:`rlvr_trainer`), they are
   merged into the base model's ``Linear`` weights in-place.  If no adapter
   is present, this step is a no-op.

2. **SafeTensors export** — The merged model weights are written to
   ``aletheia_core_v1.safetensors`` using the ``safetensors`` library.
   If ``safetensors`` is not installed, the export falls back to
   ``torch.save`` with a ``.pt`` extension and prints a warning.

3. **Model config** — A ``model_config.json`` file is written next to the
   weights file.  The config includes:
   * Model architecture hyperparameters (``hidden_dim``, ``num_layers``, …)
   * The full ``reasoning_head_labels`` list so that the UI / API can map
     action-index predictions to human-readable labels.
   * Vocabulary size and special token ids.

Usage
-----
    # Export from an RLVR checkpoint:
    python export.py \\
        --checkpoint checkpoints/aletheia_rlvr_final.pt \\
        --output_dir deploy/

    # Specify a custom weights filename:
    python export.py \\
        --checkpoint checkpoints/aletheia_rlvr_final.pt \\
        --output_dir deploy/ \\
        --filename aletheia_core_v2.safetensors

Output files
------------
    deploy/aletheia_core_v1.safetensors   ← merged weights
    deploy/model_config.json              ← architecture + action vocab
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from model import AletheiaCore, build_aletheia_core, ACTION_VOCAB, NUM_ACTIONS
from trainer import SimpleTokenizer, build_tokenizer

# Optional safetensors import
try:
    from safetensors.torch import save_file as safetensors_save  # type: ignore
    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExportConfig:
    """Configuration for the production export pipeline."""

    checkpoint: Optional[str] = None
    """Path to the source .pt state-dict (from trainer.py or rlvr_trainer.py)."""

    output_dir: str = "deploy"
    """Directory where exported files will be written."""

    weights_filename: str = "aletheia_core_v1.safetensors"
    """Filename for the exported weights (must end in .safetensors or .pt)."""

    config_filename: str = "model_config.json"
    """Filename for the model configuration JSON."""

    device: str = "cpu"
    """Device to use when loading and merging weights."""


# =============================================================================
# LoRA merge utilities
# =============================================================================

def _has_lora_state(checkpoint: Dict[str, Any]) -> bool:
    """Return True if the checkpoint dict contains LoRA adapter weights."""
    return "lora_state" in checkpoint


def merge_lora_adapters(
    model: AletheiaCore,
    lora_state: Dict[str, torch.Tensor],
) -> AletheiaCore:
    """Merge LoRA adapter weights into the base model's Linear layers.

    Expects *lora_state* to map parameter names of the form
    ``"<base_param>.lora_A"`` and ``"<base_param>.lora_B"`` to tensors of
    shapes ``(r, fan_in)`` and ``(fan_out, r)`` respectively.  The merged
    weight is:  ``W_merged = W_base + alpha * (B @ A)``

    where ``alpha / r`` is the LoRA scaling (defaults to 1.0 if not present
    in the state dict).

    Parameters
    ----------
    model:
        The base model with original (un-adapted) weights.
    lora_state:
        Dict produced by a LoRA training run containing ``*.lora_A`` and
        ``*.lora_B`` tensors.

    Returns
    -------
    AletheiaCore
        The same *model* object with weights modified in-place.
    """
    # Gather all LoRA pairs: base_name → (lora_A, lora_B, scale)
    lora_pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in lora_state.items():
        if key.endswith(".lora_A"):
            base = key[: -len(".lora_A")]
            lora_pairs.setdefault(base, {})["A"] = tensor
        elif key.endswith(".lora_B"):
            base = key[: -len(".lora_B")]
            lora_pairs.setdefault(base, {})["B"] = tensor
        elif key.endswith(".lora_scale"):
            base = key[: -len(".lora_scale")]
            lora_pairs.setdefault(base, {})["scale"] = tensor

    if not lora_pairs:
        print("  No LoRA adapter pairs found in lora_state — skipping merge.")
        return model

    merged_count = 0
    model_state = dict(model.named_parameters())
    for base_name, parts in lora_pairs.items():
        if "A" not in parts or "B" not in parts:
            continue
        if base_name not in model_state:
            continue

        lora_A = parts["A"].to(model_state[base_name].device, dtype=model_state[base_name].dtype)
        lora_B = parts["B"].to(model_state[base_name].device, dtype=model_state[base_name].dtype)
        scale = float(parts["scale"]) if "scale" in parts else 1.0

        with torch.no_grad():
            model_state[base_name].add_(scale * (lora_B @ lora_A))
        merged_count += 1

    print(f"  Merged {merged_count} LoRA adapter(s) into base weights.")
    return model


# =============================================================================
# SafeTensors export
# =============================================================================

def export_weights(
    model: AletheiaCore,
    output_path: Path,
) -> Path:
    """Export the model's state dict to *output_path*.

    Uses ``safetensors`` when available; otherwise falls back to
    ``torch.save``.

    Returns
    -------
    Path
        The path of the file that was written (may differ from *output_path*
        if a .pt fallback was needed).
    """
    # Flatten state dict — safetensors requires contiguous, non-aliased tensors.
    # The model uses weight-tying (lm_head.weight == token_embedding.weight), so we
    # must clone every tensor to ensure no two entries share the same storage.
    # Preserve the original dtype to avoid unnecessary precision loss or bloat.
    state_dict = {
        k: v.detach().contiguous().clone()
        for k, v in model.state_dict().items()
    }

    if _SAFETENSORS_AVAILABLE and str(output_path).endswith(".safetensors"):
        safetensors_save(state_dict, str(output_path))
        print(f"  Weights saved (safetensors) → {output_path}")
    else:
        if not _SAFETENSORS_AVAILABLE:
            print(
                "  Warning: 'safetensors' is not installed.  "
                "Falling back to torch.save (.pt).\n"
                "  Install with:  pip install safetensors"
            )
        pt_path = output_path.with_suffix(".pt")
        torch.save(state_dict, pt_path)
        print(f"  Weights saved (torch) → {pt_path}")
        output_path = pt_path

    return output_path


# =============================================================================
# Model config generation
# =============================================================================

def generate_model_config(
    model: AletheiaCore,
    tokenizer: SimpleTokenizer,
    output_path: Path,
    weights_filename: str,
) -> Dict[str, Any]:
    """Write ``model_config.json`` with all information needed by the UI/API.

    The config includes:
    * Architecture hyperparameters
    * Reasoning head action labels (for the UI to display action names)
    * Vocabulary size and special token ids
    * The filename of the companion weights file

    Returns
    -------
    dict
        The config dict that was written.
    """
    config: Dict[str, Any] = {
        # Architecture
        "model_type": "AletheiaCore",
        "hidden_dim": model.hidden_dim,
        "num_layers": len(model.layers),
        "num_heads": model.num_heads,
        "head_dim": model.head_dim,
        "max_seq_len": model.max_seq_len,

        # Tokeniser
        "vocab_size": model.token_embedding.num_embeddings,
        "pad_token_id": tokenizer.PAD_ID,
        "bos_token_id": tokenizer.BOS_ID,
        "eos_token_id": tokenizer.EOS_ID,

        # Reasoning Head
        "num_actions": NUM_ACTIONS,
        "reasoning_head_labels": ACTION_VOCAB,
        "reasoning_head_label_descriptions": {
            "Expand": "Expand a product or power into a sum",
            "Factor": "Factor an expression into a product",
            "Simplify": "Simplify or combine like terms",
            "Substitute": "Substitute a known value into an expression",
            "Transpose": "Rearrange terms across an equation boundary",
            "Combine": "Combine fractions or collect terms",
            "Evaluate": "Evaluate a numeric sub-expression",
            "Done": "The solution has been fully determined",
        },

        # Companion weights
        "weights_file": weights_filename,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  Model config saved → {output_path}")
    return config


# =============================================================================
# Main export pipeline
# =============================================================================

def run_export(config: ExportConfig) -> None:
    """Execute the full export pipeline.

    Steps
    -----
    1. Load model from checkpoint (or initialise fresh weights if none given).
    2. Merge LoRA adapters if present in the checkpoint.
    3. Export merged weights to SafeTensors (or .pt fallback).
    4. Write model_config.json.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    tokenizer = build_tokenizer()
    model = build_aletheia_core(vocab_size=tokenizer.vocab_size)

    if config.checkpoint:
        ckpt = torch.load(config.checkpoint, map_location=config.device)
        model_state = ckpt.get("model_state", ckpt)
        model.load_state_dict(model_state)
        print(f"Loaded checkpoint: {config.checkpoint}")

        # --- Merge LoRA adapters if present ---
        if _has_lora_state(ckpt):
            print("LoRA adapter state detected — merging into base weights …")
            merge_lora_adapters(model, ckpt["lora_state"])
        else:
            print("No LoRA adapter state found — skipping merge.")
    else:
        print("Warning: no checkpoint provided — exporting random-weight model.")

    model = model.to(torch.device(config.device))
    model.eval()

    # --- Export weights ---
    weights_path = output_dir / config.weights_filename
    actual_weights_path = export_weights(model, weights_path)

    # --- Write model config ---
    config_path = output_dir / config.config_filename
    generate_model_config(
        model,
        tokenizer,
        config_path,
        weights_filename=actual_weights_path.name,
    )

    print(f"\nExport complete.  Files written to: {output_dir}/")
    print(f"  {actual_weights_path.name}")
    print(f"  {config_path.name}")


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Export Aletheia-Core to production (SafeTensors + config).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=None,
                   help="Source .pt checkpoint (from trainer.py or rlvr_trainer.py)")
    p.add_argument("--output_dir", default="deploy",
                   help="Directory to write exported files")
    p.add_argument("--filename", default="aletheia_core_v1.safetensors",
                   help="Weights filename (use .safetensors or .pt extension)")
    p.add_argument("--config_filename", default="model_config.json")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    cfg = ExportConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        weights_filename=args.filename,
        config_filename=args.config_filename,
        device=args.device,
    )
    run_export(cfg)
