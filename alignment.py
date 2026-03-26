"""
alignment.py — Reasoning Head Alignment for Aletheia-Core
===========================================================
Trains the Reasoning Head (the 8-action MLP in model.py) to predict the
correct algebraic *action* BEFORE the next token is generated.

Two complementary mechanisms are provided:

1. **Supervised Alignment** (:class:`ReasoningHeadAligner`)
   • Uses the labelled steps from the JSONL dataset (each step already has
     an ``"action"`` field: Expand, Factor, Simplify, …).
   • Runs a forward pass of the frozen (or fine-tuned) base model on the
     prompt up to the current step, extracts the last hidden state, feeds it
     through the Reasoning Head, and minimises cross-entropy against the
     ground-truth label.

2. **Symbolic Action Mapper** (:class:`SymbolicActionMapper`)
   • Given two consecutive expression strings, tries to determine *which*
     algebraic action best describes the transformation by running SymPy
     checks (expand, factor, simplify, numeric evaluation, …).
   • This lets us derive action labels even for traces that were generated
     by the model and have no explicit label.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import AletheiaCore, build_aletheia_core, ACTION_VOCAB, NUM_ACTIONS
from verify import _safe_parse  # reuse the sanitised SymPy parser

# Optional integrations
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# =============================================================================
# Action label utilities
# =============================================================================

ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTION_VOCAB)}
IDX_TO_ACTION: Dict[int, str] = {i: a for a, i in ACTION_TO_IDX.items()}


class SymbolicActionMapper:
    """Infer the algebraic action label from two consecutive expression strings.

    The mapper applies a cascade of SymPy checks and returns the *first*
    matching action from the priority list below.

    Priority
    --------
    1. **Done** — the expression contains a complete ``x = <number>`` solution.
    2. **Expand** — ``sympy.expand(prev) == curr`` (or vice-versa).
    3. **Factor** — ``sympy.factor(prev) == curr``.
    4. **Simplify** / **Combine** — ``sympy.simplify(prev) == curr`` or
       ``sympy.radsimp`` / ``sympy.trigsimp`` matches.
    5. **Evaluate** — the new expression is purely numeric.
    6. **Substitute** — a free symbol count decreased.
    7. **Transpose** — linear rearrangement (moving terms across ``=``).
    8. **Simplify** (fallback) — any other verified simplification.
    """

    _DONE_RE = re.compile(r"^[a-zA-Z]\s*=\s*-?\d+(\.\d+)?$")

    def infer(self, prev_expr: str, curr_expr: str) -> str:
        """Return the best-matching action name, or ``'Simplify'`` as fallback."""
        _SYMPY_ERRORS = (
            TypeError, ValueError, sympy.SympifyError,
            AttributeError, NotImplementedError,
        )

        # --- Done: check pattern before SymPy parsing (equations are not valid exprs) ---
        if self._DONE_RE.match(curr_expr.strip()):
            return "Done"

        try:
            sym_prev = _safe_parse(prev_expr)
            sym_curr = _safe_parse(curr_expr)
        except Exception:  # noqa: BLE001 — SymPy/tokenize can raise varied errors
            return "Simplify"

        # --- Evaluate: result is a pure number ---
        if sym_curr.is_number:
            return "Evaluate"

        # --- Expand ---
        try:
            if sympy.expand(sym_prev) == sympy.expand(sym_curr):
                if sympy.expand(sym_curr) == sym_curr:
                    return "Expand"
        except _SYMPY_ERRORS:
            pass

        # --- Factor ---
        try:
            if sympy.factor(sym_prev) == sym_curr:
                return "Factor"
        except _SYMPY_ERRORS:
            pass

        # --- Substitute: fewer free symbols ---
        try:
            if len(sym_curr.free_symbols) < len(sym_prev.free_symbols):
                return "Substitute"
        except _SYMPY_ERRORS:
            pass

        # --- Transpose: terms moved across equality (detect linear shift) ---
        try:
            diff = sympy.simplify(sym_curr - sym_prev)
            if diff.is_number and diff != 0:
                return "Transpose"
        except _SYMPY_ERRORS:
            pass

        # --- Combine: simplify reduces term count ---
        try:
            simplified = sympy.simplify(sym_prev)
            if simplified == sym_curr and sympy.count_ops(sym_curr) < sympy.count_ops(sym_prev):
                return "Combine"
        except _SYMPY_ERRORS:
            pass

        # --- Simplify: fallback ---
        return "Simplify"


# =============================================================================
# Dataset for alignment training
# =============================================================================

@dataclass
class AlignmentSample:
    """One (prompt, action_label) pair for the Reasoning Head."""
    prompt_text: str        # Everything up to and including the current step
    action_label: int       # Ground-truth action index in ACTION_VOCAB


class AlignmentDataset(Dataset):
    """Builds :class:`AlignmentSample` objects from a JSONL algebra dataset.

    For each step in each problem, we construct:
      • A prompt = "Problem: … \\n<think>\\nStep 1…\\nStep k"
      • A label  = the action associated with Step k+1
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        mapper: Optional[SymbolicActionMapper] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mapper = mapper or SymbolicActionMapper()
        self.samples: List[AlignmentSample] = []
        self._build(data_path)

    def _build(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._extract_samples(record)

    def _extract_samples(self, record: Dict) -> None:
        problem = record["problem"]
        steps: List[Dict] = record.get("steps", [])
        if not steps:
            return

        prefix = f"Problem: {problem}\n<think>\n"
        running = prefix

        for i, step in enumerate(steps):
            action_str = step.get("action", "")
            label = ACTION_TO_IDX.get(action_str)
            if label is None:
                # Try to infer from consecutive expressions
                if i > 0:
                    label = ACTION_TO_IDX.get(
                        self.mapper.infer(steps[i - 1]["expression"], step["expression"]),
                        ACTION_TO_IDX["Simplify"],
                    )
                else:
                    label = ACTION_TO_IDX["Simplify"]

            # The prompt is everything up to (but not including) this step
            self.samples.append(AlignmentSample(
                prompt_text=running,
                action_label=label,
            ))
            # Advance the running prefix
            running += f"Step {i + 1}: {step['expression']}\n"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        ids = self.tokenizer.encode(sample.prompt_text)[-self.max_seq_len:]
        return torch.tensor(ids, dtype=torch.long), sample.action_label


def _collate_alignment(batch: List[Tuple[torch.Tensor, int]]):
    """Pad a batch of (ids, label) pairs."""
    ids_list, labels = zip(*batch)
    max_len = max(t.shape[0] for t in ids_list)
    padded = torch.zeros(len(ids_list), max_len, dtype=torch.long)
    for i, t in enumerate(ids_list):
        padded[i, : t.shape[0]] = t
    return padded, torch.tensor(labels, dtype=torch.long)


# =============================================================================
# Alignment trainer
# =============================================================================

@dataclass
class AlignmentConfig:
    """Configuration for the Reasoning Head alignment training."""

    data_path: str = "algebra_dataset.jsonl"
    model_checkpoint: Optional[str] = None
    output_dir: str = "checkpoints"

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 5
    max_seq_len: int = 512
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 50

    log_every: int = 20
    save_every: int = 500

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"

    use_wandb: bool = False
    wandb_project: str = "aletheia-core-alignment"

    freeze_backbone: bool = True
    """When True, only the reasoning_head parameters are updated."""


class ReasoningHeadAligner:
    """Trains the Reasoning Head via supervised cross-entropy.

    The backbone (transformer layers + embeddings) is frozen by default so
    that we align the Reasoning Head without disturbing the LM weights.

    Parameters
    ----------
    config:
        :class:`AlignmentConfig` instance.
    model:
        An :class:`AletheiaCore` whose ``reasoning_head`` will be trained.
    tokenizer:
        Any tokeniser with an ``encode`` method.
    """

    def __init__(
        self,
        config: AlignmentConfig,
        model: AletheiaCore,
        tokenizer,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype, torch.float32)

        self.model = self.model.to(self.device).to(self.dtype)

        if config.freeze_backbone:
            self._freeze_backbone()

        self.optimiser = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

        if _WANDB_AVAILABLE and config.use_wandb:
            wandb.init(project=config.wandb_project, config=vars(config))

        self.global_step = 0

    def _freeze_backbone(self) -> None:
        """Freeze everything except the reasoning_head parameters."""
        for name, param in self.model.named_parameters():
            if "reasoning_head" not in name:
                param.requires_grad_(False)

    def _compute_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item()

    def train(self, dataset: AlignmentDataset) -> None:
        """Run the alignment training loop."""
        cfg = self.config
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=_collate_alignment,
        )

        total_steps = len(loader) * cfg.num_epochs // max(1, cfg.grad_accum_steps)

        def _lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return float(step + 1) / float(max(1, cfg.warmup_steps))
            progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, _lr_lambda)

        print(
            f"Alignment training: {cfg.num_epochs} epochs, "
            f"{len(dataset):,} samples, backbone_frozen={cfg.freeze_backbone}"
        )

        accum_count = 0
        for epoch in range(1, cfg.num_epochs + 1):
            for batch_ids, batch_labels in loader:
                batch_ids = batch_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_ids)
                # Use the last-token hidden state as the action predictor input
                # action_logits: (B, T, NUM_ACTIONS) → take position T-1
                action_logits = outputs["action_logits"][:, -1, :]  # (B, NUM_ACTIONS)

                loss = self.criterion(action_logits, batch_labels) / cfg.grad_accum_steps
                loss.backward()
                accum_count += 1

                if accum_count % cfg.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        cfg.max_grad_norm,
                    )
                    self.optimiser.step()
                    self.optimiser.zero_grad()
                    scheduler.step()
                    self.global_step += 1

                    if self.global_step % cfg.log_every == 0:
                        acc = self._compute_accuracy(action_logits.detach(), batch_labels)
                        lr = self.optimiser.param_groups[0]["lr"]
                        print(
                            f"[epoch {epoch} step {self.global_step}] "
                            f"ce_loss={loss.item() * cfg.grad_accum_steps:.4f} "
                            f"acc={acc:.3f} lr={lr:.2e}"
                        )
                        if _WANDB_AVAILABLE and cfg.use_wandb:
                            wandb.log(
                                {
                                    "ce_loss": loss.item() * cfg.grad_accum_steps,
                                    "accuracy": acc,
                                    "lr": lr,
                                    "epoch": epoch,
                                },
                                step=self.global_step,
                            )

                    if self.global_step % cfg.save_every == 0:
                        self._save_checkpoint(epoch)

        self._save_checkpoint(cfg.num_epochs, final=True)
        if _WANDB_AVAILABLE and cfg.use_wandb:
            wandb.finish()

    def _save_checkpoint(self, epoch: int, final: bool = False) -> None:
        tag = "final" if final else f"step{self.global_step}"
        path = Path(self.config.output_dir) / f"aletheia_alignment_{tag}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "global_step": self.global_step,
                "epoch": epoch,
            },
            path,
        )
        print(f"  Alignment checkpoint saved → {path}")

    @torch.no_grad()
    def evaluate(self, dataset: AlignmentDataset) -> Dict[str, float]:
        """Compute cross-entropy and accuracy on *dataset*."""
        self.model.eval()
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, collate_fn=_collate_alignment
        )
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch_ids, batch_labels in loader:
            batch_ids = batch_ids.to(self.device)
            batch_labels = batch_labels.to(self.device)
            out = self.model(batch_ids)
            logits = out["action_logits"][:, -1, :]
            loss = self.criterion(logits, batch_labels)
            total_loss += loss.item() * batch_labels.shape[0]
            total_correct += (logits.argmax(-1) == batch_labels).sum().item()
            total_samples += batch_labels.shape[0]
        self.model.train()
        return {
            "eval_loss": total_loss / max(1, total_samples),
            "eval_accuracy": total_correct / max(1, total_samples),
        }


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Train the Aletheia-Core Reasoning Head (alignment).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="algebra_dataset.jsonl")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--no_freeze", action="store_true", help="Fine-tune the entire model")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    # Import here to avoid circular dependency at module level
    from trainer import build_tokenizer

    args = _parse_cli()
    cfg = AlignmentConfig(
        data_path=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        model_checkpoint=args.checkpoint,
        freeze_backbone=not args.no_freeze,
        use_wandb=args.wandb,
        device=args.device,
    )

    tokenizer = build_tokenizer()
    model = build_aletheia_core(vocab_size=tokenizer.vocab_size)

    if cfg.model_checkpoint:
        ckpt = torch.load(cfg.model_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

    align_dataset = AlignmentDataset(cfg.data_path, tokenizer, cfg.max_seq_len)
    aligner = ReasoningHeadAligner(cfg, model, tokenizer)
    aligner.train(align_dataset)
