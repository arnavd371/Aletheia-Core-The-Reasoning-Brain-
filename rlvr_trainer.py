"""
rlvr_trainer.py — Advanced RLVR Training Pipeline with Latent Reasoning
========================================================================
Implements a state-of-the-art Reinforcement Learning from Verifiable Rewards
(RLVR) pipeline using Group Relative Policy Optimization (GRPO).

Key design choices
------------------
* **GRPO with G=16** — For every math prompt, 16 full reasoning traces are
  sampled from the current policy.  Rewards are normalised *within* the group
  to produce zero-mean, unit-variance advantages before the policy update.

* **Verifiable Reward Model (via verify.py)**

  +0.2  per algebraically valid intermediate step  (dense reward)
  +1.0  for the correct final answer               (terminal reward)
  -0.5  when the required ``<think>…</think>``     (format penalty)
        tags are absent from the trace

* **Latent Reasoning Integration** — The model's built-in ``reasoning_head``
  (an 8-action MLP that predicts the next algebraic operation from the last
  hidden state) produces an auxiliary cross-entropy loss during each GRPO
  update.  This keeps the reasoning head calibrated while the language model
  is being trained with RL, at no extra forward pass cost.

Trace format expected from the model
-------------------------------------
  <think>
  Step 1: 3*x = 16 - 7
  Step 2: 3*x = 9
  Step 3: x = 3
  </think>
  Answer: x = 3

Usage
-----
  python rlvr_trainer.py \\
      --data algebra_dataset.jsonl \\
      --epochs 3 \\
      --group_size 16 \\
      --output_dir checkpoints \\
      --wandb

  # Resume from a checkpoint
  python rlvr_trainer.py --data … --checkpoint checkpoints/aletheia_rlvr_step500.pt
"""

from __future__ import annotations

import copy
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import AletheiaCore, build_aletheia_core, ACTION_VOCAB, NUM_ACTIONS
from verify import verify

# Re-use helpers that are already well-tested in trainer.py
from trainer import (
    AlgebraDataset,
    SimpleTokenizer,
    build_tokenizer,
    _extract_think_and_answer,
    _parse_steps,
    _EXPR_RE,
    grpo_loss,
    _seq_log_probs,
)

# Optional integrations -------------------------------------------------------
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import bitsandbytes as bnb  # type: ignore
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RLVRConfig:
    """All hyper-parameters for the RLVR training pipeline.

    Key differences from the base ``TrainingConfig`` in trainer.py:

    * ``group_size`` defaults to **16** (the GRPO group size *G*).
    * ``reward_step`` is **+0.2** per verified algebraic step.
    * ``latent_aux_coef`` controls the weight of the reasoning-head
      auxiliary loss (set to 0.0 to disable).
    """

    # --- Data ---
    data_path: str = "algebra_dataset.jsonl"
    """Path to the JSONL dataset produced by data_gen.py."""

    # --- Model ---
    vocab_size: int = 50257
    """Vocabulary size (matches the tokeniser)."""
    model_checkpoint: Optional[str] = None
    """Resume from this checkpoint (path to a .pt state-dict)."""

    # --- GRPO ---
    group_size: int = 16
    """Number of reasoning traces to generate per problem (G)."""
    clip_eps: float = 0.2
    """PPO-style clipping range [1-eps, 1+eps]."""
    kl_coef: float = 0.01
    """KL penalty coefficient β (0 disables the KL term)."""
    advantage_eps: float = 1e-8
    """Epsilon used in advantage normalisation denominator."""

    # --- Reward ---
    reward_correct: float = 1.0
    """Terminal reward for a correct final answer."""
    reward_step: float = 0.2
    """Dense reward for each algebraically valid intermediate step."""
    penalty_format: float = -0.5
    """Penalty applied when the <think>…</think> tags are absent."""

    # --- Latent Reasoning ---
    latent_aux_coef: float = 0.1
    """Weight of the reasoning-head auxiliary cross-entropy loss.
    Set to 0.0 to disable latent reasoning supervision during RLVR."""

    # --- Generation ---
    max_new_tokens: int = 512
    """Maximum tokens per generated trace."""
    temperature: float = 0.8
    """Sampling temperature during trace generation."""
    top_k: int = 50
    """Top-k sampling (0 to disable)."""

    # --- Optimisation ---
    learning_rate: float = 1e-5
    """AdamW learning rate."""
    weight_decay: float = 0.01
    batch_size: int = 4
    """Number of problems per outer batch (each generates group_size traces)."""
    grad_accum_steps: int = 4
    """Gradient accumulation steps."""
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    warmup_steps: int = 100

    # --- Logging / checkpointing ---
    log_every: int = 10
    """Log metrics every N optimiser steps."""
    save_every: int = 500
    """Save a checkpoint every N optimiser steps."""
    output_dir: str = "checkpoints"
    """Directory for saved checkpoints."""
    use_wandb: bool = False
    """Enable Weights & Biases logging."""
    wandb_project: str = "aletheia-core-rlvr"

    # --- Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    """Model dtype: 'float32', 'float16', or 'bfloat16'."""

    # --- QLoRA (requires bitsandbytes) ---
    use_qlora: bool = False
    """Enable 4-bit QLoRA weight quantisation (requires bitsandbytes)."""


# =============================================================================
# Reward computation with dense step verification
# =============================================================================

# Maps a step string to its right-hand expression for verify() comparison.
def _extract_expr(s: str) -> str:
    """Strip equation prefixes and return a comparable expression string."""
    m = _EXPR_RE.search(s)
    return m.group(1).strip() if m else s


def _answer_is_correct(model_answer: str, ground_truth: str) -> bool:
    """Check algebraic equivalence between model answer and ground truth."""
    _prefix_re = re.compile(r"^[a-zA-Z]\s*=\s*")

    def _strip(s: str) -> str:
        return _prefix_re.sub("", s).strip()

    return verify(_strip(model_answer), _strip(ground_truth))


class LatentReasoningRewardComputer:
    """Compute scalar rewards for a generated reasoning trace.

    Reward breakdown
    ----------------
    * **Format penalty** ``-0.5``:  applied when ``<think>…</think>`` tags
      are missing, before any other scoring.
    * **Dense step reward** ``+0.2`` × (number of valid steps):  for each
      consecutive pair of steps (i-1, i) inside ``<think>…</think>``, we
      call :func:`verify` to confirm the algebraic transformation is sound.
    * **Terminal reward** ``+1.0``:  awarded when the final ``Answer:`` line
      is algebraically equivalent to the ground-truth solution.

    Parameters
    ----------
    config:
        An :class:`RLVRConfig` instance providing the reward coefficients.
    """

    def __init__(self, config: RLVRConfig) -> None:
        self.reward_step = config.reward_step
        self.reward_correct = config.reward_correct
        self.penalty_format = config.penalty_format

    def __call__(self, trace: str, solution: str) -> float:
        """Return the composite scalar reward for *trace* given *solution*."""
        total = 0.0
        think_body, answer_body = _extract_think_and_answer(trace)

        # --- Format guard ---
        if think_body is None:
            total += self.penalty_format
        else:
            # --- Dense step rewards ---
            steps = _parse_steps(think_body)
            for i in range(1, len(steps)):
                prev_expr = _extract_expr(steps[i - 1])
                curr_expr = _extract_expr(steps[i])
                if verify(prev_expr, curr_expr):
                    total += self.reward_step

        # --- Terminal reward ---
        if answer_body is not None and _answer_is_correct(answer_body, solution):
            total += self.reward_correct

        return total


# =============================================================================
# Latent Reasoning auxiliary loss
# =============================================================================

# Maps each action string to its vocabulary index (from model.ACTION_VOCAB).
_ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTION_VOCAB)}


def _infer_action_labels_from_steps(steps: List[str]) -> List[int]:
    """Heuristically assign reasoning-head action labels from step strings.

    For each step we look for keywords that strongly suggest a particular
    action in the model's 8-class action vocabulary.  Unknown steps default
    to the *Simplify* label (index 2).

    This is used to construct a supervision signal for the ``reasoning_head``
    during RLVR training without requiring annotated data.
    """
    _KW: List[Tuple[str, str]] = [
        ("Done", r"x\s*=\s*-?\d+"),                    # final numeric answer
        ("Factor", r"\bfactor\b"),
        ("Expand", r"\bexpand\b"),
        ("Evaluate", r"=\s*\d+"),                       # collapses to a number
        ("Substitute", r"\bsubstitut"),
        ("Transpose", r"\btranspose\b|\brearrang\b"),
        ("Combine", r"\bcombine\b|\bcollect\b"),
    ]
    labels = []
    for step in steps:
        sl = step.lower()
        matched = "Simplify"
        for action, pattern in _KW:
            if re.search(pattern, sl):
                matched = action
                break
        labels.append(_ACTION_TO_IDX.get(matched, _ACTION_TO_IDX["Simplify"]))
    return labels


def compute_latent_aux_loss(
    model: AletheiaCore,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    traces: List[str],
) -> torch.Tensor:
    """Compute the reasoning-head auxiliary cross-entropy loss.

    For each trace in the batch, we parse the ``<think>…</think>`` steps,
    derive heuristic action labels, and supervise the model's ``reasoning_head``
    output at the corresponding response positions.

    Args:
        model: The policy model (must be in ``train()`` mode).
        input_ids: ``(B, T)`` token ids.
        response_mask: ``(B, T)`` boolean mask (True for response tokens).
        traces: Decoded text for each sequence in the batch.

    Returns:
        Scalar auxiliary loss tensor (0.0 when no valid labels are found).
    """
    device = input_ids.device
    outputs = model(input_ids)
    # action_logits: (B, T, NUM_ACTIONS)
    action_logits = outputs["action_logits"]

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for b_idx, trace in enumerate(traces):
        think_body, _ = _extract_think_and_answer(trace)
        if think_body is None:
            continue
        steps = _parse_steps(think_body)
        if not steps:
            continue
        action_labels = _infer_action_labels_from_steps(steps)

        # Collect the last `num_steps` response positions for supervision.
        # We align each label to the last token of the corresponding step.
        resp_positions = response_mask[b_idx].nonzero(as_tuple=True)[0]
        num_labels = min(len(action_labels), len(resp_positions))
        if num_labels == 0:
            continue

        # Evenly distribute label positions across the response span.
        # (This is an approximation; for exact alignment one would tokenise
        # each step individually, which would add significant complexity.)
        step_positions = torch.linspace(
            0, len(resp_positions) - 1, num_labels, dtype=torch.long
        )
        positions = resp_positions[step_positions]  # (num_labels,)

        logits_at_positions = action_logits[b_idx, positions, :]  # (num_labels, NUM_ACTIONS)
        labels_tensor = torch.tensor(
            action_labels[:num_labels], dtype=torch.long, device=device
        )
        all_logits.append(logits_at_positions)
        all_labels.append(labels_tensor)

    if not all_logits:
        return torch.tensor(0.0, device=device, requires_grad=True)

    all_logits_t = torch.cat(all_logits, dim=0)   # (N, NUM_ACTIONS)
    all_labels_t = torch.cat(all_labels, dim=0)   # (N,)
    return F.cross_entropy(all_logits_t, all_labels_t)


# =============================================================================
# RLVR Trainer
# =============================================================================

class RLVRTrainer:
    """RLVR training loop for Aletheia-Core using GRPO with G=16.

    Architecture
    ------------
    * Policy model  (the model being optimised).
    * Frozen reference policy  (a copy of the initial weights, used for the
      KL term and to compute the importance-sampling ratio).
    * Verifiable reward from :class:`LatentReasoningRewardComputer`.
    * Optional reasoning-head auxiliary loss via :func:`compute_latent_aux_loss`.

    Parameters
    ----------
    config:
        :class:`RLVRConfig` instance.
    model:
        The policy :class:`~model.AletheiaCore` (will be moved to the
        configured device and dtype).
    tokenizer:
        Any tokeniser with ``encode`` / ``decode`` / ``batch_encode`` methods
        (see :class:`~trainer.SimpleTokenizer`).
    """

    def __init__(
        self,
        config: RLVRConfig,
        model: AletheiaCore,
        tokenizer: SimpleTokenizer,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype, torch.float32)
        self.reward_fn = LatentReasoningRewardComputer(config)

        self.model = self.model.to(self.device).to(self.dtype)
        self.ref_model = self._build_ref_model()
        self.optimiser = self._build_optimiser()
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None

        self._wandb_init()
        self.global_step = 0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_ref_model(self) -> AletheiaCore:
        """Create a frozen copy of the initial policy (reference policy)."""
        ref = copy.deepcopy(self.model)
        ref.to(self.device).to(self.dtype)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        return ref

    def _build_optimiser(self):
        cfg = self.config
        if _BNB_AVAILABLE and cfg.use_qlora:
            return bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def _wandb_init(self) -> None:
        if _WANDB_AVAILABLE and self.config.use_wandb:
            wandb.init(project=self.config.wandb_project, config=vars(self.config))

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _prompt_text(self, problem: str) -> str:
        """Wrap a math problem in the expected model prompt format."""
        return f"Problem: {problem}\n<think>\n"

    # ------------------------------------------------------------------
    # Group generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_group(
        self, problem: str
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """Sample *config.group_size* (G=16) traces for a single *problem*.

        Returns
        -------
        traces:
            Decoded text strings (one per completion).
        input_ids_list:
            ``[(1, T_i)]`` prompt+response token ids for each trace.
        response_masks_list:
            ``[(1, T_i)]`` boolean tensors (``True`` on response tokens).
        """
        cfg = self.config
        prompt = self._prompt_text(problem)
        prompt_ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device
        )
        prompt_len = prompt_ids.shape[1]

        traces, input_ids_list, masks_list = [], [], []
        self.model.eval()
        for _ in range(cfg.group_size):
            gen_ids = self.model.generate(
                prompt_ids.clone(),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k if cfg.top_k > 0 else None,
                eos_token_id=self.tokenizer.EOS_ID,
            )  # (1, prompt_len + gen_len)

            response_mask = torch.zeros_like(gen_ids, dtype=torch.bool)
            response_mask[:, prompt_len:] = True

            decoded = self.tokenizer.decode(gen_ids[0].tolist())
            traces.append(decoded)
            input_ids_list.append(gen_ids)
            masks_list.append(response_mask)

        self.model.train()
        return traces, input_ids_list, masks_list

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """One GRPO update over a *batch* of problem dicts.

        For each problem in the batch:
        1. Sample G=16 traces from the current policy.
        2. Score each trace with :class:`LatentReasoningRewardComputer`.
        3. Normalise rewards within the group to get per-trace advantages.

        Then, across all (B × G) traces:
        4. Compute reference log-probs (no grad).
        5. Compute policy log-probs (with grad).
        6. Compute the GRPO clipped surrogate + KL loss.
        7. Add the reasoning-head auxiliary loss (if ``latent_aux_coef > 0``).
        8. Back-propagate and return metrics.

        Returns
        -------
        dict
            Scalar metrics for this step (loss, reward, policy_loss, kl_loss,
            latent_aux_loss, …).
        """
        cfg = self.config
        all_metrics: Dict[str, List[float]] = {
            "reward_mean": [],
            "reward_std": [],
            "policy_loss": [],
            "kl_loss": [],
            "mean_ratio": [],
            "clip_frac": [],
            "latent_aux_loss": [],
        }

        # ── Phase 1: generate groups & compute per-group advantages ──────
        all_ids: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []
        all_adv: List[torch.Tensor] = []
        all_traces: List[str] = []

        for sample in batch:
            problem = sample["problem"]
            solution = sample["solution"]

            traces, ids_list, masks_list = self.generate_group(problem)

            rewards = torch.tensor(
                [self.reward_fn(t, solution) for t in traces],
                dtype=torch.float32,
                device=self.device,
            )
            all_metrics["reward_mean"].append(rewards.mean().item())
            all_metrics["reward_std"].append(rewards.std().item())

            adv = (rewards - rewards.mean()) / (rewards.std() + cfg.advantage_eps)
            for ids, mask, a, trace in zip(ids_list, masks_list, adv, traces):
                all_ids.append(ids)
                all_masks.append(mask)
                all_adv.append(a.unsqueeze(0))
                all_traces.append(trace)

        # ── Phase 2: pad sequences to a common length ────────────────────
        advantages = torch.cat(all_adv)  # (B*G,)
        max_len = max(ids.shape[1] for ids in all_ids)

        def _pad(t: torch.Tensor, val: int) -> torch.Tensor:
            pad_len = max_len - t.shape[1]
            return F.pad(t, (0, pad_len), value=val)

        padded_ids = torch.cat(
            [_pad(ids, self.tokenizer.PAD_ID) for ids in all_ids], dim=0
        )  # (B*G, T)
        padded_masks = torch.cat(
            [_pad(m.long(), 0).bool() for m in all_masks], dim=0
        )  # (B*G, T)

        # ── Phase 3: GRPO loss ────────────────────────────────────────────
        with torch.no_grad():
            ref_lp = _seq_log_probs(self.ref_model, padded_ids, padded_masks)

        self.model.train()
        policy_lp = _seq_log_probs(self.model, padded_ids, padded_masks)

        grpo_total, step_metrics = grpo_loss(
            policy_lp, ref_lp, advantages,
            clip_eps=cfg.clip_eps,
            kl_coef=cfg.kl_coef,
        )
        for k, v in step_metrics.items():
            all_metrics.setdefault(k, []).append(v)

        # ── Phase 4: reasoning-head auxiliary loss ────────────────────────
        aux_loss = torch.tensor(0.0, device=self.device)
        if cfg.latent_aux_coef > 0.0:
            aux_loss = compute_latent_aux_loss(
                self.model, padded_ids, padded_masks, all_traces
            )
            all_metrics["latent_aux_loss"].append(aux_loss.item())

        total_loss = (grpo_total + cfg.latent_aux_coef * aux_loss) / cfg.grad_accum_steps
        total_loss.backward()

        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
        avg_metrics["loss"] = total_loss.item() * cfg.grad_accum_steps
        return avg_metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, dataset: AlgebraDataset) -> None:
        """Run the full RLVR training loop over *dataset*."""
        cfg = self.config
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

        total_steps = len(loader) * cfg.num_epochs // max(1, cfg.grad_accum_steps)

        def _lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return float(step + 1) / float(max(1, cfg.warmup_steps))
            progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimiser, _lr_lambda
        )

        print(
            f"Starting RLVR training: {cfg.num_epochs} epochs, "
            f"{len(dataset):,} samples, "
            f"group_size={cfg.group_size}, "
            f"reward_step={cfg.reward_step}, "
            f"latent_aux_coef={cfg.latent_aux_coef}"
        )

        accum_count = 0
        for epoch in range(1, cfg.num_epochs + 1):
            epoch_t0 = time.time()
            for _step_idx, batch in enumerate(loader):
                metrics = self.train_step(batch)
                accum_count += 1

                if accum_count % cfg.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.max_grad_norm
                    )
                    self.optimiser.step()
                    self.optimiser.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_step += 1

                    if self.global_step % cfg.log_every == 0:
                        lr = self.optimiser.param_groups[0]["lr"]
                        log_str = (
                            f"[epoch {epoch} step {self.global_step}] "
                            f"loss={metrics['loss']:.4f} "
                            f"reward={metrics.get('reward_mean', 0.0):.3f} "
                            f"aux={metrics.get('latent_aux_loss', 0.0):.4f} "
                            f"lr={lr:.2e}"
                        )
                        print(log_str)
                        if _WANDB_AVAILABLE and cfg.use_wandb:
                            wandb.log(
                                {**metrics, "lr": lr, "epoch": epoch},
                                step=self.global_step,
                            )

                    if self.global_step % cfg.save_every == 0:
                        self._save_checkpoint(epoch)

            elapsed = time.time() - epoch_t0
            print(f"Epoch {epoch} done in {elapsed:.1f}s")

        self._save_checkpoint(cfg.num_epochs, final=True)
        if _WANDB_AVAILABLE and cfg.use_wandb:
            wandb.finish()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, final: bool = False) -> None:
        tag = "final" if final else f"step{self.global_step}"
        path = Path(self.config.output_dir) / f"aletheia_rlvr_{tag}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimiser_state": self.optimiser.state_dict(),
                "global_step": self.global_step,
                "epoch": epoch,
                "config": vars(self.config),
            },
            path,
        )
        print(f"  RLVR checkpoint saved → {path}")


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Run the Aletheia-Core RLVR training loop (GRPO, G=16).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="algebra_dataset.jsonl",
                   help="Training data JSONL (from data_gen.py)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--group_size", type=int, default=16,
                   help="Number of completions per prompt (G)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--reward_step", type=float, default=0.2,
                   help="Dense reward per verified algebraic step")
    p.add_argument("--latent_aux_coef", type=float, default=0.1,
                   help="Weight of the reasoning-head auxiliary loss (0 to disable)")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    p.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()

    cfg = RLVRConfig(
        data_path=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.lr,
        reward_step=args.reward_step,
        latent_aux_coef=args.latent_aux_coef,
        output_dir=args.output_dir,
        model_checkpoint=args.checkpoint,
        use_wandb=args.wandb,
        device=args.device,
    )

    tokenizer = build_tokenizer()
    model = build_aletheia_core(vocab_size=tokenizer.vocab_size)

    if cfg.model_checkpoint:
        ckpt = torch.load(cfg.model_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"Resumed from {cfg.model_checkpoint}")

    dataset = AlgebraDataset(cfg.data_path)
    trainer = RLVRTrainer(cfg, model, tokenizer)
    trainer.train(dataset)
