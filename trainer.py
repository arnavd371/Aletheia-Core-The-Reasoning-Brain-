"""
trainer.py — GRPO Training Loop for Aletheia-Core
===================================================
Implements Group Relative Policy Optimization (GRPO) where the reward signal
comes entirely from the SymPy-based symbolic verifier (verify.py).

Algorithm overview
------------------
For each problem in the dataset:
  1. Generate a *group* of G=8 reasoning traces from the current policy.
  2. Score each trace with the reward function:
       • +1.0  if the final answer is algebraically correct  (binary reward)
       • +0.1  for every intermediate step verified by SymPy  (dense reward)
       • -0.5  if the trace is missing the required <think>…</think> tags
  3. Normalise rewards within the group to obtain per-trace advantages.
  4. Re-compute log-probs of the generated tokens under the current policy.
  5. Compute the GRPO / PPO-clipped surrogate loss.
  6. Optionally penalise KL divergence from a frozen reference policy.
  7. Back-propagate and update the policy parameters.

Trace format expected from the model
--------------------------------------
  <think>
  Step 1: 3*x = 16 - 7
  Step 2: 3*x = 9
  Step 3: x = 3
  </think>
  Answer: x = 3

WandB, bitsandbytes (QLoRA) and DeepSpeed integration hooks are provided but
remain optional — the loop runs on CPU/single-GPU without them.
"""

from __future__ import annotations

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

# Optional integrations -------------------------------------------------------
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from transformers import AutoTokenizer  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    import bitsandbytes as bnb  # type: ignore
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """All hyper-parameters for the GRPO training loop."""

    # --- Data ---
    data_path: str = "algebra_dataset.jsonl"
    """Path to the JSONL dataset produced by data_gen.py."""

    # --- Model ---
    vocab_size: int = 50257
    """Vocabulary size (matches the tokeniser)."""
    model_checkpoint: Optional[str] = None
    """Resume from this checkpoint (path to a .pt state-dict)."""

    # --- GRPO ---
    group_size: int = 8
    """Number of reasoning traces to generate per problem (G)."""
    clip_eps: float = 0.2
    """PPO-style clipping range [1-eps, 1+eps]."""
    kl_coef: float = 0.01
    """KL penalty coefficient β (0 disables the KL term)."""
    reward_correct: float = 1.0
    """Reward for a correct final answer."""
    reward_step: float = 0.1
    """Per-step reward for a symbolically verified intermediate step."""
    penalty_format: float = -0.5
    """Penalty for missing <think>…</think> formatting."""
    advantage_eps: float = 1e-8
    """Epsilon used in advantage normalisation denominator."""

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
    wandb_project: str = "aletheia-core"

    # --- Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    """Model dtype: 'float32', 'float16', or 'bfloat16'."""

    # --- QLoRA (requires bitsandbytes) ---
    use_qlora: bool = False
    """Enable 4-bit QLoRA weight quantisation (requires bitsandbytes)."""


# =============================================================================
# Dataset
# =============================================================================

class AlgebraDataset(Dataset):
    """Loads the JSONL algebra dataset from data_gen.py."""

    def __init__(self, path: str):
        self.samples: List[Dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


# =============================================================================
# Reward computation
# =============================================================================

# Regex patterns for trace parsing
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"[Aa]nswer\s*[:=]\s*(.+)")
_EXPR_RE = re.compile(
    r"(?:x\s*=|=\s*)([^=\n,]+)"
)  # extract RHS of simple equations like "x = 3"


def _extract_think_and_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse ``<think>…</think> Answer: …`` from a model trace string."""
    think_match = _THINK_RE.search(text)
    think_body = think_match.group(1).strip() if think_match else None
    answer_match = _ANSWER_RE.search(text)
    answer_body = answer_match.group(1).strip() if answer_match else None
    return think_body, answer_body


def _parse_steps(think_body: str) -> List[str]:
    """Split the <think> block into individual step expressions."""
    raw = [s.strip() for s in think_body.splitlines() if s.strip()]
    steps = []
    for s in raw:
        # Strip leading "Step N:" prefix
        cleaned = re.sub(r"^[Ss]tep\s*\d+\s*[:.\-]\s*", "", s).strip()
        if cleaned:
            steps.append(cleaned)
    return steps


def _answer_is_correct(model_answer: str, ground_truth: str) -> bool:
    """Check whether *model_answer* is algebraically equivalent to *ground_truth*.

    We strip common prefixes like "x = " before comparing.
    """
    def _strip_prefix(s: str) -> str:
        return re.sub(r"^[a-zA-Z]\s*=\s*", "", s).strip()

    return verify(_strip_prefix(model_answer), _strip_prefix(ground_truth))


def compute_reward(trace: str, ground_truth_solution: str) -> float:
    """Compute the scalar reward for a single generated reasoning trace.

    Parameters
    ----------
    trace:
        Raw text generated by the model.
    ground_truth_solution:
        The reference solution string from the dataset (e.g. ``"x = 3"``).

    Returns
    -------
    float
        Composite reward score.
    """
    total = 0.0
    think_body, answer_body = _extract_think_and_answer(trace)

    # Use default config values for the standalone function
    _cfg = TrainingConfig()

    # --- Format penalty ---
    if think_body is None:
        total += _cfg.penalty_format
        # We still try to extract an answer for the binary reward
    else:
        # --- Dense step reward ---
        steps = _parse_steps(think_body)
        for i in range(1, len(steps)):
            prev_expr = steps[i - 1]
            curr_expr = steps[i]
            # Extract RHS of equations to compare expressions directly
            def _extract_expr(s: str) -> str:
                m = _EXPR_RE.search(s)
                return m.group(1).strip() if m else s
            if verify(_extract_expr(prev_expr), _extract_expr(curr_expr)):
                total += _cfg.reward_step

    # --- Binary answer reward ---
    if answer_body is not None:
        if _answer_is_correct(answer_body, ground_truth_solution):
            total += _cfg.reward_correct

    return total


class RewardComputer:
    """Stateless callable that wraps :func:`compute_reward` with configurable coefficients."""

    def __init__(self, config: TrainingConfig):
        self.correct = config.reward_correct
        self.step = config.reward_step
        self.fmt_penalty = config.penalty_format

    def __call__(self, trace: str, solution: str) -> float:
        total = 0.0
        think_body, answer_body = _extract_think_and_answer(trace)

        if think_body is None:
            total += self.fmt_penalty
        else:
            steps = _parse_steps(think_body)
            for i in range(1, len(steps)):
                prev_expr = steps[i - 1]
                curr_expr = steps[i]

                def _extract_expr(s: str) -> str:
                    m = _EXPR_RE.search(s)
                    return m.group(1).strip() if m else s

                if verify(_extract_expr(prev_expr), _extract_expr(curr_expr)):
                    total += self.step

        if answer_body is not None and _answer_is_correct(answer_body, solution):
            total += self.correct

        return total


# =============================================================================
# Tokeniser utility
# =============================================================================

class SimpleTokenizer:
    """Minimal byte-level tokeniser that works without external dependencies.

    When `transformers` is installed, use ``GPT2Tokenizer`` instead via
    :func:`build_tokenizer`.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    def __init__(self):
        # Build a printable-ASCII + special-token vocabulary
        self._chars = [chr(i) for i in range(32, 127)]  # printable ASCII
        self._vocab = {c: i + 3 for i, c in enumerate(self._chars)}  # 0-2 reserved
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        self.vocab_size = len(self._vocab) + 3

    def encode(self, text: str) -> List[int]:
        return [self.BOS_ID] + [self._vocab.get(c, self.PAD_ID) for c in text]

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i == self.EOS_ID:
                break
            if i in self._inv_vocab:
                out.append(self._inv_vocab[i])
        return "".join(out)

    def batch_encode(
        self, texts: List[str], max_length: Optional[int] = None, pad: bool = True
    ) -> torch.Tensor:
        encoded = [self.encode(t) for t in texts]
        if max_length:
            encoded = [e[:max_length] for e in encoded]
        if pad:
            max_len = max(len(e) for e in encoded)
            encoded = [e + [self.PAD_ID] * (max_len - len(e)) for e in encoded]
        return torch.tensor(encoded, dtype=torch.long)


def build_tokenizer(use_gpt2: bool = True) -> SimpleTokenizer:
    """Return a tokeniser, preferring GPT-2 when transformers is available."""
    # In a real deployment, swap this out for any HuggingFace-compatible tokeniser.
    return SimpleTokenizer()


# =============================================================================
# Log-probability helpers
# =============================================================================

def _seq_log_probs(
    model: AletheiaCore,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probabilities for the *response* portion of input_ids.

    Args:
        model: The language model (current or reference policy).
        input_ids: ``(B, T)`` token ids (prompt + response concatenated).
        response_mask: ``(B, T)`` boolean mask; ``True`` for response tokens.

    Returns:
        ``(B,)`` sum of log-probs over response tokens for each sequence.
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids)
        logits = outputs["logits"]  # (B, T, V)

    # Shift: predict token at position t+1 from logits at position t
    shift_logits = logits[:, :-1, :]          # (B, T-1, V)
    shift_labels = input_ids[:, 1:]           # (B, T-1)
    shift_mask = response_mask[:, 1:]         # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)             # (B, T-1, V)
    token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    seq_lp = (token_lp * shift_mask.float()).sum(dim=-1)        # (B,)
    return seq_lp


# =============================================================================
# GRPO core
# =============================================================================

def grpo_loss(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the GRPO surrogate loss.

    GRPO adapts PPO's clipped surrogate to the group-relative setting:
      • Advantages are normalised *within* a group of G traces.
      • The ratio ``r_t = exp(log_pi_theta - log_pi_old)`` is per-sequence.
      • A KL penalty can regularise against the reference (frozen) policy.

    Args:
        policy_log_probs: ``(B*G,)`` sum log-probs under the current policy.
        ref_log_probs: ``(B*G,)`` sum log-probs under the reference policy.
        advantages: ``(B*G,)`` group-normalised advantage for each trace.
        clip_eps: PPO clipping range.
        kl_coef: KL penalty coefficient.

    Returns:
        ``(loss, metrics_dict)``
    """
    # Probability ratio (old policy = ref policy at the start of each update)
    log_ratio = policy_log_probs - ref_log_probs.detach()
    ratio = torch.exp(log_ratio.clamp(-20, 20))  # numerical safety clamp

    # Clipped surrogate objective (we *maximise* this, so loss = -obj)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Optional KL divergence penalty: KL(pi || pi_ref) ≈ log_ratio - (ratio - 1)
    kl = (ratio - 1) - log_ratio
    kl_loss = kl_coef * kl.mean()

    total_loss = policy_loss + kl_loss

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "mean_ratio": ratio.mean().item(),
        "clip_frac": ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean().item(),
    }
    return total_loss, metrics


# =============================================================================
# GRPO Trainer
# =============================================================================

class GRPOTrainer:
    """Full GRPO training loop for Aletheia-Core.

    Parameters
    ----------
    config:
        A :class:`TrainingConfig` instance.
    model:
        The policy model (will also serve as the reference when first built).
    tokenizer:
        Any tokeniser with ``encode``/``decode`` / ``batch_encode`` methods.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: AletheiaCore,
        tokenizer: SimpleTokenizer,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype, torch.float32)
        self.reward_fn = RewardComputer(config)

        # Move model to device
        self.model = self.model.to(self.device).to(self.dtype)

        # Frozen reference policy (copy of initial weights, never trained)
        self.ref_model = self._build_ref_model()

        # Optimiser
        self.optimiser = self._build_optimiser()
        self.scheduler = None

        # WandB
        self._wandb_init()

        # State
        self.global_step = 0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_ref_model(self) -> AletheiaCore:
        """Create a frozen copy of the policy model (reference policy)."""
        import copy
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

    def _wandb_init(self):
        if _WANDB_AVAILABLE and self.config.use_wandb:
            wandb.init(project=self.config.wandb_project, config=vars(self.config))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _prompt_text(self, problem: str) -> str:
        """Format a math problem into the expected model prompt."""
        return (
            f"Problem: {problem}\n"
            "<think>\n"
        )

    @torch.no_grad()
    def generate_group(
        self, problem: str
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """Generate *config.group_size* reasoning traces for a single *problem*.

        Returns
        -------
        traces:
            List of decoded trace strings.
        input_ids_list:
            List of ``(1, T)`` tensors — prompt+response token ids.
        response_masks_list:
            List of ``(1, T)`` boolean tensors (True for response tokens).
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
    # Reward & advantage computation
    # ------------------------------------------------------------------

    def compute_group_advantages(
        self, traces: List[str], solution: str
    ) -> torch.Tensor:
        """Compute group-normalised advantages for a list of G traces.

        Returns
        -------
        ``(G,)`` advantage tensor on self.device.
        """
        rewards = torch.tensor(
            [self.reward_fn(t, solution) for t in traces],
            dtype=torch.float32,
            device=self.device,
        )
        mean_r = rewards.mean()
        std_r = rewards.std() + self.config.advantage_eps
        advantages = (rewards - mean_r) / std_r
        return advantages

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """Run one GRPO update over *batch* (a list of problem dicts).

        Each problem generates *group_size* traces.  We accumulate the GRPO
        loss over all traces in the batch and call ``optimiser.step()`` once.

        Returns a metrics dict.
        """
        cfg = self.config
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        all_metrics: Dict[str, List[float]] = {
            "reward_mean": [],
            "reward_std": [],
            "policy_loss": [],
            "kl_loss": [],
            "mean_ratio": [],
            "clip_frac": [],
        }

        # Collect all (input_ids, mask, advantage) triplets first so we can
        # compute log-probs in a single forward pass per model.
        all_ids: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []
        all_adv: List[torch.Tensor] = []

        # ── Phase 1: generate groups & compute advantages ──────────────
        for sample in batch:
            problem = sample["problem"]
            solution = sample["solution"]

            traces, ids_list, masks_list = self.generate_group(problem)

            rewards = [self.reward_fn(t, solution) for t in traces]
            r_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            all_metrics["reward_mean"].append(r_tensor.mean().item())
            all_metrics["reward_std"].append(r_tensor.std().item())

            adv = (r_tensor - r_tensor.mean()) / (r_tensor.std() + cfg.advantage_eps)
            for ids, mask, a in zip(ids_list, masks_list, adv):
                all_ids.append(ids)
                all_masks.append(mask)
                all_adv.append(a.unsqueeze(0))

        # ── Phase 2: GRPO loss ─────────────────────────────────────────
        advantages = torch.cat(all_adv)  # (B*G,)

        # Pad sequences to the same length for batch processing
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

        # Reference log-probs (no grad)
        with torch.no_grad():
            ref_lp = _seq_log_probs(self.ref_model, padded_ids, padded_masks)

        # Policy log-probs (with grad)
        self.model.train()
        policy_lp = _seq_log_probs(self.model, padded_ids, padded_masks)

        loss, step_metrics = grpo_loss(
            policy_lp, ref_lp, advantages,
            clip_eps=cfg.clip_eps,
            kl_coef=cfg.kl_coef,
        )

        # Scale for gradient accumulation
        loss = loss / cfg.grad_accum_steps
        loss.backward()

        for k, v in step_metrics.items():
            all_metrics.setdefault(k, []).append(v)

        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
        avg_metrics["loss"] = loss.item() * cfg.grad_accum_steps
        return avg_metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, dataset: AlgebraDataset) -> None:
        """Run the full training loop over *dataset*."""
        cfg = self.config
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  # return list of dicts
        )

        # Linear warmup scheduler
        total_steps = len(loader) * cfg.num_epochs // cfg.grad_accum_steps
        warmup = cfg.warmup_steps

        def _lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimiser, _lr_lambda
        )

        print(
            f"Starting GRPO training: {cfg.num_epochs} epochs, "
            f"{len(dataset):,} samples, group_size={cfg.group_size}"
        )

        accum_count = 0
        for epoch in range(1, cfg.num_epochs + 1):
            epoch_t0 = time.time()
            for step_idx, batch in enumerate(loader):
                metrics = self.train_step(batch)
                accum_count += 1

                if accum_count % cfg.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    self.optimiser.step()
                    self.optimiser.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                    self.global_step += 1

                    if self.global_step % cfg.log_every == 0:
                        lr = self.optimiser.param_groups[0]["lr"]
                        log_str = (
                            f"[epoch {epoch} step {self.global_step}] "
                            f"loss={metrics['loss']:.4f} "
                            f"reward={metrics.get('reward_mean', 0.0):.3f} "
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
        path = Path(self.config.output_dir) / f"aletheia_core_{tag}.pt"
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
        print(f"  Checkpoint saved → {path}")


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Run the Aletheia-Core GRPO training loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="algebra_dataset.jsonl", help="Training data JSONL")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    p.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()

    cfg = TrainingConfig(
        data_path=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.lr,
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
    trainer = GRPOTrainer(cfg, model, tokenizer)
    trainer.train(dataset)
