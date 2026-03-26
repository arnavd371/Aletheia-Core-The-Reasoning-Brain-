"""
discovery.py — MCTS Discovery Loop for Self-Improvement
=========================================================
Runs a continuous self-play loop that pits **MCTS search** against **greedy
decoding** on the algebra dataset.  Whenever MCTS finds a reasoning trace
that scores *strictly higher* than the greedy baseline *and* produces a
symbolically correct final answer, the trace is recorded as a
**Verified Golden Path** and saved to ``discovered_logic.jsonl``.

The loop terminates once *target_paths* (default 5 000) golden paths have
been collected.

Saved record format (one JSON object per line)
-----------------------------------------------
::

    {
        "problem":       "Solve: 3*x + 7 = 16",
        "solution":      "x = 3",
        "mcts_trace":    "<think>\\nStep 1: …\\n</think>\\nAnswer: x = 3",
        "greedy_trace":  "<think>\\nStep 1: …\\n</think>\\nAnswer: x = 5",
        "mcts_reward":   1.4,
        "greedy_reward": 0.0,
        "improvement":   1.4,
        "steps":         ["3*x = 9", "x = 3"],
        "verified":      true
    }

Usage
-----
    # Collect 5 000 golden paths using an RL-trained checkpoint:
    python discovery.py \\
        --checkpoint checkpoints/aletheia_rlvr_final.pt \\
        --data algebra_dataset.jsonl \\
        --target_paths 5000 \\
        --output discovered_logic.jsonl

    # Quick smoke-test (10 paths, tiny MCTS):
    python discovery.py \\
        --data algebra_dataset.jsonl \\
        --target_paths 10 \\
        --mcts_sims 4 \\
        --mcts_depth 3
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from model import AletheiaCore, build_aletheia_core, ACTION_VOCAB, NUM_ACTIONS
from trainer import (
    AlgebraDataset,
    RewardComputer,
    SimpleTokenizer,
    TrainingConfig,
    _extract_think_and_answer,
    _parse_steps,
    build_tokenizer,
)
from verify import verify

# Re-use the MCTS and Greedy evaluators already implemented in eval.py
from eval import (
    EvalConfig,
    GreedyEvaluator,
    MCTSEvaluator,
    _answer_is_correct,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DiscoveryConfig:
    """Configuration for the MCTS discovery loop."""

    # --- Data ---
    data_path: str = "algebra_dataset.jsonl"
    """JSONL file produced by data_gen.py (training distribution)."""

    # --- Output ---
    output_path: str = "discovered_logic.jsonl"
    """Destination file for Verified Golden Paths."""
    target_paths: int = 5_000
    """Stop once this many golden paths have been collected."""

    # --- Model ---
    checkpoint: Optional[str] = None
    """Path to a .pt state-dict (from rlvr_trainer.py or trainer.py)."""
    vocab_size: int = 50_257

    # --- MCTS ---
    mcts_simulations: int = 16
    """MCTS rollouts per problem."""
    mcts_c_puct: float = 1.4
    mcts_depth: int = 6
    mcts_temperature: float = 0.8

    # --- Greedy ---
    greedy_temperature: float = 1e-6   # ≈ argmax

    # --- Generation ---
    max_new_tokens: int = 512

    # --- Progress ---
    log_every: int = 50
    """Print a status line every N problems evaluated."""

    # --- Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"


# =============================================================================
# Golden Path verification helper
# =============================================================================

_EXPR_RE = re.compile(r"(?:x\s*=|=\s*)([^=\n,]+)")


def _extract_expr(s: str) -> str:
    m = _EXPR_RE.search(s)
    return m.group(1).strip() if m else s


def _trace_is_correct(trace: str, solution: str) -> bool:
    """Return True if the trace contains a correct final Answer."""
    _, answer_body = _extract_think_and_answer(trace)
    if answer_body is None:
        return False
    return _answer_is_correct(answer_body, solution)


def _extract_verified_steps(trace: str) -> List[str]:
    """Return the list of cleaned step expressions from a trace."""
    think_body, _ = _extract_think_and_answer(trace)
    if think_body is None:
        return []
    return _parse_steps(think_body)


# =============================================================================
# Discovery loop
# =============================================================================

class GoldenPathDiscovery:
    """Runs MCTS vs Greedy on every dataset sample, collecting golden paths.

    A *Verified Golden Path* is a trace where:
    1. MCTS reward  > greedy reward  (MCTS strictly outperformed greedy).
    2. The final answer in the MCTS trace is algebraically correct.

    The loop iterates over the dataset repeatedly (cycling) until
    *target_paths* records are saved.

    Parameters
    ----------
    config:
        :class:`DiscoveryConfig` controlling all aspects of the run.
    model:
        An :class:`~model.AletheiaCore` instance.
    tokenizer:
        Tokeniser (e.g. :class:`~trainer.SimpleTokenizer`).
    """

    def __init__(
        self,
        config: DiscoveryConfig,
        model: AletheiaCore,
        tokenizer: SimpleTokenizer,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)

        # Build a shared EvalConfig for both evaluators
        eval_cfg = EvalConfig(
            checkpoint=None,          # model already loaded
            max_new_tokens=config.max_new_tokens,
            greedy_temperature=config.greedy_temperature,
            mcts_simulations=config.mcts_simulations,
            mcts_c_puct=config.mcts_c_puct,
            mcts_depth=config.mcts_depth,
            mcts_temperature=config.mcts_temperature,
            device=config.device,
            dtype=config.dtype,
        )

        self.greedy = GreedyEvaluator(model, tokenizer, eval_cfg)
        self.mcts = MCTSEvaluator(model, tokenizer, eval_cfg)

        # Shared reward function (same coefficients as rlvr_trainer defaults)
        self.reward_fn = RewardComputer(
            TrainingConfig(
                reward_correct=1.0,
                reward_step=0.2,
                penalty_format=-0.5,
            )
        )

    # ------------------------------------------------------------------
    # Core comparison
    # ------------------------------------------------------------------

    def _compare_one(self, sample: Dict) -> Optional[Dict]:
        """Decode one problem with both strategies; return a golden record or None.

        A golden record is returned only when:
        * MCTS reward strictly exceeds greedy reward.
        * The MCTS trace contains a correct final answer.
        """
        problem = sample["problem"]
        solution = sample["solution"]

        # Greedy decode
        greedy_trace = self.greedy.decode(problem)
        greedy_reward = self.reward_fn(greedy_trace, solution)

        # MCTS decode
        mcts_trace = self.mcts.decode(problem, solution)
        mcts_reward = self.reward_fn(mcts_trace, solution)

        improvement = mcts_reward - greedy_reward

        if improvement <= 0.0:
            return None  # MCTS did not outperform greedy
        if not _trace_is_correct(mcts_trace, solution):
            return None  # Answer is not algebraically verified

        return {
            "problem": problem,
            "solution": solution,
            "mcts_trace": mcts_trace,
            "greedy_trace": greedy_trace,
            "mcts_reward": float(mcts_reward),
            "greedy_reward": float(greedy_reward),
            "improvement": float(improvement),
            "steps": _extract_verified_steps(mcts_trace),
            "verified": True,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, dataset: AlgebraDataset) -> int:
        """Run the discovery loop until *target_paths* are collected.

        Parameters
        ----------
        dataset:
            The source :class:`~trainer.AlgebraDataset`.

        Returns
        -------
        int
            Total number of golden paths written to disk.
        """
        cfg = self.config
        output_path = Path(cfg.output_path)

        # Count already-saved paths so we can resume
        already_saved = 0
        if output_path.exists():
            with open(output_path, encoding="utf-8") as f:
                already_saved = sum(1 for line in f if line.strip())
            print(f"Resuming: {already_saved} paths already saved in {output_path}")

        collected = already_saved
        evaluated = 0
        t0 = time.time()

        self.model.eval()

        with open(output_path, "a", encoding="utf-8") as fout:
            cycle = 0
            while collected < cfg.target_paths:
                cycle += 1
                for sample in dataset.samples:
                    if collected >= cfg.target_paths:
                        break

                    record = self._compare_one(sample)
                    evaluated += 1

                    if record is not None:
                        fout.write(json.dumps(record) + "\n")
                        fout.flush()
                        collected += 1

                    if evaluated % cfg.log_every == 0:
                        elapsed = time.time() - t0
                        rate = collected / max(1, evaluated)
                        print(
                            f"  [discovery] evaluated={evaluated:,} "
                            f"golden={collected:,}/{cfg.target_paths:,} "
                            f"hit_rate={rate:.3f} "
                            f"elapsed={elapsed:.1f}s"
                        )

        elapsed = time.time() - t0
        print(
            f"\nDiscovery complete: {collected:,} golden paths collected "
            f"({evaluated:,} problems evaluated in {elapsed:.1f}s).\n"
            f"Output → {output_path}"
        )
        return collected


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Aletheia-Core MCTS Discovery Loop — collect Verified Golden Paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="algebra_dataset.jsonl",
                   help="Algebra JSONL dataset (from data_gen.py)")
    p.add_argument("--checkpoint", default=None,
                   help="RL-trained model checkpoint (.pt)")
    p.add_argument("--output", default="discovered_logic.jsonl",
                   help="Output JSONL file for golden paths")
    p.add_argument("--target_paths", type=int, default=5_000,
                   help="Stop after collecting this many golden paths")
    p.add_argument("--mcts_sims", type=int, default=16,
                   help="MCTS simulations per problem")
    p.add_argument("--mcts_depth", type=int, default=6,
                   help="Maximum MCTS tree depth")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()

    cfg = DiscoveryConfig(
        data_path=args.data,
        checkpoint=args.checkpoint,
        output_path=args.output,
        target_paths=args.target_paths,
        mcts_simulations=args.mcts_sims,
        mcts_depth=args.mcts_depth,
        max_new_tokens=args.max_new_tokens,
        log_every=args.log_every,
        device=args.device,
    )

    tokenizer = build_tokenizer()
    model = build_aletheia_core(vocab_size=tokenizer.vocab_size)

    if cfg.checkpoint:
        ckpt = torch.load(cfg.checkpoint, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {cfg.checkpoint}")
    else:
        print("Warning: no checkpoint provided — using random-weight model.")

    dtype = getattr(torch, cfg.dtype, torch.float32)
    model = model.to(torch.device(cfg.device)).to(dtype)

    dataset = AlgebraDataset(cfg.data_path)
    loop = GoldenPathDiscovery(cfg, model, tokenizer)
    loop.run(dataset)
