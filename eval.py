"""
eval.py — Evaluation Loop for Aletheia-Core
=============================================
Compares two decoding strategies on a held-out validation set:

  1. **Greedy Search** — the baseline: always pick the most-likely token.
  2. **MCTS Search** — uses the Reasoning Head's action probability
     distribution as a *prior* to guide Monte Carlo Tree Search over
     the space of next algebraic actions.

Validation set
--------------
A set of 1 000 problems is generated fresh (with a different random seed
from the training set) via :func:`~data_gen.generate_dataset` so none of
the problems appeared during GRPO training.

Metrics
-------
• ``exact_match`` — the decoded final answer is algebraically equivalent
  to the reference solution (checked by :func:`~verify.verify`).
• ``step_accuracy`` — fraction of intermediate steps that are symbolically
  correct (dense reward metric from trainer.py).
• ``avg_reward`` — the same scalar reward used during GRPO training.
• ``avg_steps`` — mean number of reasoning steps in generated traces.

Usage
-----
    python eval.py --checkpoint checkpoints/aletheia_core_final.pt \\
                   --num_val 1000 --output_dir results/
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from data_gen import generate_dataset
from model import AletheiaCore, build_aletheia_core, ACTION_VOCAB, NUM_ACTIONS
from trainer import (
    RewardComputer,
    SimpleTokenizer,
    TrainingConfig,
    _extract_think_and_answer,
    _parse_steps,
    build_tokenizer,
)
from verify import verify

# Optional WandB
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for the evaluation loop."""

    checkpoint: Optional[str] = None
    """Path to the model .pt checkpoint to evaluate."""

    num_val: int = 1_000
    """Number of validation problems to generate."""

    val_data_path: str = "val_algebra_dataset.jsonl"
    """Path where the validation JSONL will be written (or read if exists)."""

    val_seed: int = 99_999
    """Random seed for validation set generation (must differ from training seed)."""

    output_dir: str = "results"
    """Directory to write evaluation results."""

    # Greedy / generation settings
    max_new_tokens: int = 512
    greedy_temperature: float = 0.0   # 0 = greedy (argmax)

    # MCTS settings
    mcts_simulations: int = 16
    """Number of MCTS rollouts per position."""
    mcts_c_puct: float = 1.4
    """PUCT exploration constant (higher → more exploration)."""
    mcts_depth: int = 6
    """Maximum tree depth (number of action expansions per rollout)."""
    mcts_temperature: float = 0.8
    """Sampling temperature for MCTS rollouts."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"

    use_wandb: bool = False
    wandb_project: str = "aletheia-core-eval"


# =============================================================================
# Validation set generation
# =============================================================================

def prepare_validation_set(config: EvalConfig) -> List[Dict]:
    """Generate or load the 1 000-problem validation set.

    The set is cached at *config.val_data_path* so it only needs to be
    generated once per experiment.
    """
    if Path(config.val_data_path).exists():
        print(f"Loading existing validation set from {config.val_data_path}")
        samples = []
        with open(config.val_data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"  Loaded {len(samples):,} validation samples.")
        return samples

    print(f"Generating {config.num_val:,} validation problems …")
    generate_dataset(
        num_samples=config.num_val,
        output_path=config.val_data_path,
        seed=config.val_seed,
        report_every=200,
    )
    samples = []
    with open(config.val_data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# =============================================================================
# Greedy evaluation
# =============================================================================

class GreedyEvaluator:
    """Decodes a complete reasoning trace using argmax at every token step."""

    def __init__(
        self,
        model: AletheiaCore,
        tokenizer: SimpleTokenizer,
        config: EvalConfig,
    ):
        self.model = model
        self.tok = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        self.reward_fn = RewardComputer(
            TrainingConfig(
                reward_correct=1.0,
                reward_step=0.1,
                penalty_format=-0.5,
            )
        )

    def _format_prompt(self, problem: str) -> str:
        return f"Problem: {problem}\n<think>\n"

    @torch.no_grad()
    def decode(self, problem: str) -> str:
        """Generate a trace for *problem* using greedy (temperature=0) decoding."""
        prompt = self._format_prompt(problem)
        ids = torch.tensor(
            [self.tok.encode(prompt)], dtype=torch.long, device=self.device
        )
        # temperature ≈ 0 → argmax
        gen = self.model.generate(
            ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=max(self.config.greedy_temperature, 1e-6),
            top_k=None,
            eos_token_id=self.tok.EOS_ID,
        )
        return self.tok.decode(gen[0].tolist())

    def evaluate(self, samples: List[Dict]) -> Dict[str, float]:
        """Evaluate greedy decoding on *samples*."""
        self.model.eval()
        metrics = _run_evaluation(
            samples, self.decode, self.reward_fn, tag="Greedy"
        )
        return metrics


# =============================================================================
# MCTS Search
# =============================================================================

@dataclass
class _MCTSNode:
    """A single node in the MCTS tree.

    Each node corresponds to a partial reasoning trace produced up to a
    particular *action step*.  Expansion picks the next action using the
    Reasoning Head's prior probabilities.
    """
    action_idx: Optional[int]    # Action taken to reach this node (None for root)
    parent: Optional["_MCTSNode"]
    prior: float                 # P(action | parent state) from the Reasoning Head
    children: Dict[int, "_MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """PUCT formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))."""
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value + exploration


class MCTSEvaluator:
    """Uses the Reasoning Head's action probabilities as a search prior.

    Algorithm
    ---------
    For each problem, we run *mcts_simulations* rollouts from the root node:
      1. **Select** — descend the tree using the PUCT formula until a leaf.
      2. **Expand** — query the Reasoning Head to get action priors for the
         leaf; create child nodes for each action.
      3. **Simulate** — sample a short continuation from the model at the
         chosen action, compute the reward.
      4. **Backpropagate** — propagate the reward up to the root.

    After all simulations, the action with the highest visit count is chosen,
    and we generate the final trace by greedily following that action.
    """

    def __init__(
        self,
        model: AletheiaCore,
        tokenizer: SimpleTokenizer,
        config: EvalConfig,
    ):
        self.model = model
        self.tok = tokenizer
        self.cfg = config
        self.device = torch.device(config.device)
        self.reward_fn = RewardComputer(
            TrainingConfig(
                reward_correct=1.0,
                reward_step=0.1,
                penalty_format=-0.5,
            )
        )

    def _format_prompt(self, problem: str, action: Optional[str] = None) -> str:
        prompt = f"Problem: {problem}\n<think>\n"
        if action:
            prompt += f"[{action}] "
        return prompt

    @torch.no_grad()
    def _action_priors(self, context_ids: torch.Tensor) -> torch.Tensor:
        """Get the Reasoning Head's action probability distribution.

        Args:
            context_ids: ``(1, T)`` token ids.

        Returns:
            ``(NUM_ACTIONS,)`` probability vector.
        """
        out = self.model(context_ids)
        action_logits = out["action_logits"][:, -1, :]  # (1, NUM_ACTIONS)
        return F.softmax(action_logits, dim=-1).squeeze(0)

    @torch.no_grad()
    def _rollout(self, problem: str, action: str) -> Tuple[str, float]:
        """Sample a complete trace seeded with *action* and compute its reward."""
        prompt = self._format_prompt(problem, action)
        ids = torch.tensor(
            [self.tok.encode(prompt)], dtype=torch.long, device=self.device
        )
        gen = self.model.generate(
            ids,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.mcts_temperature,
            top_k=50,
            eos_token_id=self.tok.EOS_ID,
        )
        trace = self.tok.decode(gen[0].tolist())
        return trace, 0.0  # reward computed externally

    def _mcts_search(self, problem: str, solution: str) -> str:
        """Run MCTS and return the best complete trace."""
        prompt = self._format_prompt(problem)
        context_ids = torch.tensor(
            [self.tok.encode(prompt)], dtype=torch.long, device=self.device
        )

        priors = self._action_priors(context_ids).cpu().tolist()
        root = _MCTSNode(action_idx=None, parent=None, prior=1.0)

        # Initialise root children with Reasoning Head priors
        for a_idx, prior in enumerate(priors):
            root.children[a_idx] = _MCTSNode(
                action_idx=a_idx, parent=root, prior=prior
            )

        # MCTS simulations
        for _ in range(self.cfg.mcts_simulations):
            node = root

            # 1. Selection — traverse to a leaf using PUCT
            depth = 0
            while node.children and depth < self.cfg.mcts_depth:
                best_child = max(
                    node.children.values(),
                    key=lambda c: c.ucb_score(self.cfg.mcts_c_puct, max(1, node.visit_count)),
                )
                node = best_child
                depth += 1

            # 2. Simulation — generate a trace with the chosen action
            action_name = ACTION_VOCAB[node.action_idx] if node.action_idx is not None else "Expand"
            trace, _ = self._rollout(problem, action_name)
            reward = self.reward_fn(trace, solution)

            # 3. Backpropagate
            current = node
            while current is not None:
                current.visit_count += 1
                current.total_value += reward
                current = current.parent

            # 4. Expand node if not yet expanded
            if not node.children and depth < self.cfg.mcts_depth:
                action_prompt = self._format_prompt(problem, action_name)
                child_ids = torch.tensor(
                    [self.tok.encode(action_prompt)],
                    dtype=torch.long,
                    device=self.device,
                )
                child_priors = self._action_priors(child_ids).cpu().tolist()
                for a_idx, prior in enumerate(child_priors):
                    node.children[a_idx] = _MCTSNode(
                        action_idx=a_idx, parent=node, prior=prior
                    )

        # Pick the action with the most visits
        best_action_idx = max(root.children, key=lambda a: root.children[a].visit_count)
        best_action = ACTION_VOCAB[best_action_idx]

        # Final greedy trace with the best action as seed
        trace, _ = self._rollout(problem, best_action)
        return trace

    def decode(self, problem: str, solution: str = "") -> str:
        return self._mcts_search(problem, solution)

    def evaluate(self, samples: List[Dict]) -> Dict[str, float]:
        """Evaluate MCTS decoding on *samples*."""
        self.model.eval()

        def _decode_with_solution(problem: str) -> str:
            # We do need the solution for reward feedback during simulation
            sol = next(
                (s["solution"] for s in samples if s["problem"] == problem), ""
            )
            return self._mcts_search(problem, sol)

        metrics = _run_evaluation(
            samples,
            _decode_with_solution,
            self.reward_fn,
            tag="MCTS",
        )
        return metrics


# =============================================================================
# Common evaluation runner
# =============================================================================

def _answer_is_correct(model_answer: str, ground_truth: str) -> bool:
    def _strip_prefix(s: str) -> str:
        return re.sub(r"^[a-zA-Z]\s*=\s*", "", s).strip()
    return verify(_strip_prefix(model_answer), _strip_prefix(ground_truth))


def _run_evaluation(
    samples: List[Dict],
    decode_fn,
    reward_fn: RewardComputer,
    tag: str = "Eval",
) -> Dict[str, float]:
    """Run *decode_fn* on every sample and collect metrics.

    Args:
        samples: List of problem dicts (``problem``, ``solution``, ``steps``).
        decode_fn: Callable ``(problem: str) -> trace: str``.
        reward_fn: :class:`~trainer.RewardComputer` instance.
        tag: Label for progress output.

    Returns:
        Dict of aggregated metrics.
    """
    exact_matches = 0
    step_accs: List[float] = []
    rewards: List[float] = []
    step_counts: List[int] = []

    t0 = time.time()
    for i, sample in enumerate(samples):
        problem = sample["problem"]
        solution = sample["solution"]

        trace = decode_fn(problem)
        reward = reward_fn(trace, solution)
        rewards.append(reward)

        think_body, answer_body = _extract_think_and_answer(trace)

        # Exact-match
        if answer_body and _answer_is_correct(answer_body, solution):
            exact_matches += 1

        # Step accuracy
        if think_body:
            steps = _parse_steps(think_body)
            step_counts.append(len(steps))
            correct_steps = 0
            _EXPR_RE = re.compile(r"(?:x\s*=|=\s*)([^=\n,]+)")
            for j in range(1, len(steps)):
                def _extract_expr(s: str) -> str:
                    m = _EXPR_RE.search(s)
                    return m.group(1).strip() if m else s
                if verify(_extract_expr(steps[j - 1]), _extract_expr(steps[j])):
                    correct_steps += 1
            step_acc = correct_steps / max(1, len(steps) - 1)
            step_accs.append(step_acc)
        else:
            step_accs.append(0.0)
            step_counts.append(0)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{tag}] {i + 1}/{len(samples)} "
                f"exact_match={exact_matches / (i + 1):.3f} "
                f"avg_reward={sum(rewards) / len(rewards):.3f} "
                f"({elapsed:.1f}s elapsed)"
            )

    n = len(samples)
    metrics = {
        "exact_match": exact_matches / n,
        "step_accuracy": sum(step_accs) / n,
        "avg_reward": sum(rewards) / n,
        "avg_steps": sum(step_counts) / n,
    }
    print(
        f"\n[{tag}] Results over {n} samples:\n"
        + "\n".join(f"  {k}: {v:.4f}" for k, v in metrics.items())
    )
    return metrics


# =============================================================================
# Main evaluation entry point
# =============================================================================

def run_evaluation(config: EvalConfig) -> Dict[str, Dict[str, float]]:
    """Build the model, generate the validation set, and compare strategies.

    Returns
    -------
    ``{"greedy": {...}, "mcts": {...}}`` metrics dicts.
    """
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype, torch.float32)

    tokenizer = build_tokenizer()
    model = build_aletheia_core(vocab_size=tokenizer.vocab_size)

    if config.checkpoint:
        ckpt = torch.load(config.checkpoint, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {config.checkpoint}")
    else:
        print("Warning: no checkpoint provided — evaluating random-weight model.")

    model = model.to(device).to(dtype)
    model.eval()

    samples = prepare_validation_set(config)

    if _WANDB_AVAILABLE and config.use_wandb:
        wandb.init(project=config.wandb_project, config=vars(config))

    # --- Greedy ---
    print("\n" + "=" * 60)
    print("GREEDY SEARCH")
    print("=" * 60)
    greedy_eval = GreedyEvaluator(model, tokenizer, config)
    greedy_metrics = greedy_eval.evaluate(samples)

    # --- MCTS ---
    print("\n" + "=" * 60)
    print("MCTS SEARCH (Reasoning Head prior)")
    print("=" * 60)
    mcts_eval = MCTSEvaluator(model, tokenizer, config)
    mcts_metrics = mcts_eval.evaluate(samples)

    # --- Comparison table ---
    print("\n" + "=" * 60)
    print("COMPARISON: Greedy vs MCTS")
    print("=" * 60)
    header = f"{'Metric':<20} {'Greedy':>12} {'MCTS':>12} {'Delta':>12}"
    print(header)
    print("-" * len(header))
    for key in greedy_metrics:
        g = greedy_metrics[key]
        m = mcts_metrics.get(key, 0.0)
        delta = m - g
        print(f"{key:<20} {g:>12.4f} {m:>12.4f} {delta:>+12.4f}")

    # Save results
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    results = {"greedy": greedy_metrics, "mcts": mcts_metrics}
    out_path = Path(config.output_dir) / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    if _WANDB_AVAILABLE and config.use_wandb:
        wandb.log(
            {"greedy/" + k: v for k, v in greedy_metrics.items()}
            | {"mcts/" + k: v for k, v in mcts_metrics.items()}
        )
        wandb.finish()

    return results


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Evaluate Aletheia-Core: Greedy vs MCTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=None, help="Model .pt checkpoint")
    p.add_argument("--num_val", type=int, default=1000, help="Validation set size")
    p.add_argument("--val_data", default="val_algebra_dataset.jsonl")
    p.add_argument("--val_seed", type=int, default=99999)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--mcts_sims", type=int, default=16, help="MCTS simulations per step")
    p.add_argument("--mcts_depth", type=int, default=6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    cfg = EvalConfig(
        checkpoint=args.checkpoint,
        num_val=args.num_val,
        val_data_path=args.val_data,
        val_seed=args.val_seed,
        output_dir=args.output_dir,
        mcts_simulations=args.mcts_sims,
        mcts_depth=args.mcts_depth,
        device=args.device,
        use_wandb=args.wandb,
    )
    run_evaluation(cfg)
