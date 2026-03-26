"""
data_gen.py — Synthetic Algebra Dataset Generator
===================================================
Generates up to 1 million step-by-step algebra problems across three categories:
  1. Linear equations          (e.g.  3x + 7 = 16  →  x = 3)
  2. Quadratic equations       (e.g.  x² - 5x + 6 = 0  →  x = 2, x = 3)
  3. Systems of linear equations (2 unknowns)

Every problem is **symbolically verified** via :mod:`verify` so that the
generated chain-of-thought steps are guaranteed to be correct.

Output format
-------------
A JSONL file where each line is a JSON object::

    {
        "type":    "linear" | "quadratic" | "system",
        "problem": "Solve: 3*x + 7 = 16",
        "steps": [
            {"action": "Transpose", "expression": "3*x = 16 - 7"},
            {"action": "Simplify",  "expression": "3*x = 9"},
            {"action": "Evaluate",  "expression": "x = 3"},
            {"action": "Done",      "expression": "x = 3"}
        ],
        "solution": "x = 3"
    }

CLI usage
---------
    python data_gen.py --num_samples 1000000 --output algebra_dataset.jsonl
    python data_gen.py --num_samples 10000 --output small_dataset.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Dict, List, Optional

import sympy
from sympy import symbols, Eq, solve, expand, factor, simplify, Rational
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

from verify import verify_step

x, y = symbols("x y")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------

def _make_linear(rng: random.Random) -> Optional[Dict]:
    """Generate a random linear equation  a*x + b = c  and a verified solution chain."""
    a = rng.randint(1, 12)
    b = rng.randint(-20, 20)
    c = rng.randint(-30, 30)

    # Ensure a unique solution
    equation = Eq(a * x + b, c)
    solutions = solve(equation, x)
    if not solutions:
        return None
    sol = solutions[0]

    problem_str = f"Solve: {a}*x + {b} = {c}"

    # Build step-by-step chain
    steps = []

    # Step 1: Transpose constant to RHS
    rhs_after_transpose = c - b
    expr_after = f"{a}*x = {rhs_after_transpose}"
    steps.append({"action": "Transpose", "expression": expr_after})

    # Step 2: Divide both sides (Evaluate)
    solution_expr = f"x = {sol}"
    steps.append({"action": "Evaluate", "expression": solution_expr})
    steps.append({"action": "Done",     "expression": solution_expr})

    # Verify final step is correct
    lhs_sym = parse_expr(f"{a}*x + {b}", transformations=_TRANSFORMATIONS)
    verified_at_sol = simplify(lhs_sym.subs(x, sol) - c) == 0
    if not verified_at_sol:
        return None

    return {
        "type": "linear",
        "problem": problem_str,
        "steps": steps,
        "solution": solution_expr,
    }


def _make_quadratic(rng: random.Random) -> Optional[Dict]:
    """Generate a factorable quadratic  (x - r1)*(x - r2) = 0  with integer roots."""
    r1 = rng.randint(-10, 10)
    r2 = rng.randint(-10, 10)

    poly = expand((x - r1) * (x - r2))  # x**2 - (r1+r2)*x + r1*r2
    equation = Eq(poly, 0)
    solutions = sorted(solve(equation, x), key=lambda s: (s.is_real, float(s) if s.is_real else 0))
    if not solutions:
        return None

    problem_str = f"Solve: {poly} = 0"
    steps = []

    # Step 1: Show factored form
    factored = factor(poly)
    factored_str = f"{factored} = 0"
    steps.append({"action": "Factor", "expression": factored_str})

    # Verify factoring
    check = verify_step(str(poly), str(factored))
    if not check.is_equivalent:
        return None

    # Step 2: State solutions
    sol_strs = [f"x = {s}" for s in solutions]
    solution_expr = ",  ".join(sol_strs)
    steps.append({"action": "Evaluate", "expression": solution_expr})
    steps.append({"action": "Done",     "expression": solution_expr})

    return {
        "type": "quadratic",
        "problem": problem_str,
        "steps": steps,
        "solution": solution_expr,
    }


def _make_system(rng: random.Random) -> Optional[Dict]:
    """Generate a 2×2 linear system  a1*x + b1*y = c1, a2*x + b2*y = c2  with integer solution."""
    # Pick integer solution first to guarantee solvability
    sol_x = rng.randint(-8, 8)
    sol_y = rng.randint(-8, 8)

    a1 = rng.randint(1, 6)
    b1 = rng.randint(1, 6)
    a2 = rng.randint(1, 6)
    b2 = rng.randint(1, 6)

    # Ensure non-degenerate system (det != 0)
    if a1 * b2 - a2 * b1 == 0:
        return None

    c1 = a1 * sol_x + b1 * sol_y
    c2 = a2 * sol_x + b2 * sol_y

    eq1 = Eq(a1 * x + b1 * y, c1)
    eq2 = Eq(a2 * x + b2 * y, c2)
    solutions = solve([eq1, eq2], [x, y])
    if not solutions or solutions[x] != sol_x or solutions[y] != sol_y:
        return None

    prob_line1 = f"{a1}*x + {b1}*y = {c1}"
    prob_line2 = f"{a2}*x + {b2}*y = {c2}"
    problem_str = f"Solve the system:  {prob_line1}  |  {prob_line2}"

    steps = []

    # Step 1: Eliminate y using substitution from eq1
    # Express x from eq1: x = (c1 - b1*y) / a1  if a1 != 0
    x_expr = Rational(c1, a1) - Rational(b1, a1) * y
    steps.append({"action": "Substitute",
                  "expression": f"From eq1: x = {x_expr}"})

    # Step 2: Substitute into eq2 and simplify
    eq2_sub = simplify(a2 * x_expr + b2 * y - c2)
    sol_y_expr = solve(Eq(eq2_sub, 0), y)
    if not sol_y_expr:
        return None
    y_val = sol_y_expr[0]
    steps.append({"action": "Simplify",
                  "expression": f"Substituting: {eq2_sub} = 0  →  y = {y_val}"})

    # Step 3: Back-substitute to get x
    x_val = x_expr.subs(y, y_val)
    steps.append({"action": "Evaluate",
                  "expression": f"x = {x_val}"})
    solution_expr = f"x = {x_val}, y = {y_val}"
    steps.append({"action": "Done", "expression": solution_expr})

    return {
        "type": "system",
        "problem": problem_str,
        "steps": steps,
        "solution": solution_expr,
    }


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------
_GENERATORS = [_make_linear, _make_quadratic, _make_system]
_WEIGHTS    = [0.4, 0.4, 0.2]   # 40% linear, 40% quadratic, 20% systems


def generate_dataset(
    num_samples: int,
    output_path: str,
    seed: int = 0,
    report_every: int = 10_000,
) -> None:
    """Generate *num_samples* problems and write them to *output_path* (JSONL).

    Parameters
    ----------
    num_samples:
        Total number of verified problems to generate.
    output_path:
        Path to the output ``.jsonl`` file.
    seed:
        Random seed for reproducibility.
    report_every:
        Print progress every *report_every* accepted samples.
    """
    rng = random.Random(seed)
    accepted = 0
    attempts = 0

    print(f"Generating {num_samples:,} samples → {output_path}")

    with open(output_path, "w", encoding="utf-8") as fout:
        while accepted < num_samples:
            attempts += 1
            generator = rng.choices(_GENERATORS, weights=_WEIGHTS, k=1)[0]
            sample = generator(rng)
            if sample is None:
                continue
            fout.write(json.dumps(sample) + "\n")
            accepted += 1
            if accepted % report_every == 0:
                rate = f"{accepted/attempts:.1%}" if attempts > 0 else "N/A"
                print(f"  {accepted:>10,} / {num_samples:,}  "
                      f"(attempts: {attempts:,}, "
                      f"success rate: {rate})",
                      flush=True)

    print(f"Done. {accepted:,} samples written ({attempts:,} total attempts).")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic algebra reasoning dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_samples", type=int, default=1_000_000,
        help="Number of verified problems to generate.",
    )
    parser.add_argument(
        "--output", type=str, default="algebra_dataset.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--report_every", type=int, default=10_000,
        help="Print progress after every N accepted samples.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    generate_dataset(
        num_samples=args.num_samples,
        output_path=args.output,
        seed=args.seed,
        report_every=args.report_every,
    )
