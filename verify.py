"""
verify.py — Symbolic Algebraic Verification Bridge
====================================================
Provides a thin wrapper around SymPy that answers a single question:

    "Is expression A algebraically identical to expression B?"

This is the reward guardrail used by Aletheia-Core: after the model proposes
a rewrite of a math expression, ``verify`` is called to confirm the step is
*symbolically correct* before awarding a training signal.

Usage
-----
>>> from verify import verify
>>> verify("x**2 - 1", "(x-1)*(x+1)")
True
>>> verify("x**2 + 2*x + 1", "(x+1)**2")
True
>>> verify("x + 1", "x + 2")
False
>>> from verify import verify_step
>>> verify_step("x**2 - 1", "(x - 1)*(x + 1)")
VerificationResult(is_equivalent=True, simplified_diff='0', error=None)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# Transformations that allow "2x" to mean "2*x" and similar conveniences
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# Characters that should never appear in a trusted math expression
_DANGEROUS_PATTERN = re.compile(r"(import|exec|eval|open|__)", re.IGNORECASE)


@dataclass
class VerificationResult:
    """The result of a symbolic verification check.

    Attributes
    ----------
    is_equivalent:
        ``True`` if the two expressions are algebraically identical.
    simplified_diff:
        String representation of ``simplify(a - b)``.  Will be ``'0'`` when
        the expressions are equivalent.
    error:
        If parsing or simplification raised an exception, the error message is
        stored here and ``is_equivalent`` is set to ``False``.
    """

    is_equivalent: bool
    simplified_diff: str = "N/A"
    error: Optional[str] = None


def _safe_parse(expr_str: str) -> sympy.Expr:
    """Parse *expr_str* into a SymPy expression with basic safety checks.

    Raises
    ------
    ValueError
        If the string contains potentially dangerous tokens.
    sympy.SympifyError
        If SymPy cannot parse the expression.
    """
    if not isinstance(expr_str, str):
        raise TypeError(f"Expected a string expression, got {type(expr_str).__name__}")
    if _DANGEROUS_PATTERN.search(expr_str):
        raise ValueError(
            f"Expression contains disallowed token: {expr_str!r}"
        )
    return parse_expr(expr_str, transformations=_TRANSFORMATIONS)


def verify(expr_a: str, expr_b: str) -> bool:
    """Return ``True`` if *expr_a* and *expr_b* are algebraically equivalent.

    This is the primary entry point used by the training loop reward signal.

    Parameters
    ----------
    expr_a, expr_b:
        String representations of math expressions understood by SymPy
        (e.g. ``"x**2 - 1"`` or ``"(x-1)*(x+1)"``).

    Returns
    -------
    bool
        ``True`` when ``simplify(a - b) == 0``, ``False`` otherwise
        (including on parse errors).
    """
    result = verify_step(expr_a, expr_b)
    return result.is_equivalent


def verify_step(expr_a: str, expr_b: str) -> VerificationResult:
    """Full verification with diagnostic information.

    Same logic as :func:`verify` but returns a :class:`VerificationResult`
    containing the simplified difference and any error encountered.

    Parameters
    ----------
    expr_a, expr_b:
        String representations of math expressions.

    Returns
    -------
    VerificationResult
    """
    try:
        sym_a = _safe_parse(expr_a)
        sym_b = _safe_parse(expr_b)
        diff = sympy.simplify(sym_a - sym_b)
        is_equiv = diff == sympy.Integer(0)
        return VerificationResult(
            is_equivalent=bool(is_equiv),
            simplified_diff=str(diff),
        )
    except (TypeError, ValueError, sympy.SympifyError, SyntaxError) as exc:
        return VerificationResult(
            is_equivalent=False,
            simplified_diff="N/A",
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 — broad catch for unexpected SymPy errors
        return VerificationResult(
            is_equivalent=False,
            simplified_diff="N/A",
            error=f"Unexpected error: {exc}",
        )


def verify_equation(lhs_a: str, rhs_a: str, lhs_b: str, rhs_b: str) -> bool:
    """Check whether two equations are the same (both sides separately).

    Useful when a model rewrites an equation rather than a single expression.

    Parameters
    ----------
    lhs_a, rhs_a:
        Left-hand side and right-hand side of equation A.
    lhs_b, rhs_b:
        Left-hand side and right-hand side of equation B.

    Returns
    -------
    bool
        ``True`` if both ``lhs_a == lhs_b`` and ``rhs_a == rhs_b``
        (algebraically), OR if ``lhs_a - rhs_a == lhs_b - rhs_b``
        (i.e. the equations describe the same relationship).
    """
    # Check sides individually first (same normal form)
    if verify(lhs_a, lhs_b) and verify(rhs_a, rhs_b):
        return True
    # Alternatively, the "equation" is the same if (lhs_a - rhs_a) == (lhs_b - rhs_b)
    return verify(f"({lhs_a}) - ({rhs_a})", f"({lhs_b}) - ({rhs_b})")
