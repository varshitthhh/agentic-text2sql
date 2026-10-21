"""Result equivalence for execution accuracy (docs/eval-protocol.md section 4).

Rules: columns are compared by position (names ignored); NULL equals only NULL; integers compare exactly; if either value is a
float they compare with math.isclose(rel_tol=1e-9, abs_tol=1e-6); strings compare exactly (case-sensitive) and never equal numbers;
if the gold result is ordered rows are compared in order, otherwise as a multiset (duplicates matter); truncated results never match.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .execution import QueryResult

REL_TOL = 1e-9
ABS_TOL = 1e-6


@dataclass(frozen=True)
class Equivalence:
    ok: bool
    reason: str  # "equal", or why not: truncated | column_count | row_count | order | values


def values_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return type(a) is type(b) and a == b


def rows_equal(r1: tuple, r2: tuple) -> bool:
    return len(r1) == len(r2) and all(values_equal(a, b) for a, b in zip(r1, r2))


def _multiset_equal(rows_a: list[tuple], rows_b: list[tuple]) -> bool:
    remaining = list(rows_b)
    for row in rows_a:
        for i, other in enumerate(remaining):
            if rows_equal(row, other):
                del remaining[i]
                break
        else:
            return False
    return not remaining


def results_equivalent(candidate: QueryResult, gold: QueryResult, ordered: bool) -> Equivalence:
    if candidate.truncated or gold.truncated:
        return Equivalence(False, "truncated")
    if len(candidate.columns) != len(gold.columns):
        return Equivalence(False, "column_count")
    if len(candidate.rows) != len(gold.rows):
        return Equivalence(False, "row_count")
    if ordered:
        if all(rows_equal(a, b) for a, b in zip(candidate.rows, gold.rows)):
            return Equivalence(True, "equal")
        return Equivalence(False, "order" if _multiset_equal(candidate.rows, gold.rows) else "values")
    return Equivalence(True, "equal") if _multiset_equal(candidate.rows, gold.rows) else Equivalence(False, "values")
