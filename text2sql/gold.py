"""Gold evaluation set: loading, format validation and verification (docs/eval-protocol.md section 2).

Usage: python -B -m text2sql.gold      (verifies every gold query: single SELECT, executes read-only, checks required_tables)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import exp

from .execution import ROOT, execute_readonly

GOLD_PATH = ROOT / "data" / "eval" / "gold.jsonl"
DIFFICULTIES = ("simple", "join", "aggregation", "business_rule")


class GoldFormatError(ValueError):
    """The gold file is malformed."""


@dataclass(frozen=True)
class GoldQuestion:
    question_id: str
    question: str
    gold_sql: str
    required_tables: tuple[str, ...]
    difficulty: str
    ordered: bool  # True when row order is part of the expected result
    notes: str


def load_gold(path: Path = GOLD_PATH) -> list[GoldQuestion]:
    items, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            q = GoldQuestion(d["question_id"], d["question"], d["gold_sql"], tuple(d["required_tables"]), d["difficulty"], d["ordered"], d["notes"])
        except (ValueError, KeyError, TypeError) as e:
            raise GoldFormatError(f"line {n}: {type(e).__name__}: {e}") from e
        problems = []
        if not all(isinstance(x, str) and x.strip() for x in (q.question_id, q.question, q.gold_sql)):
            problems.append("question_id, question and gold_sql must be non-empty strings")
        if q.question_id in seen:
            problems.append(f"duplicate question_id {q.question_id}")
        if q.difficulty not in DIFFICULTIES:
            problems.append(f"difficulty must be one of {DIFFICULTIES}")
        if not q.required_tables or not all(isinstance(t, str) for t in q.required_tables) or list(q.required_tables) != sorted(set(q.required_tables)):
            problems.append("required_tables must be a non-empty, sorted list of unique table names")
        if not isinstance(q.ordered, bool):
            problems.append("ordered must be true or false")
        if problems:
            raise GoldFormatError(f"line {n} ({q.question_id}): " + "; ".join(problems))
        seen.add(q.question_id)
        items.append(q)
    if not items:
        raise GoldFormatError("gold file is empty")
    return items


def sql_tables(sql: str) -> tuple[str, ...]:
    """Sorted table names a single SELECT statement references (bridge tables included)."""
    statements = sqlglot.parse(sql, dialect="sqlite")
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise GoldFormatError("gold SQL must be exactly one SELECT statement")
    return tuple(sorted({t.name for t in statements[0].find_all(exp.Table)}))


def main() -> int:
    gold = load_gold()
    failures = 0
    print(f"{len(gold)} gold questions from {GOLD_PATH.relative_to(ROOT)}")
    for q in gold:
        parsed = sql_tables(q.gold_sql)
        result = execute_readonly(q.gold_sql)
        ok = parsed == q.required_tables and result.rows and not result.truncated
        failures += not ok
        preview = "; ".join(str(r) for r in result.rows[:2])
        print(f"{'PASS' if ok else 'FAIL'} {q.question_id} [{q.difficulty:13}] tables {list(q.required_tables)} "
              f"{'==' if parsed == q.required_tables else '!= parsed ' + str(list(parsed))} | {len(result.rows)} rows, {result.elapsed_s * 1000:.1f} ms | {preview[:110]}")
    by = {d: sum(1 for q in gold if q.difficulty == d) for d in DIFFICULTIES}
    print(f"difficulty counts: {by} | tables referenced in labels: {sorted({t for q in gold for t in q.required_tables})}")
    print(f"{len(gold) - failures}/{len(gold)} gold questions verified")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
