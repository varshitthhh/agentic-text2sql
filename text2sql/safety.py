"""SQL safety validator (D-08): exactly one read-only SELECT, checked on the SQLGlot AST for the SQLite dialect.

Allowlist, not blocklist: the root must be a SELECT (optionally with CTEs) or a UNION/INTERSECT/EXCEPT of selects; anything
SQLGlot cannot parse as one (including statements it falls back to a generic Command for) is rejected. The query that is
executed is the canonical SQL re-rendered from the validated AST, so what runs is exactly what was checked.
This is one layer: the read-only connection, query_only, timeout and row limit in execution.py are the independent boundary.
"""
from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

# SQLGlot logs a warning when it falls back to parsing an unsupported statement as a Command; the validator rejects those.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

MAX_SQL_CHARS = 10_000
FORBIDDEN_NODES = tuple(getattr(exp, n) for n in (
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Pragma", "Attach", "Detach", "Merge", "TruncateTable",
    "Transaction", "Commit", "Rollback", "Set", "Use", "Into", "Copy", "Analyze") if hasattr(exp, n))
FORBIDDEN_FUNCTIONS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zeroblob", "randomblob"})


class UnsafeSQL(ValueError):
    """The SQL is not exactly one allowed read-only SELECT. `code` is a stable category for the failure taxonomy."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ValidatedSQL:
    sql: str  # canonical SQLite text rendered from the validated AST; this is what gets executed
    tables: tuple[str, ...]  # real tables referenced (lowercase, sorted; CTE names excluded)


def validate_select(sql: str, allowed_tables: Collection[str]) -> ValidatedSQL:
    """Return the validated canonical SQL, or raise UnsafeSQL."""
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeSQL("empty", "SQL must be a non-empty string")
    if len(sql) > MAX_SQL_CHARS:
        raise UnsafeSQL("too_long", f"SQL is longer than {MAX_SQL_CHARS} characters")
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="sqlite") if s is not None]
    except SqlglotError as e:
        raise UnsafeSQL("malformed", str(e).splitlines()[0][:160]) from e
    if not statements:
        raise UnsafeSQL("empty", "no statement found (only comments or semicolons)")
    if len(statements) > 1:
        raise UnsafeSQL("multiple_statements", f"{len(statements)} statements; exactly one SELECT is allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise UnsafeSQL("not_select", f"statement is {type(tree).__name__}, not a SELECT")
    if (bad := tree.find(*FORBIDDEN_NODES)) is not None:
        raise UnsafeSQL("forbidden_statement", f"{type(bad).__name__} is not allowed")
    if tree.find(exp.Placeholder, exp.Parameter) is not None:
        raise UnsafeSQL("placeholder", "query parameters are not supported")

    allowed = {t.lower() for t in allowed_tables}
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    used: set[str] = set()
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier) or t.args.get("db") or t.args.get("catalog"):
            raise UnsafeSQL("unknown_table", "schema-qualified names and table-valued functions are not allowed")
        name = t.name.lower()
        if name in cte_names:
            continue
        if name not in allowed:
            raise UnsafeSQL("unknown_table", f"table {t.name!r} is not part of the database schema")
        used.add(name)
    for f in tree.find_all(exp.Func):
        name = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
        if name in FORBIDDEN_FUNCTIONS:
            raise UnsafeSQL("forbidden_function", f"function {name} is not allowed")
    # comments=False: SQLGlot would otherwise re-emit comment text as /* ... */; the executed text must contain no comments.
    return ValidatedSQL(tree.sql(dialect="sqlite", comments=False), tuple(sorted(used)))
