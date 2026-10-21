"""Read-only SQLite execution with SQL validation, a wall-clock timeout and a row limit (D-08, D-14).

execute_readonly() is the only public entry point: it validates the SQL (text2sql.safety) immediately before running it and
executes the validated canonical text. The connection itself is the independent boundary: mode=ro, query_only, a progress
handler for the timeout, and a row limit. Shared by gold verification, the agent and the evaluation runner.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .safety import UnsafeSQL, ValidatedSQL, validate_select  # noqa: F401  (UnsafeSQL re-exported for callers)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "sqlite" / "food_delivery.sqlite"
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_ROWS = 1000


class QueryTimeout(Exception):
    """The query ran longer than the allowed wall-clock time and was interrupted."""


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: list[tuple]
    truncated: bool  # True when more than max_rows rows existed; rows holds the first max_rows
    elapsed_s: float
    executed_sql: str = ""  # the validated canonical SQL that actually ran
    tables: tuple[str, ...] = ()  # real tables the query referenced


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def _execute(con: sqlite3.Connection, sql: str, timeout_s: float, max_rows: int) -> QueryResult:
    start = time.monotonic()
    deadline = start + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)  # raises ProgrammingError if the text holds more than one statement
        fetched = cur.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as e:
        if time.monotonic() > deadline:
            raise QueryTimeout(f"query exceeded {timeout_s} s") from e
        raise
    columns = tuple(d[0] for d in cur.description) if cur.description else ()
    return QueryResult(columns, fetched[:max_rows], len(fetched) > max_rows, time.monotonic() - start, sql)


def execute_readonly(sql: str, db_path: Path = DB_PATH, timeout_s: float = DEFAULT_TIMEOUT_S, max_rows: int = DEFAULT_MAX_ROWS) -> QueryResult:
    """Validate, then run one SELECT on a read-only connection.

    Raises UnsafeSQL (policy), QueryTimeout, or sqlite3.Error (for example an unknown column) for the caller to handle.
    """
    con = _connect(db_path)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'")}
        validated = validate_select(sql, tables)
        result = _execute(con, validated.sql, timeout_s, max_rows)
        return QueryResult(result.columns, result.rows, result.truncated, result.elapsed_s, validated.sql, validated.tables)
    finally:
        con.close()


def _execute_unvalidated(sql: str, db_path: Path = DB_PATH, timeout_s: float = DEFAULT_TIMEOUT_S, max_rows: int = DEFAULT_MAX_ROWS) -> QueryResult:
    """Skips SQL validation. Exists only so tests can prove the connection itself refuses writes; never call it elsewhere."""
    con = _connect(db_path)
    try:
        return _execute(con, sql, timeout_s, max_rows)
    finally:
        con.close()
