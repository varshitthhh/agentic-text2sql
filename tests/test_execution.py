"""Tests for text2sql.execution and text2sql.gold. Run from the project root: python -B -m unittest -v tests.test_execution"""
import hashlib
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from unittest import mock

from text2sql import execution
from text2sql.execution import DB_PATH, QueryTimeout, UnsafeSQL, _execute_unvalidated, execute_readonly
from text2sql.gold import DIFFICULTIES, GoldFormatError, load_gold, sql_tables


class Execution(unittest.TestCase):
    def test_select_returns_columns_and_rows(self):
        r = execute_readonly("SELECT order_status, COUNT(*) AS n FROM orders GROUP BY order_status ORDER BY order_status")
        self.assertEqual(r.columns, ("order_status", "n"))
        self.assertEqual(sum(n for _, n in r.rows), 130)
        self.assertFalse(r.truncated)

    def test_row_limit_is_enforced_and_flagged(self):
        r = execute_readonly("SELECT order_id FROM orders", max_rows=3)
        self.assertEqual((len(r.rows), r.truncated), (3, True))
        self.assertEqual(execute_readonly("SELECT order_id FROM orders", max_rows=130).truncated, False)

    def test_the_connection_itself_refuses_writes_even_without_validation(self):
        # Defense in depth: bypass the validator on purpose and show the read-only connection still blocks every write.
        before = hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
        for sql in ("INSERT INTO categories (category_name) VALUES ('x')", "DELETE FROM orders", "UPDATE orders SET order_status = 'x'",
                    "CREATE TABLE t (x)", "DROP TABLE orders"):
            with self.subTest(sql=sql):
                with self.assertRaises(sqlite3.OperationalError):
                    _execute_unvalidated(sql)
        self.assertEqual(hashlib.sha256(DB_PATH.read_bytes()).hexdigest(), before)

    def test_the_connection_itself_refuses_multiple_statements_without_validation(self):
        with self.assertRaises(sqlite3.ProgrammingError):
            _execute_unvalidated("SELECT 1; SELECT 2")

    def test_unsafe_sql_is_rejected_before_it_reaches_the_database(self):
        before = hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
        for sql in ("DELETE FROM orders", "SELECT 1; SELECT 2", "DROP TABLE orders", "SELECT * FROM sqlite_master"):
            with self.subTest(sql=sql):
                with mock.patch.object(execution, "_execute") as run:
                    with self.assertRaises(UnsafeSQL):
                        execute_readonly(sql)
                    run.assert_not_called()  # never executed
        self.assertEqual(hashlib.sha256(DB_PATH.read_bytes()).hexdigest(), before)

    def test_validated_canonical_sql_is_what_runs(self):
        r = execute_readonly("select   count(*)\n  from orders -- comment")
        self.assertEqual((r.executed_sql, r.tables, r.rows), ("SELECT COUNT(*) FROM orders", ("orders",), [(130,)]))

    def test_sqlite_errors_that_pass_validation_still_propagate(self):
        with self.assertRaisesRegex(sqlite3.OperationalError, "no such column"):
            execute_readonly("SELECT no_such_column FROM orders")

    def test_runaway_query_times_out(self):
        start = time.monotonic()
        with self.assertRaises(QueryTimeout):
            execute_readonly("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c", timeout_s=0.5)
        self.assertLess(time.monotonic() - start, 5)


class Gold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = load_gold()

    def test_set_shape(self):
        self.assertEqual(len(self.gold), 40)
        self.assertEqual({d: sum(q.difficulty == d for q in self.gold) for d in DIFFICULTIES},
                         {"simple": 5, "join": 12, "aggregation": 12, "business_rule": 11})
        self.assertEqual([q.question_id for q in self.gold], [f"G{i:02d}" for i in range(1, 41)])
        self.assertEqual([q.question_id for q in self.gold if q.ordered], ["G02", "G05", "G26", "G28"])
        self.assertEqual({d: sum(q.difficulty == d for q in self.gold[:20]) for d in DIFFICULTIES}, {d: 5 for d in DIFFICULTIES})  # G01-G20 unchanged

    def test_labels_match_the_gold_sql_and_every_query_returns_rows(self):
        for q in self.gold:
            with self.subTest(q.question_id):
                self.assertEqual(sql_tables(q.gold_sql), q.required_tables)
                r = execute_readonly(q.gold_sql)
                self.assertTrue(r.rows)
                self.assertFalse(r.truncated)

    def test_all_twelve_tables_are_covered(self):
        self.assertEqual(len({t for q in self.gold for t in q.required_tables}), 12)

    def test_sql_tables_rejects_non_select_and_multiple_statements(self):
        for bad in ("DELETE FROM orders", "SELECT 1; SELECT 2", "DROP TABLE orders"):
            with self.subTest(sql=bad):
                with self.assertRaises(GoldFormatError):
                    sql_tables(bad)

    def test_malformed_gold_files_are_rejected(self):
        good = json.loads(Path(__file__).resolve().parent.parent.joinpath("data/eval/gold.jsonl").read_text(encoding="utf-8").splitlines()[0])
        cases = {
            "invalid json": "{not json",
            "missing field": json.dumps({k: v for k, v in good.items() if k != "gold_sql"}),
            "bad difficulty": json.dumps({**good, "difficulty": "hard"}),
            "unsorted tables": json.dumps({**good, "required_tables": ["orders", "customers"]}),
            "empty tables": json.dumps({**good, "required_tables": []}),
            "non-bool ordered": json.dumps({**good, "ordered": "yes"}),
            "duplicate id": json.dumps(good) + "\n" + json.dumps(good),
            "empty file": "",
        }
        for name, text in cases.items():
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / "g.jsonl"
                    p.write_text(text, encoding="utf-8")
                    with self.assertRaises(GoldFormatError):
                        load_gold(p)


if __name__ == "__main__":
    unittest.main()
