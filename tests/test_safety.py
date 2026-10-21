"""Tests for text2sql.safety. Run from the project root: python -B -m unittest -v tests.test_safety

Only validation is exercised here (nothing dangerous is executed); the executor integration is in tests.test_execution.
"""
import unittest

from text2sql.execution import execute_readonly
from text2sql.gold import load_gold
from text2sql.safety import MAX_SQL_CHARS, UnsafeSQL, validate_select

TABLES = ["categories", "coupons", "customer_addresses", "customers", "delivery_partners", "menu_items", "order_items",
          "order_tracking", "orders", "payments", "ratings", "restaurants"]


def ok(sql):
    return validate_select(sql, TABLES)


def code(sql):
    with unittest.TestCase().assertRaises(UnsafeSQL) as cm:
        validate_select(sql, TABLES)
    return cm.exception.code


class Accepted(unittest.TestCase):
    def test_plain_selects(self):
        for sql in ("SELECT 1", "SELECT COUNT(*) FROM orders", "SELECT o.order_id, c.full_name FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE o.order_id = 25",
                    "SELECT ROUND(AVG(price), 2) FROM menu_items", "SELECT a FROM (SELECT 1 AS a)", "SELECT order_status, COUNT(*) FROM orders GROUP BY 1 HAVING COUNT(*) > 3 ORDER BY 2 DESC LIMIT 5"):
            with self.subTest(sql=sql):
                self.assertTrue(ok(sql).sql.startswith("SELECT"))

    def test_cte_and_recursive_cte_and_set_operations(self):
        v = ok("WITH d AS (SELECT order_id FROM orders WHERE order_status = 'Delivered') SELECT COUNT(*) FROM d")
        self.assertEqual(v.tables, ("orders",))  # the CTE name d is not a real table
        self.assertEqual(ok("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 5) SELECT COUNT(*) FROM c").tables, ())
        self.assertEqual(ok("SELECT customer_id FROM orders UNION SELECT customer_id FROM ratings").tables, ("orders", "ratings"))
        self.assertEqual(ok("WITH a AS (SELECT 1 AS x), b AS (SELECT x FROM a) SELECT x FROM b").tables, ())

    def test_formatting_case_quoting_and_trailing_semicolon(self):
        for sql in ("select * from ORDERS", "  \n\tSELECT\n  *\nFROM\n  orders\n;  ", 'SELECT * FROM "orders"', "SELECT * FROM `orders`", "SELECT * FROM [orders]", "SELECT 1;"):
            with self.subTest(sql=sql):
                ok(sql)
        self.assertEqual(ok("select * from ORDERS").tables, ("orders",))

    def test_comments_are_inert_and_stripped_from_the_executed_text(self):
        for sql in ("SELECT 1 -- ; DROP TABLE orders", "SELECT 1 /* ; DROP TABLE orders */", "/* lead */ SELECT 1 -- trail", "SELECT '; DROP TABLE orders'"):
            with self.subTest(sql=sql):
                v = ok(sql)
                self.assertNotIn("--", v.sql)
                self.assertNotIn("/*", v.sql)
                self.assertNotIn(";", v.sql.replace("'; DROP TABLE orders'", ""))  # only the string literal may contain a semicolon
        self.assertEqual(ok("select  1   -- c").sql, "SELECT 1")

    def test_comment_text_cannot_break_out_of_the_rendered_query(self):
        for sql in ("SELECT 1 -- x */ ; DROP TABLE orders", "SELECT 1 /* x */ ; DROP TABLE orders */", "SELECT /* a */ 1 /* b */ -- c */ d"):
            with self.subTest(sql=sql):
                try:
                    v = ok(sql)
                except UnsafeSQL as e:
                    self.assertIn(e.code, {"malformed", "multiple_statements", "not_select", "forbidden_statement"})
                    continue
                self.assertNotIn("DROP", v.sql)
                self.assertNotIn("/*", v.sql)
                self.assertNotIn("*/", v.sql)
        self.assertEqual(ok("SELECT 1 -- x */ ; DROP TABLE orders").sql, "SELECT 1")  # the whole line is one inert comment

    def test_every_gold_query_is_valid_and_the_canonical_text_returns_the_same_result(self):
        for q in load_gold():
            with self.subTest(q.question_id):
                v = ok(q.gold_sql)
                self.assertEqual(v.tables, q.required_tables)  # the validator's table list agrees with the reviewed labels
                self.assertEqual(execute_readonly(q.gold_sql).rows, execute_readonly(v.sql).rows)

    def test_validation_is_deterministic(self):
        sql = "WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FROM t"
        self.assertEqual(ok(sql), ok(sql))


class Rejected(unittest.TestCase):
    def test_mutations_and_ddl(self):
        cases = ["INSERT INTO categories(category_name) VALUES ('x')", "UPDATE orders SET order_status = 'x'", "DELETE FROM orders", "DROP TABLE orders",
                 "CREATE TABLE t(x)", "CREATE TEMP TABLE t AS SELECT 1", "ALTER TABLE orders ADD COLUMN z", "REPLACE INTO categories VALUES (1, 'x', 'y')",
                 "WITH t AS (SELECT 1) DELETE FROM orders", "WITH t AS (SELECT 1) INSERT INTO categories SELECT 1, 2, 3", "SELECT * INTO t2 FROM orders"]
        for sql in cases:
            with self.subTest(sql=sql):
                self.assertIn(code(sql), {"not_select", "forbidden_statement", "unknown_table"})

    def test_other_statements_that_are_not_a_select(self):
        for sql in ("PRAGMA table_info(orders)", "ATTACH DATABASE 'x.db' AS x", "DETACH x", "VACUUM", "VACUUM INTO 'x.db'", "ANALYZE", "REINDEX", "BEGIN",
                    "COMMIT", "SAVEPOINT a", "EXPLAIN SELECT 1", "EXPLAIN QUERY PLAN SELECT 1", "VALUES (1)", "(SELECT 1)"):
            with self.subTest(sql=sql):
                self.assertIn(code(sql), {"not_select", "forbidden_statement"})

    def test_multiple_statements(self):
        for sql in ("SELECT 1; SELECT 2", "SELECT 1; DROP TABLE orders", "SELECT 1;; SELECT 2", "SELECT 1; -- c\nSELECT 2", "DELETE FROM orders; SELECT 1"):
            with self.subTest(sql=sql):
                self.assertEqual(code(sql), "multiple_statements")

    def test_empty_and_non_string_input(self):
        for sql in ("", "   ", "\n\t", ";", ";;", "-- only a comment", "/* only a comment */", None, 123, ["SELECT 1"]):
            with self.subTest(sql=sql):
                self.assertEqual(code(sql), "empty")

    def test_malformed_sql(self):
        for sql in ("SELEC 1", "SELECT FROM", "SELECT * FROM orders WHERE", "SELECT 'unterminated", "SELECT 1 /* unterminated", "SELECT (1", "SELECT * FROM"):
            with self.subTest(sql=sql):
                self.assertEqual(code(sql), "malformed")

    def test_unknown_internal_qualified_and_function_tables(self):
        for sql in ("SELECT * FROM no_such_table", "SELECT * FROM sqlite_master", "SELECT * FROM sqlite_schema", "SELECT * FROM sqlite_sequence",
                    "SELECT * FROM main.orders", "SELECT * FROM temp.sqlite_master", "SELECT * FROM pragma_table_info('orders')",
                    "SELECT * FROM json_each('[1,2]')", "SELECT * FROM orders JOIN sqlite_master ON 1 = 1",
                    "WITH x AS (SELECT * FROM sqlite_master) SELECT * FROM x", "SELECT * FROM (SELECT * FROM sqlite_master)"):
            with self.subTest(sql=sql):
                self.assertEqual(code(sql), "unknown_table")

    def test_a_cte_cannot_smuggle_a_forbidden_table_reference(self):
        # A CTE named like a real table is fine, but any reference inside it is still checked.
        self.assertEqual(code("WITH orders AS (SELECT * FROM sqlite_master) SELECT * FROM orders"), "unknown_table")

    def test_dangerous_functions(self):
        for sql in ("SELECT load_extension('x')", "SELECT readfile('x')", "SELECT writefile('x', 'y')", "SELECT zeroblob(1000000000)",
                    "SELECT randomblob(16)", "SELECT edit('x')", "SELECT fts3_tokenizer('x')", "SELECT LOAD_EXTENSION('x')",
                    "SELECT COUNT(*) FROM orders WHERE order_id IN (SELECT length(zeroblob(5)))"):
            with self.subTest(sql=sql):
                self.assertEqual(code(sql), "forbidden_function")

    def test_placeholders(self):
        for sql in ("SELECT ?", "SELECT :a", "SELECT * FROM orders WHERE order_id = ?", "SELECT @a"):
            with self.subTest(sql=sql):
                self.assertIn(code(sql), {"placeholder", "malformed"})

    def test_overlong_sql(self):
        self.assertEqual(code("SELECT 1 /*" + "x" * MAX_SQL_CHARS + "*/"), "too_long")

    def test_error_carries_a_stable_code_in_its_message(self):
        with self.assertRaisesRegex(UnsafeSQL, r"^multiple_statements:"):
            validate_select("SELECT 1; SELECT 2", TABLES)


if __name__ == "__main__":
    unittest.main()
