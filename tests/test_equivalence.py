"""Tests for text2sql.equivalence. Run from the project root: python -B -m unittest -v tests.test_equivalence"""
import unittest

from text2sql.equivalence import ABS_TOL, REL_TOL, Equivalence, results_equivalent, rows_equal, values_equal
from text2sql.execution import QueryResult, execute_readonly
from text2sql.gold import load_gold


def res(rows, cols=("a",), truncated=False):
    return QueryResult(tuple(cols), list(rows), truncated, 0.0)


def eq(cand, gold, ordered=False):
    return results_equivalent(cand, gold, ordered)


class Values(unittest.TestCase):
    def test_constants_are_the_documented_ones(self):
        self.assertEqual((REL_TOL, ABS_TOL), (1e-9, 1e-6))

    def test_null_equals_only_null(self):
        self.assertTrue(values_equal(None, None))
        for other in (0, 0.0, "", "NULL", "None", False):
            with self.subTest(other=other):
                self.assertFalse(values_equal(None, other))
                self.assertFalse(values_equal(other, None))

    def test_integers_exact_and_int_float_compare_numerically(self):
        self.assertTrue(values_equal(3, 3))
        self.assertFalse(values_equal(3, 4))
        self.assertTrue(values_equal(3, 3.0))
        self.assertTrue(values_equal(3.0, 3))

    def test_float_tolerance_forgives_summation_noise_but_not_cent_errors(self):
        self.assertTrue(values_equal(122540.76, 122540.76 + 2.1e-10))  # measured worst-case summation-order noise in the gold results
        self.assertTrue(values_equal(0.1 + 0.2, 0.3))
        self.assertTrue(values_equal(1_000_000.0, 1_000_000.0005))
        self.assertFalse(values_equal(122540.76, 122540.77))   # a one-cent error
        self.assertFalse(values_equal(4.17, 4.1667))            # a missing ROUND(.., 2) (gold G12/G17 differ from the raw average by up to 0.0045)
        self.assertFalse(values_equal(1_000_000.0, 1_000_000.01))
        self.assertFalse(values_equal(4.2, 4.0))

    def test_strings_exact_case_sensitive_and_never_numbers(self):
        self.assertTrue(values_equal("Delivered", "Delivered"))
        self.assertFalse(values_equal("Delivered", "delivered"))
        self.assertFalse(values_equal("5", 5))
        self.assertFalse(values_equal(5, "5"))
        self.assertFalse(values_equal("", 0))

    def test_rows(self):
        self.assertTrue(rows_equal((1, "a", None), (1, "a", None)))
        self.assertFalse(rows_equal((1, "a"), (1, "a", None)))
        self.assertFalse(rows_equal((1, "a"), ("a", 1)))  # column order matters


class Results(unittest.TestCase):
    def test_identical_and_alias_independent(self):
        self.assertEqual(eq(res([(1, "x")], ("p", "q")), res([(1, "x")], ("total", "name"))), Equivalence(True, "equal"))

    def test_column_count_and_order_of_columns(self):
        self.assertEqual(eq(res([(1, 2)], ("a", "b")), res([(1,)])).reason, "column_count")
        self.assertEqual(eq(res([("x", 1)], ("a", "b")), res([(1, "x")], ("a", "b"))).reason, "values")  # reordered columns are a mismatch

    def test_row_count(self):
        self.assertEqual(eq(res([(1,)]), res([(1,), (2,)])).reason, "row_count")
        self.assertEqual(eq(res([]), res([(1,)])).reason, "row_count")
        self.assertTrue(eq(res([]), res([])).ok)  # both empty

    def test_unordered_comparison_is_a_multiset(self):
        self.assertTrue(eq(res([(2,), (1,), (3,)]), res([(1,), (2,), (3,)])).ok)
        self.assertFalse(eq(res([(1,), (1,), (2,)]), res([(1,), (2,), (2,)])).ok)       # same size, different multiplicities
        self.assertEqual(eq(res([(1,), (1,), (2,)]), res([(1,), (2,), (2,)])).reason, "values")
        self.assertFalse(eq(res([(1,), (2,)]), res([(1,), (2,), (2,)])).ok)              # duplicates are not collapsed

    def test_unordered_comparison_applies_the_float_tolerance(self):
        self.assertTrue(eq(res([("a", 0.1 + 0.2), ("b", 1.0)]), res([("b", 1.0), ("a", 0.3)])).ok)

    def test_ordered_comparison_respects_order(self):
        gold = res([(3,), (2,), (1,)])
        self.assertTrue(eq(res([(3,), (2,), (1,)]), gold, ordered=True).ok)
        wrong = eq(res([(1,), (2,), (3,)]), gold, ordered=True)
        self.assertEqual((wrong.ok, wrong.reason), (False, "order"))
        self.assertEqual(eq(res([(3,), (2,), (9,)]), gold, ordered=True).reason, "values")
        self.assertTrue(eq(res([(1,), (2,), (3,)]), gold, ordered=False).ok)

    def test_nulls_in_results(self):
        self.assertTrue(eq(res([(None,), (1,)]), res([(1,), (None,)])).ok)
        self.assertFalse(eq(res([(0,), (1,)]), res([(None,), (1,)])).ok)
        self.assertFalse(eq(res([("",)]), res([(None,)])).ok)

    def test_truncated_results_never_match(self):
        self.assertEqual(eq(res([(1,)], truncated=True), res([(1,)])).reason, "truncated")
        self.assertEqual(eq(res([(1,)]), res([(1,)], truncated=True)).reason, "truncated")


class AgainstTheGoldSet(unittest.TestCase):
    def test_every_gold_result_matches_itself_with_the_gold_ordering_flag(self):
        for q in load_gold():
            with self.subTest(q.question_id):
                r = execute_readonly(q.gold_sql)
                self.assertTrue(results_equivalent(r, execute_readonly(q.gold_sql), q.ordered).ok)

    def test_a_one_cent_or_missing_round_error_is_caught_on_real_gold_results(self):
        gold = {q.question_id: q for q in load_gold()}
        g16 = execute_readonly(gold["G16"].gold_sql)
        off_by_a_cent = res([(g16.rows[0][0] + 0.01,)])
        self.assertFalse(eq(off_by_a_cent, g16).ok)
        g12 = execute_readonly(gold["G12"].gold_sql)
        unrounded = execute_readonly("SELECT c.category_name, AVG(m.price) FROM menu_items m JOIN categories c ON c.category_id = m.category_id GROUP BY c.category_id")
        self.assertFalse(eq(unrounded, g12).ok)  # AVG without ROUND differs from the rounded gold
        self.assertTrue(eq(execute_readonly(gold["G12"].gold_sql), g12).ok)

    def test_ordered_gold_questions_fail_when_reversed(self):
        q = next(q for q in load_gold() if q.question_id == "G02")
        r = execute_readonly(q.gold_sql)
        self.assertTrue(q.ordered)
        self.assertEqual(eq(res(list(reversed(r.rows)), r.columns), r, ordered=True).reason, "order")


if __name__ == "__main__":
    unittest.main()
