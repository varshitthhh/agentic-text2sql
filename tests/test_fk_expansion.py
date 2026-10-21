"""Tests for FK parent expansion. Run from the project root: python -B -m unittest -v tests.test_fk_expansion"""
import unittest

from text2sql.retrieval import CONTEXT_TOP_K, ContextTable, RetrievedTable, SchemaRetriever, expand_parents

retriever = None


def setUpModule():
    global retriever
    retriever = SchemaRetriever()


def picked(*names):
    """Retrieved tables in the given order, ranks 1..n (scores are irrelevant to expansion)."""
    return [RetrievedTable(n, i, 0.0, None, 0.0, i, 0.0, (), "") for i, n in enumerate(names, start=1)]


def names(context):
    return [c.table for c in context]


class ExpandParentsUnit(unittest.TestCase):
    PARENTS = {"a": ("p", "q"), "b": ("q", "r"), "c": ("a",), "p": (), "q": (), "r": ()}

    def test_table_without_parents_adds_nothing(self):
        ctx = expand_parents(picked("p"), self.PARENTS)
        self.assertEqual(ctx, [ContextTable("p", "retrieved", 1, ())])

    def test_unknown_table_has_no_parents(self):
        self.assertEqual(names(expand_parents(picked("zzz"), self.PARENTS)), ["zzz"])

    def test_parents_follow_retrieved_tables_and_keep_original_ranks(self):
        ctx = expand_parents(picked("a"), self.PARENTS)
        self.assertEqual(ctx, [ContextTable("a", "retrieved", 1, ()), ContextTable("p", "fk_parent", None, ("a",)),
                               ContextTable("q", "fk_parent", None, ("a",))])

    def test_retrieved_tables_stay_in_rank_order_before_all_added_tables(self):
        ctx = expand_parents(picked("b", "a"), self.PARENTS)
        self.assertEqual(names(ctx), ["b", "a", "q", "r", "p"])  # q, r from b (rank 1) first, then p from a; q not repeated
        self.assertEqual([c.rank for c in ctx], [1, 2, None, None, None])
        self.assertEqual([c.kind for c in ctx], ["retrieved", "retrieved", "fk_parent", "fk_parent", "fk_parent"])

    def test_shared_parent_is_added_once_and_records_every_table_that_needs_it(self):
        ctx = expand_parents(picked("a", "b"), self.PARENTS)
        q = [c for c in ctx if c.table == "q"]
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0].added_for, ("a", "b"))
        self.assertEqual(len(names(ctx)), len(set(names(ctx))))

    def test_a_retrieved_parent_is_not_duplicated_or_relabelled(self):
        ctx = expand_parents(picked("c", "a"), self.PARENTS)  # a is both retrieved and the parent of c
        self.assertEqual(names(ctx).count("a"), 1)
        self.assertEqual(next(c for c in ctx if c.table == "a"), ContextTable("a", "retrieved", 2, ()))

    def test_only_one_level_and_no_children(self):
        ctx = expand_parents(picked("c"), self.PARENTS)  # c -> a, but a's own parents (p, q) must not follow
        self.assertEqual(names(ctx), ["c", "a"])
        self.assertNotIn("c", names(expand_parents(picked("a"), self.PARENTS)))  # c is a child of a: never added

    def test_deterministic(self):
        first = expand_parents(picked("b", "a", "c"), self.PARENTS)
        for _ in range(5):
            self.assertEqual(expand_parents(picked("b", "a", "c"), self.PARENTS), first)

    def test_empty_input(self):
        self.assertEqual(expand_parents([], self.PARENTS), [])


class RealForeignKeys(unittest.TestCase):
    def test_parent_map_matches_the_schema_in_fk_column_order(self):
        self.assertEqual(retriever._fk_parents, {
            "customers": (), "customer_addresses": ("customers",), "restaurants": (), "categories": (),
            "menu_items": ("restaurants", "categories"), "delivery_partners": (), "coupons": (),
            "orders": ("customers", "restaurants", "customer_addresses", "delivery_partners", "coupons"),
            "order_items": ("orders", "menu_items"), "payments": ("orders",),
            "ratings": ("orders", "customers", "restaurants"), "order_tracking": ("orders",),
        })

    def test_real_tables_without_parents(self):
        for t in ("customers", "restaurants", "categories", "delivery_partners", "coupons"):
            with self.subTest(t):
                self.assertEqual(names(expand_parents(picked(t), retriever._fk_parents)), [t])

    def test_bridge_table_adds_its_two_parents_but_not_their_parents(self):
        ctx = expand_parents(picked("order_items"), retriever._fk_parents)
        self.assertEqual(names(ctx), ["order_items", "orders", "menu_items"])

    def test_children_are_never_added(self):
        self.assertEqual(names(expand_parents(picked("restaurants"), retriever._fk_parents)), ["restaurants"])  # menu_items/orders/ratings reference it


class EndToEnd(unittest.TestCase):
    def test_context_keeps_the_retrievers_top_k_and_ranks(self):
        q = "How many menu items does each restaurant have? Show the restaurant name and the count."
        base = retriever.retrieve(q, CONTEXT_TOP_K)
        ctx = retriever.retrieve_context(q)
        self.assertEqual(CONTEXT_TOP_K, 3)
        self.assertEqual([(c.table, c.rank) for c in ctx if c.kind == "retrieved"], [(r.table, r.rank) for r in base])
        self.assertTrue(all(c.rank is None and c.added_for for c in ctx if c.kind == "fk_parent"))
        self.assertEqual(len(names(ctx)), len(set(names(ctx))))
        self.assertTrue({"menu_items", "restaurants"} <= set(names(ctx)))

    def test_context_is_deterministic_and_validates_top_k(self):
        q = "What is the average customer rating of each restaurant?"
        self.assertEqual(retriever.retrieve_context(q), retriever.retrieve_context(q))
        with self.assertRaises(ValueError):
            retriever.retrieve_context(q, 0)
        with self.assertRaises(ValueError):
            retriever.retrieve_context("", 3)


if __name__ == "__main__":
    unittest.main()
