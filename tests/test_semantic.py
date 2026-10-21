"""Tests for text2sql.semantic. Run from the project root: python -B -m unittest -v tests.test_semantic"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from text2sql.execution import execute_readonly
from text2sql.gold import load_gold
from text2sql.retrieval import ContextTable, SchemaRetriever, load_index
from text2sql.semantic import LAYER_PATH, SemanticLayerError, build_prompt_context, load_layer, render_layer

docs, retriever, layer, gold = None, None, None, None


def setUpModule():
    global docs, retriever, layer, gold
    retriever = SchemaRetriever()
    docs = retriever.docs
    layer = load_layer(docs)
    gold = {q.question_id: q for q in load_gold()}


def entries():
    return [e for s in layer["sections"] for e in s["entries"]]


class Structure(unittest.TestCase):
    def test_layer_loads_with_unique_ids_and_sources(self):
        ids = [e["id"] for e in entries()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(e["source"].strip() and e["text"].strip() and e["refs"] for e in entries()))
        self.assertEqual([s["title"] for s in layer["sections"]][:2], ["Metrics", "Conventions"])

    def test_every_reference_exists_in_the_schema(self):
        known = {f"{d['table']}.{c['name']}" for d in docs for c in d["columns"]}
        for e in entries():
            with self.subTest(e["id"]):
                self.assertTrue(set(e["refs"]) <= known, set(e["refs"]) - known)

    def test_unknown_column_and_malformed_files_are_rejected(self):
        bad = copy.deepcopy(layer)
        bad["sections"][0]["entries"][0]["refs"].append("orders.no_such_column")
        cases = {"unknown column": json.dumps(bad), "invalid json": "{nope", "missing key": json.dumps({"version": 1}),
                 "wrong version": json.dumps({**layer, "version": 2})}
        for name, text in cases.items():
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / "layer.json"
                    p.write_text(text, encoding="utf-8")
                    with self.assertRaises(SemanticLayerError):
                        load_layer(docs, p)

    def test_duplicate_ids_are_rejected(self):
        dup = copy.deepcopy(layer)
        dup["sections"][1]["entries"].append(copy.deepcopy(dup["sections"][0]["entries"][0]))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "layer.json"
            p.write_text(json.dumps(dup), encoding="utf-8")
            with self.assertRaisesRegex(SemanticLayerError, "duplicate"):
                load_layer(docs, p)


class DocumentedConventions(unittest.TestCase):
    def test_every_data_claim_holds_in_the_database(self):
        checked = [e for e in entries() if "check_sql" in e]
        self.assertGreaterEqual(len(checked), 9)
        for e in checked:
            with self.subTest(e["id"]):
                self.assertEqual(execute_readonly(e["check_sql"]).rows, [(1,)])

    def test_each_gold_business_rule_is_encoded_in_the_layer_and_used_by_its_gold_sql(self):
        text = render_layer(layer)
        expected = {  # gold question -> (fragment in the layer text, the same rule as it appears in that question's gold SQL)
            "G16": ("SUM(orders.final_amount)", "SUM(final_amount)"),
            "G17": ("AVG(ratings.rating)", "AVG(g.rating)"),
            "G18": ("<> 'Cancelled'", "<> 'Cancelled'"),
            "G19": ("payments.payment_status = 'Success'", "p.payment_status = 'Success'"),
            "G20": ("orders.coupon_id IS NOT NULL", "o.coupon_id"),
        }
        for qid, (in_layer, in_sql) in expected.items():
            with self.subTest(qid):
                self.assertEqual(gold[qid].difficulty, "business_rule")
                self.assertIn(in_layer, text)
                self.assertIn(in_sql, gold[qid].gold_sql)

    def test_revenue_formula_reproduces_the_gold_answer(self):
        formula = execute_readonly("SELECT SUM(final_amount) FROM orders WHERE order_status = 'Delivered'").rows
        self.assertEqual(formula, execute_readonly(gold["G16"].gold_sql).rows)
        self.assertIn("SUM(orders.final_amount)", render_layer(layer))
        self.assertIn("Do not use orders.total_amount or payments.amount", render_layer(layer))

    def test_default_address_and_vegetarian_aliases_reproduce_gold_semantics(self):
        text = render_layer(layer)
        self.assertIn("is_default = 1", text)
        self.assertIn("menu_items.is_veg = 1", text)
        self.assertIn("is_veg = 1", gold["G10"].gold_sql)

    def test_unpopulated_column_is_documented(self):
        self.assertIn("menu_items.created_at is always NULL", render_layer(layer))

    def test_city_ambiguity_is_documented_for_all_three_tables(self):
        e = next(e for e in entries() if e["id"] == "city")
        self.assertEqual(sorted(e["refs"]), ["customer_addresses.city", "delivery_partners.city", "restaurants.city"])


class Rendering(unittest.TestCase):
    def test_render_is_deterministic_compact_and_clean(self):
        a, b = render_layer(layer), render_layer(load_layer(docs))
        self.assertEqual(a, b)
        self.assertLess(len(a), 3500)
        self.assertTrue(all(line == line.rstrip() for line in a.splitlines()))
        self.assertEqual(sum(l.startswith("- ") for l in a.splitlines()), len(entries()))

    def test_layer_file_is_the_one_in_the_data_dir(self):
        self.assertEqual(LAYER_PATH, Path(__file__).resolve().parent.parent / "data" / "semantic" / "semantic_layer.json")


class PromptContextTests(unittest.TestCase):
    CTX = [ContextTable("orders", "retrieved", 1, ()), ContextTable("order_items", "retrieved", 2, ()),
           ContextTable("customers", "fk_parent", None, ("orders",)), ContextTable("menu_items", "fk_parent", None, ("order_items",))]

    def test_sections_order_and_membership(self):
        pc = build_prompt_context(self.CTX, docs, layer)
        self.assertEqual((pc.tables, pc.n_retrieved, pc.n_fk_added), (("orders", "order_items", "customers", "menu_items"), 2, 2))
        i = {k: pc.text.index(k) for k in ("Tables most relevant", "Table: orders.", "Table: order_items.", "Related tables", "Table: customers.", "Table: menu_items.", "Business rules:")}
        self.assertEqual(list(i.values()), sorted(i.values()))
        for absent in ("Table: ratings.", "Table: payments.", "Table: coupons."):
            self.assertNotIn(absent, pc.text)
        self.assertIn(render_layer(layer), pc.text)

    def test_no_related_section_without_fk_added_tables(self):
        pc = build_prompt_context(self.CTX[:2], docs, layer)
        self.assertNotIn("Related tables", pc.text)
        self.assertEqual(pc.n_fk_added, 0)

    def test_deterministic_and_does_not_touch_retrieval_rankings(self):
        q = gold["G13"].question
        base = [(r.table, r.rank) for r in retriever.retrieve(q, 3)]
        ctx = retriever.retrieve_context(q)
        a, b = build_prompt_context(ctx, docs, layer), build_prompt_context(retriever.retrieve_context(q), docs, layer)
        self.assertEqual(a, b)
        self.assertEqual([(c.table, c.rank) for c in ctx if c.kind == "retrieved"], base)
        self.assertEqual(list(a.tables[:3]), [t for t, _ in base])
        self.assertTrue({"menu_items", "restaurants"} <= set(a.tables))

    def test_every_gold_question_gets_a_prompt_that_contains_its_required_tables(self):
        uncovered = set()
        for qid, q in gold.items():
            with self.subTest(qid):
                pc = build_prompt_context(retriever.retrieve_context(q.question), docs, layer)
                if not set(q.required_tables) <= set(pc.tables):
                    uncovered.add(qid)
                self.assertIn("Business rules:", pc.text)
        # G01-G20 are fully covered. Of G21-G40 exactly one question is not: G28 retrieves no orders table. It is a retrieval miss on a valid
        # question, recorded here so any change in coverage is noticed; the gold question is not altered to fit retrieval.
        self.assertEqual(uncovered, {"G28"})


if __name__ == "__main__":
    unittest.main()
