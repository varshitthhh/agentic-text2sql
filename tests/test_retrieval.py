"""Minimal tests for text2sql.retrieval. Run from the project root: python -B -m unittest -v tests.test_retrieval

These are smoke/regression tests on known table/column queries, not an evaluation of retrieval quality.
"""
import hashlib
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from text2sql.retrieval import (DB_PATH, INDEX_DIR, IndexProvenanceError, SchemaRetriever, load_index, rrf_fuse, tokenize)

ARTIFACTS = [DB_PATH, *(INDEX_DIR / n for n in ("schema_docs.jsonl", "schema_embeddings.npy", "index_meta.json"))]
retriever = None
hashes_before = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setUpModule():
    global retriever
    hashes_before.update({str(p): sha(p) for p in ARTIFACTS})
    retriever = SchemaRetriever()


def tearDownModule():
    changed = [p for p in ARTIFACTS if sha(p) != hashes_before[str(p)]]
    assert not changed, f"tests modified protected artifacts: {changed}"


def tables(query, top_k=5):
    return [r.table for r in retriever.retrieve(query, top_k)]


class TokenizerAndFusion(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Which customers placed the most orders?"), ["customer", "placed", "most", "order"])
        self.assertEqual(tokenize("order_status"), ["order_status", "order", "statu"])  # identifier + parts, plural folded
        self.assertEqual(tokenize("categories"), ["category"])
        self.assertEqual(tokenize("what is the"), [])  # stop words only

    def test_rrf_fuse_values(self):
        s = rrf_fuse([[0, 1, 2], [1, 0]], k=60)
        self.assertAlmostEqual(s[0], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(s[1], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(s[2], 1 / 63)
        self.assertGreater(s[0], s[2])
        self.assertEqual(rrf_fuse([[], []]), {})

    def test_rrf_item_in_one_ranking_scores_lower_than_item_in_both(self):
        s = rrf_fuse([[5, 6], [6]], k=60)
        self.assertGreater(s[6], s[5])


class KnownQueries(unittest.TestCase):
    def test_required_tables_in_top3(self):
        cases = {
            "Which customers placed the most orders?": {"customers", "orders"},
            "What is the average rating of each restaurant?": {"ratings", "restaurants"},
            "How many menu items are there in each category?": {"menu_items", "categories"},
            # order_items.order_id identifies the order, so `orders` itself is not required (it ranks 5th); item names need menu_items.
            "Show the items in each order": {"order_items", "menu_items"},
        }
        for query, required in cases.items():
            with self.subTest(query=query):
                self.assertTrue(required <= set(tables(query, 3)), tables(query, 3))

    def test_single_table_top_rank(self):
        cases = {
            "delivery partner vehicle type": "delivery_partners",
            "coupon code and discount percent": "coupons",
            "payment method used for each order": "payments",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(tables(query, 1), [expected])

    def test_column_name_query(self):
        self.assertIn("orders", tables("order_status", 2))

    def test_sample_value_query(self):
        # "Bengaluru" occurs only as a sample value of customer_addresses.city and restaurants.city.
        top = retriever.retrieve("orders placed in Bengaluru", 3)
        self.assertTrue({"customer_addresses", "restaurants"} & {r.table for r in top})
        bengaluru = [r for r in top if "bengaluru" in r.matched_terms]
        self.assertTrue(bengaluru, "expected the sample-value term to be reported as matched")

    def test_result_fields_and_explanation(self):
        top = retriever.retrieve("customer email address", 5)
        self.assertEqual([r.rank for r in top], [1, 2, 3, 4, 5])
        self.assertEqual(len({r.table for r in top}), 5)
        self.assertTrue(all(top[i].rrf_score >= top[i + 1].rrf_score for i in range(4)))
        first = top[0]
        self.assertIn("dense #", first.explanation)
        self.assertIn(first.table, [d["table"] for d in retriever.docs])
        self.assertTrue(first.matched_terms)  # lexical terms that hit this table
        self.assertGreater(first.dense_score, 0.0)

    def test_query_with_no_lexical_match_still_ranks_by_dense(self):
        top = retriever.retrieve("what is the", 5)  # only stop words
        self.assertEqual(len(top), 5)
        self.assertTrue(all(r.bm25_rank is None and r.matched_terms == () for r in top))
        self.assertTrue(all("no lexical match" in r.explanation for r in top))

    def test_top_k_bounds(self):
        self.assertEqual(len(retriever.retrieve("customers", 12)), 12)
        self.assertEqual(len(retriever.retrieve("customers", 1)), 1)


class Determinism(unittest.TestCase):
    def test_same_query_same_result(self):
        a = retriever.retrieve("Total revenue from delivered orders by payment method", 12)
        b = retriever.retrieve("Total revenue from delivered orders by payment method", 12)
        self.assertEqual(a, b)

    def test_fresh_retriever_same_result(self):
        other = SchemaRetriever()
        q = "average rating of each restaurant"
        self.assertEqual(retriever.retrieve(q, 12), other.retrieve(q, 12))

    def test_query_instruction_is_configurable_and_deterministic(self):
        plain = SchemaRetriever(use_query_instruction=False)
        q = "customer email address"
        self.assertEqual(plain.retrieve(q, 12), plain.retrieve(q, 12))


class BadInput(unittest.TestCase):
    def test_empty_or_missing_queries(self):
        for bad in (None, "", "   ", "\n\t", "???", "!!! ... ---", 123, ["customers"]):
            with self.subTest(query=bad):
                with self.assertRaises(ValueError):
                    retriever.retrieve(bad)

    def test_bad_top_k(self):
        for bad in (0, -1, 13, True, "5", 2.5, None):
            with self.subTest(top_k=bad):
                with self.assertRaises(ValueError):
                    retriever.retrieve("customers", bad)


class Provenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "index"
        shutil.copytree(INDEX_DIR, self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_index_loads(self):
        docs, emb, meta = load_index(self.dir, DB_PATH)
        self.assertEqual((len(docs), emb.shape, str(emb.dtype)), (12, (12, 384), "float32"))
        self.assertEqual(meta["model"]["id"], "BAAI/bge-small-en-v1.5")

    def test_altered_documents_rejected(self):
        p = self.dir / "schema_docs.jsonl"
        p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaisesRegex(IndexProvenanceError, "schema_docs.jsonl"):
            load_index(self.dir, DB_PATH)

    def test_altered_embeddings_rejected(self):
        p = self.dir / "schema_embeddings.npy"
        data = bytearray(p.read_bytes())
        data[-1] ^= 0xFF
        p.write_bytes(bytes(data))
        with self.assertRaisesRegex(IndexProvenanceError, "schema_embeddings.npy"):
            load_index(self.dir, DB_PATH)

    def test_changed_database_rejected_as_stale(self):
        other_db = Path(self.tmp.name) / "other.sqlite"
        con = sqlite3.connect(other_db)
        con.execute("CREATE TABLE t (x)")
        con.commit()
        con.close()
        with self.assertRaisesRegex(IndexProvenanceError, "database changed"):
            load_index(self.dir, other_db)

    def test_missing_file_rejected(self):
        (self.dir / "index_meta.json").unlink()
        with self.assertRaisesRegex(IndexProvenanceError, "missing file"):
            load_index(self.dir, DB_PATH)

    def test_different_model_revision_rejected(self):
        p = self.dir / "index_meta.json"
        meta = json.loads(p.read_text(encoding="ascii"))
        meta["model"]["revision"] = "0" * 40
        p.write_text(json.dumps(meta), encoding="ascii")
        with self.assertRaisesRegex(IndexProvenanceError, "different embedding model"):
            load_index(self.dir, DB_PATH)

    def test_damaged_meta_rejected(self):
        (self.dir / "index_meta.json").write_text("{not json", encoding="ascii")
        with self.assertRaisesRegex(IndexProvenanceError, "unreadable"):
            load_index(self.dir, DB_PATH)

    def test_retriever_refuses_a_tampered_index(self):
        p = self.dir / "schema_docs.jsonl"
        p.write_bytes(p.read_bytes() + b"\n")
        with self.assertRaises(IndexProvenanceError):
            SchemaRetriever(self.dir, DB_PATH)


if __name__ == "__main__":
    unittest.main()
