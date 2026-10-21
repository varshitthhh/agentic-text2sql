"""Tests for text2sql.eval_runner with a scripted fake model (no Qwen). Run: python -B -m unittest -v tests.test_eval_runner"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from text2sql.agent import Text2SQLAgent
from text2sql.eval_runner import (EvalRunnerError, FullSchemaRetriever, append_record, check_output, format_summary, load_run, main, run_evaluation,
                                  select_questions, summarize)
from text2sql.generation import ModelReply
from text2sql.gold import load_gold
from text2sql.retrieval import ContextTable, load_index
from text2sql.semantic import load_layer

docs, layer, gold = None, None, None
RANKING = ["orders", "customers", "payments", "menu_items", "ratings", "restaurants", "categories", "coupons", "customer_addresses",
           "delivery_partners", "order_items", "order_tracking"]


def setUpModule():
    global docs, layer, gold
    docs = load_index()[0]
    layer = load_layer(docs)
    gold = load_gold()


def ready(sql):
    return json.dumps({"status": "ready", "sql": sql, "message": "ok"})


def status(s, msg="m"):
    return json.dumps({"status": s, "sql": "", "message": msg})


class FakeModel:
    def __init__(self, *replies, tokens=(100, 20)):
        self.replies, self.calls, self.tokens = list(replies), [], tokens

    def chat(self, messages):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("unexpected model call")
        t = self.tokens(len(self.calls)) if callable(self.tokens) else self.tokens
        return ModelReply(self.replies.pop(0), *t)


class FakeRetriever:
    """Fixed ranking; context = top 3 plus one FK-added table (restaurants)."""

    docs = None
    meta = {"sources": {"database": {"sha256": "x" * 64}}}
    rrf_k, use_query_instruction = 60, True

    def __init__(self):
        self.docs = docs

    def retrieve(self, question, top_k=5):
        return [SimpleNamespace(table=t) for t in RANKING[:top_k]]

    def retrieve_context(self, question):
        return [ContextTable(t, "retrieved", i, ()) for i, t in enumerate(RANKING[:3], 1)] + [ContextTable("restaurants", "fk_parent", None, ("orders",))]


def agent_with(*replies, **kw):
    tokens = kw.pop("tokens", (100, 20))
    return Text2SQLAgent(FakeRetriever(), FakeModel(*replies, tokens=tokens), layer, **kw)


G = {q.question_id: q for q in load_gold()} if False else None
IDS = ["G01", "G04", "G03", "G05", "G02", "G16", "G17"]
SCRIPT = [
    ready("SELECT COUNT(*) FROM customers"),                    # G01 correct
    ready("SELECT COUNT(*) FROM orders"),                        # G04 valid and executes, but wrong (not equivalent)
    status("needs_clarification", "Which status?"),              # G03 clarification
    status("unsupported", "No such data"),                       # G05 unsupported
    "garbage",                                                   # G02 malformed
    ready("DELETE FROM orders"), ready("SELECT SUM(final_amount) FROM orders WHERE order_status = 'Delivered'"),  # G16 unsafe, repaired, correct
    ready("SELECT a FROM orders"), ready("SELECT b FROM orders"),  # G17 fails twice
]


def run_script(tmp, run="r1", tokens=(100, 20), ids=IDS, script=SCRIPT):
    out = Path(tmp) / "res.jsonl"
    agent = agent_with(*script, tokens=tokens)
    records = run_evaluation(agent, agent.retriever, select_questions(gold, ids), run, out)
    return agent, out, records


class Selection(unittest.TestCase):
    def test_order_is_preserved_and_bad_selections_are_errors(self):
        self.assertEqual([q.question_id for q in select_questions(gold, ["G16", "G01", "G04"])], ["G16", "G01", "G04"])
        for bad in ([], ["G99"], ["G01", "G01"], ["g01"], ["G01", "NOPE"]):
            with self.subTest(ids=bad):
                with self.assertRaises(EvalRunnerError):
                    select_questions(gold, bad)

    def test_cli_requires_an_explicit_selection_and_never_defaults_to_the_full_run(self):
        for argv in (["run1"], ["run1", "--ids"], ["run1", "--ids", "G01", "--all"]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    main(argv)

    def test_cli_rejects_unknown_ids_and_used_run_names_before_loading_any_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.jsonl"
            with self.assertRaises(EvalRunnerError):
                main(["run1", "--ids", "G99", "--out", str(out)])
            out.write_text(json.dumps({"run": "taken"}) + "\n", encoding="utf-8")
            with self.assertRaises(EvalRunnerError):
                main(["taken", "--ids", "G01", "--out", str(out)])


class OutputSafety(unittest.TestCase):
    def test_existing_run_name_is_refused_and_the_file_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out, _ = run_script(tmp, "r1", ids=["G01"], script=SCRIPT[:1])
            before = out.read_bytes()
            agent = agent_with()  # no replies: any model call would fail the test
            with self.assertRaises(EvalRunnerError):
                run_evaluation(agent, agent.retriever, select_questions(gold, ["G01"]), "r1", out)
            self.assertEqual(out.read_bytes(), before)
            self.assertEqual(agent.model.calls, [])

    def test_a_new_run_appends_and_never_rewrites_existing_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out, first = run_script(tmp, "r1", ids=["G01"], script=SCRIPT[:1])
            before = out.read_bytes()
            agent = agent_with(SCRIPT[1])
            run_evaluation(agent, agent.retriever, select_questions(gold, ["G04"]), "r2", out)
            after = out.read_bytes()
            self.assertTrue(after.startswith(before))
            self.assertEqual(len(after.splitlines()), 2)
            self.assertEqual((len(load_run(out, "r1")), len(load_run(out, "r2"))), (1, 1))

    def test_missing_directories_are_created_and_empty_run_names_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a" / "b" / "r.jsonl"
            append_record(out, {"run": "x"})
            self.assertTrue(out.exists())
            for bad in ("", "   "):
                with self.assertRaises(EvalRunnerError):
                    check_output(out, bad)


class RecordsAndSerialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.agent, cls.out, cls.records = run_script(cls.tmp.name)
        cls.by_id = {r["question_id"]: r for r in cls.records}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_jsonl_round_trips_exactly(self):
        lines = self.out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 7)
        self.assertTrue(all(json.loads(l)["run"] == "r1" and json.loads(l)["stage"] == "end_to_end" for l in lines))
        self.assertEqual(load_run(self.out, "r1"), self.records)
        self.assertEqual(summarize(load_run(self.out, "r1")), summarize(self.records))

    def test_required_fields_are_present_for_every_question(self):
        keys = {"run", "stage", "config", "question_id", "difficulty", "question", "required_tables", "retrieved_top5", "retrieved_tables",
                "fk_added_tables", "context_tables", "recall@3", "recall@5", "context_coverage", "outcome", "failure", "final_sql",
                "final_equivalence", "ex", "valid_sql", "executed_ok", "repair_attempted", "repair_succeeded", "first_attempt", "attempts",
                "retrieval_s", "total_s", "gold"}
        for r in self.records:
            with self.subTest(r["question_id"]):
                self.assertTrue(keys <= set(r), keys - set(r))
                self.assertEqual({"index", "sql", "validation_ok", "execution_ok", "failure", "prompt_tokens", "completion_tokens", "model_s", "exec_s",
                                  "elapsed_at_end_s", "equivalence"} <= set(r["first_attempt"]), True)

    def test_config_is_recorded(self):
        c = self.records[0]["config"]
        self.assertEqual((c["max_retries"], c["temperature"], c["seed"], c["context_base_k"], c["expansion"]), (1, 0, 0, 3, "direct_fk_parents_only"))
        self.assertEqual(len(c["semantic_layer_sha256"]), 64)

    def test_correct_answer(self):
        r = self.by_id["G01"]
        self.assertEqual((r["outcome"], r["ex"], r["valid_sql"], r["executed_ok"], r["repair_attempted"], r["n_attempts"]), ("answered", True, True, True, False, 1))
        self.assertEqual(r["final_equivalence"], {"ok": True, "reason": "equal"})
        self.assertEqual(r["gold"], {"sql": "SELECT COUNT(*) FROM customers", "ordered": False, "n_rows": 1, "n_columns": 1, "columns": ["COUNT(*)"], "truncated": False})

    def test_valid_but_wrong_answer_stays_a_miss(self):
        r = self.by_id["G04"]
        self.assertEqual((r["outcome"], r["ex"], r["valid_sql"], r["executed_ok"]), ("answered", False, True, True))
        self.assertEqual(r["final_equivalence"], {"ok": False, "reason": "values"})
        self.assertFalse(r["repair_attempted"])

    def test_clarification_unsupported_and_malformed_stay_in_the_denominators(self):
        for qid, outcome in (("G03", "needs_clarification"), ("G05", "unsupported"), ("G02", "failed")):
            with self.subTest(qid):
                r = self.by_id[qid]
                self.assertEqual((r["outcome"], r["ex"], r["valid_sql"], r["executed_ok"], r["final_sql"], r["n_attempts"]), (outcome, False, False, False, None, 1))
        self.assertEqual(self.by_id["G03"]["message"], "Which status?")
        self.assertEqual((self.by_id["G02"]["failure"]["stage"], self.by_id["G02"]["failure"]["code"]), ("generation", "invalid_json"))

    def test_repaired_question_keeps_first_attempt_data(self):
        r = self.by_id["G16"]
        self.assertEqual((r["outcome"], r["ex"], r["repair_attempted"], r["repair_succeeded"], r["n_attempts"]), ("answered", True, True, True, 2))
        fa = r["first_attempt"]
        self.assertEqual((fa["sql"], fa["validation_ok"], fa["execution_ok"], fa["failure"]["stage"], fa["failure"]["code"]), ("DELETE FROM orders", False, None, "validation", "not_select"))
        self.assertEqual((r["first_attempt_ex"], r["first_attempt_valid_sql"]), (False, False))
        self.assertGreater(r["attempts"][1]["elapsed_at_end_s"], fa["elapsed_at_end_s"])
        self.assertLess(fa["elapsed_at_end_s"], r["total_s"])

    def test_failed_after_repair(self):
        r = self.by_id["G17"]
        self.assertEqual((r["outcome"], r["ex"], r["valid_sql"], r["executed_ok"], r["repair_attempted"], r["repair_succeeded"]), ("failed", False, True, False, True, False))
        self.assertEqual((r["failure"]["stage"], r["failure"]["code"], r["final_sql"]), ("execution", "sqlite_error", "SELECT b FROM orders"))
        self.assertEqual((r["first_attempt_valid_sql"], r["first_attempt_ex"]), (True, False))


class MetricDenominators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.records = run_script(cls.tmp.name)[2]
        cls.s = summarize(cls.records)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_rate_uses_all_seven_questions(self):
        s = self.s
        self.assertEqual(s["n_questions"], 7)
        for key, count in (("ex", 2), ("valid_sql", 4), ("executed_ok", 3), ("repair_attempted", 2)):
            with self.subTest(key):
                self.assertEqual((s[key]["count"], s[key]["denominator"]), (count, 7))
                self.assertAlmostEqual(s[key]["rate"], count / 7)

    def test_repair_success_is_over_repaired_questions_only(self):
        self.assertEqual((self.s["repair_succeeded"]["count"], self.s["repair_succeeded"]["denominator"]), (1, 2))

    def test_outcomes_and_failure_stages_account_for_everything(self):
        self.assertEqual(self.s["outcomes"], {"answered": 3, "needs_clarification": 1, "unsupported": 1, "failed": 2})
        self.assertEqual(sum(self.s["outcomes"].values()), 7)
        self.assertEqual(self.s["failure_stages"], {"execution/sqlite_error": 1, "generation/invalid_json": 1})

    def test_recall_and_expanded_coverage_are_separate_measures(self):
        s = self.s
        self.assertAlmostEqual(s["recall@3"]["macro"], 3 / 7)
        self.assertAlmostEqual(s["recall@5"]["macro"], 5.5 / 7)
        self.assertAlmostEqual(s["context_coverage"]["macro"], 3.5 / 7)
        self.assertEqual((s["recall@3"]["complete"], s["recall@5"]["complete"], s["context_coverage"]["complete"]), (3, 5, 3))
        self.assertEqual(s["context_coverage"]["mean_context_tables"], 4)
        self.assertNotEqual(s["recall@3"]["macro"], s["context_coverage"]["macro"])

    def test_derived_zero_retry_uses_first_attempts_only(self):
        z = self.s["derived_zero_retry"]
        self.assertEqual((z["ex"]["count"], z["ex"]["denominator"], z["valid_sql"]["count"], z["valid_sql"]["denominator"]), (1, 7, 3, 7))

    def test_latency_and_tokens(self):
        lat, tok = self.s["latency_s"], self.s["tokens"]
        self.assertEqual((lat["attempts"], lat["exec_attempts"]), (9, 6))  # 9 attempts; G03, G05 and G02 proposed no SQL, so 9 - 3 went through validation/execution
        for k in ("total_median", "total_max", "retrieval_median", "first_attempt_elapsed_median", "model_per_attempt_median", "exec_per_attempt_median"):
            self.assertGreaterEqual(lat[k], 0)
        self.assertGreaterEqual(lat["total_max"], lat["total_median"])
        self.assertEqual((tok["attempts"], tok["attempts_with_counts"], tok["prompt_total"], tok["completion_total"]), (9, 9, 900, 180))
        self.assertEqual((tok["prompt_mean"], tok["completion_mean"]), (100, 20))

    def test_summary_text_states_the_denominators(self):
        text = format_summary(self.s)
        for fragment in ("N = 7", "EX                  2/7", "valid-SQL rate      4/7", "executed-OK rate    3/7", "repair succeeded    1/2", "Recall@3", "context coverage"):
            self.assertIn(fragment, text)


class EdgeCases(unittest.TestCase):
    def test_missing_token_counts_do_not_break_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = summarize(run_script(tmp, ids=["G01", "G04"], script=SCRIPT[:2], tokens=(None, None))[2])
        self.assertEqual((s["tokens"]["attempts"], s["tokens"]["attempts_with_counts"]), (2, 0))
        self.assertEqual((s["tokens"]["prompt_total"], s["tokens"]["prompt_mean"], s["tokens"]["completion_total"]), (None, None, None))
        self.assertIn("0/2 attempts have counts", format_summary(s))
        self.assertIn("prompt total n/a", format_summary(s))

    def test_partial_token_counts_only_count_complete_pairs(self):
        tokens = lambda n: (100, 20) if n == 1 else (None, 5)
        with tempfile.TemporaryDirectory() as tmp:
            s = summarize(run_script(tmp, ids=["G01", "G04"], script=SCRIPT[:2], tokens=tokens)[2])
        self.assertEqual((s["tokens"]["attempts_with_counts"], s["tokens"]["prompt_total"], s["tokens"]["completion_total"]), (1, 100, 20))

    def test_no_repair_means_repair_success_rate_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = summarize(run_script(tmp, ids=["G01"], script=SCRIPT[:1])[2])
        self.assertEqual((s["repair_attempted"]["count"], s["repair_succeeded"]["denominator"], s["repair_succeeded"]["rate"]), (0, 0, None))
        self.assertIn("n/a: denominator 0", format_summary(s))

    def test_empty_record_set_is_an_error(self):
        with self.assertRaises(EvalRunnerError):
            summarize([])

    def test_all_correct_and_all_wrong_extremes(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = summarize(run_script(tmp, ids=["G01"], script=SCRIPT[:1])[2])
        self.assertEqual((good["ex"]["rate"], good["valid_sql"]["rate"]), (1.0, 1.0))
        with tempfile.TemporaryDirectory() as tmp:
            bad = summarize(run_script(tmp, ids=["G03", "G05"], script=SCRIPT[2:4])[2])
        self.assertEqual((bad["ex"]["rate"], bad["valid_sql"]["rate"], bad["executed_ok"]["rate"]), (0.0, 0.0, 0.0))
        self.assertEqual(bad["outcomes"], {"answered": 0, "needs_clarification": 1, "unsupported": 1, "failed": 0})

    def test_records_are_written_one_by_one_as_questions_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.jsonl"
            agent = agent_with(SCRIPT[0], SCRIPT[1])
            seen = []
            run_evaluation(agent, agent.retriever, select_questions(gold, ["G01", "G04"]), "r1", out,
                           on_record=lambda r: seen.append((r["question_id"], len(out.read_text(encoding="utf-8").splitlines()))))
        self.assertEqual(seen, [("G01", 1), ("G04", 2)])


class FullSchemaContext(unittest.TestCase):
    def test_every_table_in_index_order_with_other_behaviour_delegated(self):
        inner = FakeRetriever()
        inner.docs = docs
        r = FullSchemaRetriever(inner)
        ctx = r.retrieve_context("anything")
        self.assertEqual([c.table for c in ctx], [d["table"] for d in docs])
        self.assertTrue(all(c.kind == "retrieved" for c in ctx))
        self.assertEqual((len(ctx), r.context_strategy, r.docs is docs), (12, "full_schema", True))


if __name__ == "__main__":
    unittest.main()
