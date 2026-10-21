"""Tests for text2sql.agent with a deterministic fake model. Run from the project root: python -B -m unittest -v tests.test_agent

No Qwen inference. FakeRetriever avoids loading the embedding model; one test wires in the real SchemaRetriever.
"""
import hashlib
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from text2sql import agent as agent_module
from text2sql.agent import AgentResult, Failure, MAX_RETRIES, OUTCOMES, Text2SQLAgent, attach_equivalence
from text2sql.execution import DB_PATH, QueryTimeout, execute_readonly
from text2sql.generation import ModelReply
from text2sql.gold import load_gold
from text2sql.retrieval import ContextTable, SchemaRetriever, load_index
from text2sql.semantic import load_layer

docs, layer, gold = None, None, None


def setUpModule():
    global docs, layer, gold
    docs = load_index()[0]
    layer = load_layer(docs)
    gold = {q.question_id: q for q in load_gold()}


def ready(sql, message="ok"):
    return json.dumps({"status": "ready", "sql": sql, "message": message})


def status(s, message):
    return json.dumps({"status": s, "sql": "", "message": message})


class FakeModel:
    """Returns the given replies in order; any call beyond them fails the test loudly."""

    def __init__(self, *replies, delay=0.0):
        self.replies, self.calls, self.delay = list(replies), [], delay

    def chat(self, messages):
        self.calls.append(json.loads(json.dumps(messages)))
        if not self.replies:
            raise AssertionError("unexpected extra model call")
        time.sleep(self.delay)
        return ModelReply(self.replies.pop(0), prompt_tokens=100, completion_tokens=20)


class FakeRetriever:
    def __init__(self):
        self.docs, self.questions = docs, []

    def retrieve_context(self, question):
        self.questions.append(question)
        return [ContextTable("orders", "retrieved", 1, ()), ContextTable("customers", "fk_parent", None, ("orders",))]


def make(*replies, max_retries=MAX_RETRIES, **kw):
    model = FakeModel(*replies, delay=kw.pop("delay", 0.0))
    return Text2SQLAgent(FakeRetriever(), model, layer, max_retries=max_retries, **kw), model


def db_hash():
    return hashlib.sha256(DB_PATH.read_bytes()).hexdigest()


class Success(unittest.TestCase):
    def test_generate_validate_execute(self):
        agent, model = make(ready("select count(*) from orders -- all"))
        r = agent.run("How many orders are there?")
        self.assertEqual((r.outcome, r.message, r.failure), ("answered", "", None))
        self.assertEqual((r.result.rows, r.result.columns), ([(130,)], ("COUNT(*)",)))
        self.assertEqual(r.sql, "SELECT COUNT(*) FROM orders")  # the validated canonical SQL that actually ran
        self.assertEqual(len(r.attempts), 1)
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(r.repaired)
        self.assertEqual((r.retrieved_tables, r.fk_added_tables, r.context_tables), (("orders",), ("customers",), ("orders", "customers")))
        a = r.first_attempt
        self.assertEqual((a.index, a.generation_status, a.validation_ok, a.execution_ok, a.failure), (1, "ready", True, True, None))
        self.assertEqual((a.prompt_tokens, a.completion_tokens), (100, 20))

    def test_no_gold_answer_means_no_equivalence_claim(self):
        r = make(ready("SELECT COUNT(*) FROM orders"))[0].run("q")
        self.assertIsNone(r.first_attempt.equivalence)
        self.assertIsNone(r.final_equivalence)

    def test_the_prompt_sent_to_the_model_has_static_prefix_then_schema_then_question(self):
        agent, model = make(ready("SELECT 1"), ready("SELECT 2"))
        agent.run("first question?")
        agent.run("second question?")
        a, b = model.calls
        self.assertEqual(a[0], b[0])  # system message (instructions + business rules) identical across questions
        self.assertTrue(a[1]["content"].endswith("Question: first question?"))
        self.assertTrue(b[1]["content"].endswith("Question: second question?"))
        self.assertIn("Table: orders.", a[1]["content"])
        self.assertIn("Table: customers.", a[1]["content"])
        self.assertNotIn("Table: payments.", a[1]["content"])

    def test_a_valid_but_wrong_answer_is_not_a_failure_and_is_not_repaired(self):
        wrong = "SELECT SUM(total_amount) FROM orders WHERE order_status = 'Delivered'"  # executes fine, but revenue is final_amount
        agent, model = make(ready(wrong))
        r = agent.run(gold["G16"].question)
        self.assertEqual((r.outcome, len(r.attempts), len(model.calls)), ("answered", 1, 1))
        scored = attach_equivalence(r, execute_readonly(gold["G16"].gold_sql), gold["G16"].ordered)
        self.assertEqual((scored.first_attempt.equivalence.ok, scored.final_equivalence.reason), (False, "values"))

    def test_equivalence_is_attached_only_afterwards_and_marks_a_correct_answer(self):
        r = make(ready(gold["G16"].gold_sql))[0].run(gold["G16"].question)
        self.assertIsNone(r.first_attempt.equivalence)
        scored = attach_equivalence(r, execute_readonly(gold["G16"].gold_sql), False)
        self.assertTrue(scored.final_equivalence.ok)
        self.assertIsNone(r.first_attempt.equivalence)  # the original result object is untouched


class NonReadyStatuses(unittest.TestCase):
    def test_clarification_is_returned_without_execution_or_repair(self):
        agent, model = make(status("needs_clarification", "Which city do you mean?"))
        with mock.patch.object(agent_module, "execute_readonly") as run:
            r = agent.run("How many orders came from Bengaluru?")
            run.assert_not_called()
        self.assertEqual((r.outcome, r.message, r.sql, r.result, r.failure), ("needs_clarification", "Which city do you mean?", None, None, None))
        self.assertEqual((len(model.calls), len(r.attempts)), (1, 1))  # no model call for repair
        self.assertEqual((r.first_attempt.validation_ok, r.first_attempt.execution_ok), (None, None))

    def test_unsupported_is_returned_without_execution_or_repair(self):
        agent, model = make(status("unsupported", "The schema has no weather data."))
        with mock.patch.object(agent_module, "execute_readonly") as run:
            r = agent.run("What is the weather?")
            run.assert_not_called()
        self.assertEqual((r.outcome, r.message, r.failure), ("unsupported", "The schema has no weather data.", None))
        self.assertEqual(len(model.calls), 1)


class MalformedModelOutput(unittest.TestCase):
    BAD = ["", "not json", "{", '{"status": "ready", "sql": "SELECT 1"}', json.dumps({"status": "maybe", "sql": "", "message": "m"}),
           json.dumps({"status": "ready", "sql": "", "message": "m"}), json.dumps({"status": "unsupported", "sql": "SELECT 1", "message": "m"}),
           json.dumps({"status": "ready", "sql": "SELECT 1", "message": "m", "x": 1}), "[1]"]

    def test_each_malformed_reply_ends_safely_without_execution_or_repair(self):
        for text in self.BAD:
            with self.subTest(text=text):
                agent, model = make(text)
                with mock.patch.object(agent_module, "execute_readonly") as run:
                    r = agent.run("q")
                    run.assert_not_called()
                self.assertEqual((r.outcome, r.result, r.sql), ("failed", None, None))
                self.assertEqual((r.failure.stage, len(model.calls), len(r.attempts)), ("generation", 1, 1))
                self.assertEqual(r.first_attempt.raw_response, text)
                self.assertIsNone(r.first_attempt.generation_status)

    def test_a_malformed_repair_reply_ends_the_run(self):
        agent, model = make(ready("SELECT nope FROM orders"), "garbage")
        r = agent.run("q")
        self.assertEqual((r.outcome, r.failure.stage, r.failure.code, len(model.calls)), ("failed", "repair", "invalid_json", 2))
        self.assertEqual(r.first_attempt.failure.stage, "execution")  # the first attempt's own failure is preserved

    def test_a_repair_reply_that_is_not_ready_ends_the_run(self):
        agent, model = make(ready("SELECT nope FROM orders"), status("unsupported", "cannot"))
        r = agent.run("q")
        self.assertEqual((r.outcome, r.failure.stage, r.failure.code, len(model.calls)), ("failed", "repair", "repair_not_ready", 2))


class UnsafeSql(unittest.TestCase):
    def test_unsafe_sql_is_never_executed_and_is_repaired_once(self):
        before = db_hash()
        agent, model = make(ready("DELETE FROM orders"), ready("SELECT COUNT(*) FROM orders"))
        r = agent.run("q")
        first = r.first_attempt
        self.assertEqual((first.validation_ok, first.execution_ok, first.result), (False, None, None))
        self.assertEqual((first.failure.stage, first.failure.code), ("validation", "not_select"))
        self.assertEqual((r.outcome, r.repaired, len(r.attempts), len(model.calls)), ("answered", True, 2, 2))
        self.assertEqual(r.result.rows, [(130,)])
        self.assertIn("validator rejected", model.calls[1][-1]["content"])
        self.assertEqual(db_hash(), before)

    def test_unsafe_sql_twice_fails_without_a_third_attempt(self):
        before = db_hash()
        agent, model = make(ready("DROP TABLE orders"), ready("SELECT 1; DROP TABLE orders"))
        r = agent.run("q")
        self.assertEqual((r.outcome, r.failure.stage, r.failure.code, len(model.calls), len(r.attempts)), ("failed", "validation", "multiple_statements", 2, 2))
        self.assertIsNone(r.result)
        self.assertEqual(db_hash(), before)


class Repair(unittest.TestCase):
    def test_execution_error_then_one_successful_repair(self):
        agent, model = make(ready("SELECT order_total FROM orders"), ready("SELECT COUNT(*) FROM orders"))
        r = agent.run("How many orders?")
        self.assertEqual((r.outcome, r.repaired, r.result.rows, len(model.calls)), ("answered", True, [(130,)], 2))
        feedback = model.calls[1]
        self.assertEqual([m["role"] for m in feedback], ["system", "user", "assistant", "user"])
        self.assertEqual(feedback[:2], model.calls[0])  # same prefix as the first call
        self.assertIn("SELECT order_total FROM orders", feedback[3]["content"])
        self.assertIn("no such column: order_total", feedback[3]["content"])
        self.assertNotIn("Traceback", feedback[3]["content"])
        self.assertEqual(feedback[2]["content"], ready("SELECT order_total FROM orders"))

    def test_repair_failure_returns_a_structured_failure_without_a_third_attempt(self):
        agent, model = make(ready("SELECT a FROM orders"), ready("SELECT b FROM orders"))
        r = agent.run("q")
        self.assertEqual((r.outcome, len(model.calls), len(r.attempts)), ("failed", 2, 2))
        self.assertEqual((r.failure.stage, r.failure.code), ("execution", "sqlite_error"))
        self.assertIn("no such column: b", r.failure.message)
        self.assertIsNone(r.result)
        self.assertEqual(r.sql, "SELECT b FROM orders")

    def test_timeout_is_a_repairable_failure_with_a_concise_message(self):
        agent, model = make(ready("SELECT 1"), ready("SELECT 2"))
        real = agent_module.execute_readonly
        calls = []

        def flaky(sql, **kw):
            calls.append(sql)
            if len(calls) == 1:
                raise QueryTimeout("query exceeded 10 s")
            return real(sql, **kw)
        with mock.patch.object(agent_module, "execute_readonly", side_effect=flaky):
            r = agent.run("q")
        self.assertEqual((r.outcome, r.first_attempt.failure.code, len(model.calls)), ("answered", "timeout", 2))
        self.assertIn("time limit", model.calls[1][-1]["content"])

    def test_max_retries_zero_never_repairs(self):
        agent, model = make(ready("SELECT nope FROM orders"), max_retries=0)
        r = agent.run("q")
        self.assertEqual((r.outcome, len(model.calls), len(r.attempts)), ("failed", 1, 1))

    def test_max_retries_is_validated_and_defaults_to_one(self):
        self.assertEqual(MAX_RETRIES, 1)
        for bad in (-1, 1.5, "1", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    make(max_retries=bad)

    def test_two_retries_allow_exactly_two_repairs(self):
        agent, model = make(ready("SELECT a FROM orders"), ready("SELECT b FROM orders"), ready("SELECT COUNT(*) FROM orders"), max_retries=2)
        r = agent.run("q")
        self.assertEqual((r.outcome, len(model.calls), len(r.attempts)), ("answered", 3, 3))
        self.assertEqual([m["role"] for m in model.calls[2]], ["system", "user", "assistant", "user", "assistant", "user"])


class FirstAttemptAndTiming(unittest.TestCase):
    def test_first_attempt_fields_survive_a_repair(self):
        agent, model = make(ready("SELECT order_total FROM orders"), ready("SELECT COUNT(*) FROM orders"), delay=0.02)
        r = agent.run("q")
        first, second = r.attempts
        self.assertEqual((first.sql, first.generation_status, first.validation_ok, first.execution_ok), ("SELECT order_total FROM orders", "ready", True, False))
        self.assertEqual((first.failure.stage, first.failure.code), ("execution", "sqlite_error"))
        self.assertIsNone(first.result)
        self.assertEqual((second.index, second.execution_ok, second.failure), (2, True, None))
        self.assertEqual(r.first_attempt, first)
        self.assertEqual(r.sql, "SELECT COUNT(*) FROM orders")

    def test_timing_fields(self):
        agent, _ = make(ready("SELECT order_total FROM orders"), ready("SELECT COUNT(*) FROM orders"), delay=0.03)
        r = agent.run("q")
        a1, a2 = r.attempts
        self.assertGreaterEqual(a1.model_s, 0.03)
        self.assertGreaterEqual(a2.model_s, 0.03)
        self.assertGreater(a1.exec_s, 0)
        self.assertGreater(a1.elapsed_at_end_s, a1.model_s)          # includes retrieval + generation + validation/execution of attempt 1
        self.assertGreater(a2.elapsed_at_end_s, a1.elapsed_at_end_s)  # cumulative
        self.assertGreaterEqual(r.total_s, a2.elapsed_at_end_s)
        self.assertGreaterEqual(r.total_s, a1.model_s + a2.model_s)
        self.assertGreaterEqual(r.retrieval_s, 0)
        self.assertLess(a1.elapsed_at_end_s, r.total_s)  # first-attempt elapsed time is separable for Experiment B

    def test_equivalence_is_recorded_per_attempt_for_experiment_b(self):
        agent, _ = make(ready("SELECT SUM(total_amount) FROM orders WHERE order_status = 'Delivered'"), ready("SELECT x FROM orders"), max_retries=1)
        r = agent.run(gold["G16"].question)  # first attempt executes (wrong value) -> no repair
        g = execute_readonly(gold["G16"].gold_sql)
        scored = attach_equivalence(r, g, False)
        self.assertEqual(len(scored.attempts), 1)
        self.assertFalse(scored.first_attempt.equivalence.ok)


class WorkflowStructure(unittest.TestCase):
    def test_graph_nodes_and_outcome_vocabulary(self):
        agent, _ = make()
        self.assertEqual(set(agent.graph.get_graph().nodes) - {"__start__", "__end__"}, {"retrieve", "generate", "run_sql", "assess", "repair", "finish"})
        self.assertEqual(OUTCOMES, ("answered", "needs_clarification", "unsupported", "failed"))

    def test_runs_are_deterministic_apart_from_timing(self):
        def summary():
            agent, _ = make(ready("SELECT nope FROM orders"), ready("SELECT COUNT(*) FROM orders"))
            r = agent.run("q")
            return (r.outcome, r.sql, r.result.rows, [(a.sql, a.failure, a.validation_ok, a.execution_ok) for a in r.attempts], r.context_tables)
        self.assertEqual(summary(), summary())

    def test_retriever_input_errors_propagate(self):
        class Strict(FakeRetriever):
            def retrieve_context(self, q):
                raise ValueError("empty question")
        agent = Text2SQLAgent(Strict(), FakeModel(), layer)
        with self.assertRaises(ValueError):
            agent.run("")


class TemporaryDatabase(unittest.TestCase):
    def test_db_path_is_honoured_and_validation_uses_that_databases_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            con = sqlite3.connect(db)
            con.executescript("CREATE TABLE things (x INTEGER); INSERT INTO things VALUES (1), (2), (3);")
            con.close()
            ok = Text2SQLAgent(FakeRetriever(), FakeModel(ready("SELECT COUNT(*) FROM things")), layer, db_path=db).run("q")
            self.assertEqual((ok.outcome, ok.result.rows), ("answered", [(3,)]))
            bad = Text2SQLAgent(FakeRetriever(), FakeModel(ready("SELECT COUNT(*) FROM orders"), ready("SELECT 1")), layer, db_path=db, max_retries=0).run("q")
            self.assertEqual((bad.outcome, bad.failure.code), ("failed", "unknown_table"))


class WithTheRealRetriever(unittest.TestCase):
    def test_context_comes_from_the_existing_retriever_and_the_prefix_is_shared(self):
        retriever = SchemaRetriever()
        model = FakeModel(ready(gold["G16"].gold_sql), ready("SELECT 1"))
        agent = Text2SQLAgent(retriever, model, load_layer(retriever.docs))
        r = agent.run(gold["G16"].question)
        expected = retriever.retrieve_context(gold["G16"].question)
        self.assertEqual(r.context_tables, tuple(c.table for c in expected))
        self.assertEqual(r.outcome, "answered")
        self.assertEqual(r.result.rows, execute_readonly(gold["G16"].gold_sql).rows)
        agent.run(gold["G01"].question)
        self.assertEqual(model.calls[0][0], model.calls[1][0])
        self.assertTrue(model.calls[1][1]["content"].endswith("Question: " + gold["G01"].question))


if __name__ == "__main__":
    unittest.main()
