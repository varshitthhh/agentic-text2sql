"""Agent Eval gap-fillers (docs/eval-protocol.md section 7), deterministic, fake model only.

tests.test_agent already covers: unsupported routing, malformed replies, repair after a validation or execution failure, failed repair with no
third attempt, and the retry budget. This file adds only what was missing at the agent level. No real-Qwen inference.
Run from the project root: python -B -m unittest -v tests.test_agent_eval
"""
import hashlib
import itertools
import json
import unittest
from unittest import mock

from tests import test_agent as ta
from text2sql import agent as agent_module
from text2sql import execution
from text2sql.agent import OUTCOMES, attach_equivalence
from text2sql.execution import DB_PATH, execute_readonly
from text2sql.generation import INSTRUCTIONS, STATUSES, ModelResponseError, parse_response


def setUpModule():
    ta.setUpModule()  # loads the shared docs/layer/gold fixtures that test_agent's helpers read


def db_hash():
    return hashlib.sha256(DB_PATH.read_bytes()).hexdigest()


class Clarification(unittest.TestCase):
    QUESTION = "How many orders came from Bengaluru?"  # 'city' exists in three tables with no default (semantic layer: ambiguity)

    def test_ambiguity_rule_and_clarification_instruction_reach_the_model_and_the_reply_is_routed(self):
        agent, model = ta.make(ta.status("needs_clarification", "Do you mean the customer's, the restaurant's or the delivery partner's city?"))
        with mock.patch.object(execution, "_execute") as database:
            r = agent.run(self.QUESTION)
            database.assert_not_called()
        system = model.calls[0][0]["content"]
        self.assertIn("city exists in customer_addresses, restaurants and delivery_partners", system)
        self.assertIn("ask which city is meant", system)
        self.assertIn("needs_clarification", INSTRUCTIONS)
        self.assertTrue(model.calls[0][1]["content"].endswith("Question: " + self.QUESTION))
        self.assertEqual((r.outcome, r.sql, r.result, r.failure), ("needs_clarification", None, None, None))
        self.assertIn("city", r.message)
        self.assertEqual((len(model.calls), len(r.attempts)), (1, 1))  # a clarification is never repaired

    def test_a_clarification_is_a_distinct_outcome_not_an_error(self):
        r = ta.make(ta.status("needs_clarification", "Which city?"))[0].run(self.QUESTION)
        self.assertNotEqual(r.outcome, "failed")
        self.assertIsNone(r.failure)


class Unsupported(unittest.TestCase):
    def test_instruction_defines_unsupported_and_a_reply_is_routed_without_sql(self):
        agent, model = ta.make(ta.status("unsupported", "The schema has no delivery-time estimates."))
        with mock.patch.object(execution, "_execute") as database:
            r = agent.run("What is the estimated delivery time for order 25?")
            database.assert_not_called()
        self.assertIn("unsupported", INSTRUCTIONS)
        self.assertEqual((r.outcome, r.sql, r.result, r.failure, len(model.calls)), ("unsupported", None, None, None, 1))


class UnsafeSqlNeverReachesTheDatabase(unittest.TestCase):
    UNSAFE = [
        "INSERT INTO categories (category_name) VALUES ('x')", "UPDATE orders SET order_status = 'x'", "DELETE FROM orders", "DROP TABLE orders",
        "CREATE TABLE t (x)", "ALTER TABLE orders ADD COLUMN z", "REPLACE INTO categories VALUES (1, 'x', 'y')", "PRAGMA table_info(orders)",
        "ATTACH DATABASE 'x.db' AS x", "VACUUM", "SELECT 1; DROP TABLE orders", "SELECT * FROM sqlite_master", "SELECT load_extension('x')",
        "WITH t AS (SELECT 1) DELETE FROM orders", "SELECT * INTO t2 FROM orders",
    ]

    def test_every_disallowed_statement_is_rejected_before_any_database_call(self):
        before = db_hash()
        for sql in self.UNSAFE:
            with self.subTest(sql=sql):
                agent, model = ta.make(ta.ready(sql), max_retries=0)
                with mock.patch.object(execution, "_execute") as database:  # the only function that runs SQL on the connection
                    r = agent.run("q")
                    database.assert_not_called()
                self.assertEqual((r.outcome, r.failure.stage, r.result), ("failed", "validation", None))
                self.assertEqual((r.first_attempt.validation_ok, r.first_attempt.execution_ok), (False, None))
                self.assertEqual(len(model.calls), 1)
        self.assertEqual(db_hash(), before)


class RepairedQueriesAreRevalidated(unittest.TestCase):
    def test_a_repair_after_an_execution_failure_that_proposes_unsafe_sql_is_rejected(self):
        before = db_hash()
        for unsafe in ("DROP TABLE orders", "SELECT 1; DELETE FROM orders", "SELECT * FROM sqlite_master"):
            with self.subTest(unsafe=unsafe):
                agent, model = ta.make(ta.ready("SELECT no_such_column FROM orders"), ta.ready(unsafe))
                r = agent.run("q")
                first, second = r.attempts
                self.assertEqual((first.validation_ok, first.execution_ok, first.failure.stage), (True, False, "execution"))
                self.assertEqual((second.validation_ok, second.execution_ok, second.result), (False, None, None))  # the repaired query was validated again
                self.assertEqual((r.outcome, r.failure.stage, len(model.calls)), ("failed", "validation", 2))
        self.assertEqual(db_hash(), before)

    def test_only_the_validated_repair_text_is_ever_executed(self):
        agent, model = ta.make(ta.ready("SELECT no_such_column FROM orders"), ta.ready("select  count(*)  from orders  -- fixed"))
        r = agent.run("q")
        self.assertEqual((r.outcome, r.sql, r.result.rows), ("answered", "SELECT COUNT(*) FROM orders", [(130,)]))

    def test_the_default_budget_leaves_a_third_model_reply_unused(self):
        agent, model = ta.make(ta.ready("SELECT a FROM orders"), ta.ready("SELECT b FROM orders"), ta.ready("SELECT COUNT(*) FROM orders"))
        r = agent.run("q")
        self.assertEqual((r.outcome, len(r.attempts), len(model.calls)), ("failed", 2, 2))
        self.assertEqual(len(model.replies), 1)  # the would-be third attempt was never requested


# ---------------------------------------------------------------------------------------------------------------------------------
# Scenario matrix: the public entry point (Text2SQLAgent.run) observed at three levels. "executor calls" = calls to execute_readonly
# (validate, then run); "validations" = calls to the SQLGlot validator; "db executions" = calls that actually run SQL on the SQLite
# connection. Internal graph paths are not asserted.
# ---------------------------------------------------------------------------------------------------------------------------------
def observe(agent, question="q"):
    events = []
    real_validate, real_execute = execution.validate_select, execution._execute

    def validate(sql, tables):
        events.append("validate")
        return real_validate(sql, tables)

    def run_sql(con, sql, *args, **kwargs):
        events.append(("execute", sql))
        return real_execute(con, sql, *args, **kwargs)

    with mock.patch.object(agent_module, "execute_readonly", wraps=agent_module.execute_readonly) as public, \
            mock.patch.object(execution, "validate_select", side_effect=validate), \
            mock.patch.object(execution, "_execute", side_effect=run_sql):
        result = agent.run(question)
    counts = {"executor_calls": public.call_count, "validations": events.count("validate"),
              "db_executions": sum(isinstance(e, tuple) for e in events), "events": events, "executor_args": [c.args[0] for c in public.call_args_list]}
    return result, counts


def code_of(failure):
    return f"{failure.stage}/{failure.code}" if failure else None


R = ta.ready
S = ta.status
BAD_COLUMN = "SELECT order_total FROM orders"
COUNT = "SELECT COUNT(*) FROM orders"
CLARIFY = S("needs_clarification", "Do you mean the customer's, the restaurant's or the delivery partner's city?")
UNSUPPORTED = S("unsupported", "The schema has no delivery-time data, so this cannot be answered.")
MALFORMED = {  # raw reply -> stable error code
    "garbage": "invalid_json", '{"status": "ready", "sql": "SELECT 1"}': "missing_keys",
    json.dumps({"status": "ready", "sql": "SELECT 1", "message": "m", "x": 1}): "unexpected_keys",
    json.dumps({"status": "ready", "sql": None, "message": "m"}): "wrong_types", json.dumps({"status": "done", "sql": "", "message": "m"}): "invalid_status",
    json.dumps({"status": "ready", "sql": "", "message": "m"}): "status_sql_mismatch",
    json.dumps({"status": "needs_clarification", "sql": "SELECT 1", "message": "m"}): "status_sql_mismatch",
    json.dumps({"status": "needs_clarification", "sql": "", "message": ""}): "empty_message",
    json.dumps({"status": "unsupported", "sql": "", "message": "   "}): "empty_message", "[1, 2]": "invalid_json", "": "invalid_json",
}
UNSAFE = {"DELETE FROM orders": "not_select", "SELECT 1; DROP TABLE orders": "multiple_statements", "SELECT * FROM sqlite_master": "unknown_table",
          "SELECT load_extension('x')": "forbidden_function", "INSERT INTO categories (category_name) VALUES ('x')": "not_select", "PRAGMA table_info(orders)": "not_select"}


class ScenarioMatrix(unittest.TestCase):
    def expect(self, name, replies, outcome, model_calls, executor_calls, db_executions, repairs, code, first_code=None, max_retries=1, question="q", unused=0):
        agent, model = ta.make(*replies, max_retries=max_retries)
        before = hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
        r, c = observe(agent, question)
        got = (r.outcome, len(model.calls), c["executor_calls"], c["db_executions"], len(r.attempts) - 1, code_of(r.failure), code_of(r.first_attempt.failure))
        self.assertEqual(got, (outcome, model_calls, executor_calls, db_executions, repairs, code, first_code), name)
        self.assertEqual(c["validations"], c["executor_calls"], name)  # every executor call validated first
        self.assertEqual(len(model.replies), unused, name)  # no model reply beyond the budget was consumed
        self.assertEqual(hashlib.sha256(DB_PATH.read_bytes()).hexdigest(), before, name)
        self.assertIn(r.outcome, OUTCOMES)
        if r.outcome == "answered":  # a success carries a result and no failure
            self.assertIsNotNone(r.result)
            self.assertIsNone(r.failure)
        else:  # nothing else may look like an answer
            self.assertIsNone(r.result, name)
        if r.outcome == "failed":
            self.assertIsNotNone(r.failure, name)
        else:
            self.assertIsNone(r.failure, name)
        return r, c

    def test_1_valid_sql_is_validated_executed_and_answered(self):
        r, _ = self.expect("valid", [R(COUNT)], "answered", 1, 1, 1, 0, None)
        self.assertEqual((r.result.rows, r.message, r.repaired), ([(130,)], "", False))

    def test_2_invalid_sql_is_repaired_once_and_answered(self):
        self.expect("execution error then repair", [R(BAD_COLUMN), R(COUNT)], "answered", 2, 2, 2, 1, None, "execution/sqlite_error")
        r, _ = self.expect("validation failure then repair", [R("DELETE FROM orders"), R(COUNT)], "answered", 2, 2, 1, 1, None, "validation/not_select")
        self.assertTrue(r.repaired)

    def test_3_failed_repair_is_a_structured_failure_with_no_third_attempt(self):
        self.expect("repair still fails to execute", [R("SELECT a FROM orders"), R("SELECT b FROM orders"), R(COUNT)], "failed", 2, 2, 2, 1,
                    "execution/sqlite_error", "execution/sqlite_error", unused=1)
        self.expect("repair is unsafe", [R("SELECT a FROM orders"), R("DROP TABLE orders"), R(COUNT)], "failed", 2, 2, 1, 1,
                    "validation/not_select", "execution/sqlite_error", unused=1)

    def test_4_clarification_asks_a_question_and_executes_nothing(self):
        r, c = self.expect("clarification", [CLARIFY], "needs_clarification", 1, 0, 0, 0, None)
        self.assertEqual((r.sql, c["events"]), (None, []))
        self.assertTrue(r.message.endswith("?"))
        self.assertIn("city", r.message)  # the specific ambiguity, passed through unchanged

    def test_5_unsupported_explains_the_limitation_and_fabricates_nothing(self):
        r, c = self.expect("unsupported", [UNSUPPORTED], "unsupported", 1, 0, 0, 0, None)
        self.assertEqual((r.sql, r.result, c["events"]), (None, None, []))
        self.assertIn("cannot be answered", r.message)

    def test_6_malformed_model_output_fails_safely_without_execution_or_repair(self):
        for raw, code in MALFORMED.items():
            with self.subTest(raw=raw[:60], code=code):
                r, c = self.expect("malformed", [raw], "failed", 1, 0, 0, 0, f"generation/{code}", f"generation/{code}")
                self.assertEqual((c["events"], r.first_attempt.raw_response, r.sql), ([], raw, None))

    def test_7_unsafe_sql_is_rejected_before_the_database(self):
        for sql, code in UNSAFE.items():
            with self.subTest(sql=sql):
                self.expect("unsafe, one repair allowed", [R(sql), R(sql)], "failed", 2, 2, 0, 1, f"validation/{code}", f"validation/{code}")
                self.expect("unsafe, no repair allowed", [R(sql)], "failed", 1, 1, 0, 0, f"validation/{code}", f"validation/{code}", max_retries=0)

    def test_8_valid_sql_that_disagrees_with_gold_is_reported_as_a_mismatch_not_a_failure(self):
        g = ta.gold["G16"]
        wrong = "SELECT SUM(total_amount) FROM orders WHERE order_status = 'Delivered'"
        r, _ = self.expect("valid but wrong", [R(wrong), R(g.gold_sql)], "answered", 1, 1, 1, 0, None, unused=1, question=g.question)
        self.assertEqual((r.first_attempt.validation_ok, r.first_attempt.execution_ok, r.first_attempt.failure), (True, True, None))
        scored = attach_equivalence(r, execute_readonly(g.gold_sql), g.ordered)
        self.assertEqual((scored.first_attempt.equivalence.ok, scored.first_attempt.equivalence.reason), (False, "values"))
        self.assertEqual((scored.outcome, len(scored.attempts)), ("answered", 1))  # execution succeeded and nothing was repaired automatically

    def test_the_four_public_outcomes_are_distinguishable(self):
        results = {}
        for name, replies in (("answered", [R(COUNT)]), ("needs_clarification", [CLARIFY]), ("unsupported", [UNSUPPORTED]), ("failed", ["garbage"])):
            results[name] = ta.make(*replies)[0].run("q")
        self.assertEqual({k: v.outcome for k, v in results.items()}, {k: k for k in OUTCOMES})
        shape = {k: (v.result is not None, v.failure is not None, bool(v.message), v.sql is not None) for k, v in results.items()}
        self.assertEqual(shape, {"answered": (True, False, False, True), "needs_clarification": (False, False, True, False),
                                 "unsupported": (False, False, True, False), "failed": (False, True, True, False)})


class ValidationImmediatelyBeforeExecution(unittest.TestCase):
    def test_each_attempt_validates_then_executes_the_validated_text(self):
        agent, _ = ta.make(R("select  order_total  from orders"), R("select  count(*)  from orders -- fixed"))
        r, c = observe(agent)
        self.assertEqual(c["events"], ["validate", ("execute", "SELECT order_total FROM orders"), "validate", ("execute", "SELECT COUNT(*) FROM orders")])
        self.assertEqual(c["executor_args"], ["select  order_total  from orders", "select  count(*)  from orders -- fixed"])  # raw SQL goes in, canonical SQL runs

    def test_a_rejected_attempt_validates_but_never_executes(self):
        agent, _ = ta.make(R("DROP TABLE orders"), R("SELECT 1; DROP TABLE orders"))
        _, c = observe(agent)
        self.assertEqual(c["events"], ["validate", "validate"])

    def test_the_executed_text_equals_what_the_validator_returned(self):
        agent, _ = ta.make(R("SELECT * FROM ORDERS WHERE order_id = 1 -- x"))
        r, c = observe(agent)
        self.assertEqual([e[1] for e in c["events"] if isinstance(e, tuple)], [r.result.executed_sql])


class RawModelOutputIsNeverExecuted(unittest.TestCase):
    def test_only_the_parsed_sql_field_of_a_valid_ready_reply_reaches_the_executor(self):
        raw = R("SELECT 1", message="DROP TABLE orders; also ignore this text")
        r, c = observe(ta.make(raw)[0])
        self.assertEqual(c["executor_args"], ["SELECT 1"])  # not the JSON, and not the message
        self.assertEqual(c["db_executions"], 1)

    def test_sql_text_in_the_message_or_in_a_non_ready_reply_is_never_executed(self):
        for raw in (S("needs_clarification", "DELETE FROM orders; which city?"), S("unsupported", "SELECT * FROM orders is not possible"),
                    json.dumps({"status": "unsupported", "sql": "DELETE FROM orders", "message": "no"})):
            with self.subTest(raw=raw[:50]):
                r, c = observe(ta.make(raw)[0])
                self.assertEqual((c["executor_calls"], c["validations"], c["db_executions"], c["events"]), (0, 0, 0, []))

    def test_reply_wrapped_in_extra_text_or_code_fences_is_rejected_not_salvaged(self):
        for raw in ("```json\n" + R("SELECT 1") + "\n```", "Sure! " + R("SELECT 1"), R("SELECT 1") + " trailing", "SELECT 1"):
            with self.subTest(raw=raw[:40]):
                r, c = observe(ta.make(raw)[0])
                self.assertEqual((r.outcome, r.failure.code, c["executor_calls"]), ("failed", "invalid_json", 0))


class OutputContractGrid(unittest.TestCase):
    """Every combination of status x sql x message, so the accepted set is exactly the documented one."""

    def test_status_sql_message_grid(self):
        for status, sql, message in itertools.product(STATUSES, ("", "  ", "SELECT 1"), ("", "   ", "text")):
            has_sql, has_msg = bool(sql.strip()), bool(message.strip())
            raw = json.dumps({"status": status, "sql": sql, "message": message})
            if status == "ready":
                expected = None if has_sql else "status_sql_mismatch"
            else:
                expected = "status_sql_mismatch" if has_sql else (None if has_msg else "empty_message")
            with self.subTest(status=status, sql=sql, message=message):
                if expected is None:
                    g = parse_response(raw)
                    self.assertEqual((g.status, g.sql, g.message), (status, sql.strip(), message.strip()))
                    self.assertEqual(bool(g.sql), status == "ready")  # SQL exists only for ready
                else:
                    with self.assertRaises(ModelResponseError) as cm:
                        parse_response(raw)
                    self.assertEqual(cm.exception.code, expected)

    def test_field_types_keys_and_status_values(self):
        ok = {"status": "ready", "sql": "SELECT 1", "message": "m"}
        for key in ok:
            for bad in (None, 5, ["x"], {"a": 1}, True):
                with self.subTest(key=key, bad=bad):
                    with self.assertRaises(ModelResponseError) as cm:
                        parse_response(json.dumps({**ok, key: bad}))
                    self.assertEqual(cm.exception.code, "wrong_types" if key != "status" or not isinstance(bad, str) else "invalid_status")
            with self.subTest(missing=key):
                with self.assertRaises(ModelResponseError) as cm:
                    parse_response(json.dumps({k: v for k, v in ok.items() if k != key}))
                self.assertEqual(cm.exception.code, "missing_keys")
        for status in ("Ready", "READY", " ready", "ready ", "done", "", "needs-clarification", "unsupported "):
            with self.subTest(status=status):
                with self.assertRaises(ModelResponseError) as cm:
                    parse_response(json.dumps({**ok, "status": status}))
                self.assertEqual(cm.exception.code, "invalid_status")
        for text in (None, "", "   ", "{", "null", "42", '"ready"', "[]", "[1]", "true"):
            with self.subTest(text=text):
                with self.assertRaises(ModelResponseError) as cm:
                    parse_response(text)
                self.assertEqual(cm.exception.code, "invalid_json")


if __name__ == "__main__":
    unittest.main()
