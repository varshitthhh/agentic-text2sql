"""Tests for text2sql.generation (no real model). Run from the project root: python -B -m unittest -v tests.test_generation"""
import json
import unittest

from text2sql.generation import (INSTRUCTIONS, MODEL, NUM_CTX, NUM_PREDICT, RESPONSE_KEYS, RESPONSE_SCHEMA, SEED, STATUSES, TEMPERATURE,
                                 Generation, ModelResponseError, OllamaModel, PromptTooLong, build_messages, parse_response, repair_messages)
from text2sql.retrieval import ContextTable, load_index
from text2sql.semantic import load_layer, render_layer

docs, layer = None, None


def setUpModule():
    global docs, layer
    docs = load_index()[0]
    layer = load_layer(docs)


def reply(status="ready", sql="SELECT 1", message="ok"):
    return json.dumps({"status": status, "sql": sql, "message": message})


class ParseResponse(unittest.TestCase):
    def test_valid_replies(self):
        self.assertEqual(parse_response(reply()), Generation("ready", "SELECT 1", "ok"))
        self.assertEqual(parse_response(reply("needs_clarification", "", "Which city?")), Generation("needs_clarification", "", "Which city?"))
        self.assertEqual(parse_response(reply("unsupported", "  ", "No such data")), Generation("unsupported", "", "No such data"))
        self.assertEqual(parse_response(reply(sql="  SELECT 1  ")).sql, "SELECT 1")

    def test_invalid_replies_are_rejected_with_a_stable_code(self):
        cases = {
            "invalid_json": ["", "not json", "{", "SELECT 1", None, "[1, 2]", "42", '"text"'],
            "missing_keys": ['{"status": "ready", "sql": "SELECT 1"}', "{}", '{"sql": "SELECT 1", "message": "m"}'],
            "unexpected_keys": [json.dumps({"status": "ready", "sql": "SELECT 1", "message": "m", "extra": 1})],
            "wrong_types": [json.dumps({"status": "ready", "sql": None, "message": "m"}), json.dumps({"status": "ready", "sql": "SELECT 1", "message": 5}),
                            json.dumps({"status": ["ready"], "sql": "SELECT 1", "message": "m"})],
            "invalid_status": [reply("done"), reply("READY"), reply("")],
            "status_sql_mismatch": [reply("ready", ""), reply("ready", "   "), reply("needs_clarification", "SELECT 1"), reply("unsupported", "DROP TABLE x")],
            "empty_message": [reply("needs_clarification", "", ""), reply("needs_clarification", "", "   "), reply("unsupported", "", ""), reply("unsupported", "", "\n\t")],
        }
        for code, texts in cases.items():
            for text in texts:
                with self.subTest(code=code, text=text):
                    with self.assertRaises(ModelResponseError) as cm:
                        parse_response(text)
                    self.assertEqual(cm.exception.code, code)

    def test_schema_matches_the_parser(self):
        self.assertEqual(RESPONSE_KEYS, ("status", "sql", "message"))
        self.assertEqual(STATUSES, ("ready", "needs_clarification", "unsupported"))
        self.assertEqual(RESPONSE_SCHEMA["properties"]["status"]["enum"], list(STATUSES))
        self.assertEqual(sorted(RESPONSE_SCHEMA["required"]), sorted(RESPONSE_KEYS))
        self.assertEqual(set(RESPONSE_SCHEMA["properties"]), set(RESPONSE_KEYS))


class Prompt(unittest.TestCase):
    CTX = [ContextTable("orders", "retrieved", 1, ()), ContextTable("order_items", "retrieved", 2, ()),
           ContextTable("customers", "fk_parent", None, ("orders",))]

    def test_static_instructions_and_rules_come_first_then_schema_then_the_question(self):
        m = build_messages("How many orders are there?", self.CTX, docs, layer)
        self.assertEqual([x["role"] for x in m], ["system", "user"])
        self.assertTrue(m[0]["content"].startswith(INSTRUCTIONS))
        self.assertIn("Business rules:\n" + render_layer(layer), m[0]["content"])
        user = m[1]["content"]
        self.assertTrue(user.endswith("Question: How many orders are there?"))
        order = [user.index(s) for s in ("Tables most relevant", "Table: orders.", "Table: order_items.", "Related tables", "Table: customers.", "Question:")]
        self.assertEqual(order, sorted(order))
        for absent in ("Table: ratings.", "Table: payments.", "Table: coupons."):
            self.assertNotIn(absent, user)  # only the retrieved and FK-added schema is included
        self.assertNotIn("Business rules", user)

    def test_prefix_is_identical_for_different_questions(self):
        a = build_messages("How many orders?", self.CTX, docs, layer)
        b = build_messages("What is the average price?", self.CTX[:1], docs, layer)
        self.assertEqual(a[0], b[0])  # the whole system message (instructions + business rules) is question-independent
        self.assertEqual(a[1]["content"].split("\n\n")[0], b[1]["content"].split("\n\n")[0])  # fixed user header

    def test_prompt_is_deterministic(self):
        self.assertEqual(build_messages("q", self.CTX, docs, layer), build_messages("q", self.CTX, docs, layer))

    def test_only_documented_facts_appear_in_the_system_prompt(self):
        system = build_messages("q", self.CTX, docs, layer)[0]["content"]
        for fragment in ("SUM(orders.final_amount)", "AVG(ratings.rating)", "payments.payment_status = 'Success'", "orders.coupon_id IS NOT NULL",
                         "menu_items.created_at is always NULL"):
            self.assertIn(fragment, system)

    def test_repair_messages_continue_the_same_conversation(self):
        base = build_messages("q", self.CTX, docs, layer)
        r = repair_messages(base, reply(sql="SELECT nope FROM orders"), "SQLite error: no such column: nope", "SELECT nope FROM orders")
        self.assertEqual(r[:2], base)  # unchanged prefix (cache reuse)
        self.assertEqual([x["role"] for x in r], ["system", "user", "assistant", "user"])
        self.assertEqual(r[2]["content"], reply(sql="SELECT nope FROM orders"))
        self.assertIn("SELECT nope FROM orders", r[3]["content"])
        self.assertIn("no such column: nope", r[3]["content"])
        self.assertNotIn("Traceback", r[3]["content"])


class FakeOllamaClient:
    def __init__(self, content=None, prompt_tokens=1200, completion_tokens=40):
        self.content = content if content is not None else reply()
        self.prompt_tokens, self.completion_tokens, self.calls = prompt_tokens, completion_tokens, []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"role": "assistant", "content": self.content}, "prompt_eval_count": self.prompt_tokens, "eval_count": self.completion_tokens}


class OllamaWrapper(unittest.TestCase):
    def test_request_uses_the_agreed_model_schema_and_deterministic_options(self):
        client = FakeOllamaClient()
        out = OllamaModel(client=client).chat([{"role": "user", "content": "hi"}])
        call = client.calls[0]
        self.assertEqual(call["model"], "qwen2.5-coder:7b-instruct-q4_K_M")
        self.assertEqual(call["format"], RESPONSE_SCHEMA)
        self.assertEqual(call["options"], {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": 400})
        self.assertEqual((MODEL, NUM_CTX, NUM_PREDICT, TEMPERATURE, SEED), (call["model"], 4096, 400, 0, 0))
        self.assertIs(call["stream"], False)
        self.assertEqual((out.content, out.prompt_tokens, out.completion_tokens), (reply(), 1200, 40))

    def test_a_prompt_that_may_have_been_truncated_is_refused(self):
        with self.assertRaises(PromptTooLong):
            OllamaModel(client=FakeOllamaClient(prompt_tokens=3800)).chat([])
        OllamaModel(client=FakeOllamaClient(prompt_tokens=3696)).chat([])  # exactly fits with the generation budget

    def test_missing_token_counts_are_tolerated(self):
        out = OllamaModel(client=FakeOllamaClient(prompt_tokens=None, completion_tokens=None)).chat([])
        self.assertEqual((out.prompt_tokens, out.completion_tokens), (None, None))

    def test_no_server_connection_is_made_when_a_client_is_supplied(self):
        client = FakeOllamaClient()
        OllamaModel(client=client)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
