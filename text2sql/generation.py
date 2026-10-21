"""SQL generation: prompt layout, structured response contract, and the model interface (D-02, D-07).

The model is reached only through ModelClient.chat(messages) -> ModelReply, so tests can use a deterministic fake. Whatever the model
returns is parsed and validated here as data; it is never executed directly (text2sql.agent routes ready SQL through execute_readonly).
Prompt layout for prefix caching: [system: stable instructions + business rules] [user: schema context, then the question last].
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import ollama

from .retrieval import ContextTable
from .semantic import render_layer, render_schema

MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
HOST = "http://127.0.0.1:11434"
NUM_CTX = 4096  # same value as the smoke test; prompts are checked against it (see OllamaModel.chat)
NUM_PREDICT = 400
TEMPERATURE = 0
SEED = 0
STATUSES = ("ready", "needs_clarification", "unsupported")
RESPONSE_KEYS = ("status", "sql", "message")
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string", "enum": list(STATUSES)}, "sql": {"type": "string"}, "message": {"type": "string"}},
    "required": list(RESPONSE_KEYS),
}

INSTRUCTIONS = (
    "You translate one question into one read-only SQLite SELECT query, using only the tables, columns and business rules provided. "
    "Reply with JSON containing exactly: status, sql, message.\n"
    "- status 'ready': you can write the query. sql holds that single SELECT statement; message is one short sentence.\n"
    "- status 'needs_clarification': the question is materially ambiguous (for example, an ambiguity listed in the business rules applies). "
    "sql must be an empty string; message is the one clarifying question to ask.\n"
    "- status 'unsupported': the schema cannot answer the question. sql must be an empty string; message says why.\n"
    "Write SQLite syntax, one statement, no comments. Follow the business rules exactly. Do not invent tables or columns."
)


class ModelResponseError(ValueError):
    """The model's reply is not a valid response. `code` is a stable category."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class PromptTooLong(Exception):
    """The prompt (plus the generation budget) does not fit the context window, so it may have been truncated."""


@dataclass(frozen=True)
class Generation:
    status: str
    sql: str  # empty unless status == "ready"
    message: str


@dataclass(frozen=True)
class ModelReply:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ModelClient(Protocol):
    def chat(self, messages: list[dict]) -> ModelReply: ...


def parse_response(text: str) -> Generation:
    """Validate the model's reply; raise ModelResponseError for anything that is not exactly the agreed shape."""
    try:
        obj = json.loads(text)
    except (TypeError, ValueError) as e:
        raise ModelResponseError("invalid_json", f"reply is not valid JSON ({type(e).__name__})") from e
    if not isinstance(obj, dict):
        raise ModelResponseError("invalid_json", "reply is JSON but not an object")
    if (missing := [k for k in RESPONSE_KEYS if k not in obj]):
        raise ModelResponseError("missing_keys", f"missing {missing}")
    if (extra := sorted(set(obj) - set(RESPONSE_KEYS))):
        raise ModelResponseError("unexpected_keys", f"unexpected {extra}")
    if not all(isinstance(obj[k], str) for k in RESPONSE_KEYS):
        raise ModelResponseError("wrong_types", "status, sql and message must all be strings")
    if obj["status"] not in STATUSES:
        raise ModelResponseError("invalid_status", f"status {obj['status']!r} is not one of {STATUSES}")
    has_sql = bool(obj["sql"].strip())
    if obj["status"] == "ready" and not has_sql:
        raise ModelResponseError("status_sql_mismatch", "status 'ready' needs non-empty sql")
    if obj["status"] != "ready" and has_sql:
        raise ModelResponseError("status_sql_mismatch", f"status {obj['status']!r} must have empty sql")
    if obj["status"] != "ready" and not obj["message"].strip():
        raise ModelResponseError("empty_message", f"status {obj['status']!r} needs a message (the clarifying question, or why it is unsupported)")
    return Generation(obj["status"], obj["sql"].strip(), obj["message"].strip())


def build_messages(question: str, context: list[ContextTable], docs: list[dict], layer: dict) -> list[dict]:
    """Static instructions + business rules first (identical for every question), then schema context, then the question."""
    system = INSTRUCTIONS + "\n\nBusiness rules:\n" + render_layer(layer)
    user = ("Database: SQLite. Use only the tables and columns listed below.\n\n" + render_schema(context, docs)
            + f"\n\nQuestion: {question}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def repair_messages(messages: list[dict], previous_reply: str, error: str, failed_sql: str) -> list[dict]:
    """Continue the same conversation (so the cached prefix is reused): the failed reply, then concise error feedback."""
    feedback = (f"That query failed.\nFailed SQL: {failed_sql}\nError: {error}\n"
                "Write a corrected query for the same question. Reply in the same JSON format.")
    return messages + [{"role": "assistant", "content": previous_reply}, {"role": "user", "content": feedback}]


class OllamaModel:
    """The real model: qwen2.5-coder:7b-instruct-q4_K_M through the local Ollama server, JSON-schema constrained."""

    def __init__(self, model: str = MODEL, host: str = HOST, num_ctx: int = NUM_CTX, timeout: float = 900, client=None):
        self.model, self.num_ctx = model, num_ctx
        self._client = client if client is not None else ollama.Client(host=host, timeout=timeout)

    def chat(self, messages: list[dict]) -> ModelReply:
        resp = self._client.chat(model=self.model, messages=messages, format=RESPONSE_SCHEMA, stream=False,
                                 options={"temperature": TEMPERATURE, "seed": SEED, "num_ctx": self.num_ctx, "num_predict": NUM_PREDICT})
        d = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        prompt_tokens, completion_tokens = d.get("prompt_eval_count"), d.get("eval_count")
        if prompt_tokens is not None and prompt_tokens + NUM_PREDICT > self.num_ctx:
            raise PromptTooLong(f"prompt of {prompt_tokens} tokens plus {NUM_PREDICT} to generate exceeds num_ctx {self.num_ctx}")
        return ModelReply(d["message"]["content"], prompt_tokens, completion_tokens)
