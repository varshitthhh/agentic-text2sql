"""Single-agent LangGraph workflow (D-06, D-07, D-09).

question -> retrieve -> generate -> route by status -> run_sql -> assess -> [repair -> run_sql] -> finish

* The model's reply is parsed as data (text2sql.generation); SQL it proposes is only ever run through execute_readonly(), which
  validates it (text2sql.safety) and runs it on the read-only connection (text2sql.execution).
* Repair happens only after a validation or execution failure, at most `max_retries` times (default 1). needs_clarification and
  unsupported replies, and malformed replies, end the run without repair. A wrong-but-valid result is not a failure here.
* Every attempt keeps its SQL, validation/execution outcome, timing and (when a gold answer is supplied afterwards through
  attach_equivalence) its equivalence outcome, so first-attempt results can be read for Experiment B.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .equivalence import Equivalence, results_equivalent
from .execution import DB_PATH, DEFAULT_MAX_ROWS, DEFAULT_TIMEOUT_S, QueryResult, QueryTimeout, UnsafeSQL, execute_readonly
from .generation import ModelClient, ModelResponseError, build_messages, parse_response, repair_messages
from .retrieval import ContextTable

MAX_RETRIES = 1  # configuration value (CLAUDE.md section 2.8); the evaluation compares 0 and 1
OUTCOMES = ("answered", "needs_clarification", "unsupported", "failed")


@dataclass(frozen=True)
class Failure:
    stage: str  # generation | validation | execution | repair
    code: str
    message: str


@dataclass(frozen=True)
class Attempt:
    index: int  # 1 = first attempt
    raw_response: str
    generation_status: str | None  # None when the reply was not a valid response
    message: str
    sql: str | None  # the SQL the model proposed (status 'ready' only); never executed except through execute_readonly
    prompt_tokens: int | None
    completion_tokens: int | None
    model_s: float
    validation_ok: bool | None = None  # None: no SQL was run
    execution_ok: bool | None = None  # None: not reached (no SQL, or validation failed)
    result: QueryResult | None = None
    failure: Failure | None = None
    exec_s: float = 0.0  # validation + execution time
    elapsed_at_end_s: float = 0.0  # seconds since the question was received, including retrieval and earlier attempts
    equivalence: Equivalence | None = None  # only filled by attach_equivalence(); None for ordinary queries


@dataclass(frozen=True)
class AgentResult:
    question: str
    outcome: str  # one of OUTCOMES
    message: str  # clarification question, unsupported reason, failure message, or empty when answered
    sql: str | None  # answered: the validated SQL that ran; otherwise the last proposed SQL, if any
    result: QueryResult | None  # answered only
    failure: Failure | None  # failed only
    attempts: tuple[Attempt, ...]
    context_tables: tuple[str, ...]
    retrieved_tables: tuple[str, ...]
    fk_added_tables: tuple[str, ...]
    retrieval_s: float
    total_s: float  # question received -> final state (retrieval, generation, validation, execution, repair)

    @property
    def first_attempt(self) -> Attempt:
        return self.attempts[0]

    @property
    def repaired(self) -> bool:
        return self.outcome == "answered" and len(self.attempts) > 1

    @property
    def final_equivalence(self) -> Equivalence | None:
        return self.attempts[-1].equivalence if self.outcome == "answered" else None


class AgentState(TypedDict, total=False):
    question: str
    started: float
    context: list[ContextTable]
    retrieval_s: float
    messages: list[dict]
    attempts: list[Attempt]
    outcome: str | None  # None while the run continues
    message: str
    failure: Failure | None
    total_s: float


def attach_equivalence(result: AgentResult, gold_result: QueryResult, ordered: bool) -> AgentResult:
    """Score every executed attempt against a gold result. Called by the evaluation runner after the run, outside the timing."""
    attempts = tuple(replace(a, equivalence=results_equivalent(a.result, gold_result, ordered)) if a.result is not None else a
                     for a in result.attempts)
    return replace(result, attempts=attempts)


def _error_text(f: Failure) -> str:
    if f.stage == "validation":
        return f"the SQL validator rejected it ({f.message[:300]})"
    if f.code == "timeout":
        return "the query exceeded the time limit"
    return f.message[:300]


class Text2SQLAgent:
    def __init__(self, retriever, model: ModelClient, layer: dict, max_retries: int = MAX_RETRIES, db_path: Path = DB_PATH,
                 timeout_s: float = DEFAULT_TIMEOUT_S, max_rows: int = DEFAULT_MAX_ROWS):
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        self.retriever, self.model, self.layer, self.max_retries = retriever, model, layer, max_retries
        self.db_path, self.timeout_s, self.max_rows = db_path, timeout_s, max_rows
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        g = StateGraph(AgentState)
        for name, fn in (("retrieve", self._retrieve), ("generate", self._generate), ("run_sql", self._run_sql),
                         ("assess", self._assess), ("repair", self._repair), ("finish", self._finish)):
            g.add_node(name, fn)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_conditional_edges("generate", lambda s: "finish" if s["outcome"] is not None else "run_sql", {"run_sql": "run_sql", "finish": "finish"})
        g.add_edge("run_sql", "assess")
        g.add_conditional_edges("assess", lambda s: "repair" if s["outcome"] is None else "finish", {"repair": "repair", "finish": "finish"})
        g.add_conditional_edges("repair", lambda s: "run_sql" if s["outcome"] is None else "finish", {"run_sql": "run_sql", "finish": "finish"})
        g.add_edge("finish", END)
        return g.compile()

    def run(self, question: str) -> AgentResult:
        final = self.graph.invoke({"question": question, "started": time.perf_counter(), "attempts": [], "outcome": None, "failure": None, "message": ""})
        attempts = tuple(final["attempts"])
        last = attempts[-1]
        context = final["context"]
        answered = final["outcome"] == "answered"
        return AgentResult(
            question=question, outcome=final["outcome"], message=final["message"],
            sql=last.result.executed_sql if answered else last.sql, result=last.result if answered else None,
            failure=final["failure"] if final["outcome"] == "failed" else None, attempts=attempts,
            context_tables=tuple(c.table for c in context), retrieved_tables=tuple(c.table for c in context if c.kind == "retrieved"),
            fk_added_tables=tuple(c.table for c in context if c.kind == "fk_parent"), retrieval_s=final["retrieval_s"], total_s=final["total_s"])

    # ------------------------------------------------------------------ nodes
    def _retrieve(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        context = self.retriever.retrieve_context(state["question"])
        return {"context": context, "messages": build_messages(state["question"], context, self.retriever.docs, self.layer),
                "retrieval_s": time.perf_counter() - t0}

    def _ask_model(self, state: AgentState, messages: list[dict]) -> Attempt:
        t0 = time.perf_counter()
        reply = self.model.chat(messages)
        model_s = time.perf_counter() - t0
        try:
            gen, failure = parse_response(reply.content), None
        except ModelResponseError as e:
            gen, failure = None, Failure("generation", e.code, str(e))
        return Attempt(index=len(state["attempts"]) + 1, raw_response=reply.content, generation_status=gen.status if gen else None,
                       message=gen.message if gen else "", sql=gen.sql if gen and gen.status == "ready" else None,
                       prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens, model_s=model_s, failure=failure,
                       elapsed_at_end_s=time.perf_counter() - state["started"])

    def _generate(self, state: AgentState) -> dict:
        attempt = self._ask_model(state, state["messages"])
        update: dict = {"attempts": state["attempts"] + [attempt]}
        if attempt.failure is not None:
            update |= {"outcome": "failed", "failure": attempt.failure, "message": attempt.failure.message}
        elif attempt.generation_status != "ready":
            update |= {"outcome": attempt.generation_status, "message": attempt.message}  # needs_clarification | unsupported: no repair
        return update

    def _run_sql(self, state: AgentState) -> dict:
        attempt = state["attempts"][-1]
        t0 = time.perf_counter()
        validation_ok, execution_ok, result, failure = True, None, None, None
        try:
            result = execute_readonly(attempt.sql, db_path=self.db_path, timeout_s=self.timeout_s, max_rows=self.max_rows)
            execution_ok = True
        except UnsafeSQL as e:
            validation_ok, failure = False, Failure("validation", e.code, str(e))
        except QueryTimeout as e:
            execution_ok, failure = False, Failure("execution", "timeout", str(e))
        except sqlite3.Error as e:
            execution_ok, failure = False, Failure("execution", "sqlite_error", f"{type(e).__name__}: {e}")
        done = replace(attempt, validation_ok=validation_ok, execution_ok=execution_ok, result=result, failure=failure,
                       exec_s=time.perf_counter() - t0, elapsed_at_end_s=time.perf_counter() - state["started"])
        return {"attempts": state["attempts"][:-1] + [done]}

    def _assess(self, state: AgentState) -> dict:
        last = state["attempts"][-1]
        if last.failure is None:
            return {"outcome": "answered", "message": ""}
        if len(state["attempts"]) <= self.max_retries:
            return {"outcome": None}  # repair
        return {"outcome": "failed", "failure": last.failure, "message": last.failure.message}

    def _repair(self, state: AgentState) -> dict:
        last = state["attempts"][-1]
        messages = repair_messages(state["messages"], last.raw_response, _error_text(last.failure), last.sql)
        attempt = self._ask_model(state, messages)
        update: dict = {"messages": messages, "attempts": state["attempts"] + [attempt], "outcome": None}
        if attempt.failure is not None:
            update |= {"outcome": "failed", "failure": Failure("repair", attempt.failure.code, attempt.failure.message), "message": attempt.failure.message}
        elif attempt.generation_status != "ready":
            msg = f"the model answered {attempt.generation_status!r} instead of a corrected query: {attempt.message}"
            update |= {"outcome": "failed", "failure": Failure("repair", "repair_not_ready", msg), "message": msg}
        return update

    def _finish(self, state: AgentState) -> dict:
        return {"total_s": time.perf_counter() - state["started"]}
