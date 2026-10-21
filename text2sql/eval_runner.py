"""End-to-end evaluation runner (docs/eval-protocol.md sections 3 and 6): gold questions -> compiled LangGraph agent -> JSONL records.

Metric denominators (N = number of gold questions run; every question stays in every denominator):
  EX                = final outcome 'answered' AND result equivalent to gold / N
  valid-SQL rate    = final SQL candidate (last attempt that proposed SQL) passed validation / N   (no SQL candidate = invalid)
  executed-OK rate  = final outcome 'answered' / N
  repair rate       = questions that used a repair attempt / N
  repair success    = repaired questions that ended 'answered' / repaired questions   (n/a when none were repaired)
  Recall@3, @5      = macro mean over N of |required tables in top k| / |required tables|, before FK expansion
  context coverage  = same fraction over the FK-expanded prompt context (reported separately from Recall)
Derived zero-retry results (Experiment B) are read from the first-attempt fields of the same records.

Usage: python -B -m text2sql.eval_runner RUN_NAME (--ids G04 G13 ... | --all) [--out PATH] [--max-retries 1] [--context retrieval|full]
--context full is Experiment A's full-schema condition: every table's schema document goes into the prompt, nothing else changes.
The output is append-only JSONL; an existing run name in the file is refused and existing lines are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .agent import MAX_RETRIES, AgentResult, Text2SQLAgent, attach_equivalence
from .execution import ROOT, execute_readonly
from .generation import MODEL, NUM_CTX, NUM_PREDICT, SEED, TEMPERATURE, OllamaModel, PromptTooLong
from .gold import GoldQuestion, load_gold
from .retrieval import CONTEXT_TOP_K, RRF_K, ContextTable, SchemaRetriever
from .retrieval_eval import recall_at_k
from .semantic import LAYER_PATH, load_layer

DEFAULT_OUT = ROOT / "data" / "eval" / "results.jsonl"
STAGE = "end_to_end"


class EvalRunnerError(ValueError):
    """Bad selection, an already-used run name, or an empty record set."""


class FullSchemaRetriever:
    """Experiment A condition: the prompt context is all tables, in index order, rendered by the unchanged prompt code.
    Everything else (retrieve() for Recall@k, docs, index metadata) is delegated, so only the context strategy differs."""
    context_strategy = "full_schema"

    def __init__(self, retriever: SchemaRetriever):
        self._r = retriever

    def __getattr__(self, name):
        return getattr(self._r, name)

    def retrieve_context(self, query: str, top_k: int = CONTEXT_TOP_K) -> list[ContextTable]:
        return [ContextTable(d["table"], "retrieved", i, ()) for i, d in enumerate(self._r.docs, 1)]


# ----------------------------------------------------------------------------- selection and output safety
def select_questions(gold: list[GoldQuestion], ids: list[str]) -> list[GoldQuestion]:
    """The requested questions in the requested order; unknown, duplicate or empty selections are errors."""
    if not ids:
        raise EvalRunnerError("no question ids selected")
    if len(set(ids)) != len(ids):
        raise EvalRunnerError(f"duplicate question ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    by_id = {q.question_id: q for q in gold}
    if (unknown := [i for i in ids if i not in by_id]):
        raise EvalRunnerError(f"unknown question ids: {unknown}")
    return [by_id[i] for i in ids]


def _existing_runs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["run"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def check_output(path: Path, run: str) -> None:
    if not run or not run.strip():
        raise EvalRunnerError("run name must not be empty")
    if run in _existing_runs(path):
        raise EvalRunnerError(f"run name {run!r} already exists in {path}; choose a new name")


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:  # append only: existing lines are never rewritten
        f.write(json.dumps(record) + "\n")


def load_run(path: Path, run: str) -> list[dict]:
    return [r for r in (json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()) if r["run"] == run]


# ----------------------------------------------------------------------------- one question -> one record
def _attempt_dict(a) -> dict:
    return {"index": a.index, "generation_status": a.generation_status, "sql": a.sql, "validation_ok": a.validation_ok,
            "execution_ok": a.execution_ok, "failure": asdict(a.failure) if a.failure else None,
            "prompt_tokens": a.prompt_tokens, "completion_tokens": a.completion_tokens, "model_s": a.model_s, "exec_s": a.exec_s,
            "elapsed_at_end_s": a.elapsed_at_end_s, "n_rows": len(a.result.rows) if a.result else None,
            "truncated": a.result.truncated if a.result else None,
            "equivalence": {"ok": a.equivalence.ok, "reason": a.equivalence.reason} if a.equivalence else None}


def build_record(run: str, config: dict, q: GoldQuestion, result: AgentResult, gold_result, top5: list[str]) -> dict:
    """`result` must already carry equivalence outcomes (attach_equivalence). All per-question flags are computed here, once."""
    attempts = [_attempt_dict(a) for a in result.attempts]
    first = attempts[0]
    candidate = next((a for a in reversed(attempts) if a["sql"] is not None), None)  # final SQL candidate
    answered = result.outcome == "answered"
    final_eq = result.final_equivalence
    required = set(q.required_tables)
    return {
        "run": run, "stage": STAGE, "config": config, "question_id": q.question_id, "difficulty": q.difficulty, "question": q.question,
        "required_tables": list(q.required_tables), "retrieved_top5": top5, "retrieved_tables": list(result.retrieved_tables),
        "fk_added_tables": list(result.fk_added_tables), "context_tables": list(result.context_tables), "context_size": len(result.context_tables),
        "recall@3": recall_at_k(q.required_tables, top5, 3), "recall@5": recall_at_k(q.required_tables, top5, 5),
        "context_coverage": len(required & set(result.context_tables)) / len(required),
        "outcome": result.outcome, "message": result.message, "failure": asdict(result.failure) if result.failure else None,
        "final_sql": result.sql, "final_equivalence": {"ok": final_eq.ok, "reason": final_eq.reason} if final_eq else None,
        "ex": bool(answered and final_eq and final_eq.ok),
        "valid_sql": bool(candidate and candidate["validation_ok"] is True),
        "executed_ok": answered,
        "repair_attempted": len(attempts) > 1,
        "repair_succeeded": len(attempts) > 1 and answered,
        "first_attempt": first,
        "first_attempt_ex": bool(first["equivalence"] and first["equivalence"]["ok"]),
        "first_attempt_valid_sql": bool(first["sql"] is not None and first["validation_ok"] is True),
        "attempts": attempts, "n_attempts": len(attempts),
        "retrieval_s": result.retrieval_s, "total_s": result.total_s,
        "gold": {"sql": q.gold_sql, "ordered": q.ordered, "n_rows": len(gold_result.rows), "n_columns": len(gold_result.columns),
                 "columns": list(gold_result.columns), "truncated": gold_result.truncated},
    }


def evaluate_question(agent: Text2SQLAgent, retriever, q: GoldQuestion, run: str, config: dict) -> dict:
    result = agent.run(q.question)  # timed by the agent itself (total_s); everything below is outside the timing
    gold_result = execute_readonly(q.gold_sql, db_path=agent.db_path)
    result = attach_equivalence(result, gold_result, q.ordered)
    top5 = [r.table for r in retriever.retrieve(q.question, 5)]
    return build_record(run, config, q, result, gold_result, top5)


# ----------------------------------------------------------------------------- aggregation
def _rate(count: int, denominator: int) -> dict:
    return {"count": count, "denominator": denominator, "rate": count / denominator if denominator else None}


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(records: list[dict]) -> dict:
    if not records:
        raise EvalRunnerError("cannot summarize zero records")
    n = len(records)
    attempts = [a for r in records for a in r["attempts"]]
    counted = [a for a in attempts if a["prompt_tokens"] is not None and a["completion_tokens"] is not None]
    repaired = [r for r in records if r["repair_attempted"]]
    mean = lambda vals: statistics.mean(vals) if vals else None
    return {
        "n_questions": n,
        "ex": _rate(sum(r["ex"] for r in records), n),
        "valid_sql": _rate(sum(r["valid_sql"] for r in records), n),
        "executed_ok": _rate(sum(r["executed_ok"] for r in records), n),
        "repair_attempted": _rate(len(repaired), n),
        "repair_succeeded": _rate(sum(r["repair_succeeded"] for r in repaired), len(repaired)),
        "outcomes": {o: sum(r["outcome"] == o for r in records) for o in ("answered", "needs_clarification", "unsupported", "failed")},
        "failure_stages": {k: sum(1 for r in records if r["failure"] and f"{r['failure']['stage']}/{r['failure']['code']}" == k)
                           for k in sorted({f"{r['failure']['stage']}/{r['failure']['code']}" for r in records if r["failure"]})},
        "recall@3": {"macro": mean([r["recall@3"] for r in records]), "complete": sum(r["recall@3"] == 1.0 for r in records), "denominator": n},
        "recall@5": {"macro": mean([r["recall@5"] for r in records]), "complete": sum(r["recall@5"] == 1.0 for r in records), "denominator": n},
        "context_coverage": {"macro": mean([r["context_coverage"] for r in records]), "complete": sum(r["context_coverage"] == 1.0 for r in records),
                             "mean_context_tables": mean([r["context_size"] for r in records]), "denominator": n},
        "latency_s": {"total_median": _median([r["total_s"] for r in records]), "total_max": max(r["total_s"] for r in records),
                      "retrieval_median": _median([r["retrieval_s"] for r in records]),
                      "first_attempt_elapsed_median": _median([r["first_attempt"]["elapsed_at_end_s"] for r in records]),
                      "model_per_attempt_median": _median([a["model_s"] for a in attempts]),
                      "exec_per_attempt_median": _median([a["exec_s"] for a in attempts if a["validation_ok"] is not None]),
                      "attempts": len(attempts), "exec_attempts": sum(a["validation_ok"] is not None for a in attempts)},
        "tokens": {"attempts": len(attempts), "attempts_with_counts": len(counted),
                   "prompt_total": sum(a["prompt_tokens"] for a in counted) if counted else None,
                   "prompt_mean": mean([a["prompt_tokens"] for a in counted]),
                   "completion_total": sum(a["completion_tokens"] for a in counted) if counted else None,
                   "completion_mean": mean([a["completion_tokens"] for a in counted])},
        "derived_zero_retry": {"ex": _rate(sum(r["first_attempt_ex"] for r in records), n),
                               "valid_sql": _rate(sum(r["first_attempt_valid_sql"] for r in records), n)},
    }


def _pct(r: dict) -> str:
    return f"{r['count']}/{r['denominator']}" + (f" ({r['rate']:.1%})" if r["rate"] is not None else " (n/a: denominator 0)")


def format_summary(s: dict) -> str:
    n, lat, tok = s["n_questions"], s["latency_s"], s["tokens"]
    f = lambda v, spec=".1f": "n/a" if v is None else format(v, spec)
    lines = [
        f"N = {n} gold questions run; every metric below uses the denominator shown (no question is dropped)",
        f"EX                  {_pct(s['ex'])}   [answered AND equivalent to gold / N]",
        f"valid-SQL rate      {_pct(s['valid_sql'])}   [final SQL candidate passed validation / N; no candidate = invalid]",
        f"executed-OK rate    {_pct(s['executed_ok'])}   [final outcome answered / N]",
        f"repair attempted    {_pct(s['repair_attempted'])}   [/ N]",
        f"repair succeeded    {_pct(s['repair_succeeded'])}   [/ repaired questions]",
        f"outcomes            {s['outcomes']}   failure stages: {s['failure_stages'] or 'none'}",
        f"Recall@3 (macro)    {f(s['recall@3']['macro'], '.3f')}  complete {s['recall@3']['complete']}/{n}   [top-3, before FK expansion]",
        f"Recall@5 (macro)    {f(s['recall@5']['macro'], '.3f')}  complete {s['recall@5']['complete']}/{n}   [top-5, before FK expansion]",
        f"context coverage    {f(s['context_coverage']['macro'], '.3f')}  complete {s['context_coverage']['complete']}/{n}, mean {f(s['context_coverage']['mean_context_tables'], '.1f')} tables   [after FK parents; separate from Recall]",
        f"latency (s)         total median {f(lat['total_median'])}, max {f(lat['total_max'])}; retrieval median {f(lat['retrieval_median'], '.3f')}; first-attempt elapsed median {f(lat['first_attempt_elapsed_median'])}",
        f"per attempt (s)     model median {f(lat['model_per_attempt_median'])} over {lat['attempts']} attempts; validate+execute median {f(lat['exec_per_attempt_median'], '.4f')} over {lat['exec_attempts']} executed attempts",
        f"tokens              {tok['attempts_with_counts']}/{tok['attempts']} attempts have counts; prompt total {f(tok['prompt_total'], 'd') if tok['prompt_total'] is not None else 'n/a'} (mean {f(tok['prompt_mean'])}), generated total {f(tok['completion_total'], 'd') if tok['completion_total'] is not None else 'n/a'} (mean {f(tok['completion_mean'])})",
        f"derived zero-retry  EX {_pct(s['derived_zero_retry']['ex'])}, valid-SQL {_pct(s['derived_zero_retry']['valid_sql'])}   [first attempts only; Experiment B, subject to its validity check]",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- orchestration
def make_config(agent: Text2SQLAgent, retriever) -> dict:
    model = agent.model
    return {"model": getattr(model, "model", None), "num_ctx": getattr(model, "num_ctx", None), "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE, "seed": SEED, "max_retries": agent.max_retries, "context_base_k": CONTEXT_TOP_K,
            "context_strategy": getattr(retriever, "context_strategy", "top3_fk_parents"),
            "expansion": "none" if hasattr(retriever, "context_strategy") else "direct_fk_parents_only", "retriever": "hybrid_bm25_dense_rrf", "rrf_k": getattr(retriever, "rrf_k", RRF_K),
            "query_instruction": getattr(retriever, "use_query_instruction", None),
            "index_db_sha256": getattr(retriever, "meta", {}).get("sources", {}).get("database", {}).get("sha256"),
            "semantic_layer_sha256": hashlib.sha256(LAYER_PATH.read_bytes()).hexdigest(),
            "executor": {"timeout_s": agent.timeout_s, "max_rows": agent.max_rows}}


def run_evaluation(agent: Text2SQLAgent, retriever, questions: list[GoldQuestion], run: str, out: Path, on_record=None) -> list[dict]:
    """Questions run sequentially; each record is appended as soon as the question finishes."""
    check_output(out, run)
    config = make_config(agent, retriever)
    records = []
    for q in questions:
        record = evaluate_question(agent, retriever, q, run, config)
        append_record(out, record)
        records.append(record)
        if on_record:
            on_record(record)
    return records


def _line(r: dict) -> str:
    a1 = r["first_attempt"]
    eq = (r["final_equivalence"] or {}).get("reason", "-")
    return (f"{r['question_id']} [{r['difficulty']:13}] outcome={r['outcome']:19} EX={str(r['ex']):5} equivalence={eq:7} attempts={r['n_attempts']} "
            f"total={r['total_s']:.1f}s (first attempt done at {a1['elapsed_at_end_s']:.1f}s) context={r['context_size']} tables")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run gold questions through the real agent and model.")
    p.add_argument("run_name")
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--ids", nargs="+", metavar="ID", help="gold question ids to run, e.g. G04 G13 G16")
    sel.add_argument("--all", action="store_true", help="run all gold questions (the full evaluation)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSONL results file (append-only)")
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    p.add_argument("--context", choices=("retrieval", "full"), default="retrieval", help="prompt schema context: top-3 + FK parents (default) or the full schema")
    args = p.parse_args(argv)

    gold = load_gold()
    questions = gold if args.all else select_questions(gold, args.ids)
    check_output(args.out, args.run_name)  # fail before loading any model

    retriever = SchemaRetriever()
    if args.context == "full":
        retriever = FullSchemaRetriever(retriever)
    agent = Text2SQLAgent(retriever, OllamaModel(), load_layer(retriever.docs), max_retries=args.max_retries)
    t0 = time.perf_counter()  # untimed warm-up: embedding model and the Qwen weights, so model loading is excluded from latency (protocol 3.4)
    retriever.retrieve(questions[0].question, 5)
    agent.model.chat([{"role": "user", "content": 'Reply with JSON: {"status": "unsupported", "sql": "", "message": "warm-up"}'}])
    print(f"warm-up (excluded from latency): {time.perf_counter() - t0:.1f} s; running {len(questions)} questions sequentially -> {args.out}", flush=True)
    try:
        records = run_evaluation(agent, retriever, questions, args.run_name, args.out, on_record=lambda r: print(_line(r), flush=True))
    except PromptTooLong as e:
        print(f"STOPPED: {e}. Nothing was truncated or reconfigured; records written so far remain in {args.out}.", file=sys.stderr)
        return 3
    reread = load_run(args.out, args.run_name)
    print("\n" + format_summary(summarize(reread)))
    print(f"\nread back {len(reread)} records from {args.out}; summary computed from the file: {summarize(reread) == summarize(records)} (equals the in-memory summary)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
