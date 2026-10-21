"""Schema-retrieval evaluation over the gold set (docs/eval-protocol.md section 3.3).

recall@k(question) = |required_tables found in the top k| / |required_tables|, measured before any FK/bridge expansion.
Macro average = mean of the per-question fractions. Recall@5 is the protocol metric; Recall@3 is a secondary diagnostic.

With --expand-parents the prompt context (top-3 retrieved tables + their direct FK parents) is evaluated as well:
coverage_after = |required_tables in the context| / |required_tables|, plus the number of tables in the context.

Usage: python -B -m text2sql.retrieval_eval RUN_NAME [--expand-parents]
       (appends one JSONL record per question to data/eval/results.jsonl; an existing run name is refused)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

from .execution import ROOT
from .gold import DIFFICULTIES, GoldQuestion, load_gold
from .retrieval import CONTEXT_TOP_K, DEFAULT_TOP_K, RRF_K, SchemaRetriever

KS = (3, 5)
RESULTS_PATH = ROOT / "data" / "eval" / "results.jsonl"


def recall_at_k(required: tuple[str, ...], ranked: list[str], k: int) -> float:
    return len(set(required) & set(ranked[:k])) / len(required)


def evaluate(retriever: SchemaRetriever, gold: list[GoldQuestion]) -> list[dict]:
    retriever.retrieve(gold[0].question, DEFAULT_TOP_K)  # untimed warm-up (model load)
    records = []
    for q in gold:
        t0 = time.perf_counter()
        ranked = [r.table for r in retriever.retrieve(q.question, max(KS))]
        ms = (time.perf_counter() - t0) * 1000
        rec = {"question_id": q.question_id, "difficulty": q.difficulty, "required_tables": list(q.required_tables),
               "retrieved_top5": ranked, "retrieval_ms": round(ms, 1)}
        for k in KS:
            rec[f"recall@{k}"] = recall_at_k(q.required_tables, ranked, k)
            rec[f"missed@{k}"] = sorted(set(q.required_tables) - set(ranked[:k]))
        records.append(rec)
    return records


def evaluate_context(retriever: SchemaRetriever, gold: list[GoldQuestion]) -> list[dict]:
    retriever.retrieve(gold[0].question, CONTEXT_TOP_K)  # untimed warm-up
    records = []
    for q in gold:
        t0 = time.perf_counter()
        ctx = retriever.retrieve_context(q.question)
        ms = (time.perf_counter() - t0) * 1000
        retrieved = [c.table for c in ctx if c.kind == "retrieved"]
        context = [c.table for c in ctx]
        records.append({
            "question_id": q.question_id, "difficulty": q.difficulty, "required_tables": list(q.required_tables),
            "retrieved_top3": retrieved, "fk_added": [c.table for c in ctx if c.kind == "fk_parent"], "context_tables": context,
            "context_size": len(context), "recall@3_before": recall_at_k(q.required_tables, retrieved, 3),
            "coverage_after": len(set(q.required_tables) & set(context)) / len(q.required_tables),
            "missed_before": sorted(set(q.required_tables) - set(retrieved)), "missed_after": sorted(set(q.required_tables) - set(context)),
            "retrieval_ms": round(ms, 1)})
    return records


def summarize(records: list[dict]) -> dict:
    def block(rs):
        out = {"n": len(rs)}
        for k in KS:
            out[f"macro_recall@{k}"] = statistics.mean(r[f"recall@{k}"] for r in rs)
            out[f"complete@{k}"] = sum(1 for r in rs if r[f"recall@{k}"] == 1.0)
        return out
    return {"overall": block(records), **{d: block([r for r in records if r["difficulty"] == d]) for d in DIFFICULTIES}}


def summarize_context(records: list[dict]) -> dict:
    def block(rs):
        return {"n": len(rs), "macro_recall@3_before": statistics.mean(r["recall@3_before"] for r in rs),
                "complete_before": sum(1 for r in rs if r["recall@3_before"] == 1.0),
                "macro_coverage_after": statistics.mean(r["coverage_after"] for r in rs),
                "complete_after": sum(1 for r in rs if r["coverage_after"] == 1.0),
                "mean_context_tables": statistics.mean(r["context_size"] for r in rs),
                "min_context": min(r["context_size"] for r in rs), "max_context": max(r["context_size"] for r in rs)}
    return {"overall": block(records), **{d: block([r for r in records if r["difficulty"] == d]) for d in DIFFICULTIES}}


def append_results(run: str, stage: str, config: dict, records: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps({"run": run, "stage": stage, "config": config, **r}) + "\n")
    print(f"appended {len(records)} records for run {run!r} to {RESULTS_PATH.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_name")
    parser.add_argument("--expand-parents", action="store_true", help="evaluate top-3 + direct FK parents instead of plain top-k")
    args = parser.parse_args()
    if RESULTS_PATH.exists() and any(json.loads(l).get("run") == args.run_name for l in RESULTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()):
        sys.exit(f"run name {args.run_name!r} already exists in {RESULTS_PATH.name}; choose a new name")
    gold = load_gold()
    retriever = SchemaRetriever()
    config = {"retriever": "hybrid_bm25_dense_rrf", "rrf_k": RRF_K, "query_instruction": retriever.use_query_instruction,
              "index_db_sha256": retriever.meta["sources"]["database"]["sha256"], "model_revision": retriever.meta["model"]["revision"]}

    if args.expand_parents:
        records = evaluate_context(retriever, gold)
        print(f"{'id':4} {'difficulty':13} {'R@3 before':>10} {'cover after':>11} {'tables':>6}  top 3 + [FK-added]")
        for r in records:
            print(f"{r['question_id']:4} {r['difficulty']:13} {r['recall@3_before']:10.2f} {r['coverage_after']:11.2f} {r['context_size']:6}  "
                  f"{', '.join(r['retrieved_top3'])} + [{', '.join(r['fk_added'])}]" + (f"   ! missed after: {','.join(r['missed_after'])}" if r["missed_after"] else ""))
        s = summarize_context(records)
        print(f"\n{'group':14} {'N':>3} {'R@3 before':>10} {'cover after':>11} {'complete before':>16} {'complete after':>15} {'mean tables (min-max)':>22}")
        for name, b in s.items():
            print(f"{name:14} {b['n']:>3} {b['macro_recall@3_before']:>10.3f} {b['macro_coverage_after']:>11.3f} {str(b['complete_before']) + '/' + str(b['n']):>16} "
                  f"{str(b['complete_after']) + '/' + str(b['n']):>15} {b['mean_context_tables']:>10.2f} ({b['min_context']}-{b['max_context']})")
        n_tables = len(retriever.docs)
        print(f"\nfull-schema context would be {n_tables} tables; mean expanded context is {s['overall']['mean_context_tables']:.2f} ({s['overall']['mean_context_tables'] / n_tables:.0%} of the schema)")
        append_results(args.run_name, "context_fk_parents", {**config, "base_k": CONTEXT_TOP_K, "expansion": "direct_fk_parents_only"}, records)
        return 0

    records = evaluate(retriever, gold)
    print(f"{'id':4} {'difficulty':13} {'R@3':>5} {'R@5':>5}  required -> top 5 ([table] = required table found)")
    for r in records:
        shown = ", ".join(f"[{t}]" if t in r["required_tables"] else t for t in r["retrieved_top5"])
        print(f"{r['question_id']:4} {r['difficulty']:13} {r['recall@3']:5.2f} {r['recall@5']:5.2f}  {','.join(r['required_tables'])} -> {shown}"
              + (f"   ! missed: {','.join(r['missed@5'])}" if r["missed@5"] else ""))
    s = summarize(records)
    n = s["overall"]["n"]
    print(f"\n{'group':14} {'N':>3} {'macro R@3':>10} {'macro R@5':>10} {'complete@3':>11} {'complete@5':>11}")
    for name, b in s.items():
        print(f"{name:14} {b['n']:>3} {b['macro_recall@3']:>10.3f} {b['macro_recall@5']:>10.3f} {str(b['complete@3']) + '/' + str(b['n']):>11} {str(b['complete@5']) + '/' + str(b['n']):>11}")
    print(f"\nreference (arithmetic, not a measurement): a random ranking of 12 tables has expected recall k/12 = {3/12:.3f} at k=3 and {5/12:.3f} at k=5")
    lat = sorted(r["retrieval_ms"] for r in records)
    print(f"retrieval latency over {n} queries (model already loaded): p50 {statistics.median(lat):.1f} ms, max {lat[-1]:.1f} ms")
    append_results(args.run_name, "retrieval", config, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
