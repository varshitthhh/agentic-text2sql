# Project Handoff: Agentic Text-to-SQL

Purpose: a briefing for a person or coding agent who upgrades this project later. It points to the files that own each fact and does not repeat them. Read `CLAUDE.md` first (scope and working rules), then this file, then `docs/decisions.md` and `docs/eval-protocol.md` for the topic you touch.

State when written: complete and published (commit `5107627`, branch `main`). All figures below are from one run and one seed and are not a general accuracy claim.

## 1. Plan (what was agreed)

A local, learning-focused portfolio system over one 12-table SQLite database. It was not built for production or deployment. The agreed architecture is in `CLAUDE.md` section 2:

1. Gold questions with verified reference SQL and results.
2. Table-level schema index (JSONL documents plus `BAAI/bge-small-en-v1.5` embeddings).
3. Hybrid retrieval: BM25 + dense, fused with RRF; top-3 tables plus direct FK parents.
4. Semantic layer: short business rules, placed whole in the system prompt.
5. `qwen2.5-coder:7b-instruct-q4_K_M` through Ollama, JSON-schema output.
6. One LangGraph workflow: retrieve, generate, route, run SQL, assess, repair, finish.
7. SQLGlot allowlist validation, read-only execution, timeout, row limit.
8. At most one repair attempt (`max_retries=1`).
9. Code-based evaluation only (EX, valid-SQL, Recall@k, latency, tokens).
10. Two experiments only: A (full schema vs. retrieval) and B (zero vs. one retry).

Out of scope unless approved: frontend, deployment, multiple agents, reranking, fine-tuning, paid APIs, LLM-as-judge, production vector DB.

## 2. What was done

| Area | Where | Notes |
|---|---|---|
| Database | `scripts/build_sqlite_db.py` | Rebuilds SQLite from the pinned upstream MySQL SQL. D-13, D-14. DB and source SQL are **not in the repo** (upstream has no license). README explains the rebuild. |
| Index and retrieval | `scripts/build_schema_index.py`, `data/index/`, `text2sql/retrieval.py` | 12 documents. Index records the DB hash and refuses a different DB. D-15. |
| Semantic layer | `data/semantic/semantic_layer.json`, `text2sql/semantic.py` | 14 rules. |
| Generation | `text2sql/generation.py` | Ollama wrapper, prompt layout, response parser. A non-`ready` reply needs a non-empty message (D-07). A prompt that would not fit `num_ctx` is refused, never truncated. |
| Agent | `text2sql/agent.py` | One graph. Repair runs only after a validation or execution error. |
| Safety | `text2sql/safety.py`, `text2sql/execution.py` | Allowlist on the parsed AST, execution of the re-rendered SQL, `mode=ro` plus `query_only`, 10 s timeout, 1000-row limit. D-08. |
| Evaluation | `text2sql/eval_runner.py`, `equivalence.py`, `gold.py`, `data/eval/` | Result-set equivalence, not SQL text match. Append-only JSONL. 40 gold questions (G01-G40). |
| Tests | `tests/` (10 modules, 192 tests) | Scripted fake model, so no Ollama needed. Agent Eval scenarios in `tests/test_agent_eval.py`, described in `eval-protocol.md` section 7.1. |

Recorded runs are listed in `eval-protocol.md` section 6; the decisions are D-01 to D-19.

## 3. Results to know (one run, one seed)

- **Final run (`e2e40_top3_fk_retry1_v1`, 40 questions):** EX 34/40, valid-SQL 40/40, executed 39/40, repairs 1 attempted and 0 succeeded. Original 20 questions 19/20; added 20 harder questions 15/20. D-19.
- **Experiment A (original 20):** retrieval and full schema both 19/20. Full schema used about 33% more prompt tokens (40,216 vs 30,271) and fits in context. Neither was shown to be better. D-17.
- **Experiment B:** only one repaired question, so there is no meaningful comparison.
- **Not measured:** the semantic layer's contribution (no ablation was run), the real model's behaviour on clarification, unsupported and unsafe-SQL cases, and variance across seeds.

## 4. Known issues and open questions

- Five of the six misses were valid SQL returning wrong results (G12, G22, G27, G31, G40). Neither validation nor repair can catch these.
- G22 and G40 skipped the requested rounding. Hypothesis: a prompt line about applying requested rounding would help. Not tested.
- G39's repair failed because the model answered `unsupported`, although the table's columns were in the prompt. Hypothesis: the "no such column" feedback misled it. Not tested.
- G28's context lacked `orders` but the answer was correct, apparently from semantic-layer text (an inference, untested). FK expansion adds parents only, so child tables can be missed.
- The semantic layer, prompt and gold questions were written by the same person, so EX measures conformity to those conventions. The original 20 questions were seen during development.
- Latency on CPU varies a lot between runs (48.3 s vs 26.9 s median for the same configuration). Treat it as indicative.

## 5. Rules for the next agent

- Work on one component at a time, and ask before creating files or running implementation commands. Show results and stop (`CLAUDE.md` section 4).
- Never modify `data/eval/gold.jsonl` G01-G40, the recorded result files, the database, source SQL or the index unless the user explicitly asks.
- Never overwrite a results file. Use a new run name and output path for every run.
- Local inference only. Do not add hosted APIs, LLM-as-judge, or new dependencies without approval.
- Do not tune prompts or rules against the 40 gold questions and then report that score as unbiased. For a clean score, write a new held-out set first.
- Report counts with denominators. Separate measured findings from hypotheses. Record decisions in `docs/decisions.md`, not in code comments.
- Run the tests from the project root with `python -B -m unittest` on the modules listed in the README.

## 6. Suggested upgrade paths

Ordered by evidence, not by effort. Each needs the user's approval before work starts.

1. **Semantic-layer ablation.** Run the 40 questions once with the business rules removed (needs a small runner option). It would measure how much the layer helps and how often it rescues cases like G28. About 20 minutes of inference.
2. **Prompt and repair-feedback fixes** for the rounding and G39 failures. Evaluate them on a new held-out set, not on the 40.
3. **A larger or held-out gold set**, including clarification and unsupported cases, to get less noisy comparisons and unbiased numbers.
4. **Scaling to about 50 tables and 2-4 databases.** Retrieval stops being optional and Recall@k becomes meaningful. Add a database-routing step, selective retrieval of semantic-layer rules, and extend FK expansion beyond parents.
5. **Repair evaluation** with many more questions, or a seeded-error probe (measures repair ability, not benefit on natural errors).

Do not build: a frontend, deployment, multiple agents, cross-encoder reranking or fine-tuning, unless a measured need is shown first.
