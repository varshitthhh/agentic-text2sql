# Agentic Text-to-SQL (local, learning project)

A small, fully local Text-to-SQL system over one 12-table SQLite database (an online food-delivery schema). A natural-language question goes through schema retrieval, a local LLM, SQL validation, read-only execution and at most one repair attempt. It is a learning and portfolio project, not a production deployment, and it was built to be explained and tested, not to chase a leaderboard.

Everything runs locally: Qwen2.5-Coder through Ollama, local embeddings, no hosted LLM API and no LLM-as-judge.

## Architecture

| Step | What it does | Code |
|---|---|---|
| Schema index | 12 table-level documents (columns, types, keys, a few sample values) and `BAAI/bge-small-en-v1.5` embeddings | `scripts/build_schema_index.py`, `data/index/` |
| Retrieval | BM25 + dense embeddings fused with Reciprocal Rank Fusion (k = 60); prompt context = top-3 tables plus their direct foreign-key parents | `text2sql/retrieval.py` |
| Semantic layer | 14 short business rules (revenue, ratings, status, coupon, flag conventions, join guidance, ambiguities) placed whole in the system prompt | `data/semantic/semantic_layer.json`, `text2sql/semantic.py` |
| Generation | `qwen2.5-coder:7b-instruct-q4_K_M` via Ollama; temperature 0, seed 0, `num_ctx` 4096, `num_predict` 400; JSON-schema output (`status`: ready / needs_clarification / unsupported) | `text2sql/generation.py` |
| Agent | One LangGraph graph: retrieve, generate, route, run SQL, assess, repair, finish | `text2sql/agent.py` |
| Safety | SQLGlot parse and allowlist (single SELECT only); read-only SQLite connection, timeout, row limit; every repaired query is validated again | `text2sql/safety.py`, `text2sql/execution.py` |
| Evaluation | Code-based execution accuracy (EX), valid-SQL rate, Recall@k, latency; append-only JSONL results | `text2sql/eval_runner.py`, `text2sql/equivalence.py`, `text2sql/gold.py`, `docs/eval-protocol.md` |

Design decisions with alternatives and reasons are in [`docs/decisions.md`](docs/decisions.md). Metric definitions and experiment procedures are in [`docs/eval-protocol.md`](docs/eval-protocol.md). Project scope and working rules are in [`CLAUDE.md`](CLAUDE.md).

## Results (measured, one run, one seed)

Run `e2e40_top3_fk_retry1_v1`, 40 gold questions (`data/eval/gold.jsonl`), output in `data/eval/e2e40_results.jsonl`:

| Metric | Result |
|---|---|
| Execution accuracy (EX) | 34/40 (85.0%) |
| Valid-SQL rate | 40/40 |
| Executed without error | 39/40 |
| Repairs | 1 attempted, 0 succeeded |
| Recall@3 / Recall@5 (before FK expansion) | 0.917 / 0.948 |
| Required-table coverage after FK expansion | 0.988 (39/40 complete) |
| Median / max latency per question (CPU only) | 27.6 s / 54.7 s |
| Prompt tokens per question | mean 1,539 (1,046 to 1,774) of a 4,096 context |

The original 20 questions score 19/20 and the 20 harder added questions score 15/20. The six misses are described in decision D-19: one unnecessary join (G12), a join fan-out (G27), a misread "and" (G31), two answers that skipped the requested rounding (G22, G40), and one missing join followed by a failed repair (G39).

Experiment A (original 20 questions): full-schema prompting and top-3 + FK retrieval both scored 19/20 EX (18 questions both correct, one question each way); the full schema used about 33% more prompt tokens (40,216 vs 30,271) and fits the context. This does not show either approach is better (decision D-17).

## Limitations

- Small evaluation: 40 questions written for this project, run once with one seed. One or two questions are not a meaningful difference, and there is no variance estimate.
- The gold SQL and the semantic layer follow the same conventions, so EX measures conformity to those conventions, not general business correctness.
- The 12-table schema is small, which is the case where full-schema prompting is least likely to need retrieval. Nothing here shows how either approach scales.
- Clarification, unsupported-request, repair and unsafe-SQL handling are verified with a scripted fake model (`tests/test_agent_eval.py`). The real model produced none of these on the evaluation questions apart from one failed repair (G39), so its behaviour there is largely unmeasured.
- Latency is machine- and load-dependent; the same configuration showed medians of 48.3 s and 26.9 s in two runs. Any explanation (for example prompt caching) is a hypothesis.
- Experiment B (zero vs. one retry) has one repaired question and is not a meaningful comparison.
- Not implemented: a frontend, deployment, multiple agents, reranking, fine-tuning.

## Setup

Requirements: Python 3.11, [conda](https://docs.conda.io/), [Ollama](https://ollama.com/), about 5 GB of disk for the model. Developed and tested on Windows 11, CPU only (15.7 GB RAM); other platforms are untested.

```
conda create -n text2sql python=3.11
conda activate text2sql
pip install -r requirements.txt          # direct dependencies; requirements-freeze.txt is the full validated environment
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

`requirements.txt` pins `torch==2.14.1`; the original environment used the CPU build. The embedding model `BAAI/bge-small-en-v1.5` is downloaded from Hugging Face the first time it is used.

## Reproducing the database

The SQLite database and the original SQL are **not included** in this repository. The data comes from [Priya-M03/online-food-delivery-sql](https://github.com/Priya-M03/online-food-delivery-sql) (pinned commit `af1cf53e8ff7c48dadfe69299ab1f84e332488d1`). That repository has no license file; its README only says "free to use, fork, and adapt", which is not a formal license, so the SQL and the generated database are left out. The data, schema and table design belong to their original author, not to this project.

To regenerate the database locally:

```
mkdir data/source
# download these two files from the pinned commit into data/source/ :
#   https://raw.githubusercontent.com/Priya-M03/online-food-delivery-sql/af1cf53e8ff7c48dadfe69299ab1f84e332488d1/01_schema.sql
#   https://raw.githubusercontent.com/Priya-M03/online-food-delivery-sql/af1cf53e8ff7c48dadfe69299ab1f84e332488d1/02_data.sql
python scripts/build_sqlite_db.py
```

The script checks the source files against pinned SHA-256 hashes, converts MySQL to SQLite, verifies the result against an independent parse of the source, and writes `data/sqlite/food_delivery.sqlite`. It refuses to overwrite an existing file. The build is deterministic; the expected database hash (`7efa6e3a56098a02...`) is recorded in decision D-14. The committed schema index (`data/index/`) records the database hash it was built from, and retrieval refuses a different database.

The committed index (`schema_docs.jsonl`) and the gold questions contain table and column descriptions and a few sample values taken from that dataset. They are small excerpts needed for the project to work.

## Running

From the project root, with the `text2sql` environment active and the database built:

```
python -B -m text2sql.gold                     # verify all 40 gold queries run and their table labels are right
python -B -m unittest tests.test_agent tests.test_agent_eval tests.test_equivalence tests.test_eval_runner tests.test_execution tests.test_fk_expansion tests.test_generation tests.test_retrieval tests.test_safety tests.test_semantic
python -m text2sql.retrieval "which customers placed orders?"   # retrieval only
```

The test suite (192 tests) uses a scripted fake model, so it does not need Ollama; it does need the database, the committed index and the embedding model in the local Hugging Face cache.

Full evaluation (needs Ollama running; about 20 to 35 minutes on CPU for 40 questions; the run name and output file must be new):

```
python -B -m text2sql.eval_runner my_run_name --all --out data/eval/my_results.jsonl --context retrieval
```

`--context full` uses the full-schema prompt instead (Experiment A). `--ids G04 G13` runs selected questions. Results are appended as one JSON line per question; an existing run name is refused.

## Repository layout

```
text2sql/        retrieval, semantic layer, generation, agent, safety, execution, evaluation runner
tests/           unit tests (stdlib unittest; scripted fake model, no inference)
scripts/         database build and schema-index build
data/eval/       gold questions and recorded run results (JSONL)
data/index/      committed schema index (documents, embeddings, provenance)
data/semantic/   semantic layer
docs/            decisions.md (D-01 to D-19) and eval-protocol.md
references/      index of source links and typed whiteboard notes
```

## License

No license has been chosen for this project's own code, so the default (all rights reserved) applies until the author adds one. Third-party data is covered above.
