# Architecture Decisions — Agentic Text-to-SQL

This file records **why** the project choices were made. Overall scope and collaboration rules live in [`../CLAUDE.md`](../CLAUDE.md); evaluation definitions and procedures live in [`eval-protocol.md`](eval-protocol.md). External source URLs live in [`../references/sources.md`](../references/sources.md).

**Source attribution:** References are sources to consult; a reference listed below does not by itself prove that it supports a decision. Treat a source as supporting a claim only when it has been accessed and the relevant content checked.

Decision status means the choice is agreed for the initial implementation, not that the component has already been implemented or benchmarked. Validate environment-dependent details before relying on them.

## D-01 — Keep the project small and single-database

- **Decision:** Build a learning-focused portfolio project for one relational database with 12 tables.
- **Alternatives considered:** Enterprise-scale deployment, multiple databases, multi-agent orchestration, frontend and hosted infrastructure.
- **Reason:** The goal is to understand and defend each component, not recreate a full enterprise platform.
- **Status:** Agreed scope.
- **References:** Whiteboard notes; [S1], [S4], [S8] for production concerns to understand without automatically implementing all of them.

## D-02 — Use Qwen2.5-Coder-7B-Instruct locally through Ollama

- **Decision:** Use the instruction-tuned Qwen2.5-Coder-7B model as the SQL-generation and repair model, running locally through Ollama with a compatible available quantization.
- **Alternatives considered:** Hosted paid inference, a larger model, a separate model for each agent stage.
- **Reason:** Local inference avoids per-request hosted-model costs and keeps the architecture easy to inspect. The model variant, quantization, runtime compatibility, memory needs, context settings, and structured-output support must be verified on the actual machine.
- **Status:** Agreed and verified: `qwen2.5-coder:7b-instruct-q4_K_M` (digest `dae161e27b0e`) on Ollama 0.40.7, running CPU-only; configuration recorded in `CLAUDE.md` section 10 and D-16.
- **References:** Whiteboard notes; [S3], [S4].

## D-03 — Start with table-level JSONL schema documents

- **Decision:** Extract schema metadata and represent each table as an initial JSONL document, including its columns, data types, primary/foreign keys, useful descriptions, and carefully selected sample values.
- **Alternatives considered:** One document per column, a relational catalog table, a document store, or a production vector database.
- **Reason:** Table-level chunks preserve local table context and are simple for a 12-table project. If retrieval results show that table-level chunking is a bottleneck, discuss column-level chunking before trying it.
- **Status:** Agreed initial representation; validate retrieval quality rather than assuming it is optimal.
- **References:** [S1], [S2], [S4].

## D-04 — Use hybrid schema retrieval: BM25 + dense embeddings + RRF

- **Decision:** Combine keyword search with dense-vector similarity and merge rankings with Reciprocal Rank Fusion (RRF). Start with `BAAI/bge-small-en-v1.5` for dense embeddings. Expand retrieved schema with required foreign-key or bridge-table relationships where needed.
- **Alternatives considered:** Full-schema prompting alone, BM25 alone, vector-only retrieval, cross-encoder reranking, and a managed vector database.
- **Reason:** BM25 helps match exact identifiers; dense retrieval helps with semantic phrasing; RRF combines the rankings without directly comparing their raw scores. Lightweight local indexes suit the current scale.
- **Status:** Agreed initial approach; compare against full-schema prompting under the evaluation protocol. Do not assume retrieval improves accuracy before measuring it.
- **References:** [S1], [S2], [S4]; RESDSQL/schema-linking discussion in the attached paper.

## D-05 — Begin with a compact context-form semantic layer

- **Decision:** Maintain a small, curated document describing business definitions, metric formulas, join rules, data conventions, and ambiguity/default rules. Initially provide the compact document as prompt context.
- **Alternatives considered:** Raw schema only, selective retrieval of semantic definitions from the start, or a full runtime semantic-layer platform.
- **Reason:** Database structure does not necessarily encode business meaning. A compact context document is straightforward to inspect and maintain; selective retrieval or deterministic runtime enforcement can be considered only if evidence or context size justifies it.
- **Status:** Agreed initial approach; business definitions must be verified and kept current.
- **References:** Attached paper, *Semantic Layers for Reliable LLM-Powered Data Analytics*; [S4].

## D-06 — Use one LangGraph workflow, not multiple agents

- **Decision:** Use one LangGraph agent/workflow for context preparation, SQL generation, structured decisions, validation, execution, and repair.
- **Alternatives considered:** Plain Python control flow, multiple specialized agents, or a separate routing model.
- **Reason:** The workflow has real conditional branches for clarification, validation failure, execution failure, and bounded repair, but the project does not need multi-agent complexity. LangGraph is the chosen orchestration framework; it is not itself the schema retriever or database.
- **Status:** Agreed. Revisit only if implementation exposes a concrete limitation.
- **References:** [S3], [S6], [S8]; whiteboard notes.

## D-07 — Use structured clarification and unsupported-request handling

- **Decision:** Have the agent return a structured status, such as `ready`, `needs_clarification`, or `unsupported`, and branch deterministically in the workflow. Use JSON-schema-constrained output if the chosen Ollama/model combination supports it.
- **Alternatives considered:** Always guessing from the question, adding a separate disambiguation model, or introducing a separate routing agent.
- **Reason:** Material ambiguity should be clarified instead of silently resolved arbitrarily. A structured branch reuses the existing model and workflow.
- **Status:** Implemented. Replies are constrained by a JSON schema (`status` enum, `sql`, `message`) and validated in code, which rejects malformed JSON, missing or extra keys and inconsistent status/SQL combinations (unit-tested with a fake model). In real use, 23 of 23 recorded replies (3 smoke, 20 evaluation) were valid and all had status `ready`; no recorded reply was `needs_clarification` or `unsupported` (the warm-up request asks for an `unsupported` reply verbatim, and its output is not recorded), so those paths are verified only with the fake model.
- **Contract clarification:** a `needs_clarification` or `unsupported` reply must carry a non-empty `message` (the clarifying question, or why it is unsupported). An empty or whitespace-only message is rejected as malformed (`empty_message`) so the user never receives a contentless response. Found by probing the status × sql × message grid; the prompt already required a message.
- **References:** [S4], [S6], [S8]; attached semantic-layer paper.

## D-08 — Apply minimal, independent SQL safety controls

- **Decision:** Use SQLGlot to parse SQL into an AST and enforce an allowlist for permitted read queries; reject disallowed/multiple statements. Enforce read-only access, a timeout, and a result limit at the database/execution layer. Revalidate every repaired SQL query.
- **Alternatives considered:** Relying on prompting, string matching alone, or implementing an elaborate semantic SQL validator.
- **Reason:** Parsing catches structural issues, policy checks restrict allowed statements, and database permissions provide an independent safety boundary. None of these alone guarantees business-semantic correctness.
- **Status:** Agreed and implemented for SQLite (D-14). The SQLGlot validator (`text2sql/safety.py`: exactly one SELECT, optionally with CTEs or set operations; table, function and placeholder allowlists; the executed text is the canonical SQL re-rendered from the validated AST) and the read-only executor (`text2sql/execution.py`: `mode=ro`, `query_only`, timeout, row limit) are unit-tested, including that unsafe SQL never reaches the database and that repaired SQL is validated again. Behaviour against real model output is covered by D-16.
- **References:** [S1], [S2], [S4], [S8].

## D-09 — Start with one repair retry

- **Decision:** Set `max_retries=1` initially, as a configuration value. Compare zero retries with one retry. Consider a second only if observed failures give a concrete reason and the user approves.
- **Alternatives considered:** No repair loop, multiple unbounded retries, or a broad retry sweep.
- **Reason:** Execution feedback can help fix repairable SQL failures, while a bounded loop avoids uncontrolled latency and repeated execution.
- **Status:** Agreed initial setting; benefit must be measured.
- **References:** [S2], [S3], [S8].

## D-10 — Use code-based evaluation and a small fixed question set

- **Decision:** Start with approximately 20–40 verified question/reference-SQL/expected-result cases, plus a small separate set of behavior tests. Use code-based evaluation rather than a paid LLM-as-judge service.
- **Alternatives considered:** A 100+ question benchmark immediately, paid LLM judging, and broad metric/ablation sweeps.
- **Reason:** This is a focused learning project. Deterministic execution/result comparison, validation checks, retrieval measurements, and workflow assertions are reproducible and inexpensive. Small-set results should be reported with raw counts and without broad statistical claims.
- **Status:** Agreed; evaluation methodology is defined in [`eval-protocol.md`](eval-protocol.md).
- **References:** [S5], [S7]; attached semantic-layer paper.

## D-11 — Use a focused metric set and two comparisons

- **Decision:** Use execution accuracy (EX), valid-SQL rate, schema Recall@5, and end-to-end p50 latency. Do not include VES initially. Run only (a) full-schema prompting vs. retrieval and (b) zero vs. one repair retry.
- **Alternatives considered:** Additional efficiency metrics, large ablation matrices, and unmeasured claims about expected accuracy.
- **Reason:** The metrics cover final correctness, syntactic/policy validity, retrieval quality, and user-visible speed. VES is omitted unless reconsidered with evidence. Details and definitions belong only in [`eval-protocol.md`](eval-protocol.md).
- **Status:** Agreed. Do not change metrics or add experiments without discussion.
- **References:** [S5], [S7]; attached semantic-layer paper.

## D-12 — Keep development interactive and changes reviewable

- **Decision:** Inspect first; discuss one component/subcomponent at a time; wait for approval before implementation actions; show actual results after each component and ask whether to proceed. Preserve unrelated Git work and require explicit approval for every commit and push.
- **Alternatives considered:** Autonomous end-to-end implementation, unreviewed environment changes, and automatic Git operations.
- **Reason:** The user wants to understand the implementation, ask questions in the terminal, and approve design changes rather than receive an opaque code dump.
- **Status:** Agreed collaboration rule, defined in [`../CLAUDE.md`](../CLAUDE.md).
- **References:** User's working preferences captured in `CLAUDE.md`.

## D-13 — Use the Priya-M03 online-food-delivery dataset (MySQL source SQL)

- **Decision:** Use `food_delivery_db` (12 tables) from https://github.com/Priya-M03/online-food-delivery-sql, pinned to commit `af1cf53e8ff7c48dadfe69299ab1f84e332488d1` (the repository's only commit at review time, 2026-07-26). Local copies of `01_schema.sql` and `02_data.sql` are kept in `data/source/`. `03_views_procedures_triggers.sql` is deliberately not downloaded or loaded, so views, functions, procedures and triggers stay out of schema extraction and out of any write path.
- **Alternatives considered:** Not recorded; the user selected this dataset.
- **Reason:** It matches the 12-table, single-database scope and targets MySQL. Source review (not yet confirmed on a live server) showed 15 declared foreign keys, `orders` as the hub, `order_items` as the bridge, and 1,303 `INSERT` tuples (130 orders). It also has semantic ambiguities to exercise the semantic layer, such as stored `restaurants.avg_rating` differing from the average of `ratings.rating`, and several possible definitions of revenue.
- **Integrity record (SHA-256):**
  - `01_schema.sql` (10,876 bytes): `6a0e6266ea6ebb5c33dbf66afa2da6521981b725943b6f5a7800633f20db12a9`
  - `02_data.sql` (71,994 bytes): `4975b09b39e5d38305cd9d421d88ceff318c73f55dafdfd4959c96be930d2f91`
  - Raw URL pattern: `https://raw.githubusercontent.com/Priya-M03/online-food-delivery-sql/af1cf53e8ff7c48dadfe69299ab1f84e332488d1/<file>`
- **License:** Unclear. GitHub reports no license, there is no `LICENSE` file, and the README says only "free to use, fork, and adapt". Do not redistribute the SQL files publicly without clarifying; cite the source instead.
- **Caution:** `01_schema.sql` begins with `DROP DATABASE IF EXISTS food_delivery_db`. Check that the target server has no database of that name before running it. This applies only to running the original SQL against MySQL; the SQLite build in D-14 never executes it.
- **Amendment (2026-10-21):** The MySQL load was abandoned and never executed. No MySQL database or account was created for this project, and the local MySQL server was not modified by it. The source SQL is instead converted to a SQLite file; see D-14. The source files in `data/source/` remain the pinned, unmodified provenance and are the input to that conversion.
- **Status:** Source files downloaded and hash-verified. Target engine changed from MySQL to SQLite (D-14); the SQLite database was built from them and verified against an independent parse of these files.
- **References:** Source repository above (read directly; the README's mention of `04_interview_queries.sql`, `sql/` and `docs/` does not match the repository contents).

## D-14 — Use SQLite as the project database

- **Decision:** Keep the dataset as a single SQLite file, `data/sqlite/food_delivery.sqlite`, built by `scripts/build_sqlite_db.py` from the pinned MySQL source SQL in `data/source/` (D-13). Gold-result generation, execution, schema extraction and evaluation run against this file. The SQLGlot dialect and the generation prompt must therefore be SQLite (not yet implemented).
- **Alternatives considered:** The MySQL 8.0 server that was the original target (D-13). It was dropped because the local server needed credentials that the agent could not enter and the user chose not to hand over, so authentication was costing time. Other engines were not evaluated.
- **Reason:** SQLite needs no server or credentials, is reproducible from the pinned source files, and gives a code-level read-only boundary through the connection mode. Python's standard `sqlite3` module removes the need for a separate database driver. Everything else in the architecture (model, retrieval, semantic layer, LangGraph workflow, validation approach, repair, evaluation scope) is unchanged.
- **Compatibility decisions made in the conversion** (`scripts/build_sqlite_db.py`; all 12 tables, 15 foreign keys, 10 named indexes and all 1,303 rows are preserved):
  - `CREATE DATABASE`, `DROP DATABASE`, `USE`, and the character set/collation clause are removed; `SET FOREIGN_KEY_CHECKS` lines are removed. Foreign keys are enforced during the build (`PRAGMA foreign_keys=ON`).
  - `INT AUTO_INCREMENT PRIMARY KEY` becomes `INTEGER PRIMARY KEY` (12 columns). `AUTOINCREMENT` is deliberately omitted so no `sqlite_sequence` table appears among the tables.
  - Six `ENUM(...)` columns become `TEXT` with a `CHECK (col IN (...))` constraint, keeping their `DEFAULT`.
  - Other declared type names (`VARCHAR(n)`, `DECIMAL(p,s)`, `BOOLEAN`, `TINYINT`, `DATETIME`, `DATE`) are kept as written, and `CHECK`, `UNIQUE`, named foreign keys with `ON DELETE` actions, `DEFAULT` values, and the stored generated column `order_items.subtotal` are unchanged.
  - The 40 `SHA2('passN',256)` calls in the data (MySQL function, unavailable in SQLite) are replaced by the equivalent lowercase SHA-256 hex computed with Python's `hashlib`. Their equivalence to MySQL's output was not checked against a MySQL server.
  - Text uses SQLite's default collation (case-sensitive `BINARY`) for now, by the user's decision. The MySQL source used a case-insensitive collation; whether to add `COLLATE NOCASE` is left open and should be revisited if case-related gold or failure cases appear.
  - `menu_items.created_at` is not supplied by the source `INSERT`, so the schema's `DEFAULT CURRENT_TIMESTAMP` would stamp the build time. The build stores `NULL` in that column for all 108 rows (the `DEFAULT` stays in the DDL) and runs `VACUUM` after the load, so builds are reproducible. This was a deliberate change on 2026-10-21; the column is unpopulated and must not be used in gold questions, sample values or the semantic layer. It is the only column in the schema with a non-deterministic default; the other seven `DEFAULT CURRENT_TIMESTAMP` columns are supplied by the source.
- **Verified (database rebuilt 2026-10-21 with the deterministic script in the `text2sql` environment, Python 3.11.17, SQLite library 3.53.4; 30 of 30 checks passed):** source hashes match D-13; 12 tables and the per-table row counts in the build script, total 1,303; all 6,810 inserted cells and all 88 NULLs equal an independent parse of the source; column names, order, types, NOT NULL, defaults, primary keys and 9 UNIQUE columns match the source DDL; the 15 foreign keys and 10 named indexes match the source; `PRAGMA foreign_key_check` reports no violations; `PRAGMA integrity_check` is `ok`; `subtotal` equals `quantity * price_at_order` for all 307 rows; enum `CHECK` and foreign-key enforcement work on an in-memory copy; a `mode=ro` connection refuses `INSERT`, `CREATE` and `DROP` and leaves the file unchanged. The current file's SHA-256 is `7efa6e3a56098a02f86785ff666f0d13686d650e42dff7d7d1002d242d451e0c` (221,184 bytes). Further builds made seconds apart on SQLite 3.53.4 produced the same bytes; byte identity on other SQLite versions was not tested, so correctness rests on the script's checks and the logical comparison. The first build (2026-10-21, base Python 3.13.5, SQLite 3.45.3, 29 of 29 checks) had SHA-256 `a6b7f903c3159bd482576608ef8663bdff51fd227bd0195e59fb5606684b4a25` (229,376 bytes) and was superseded; it differed from the current file only in `menu_items.created_at`, which held that build's timestamp (`2026-10-21 15:18:14`).
- **Known dialect differences and implications** (observed in-memory on SQLite 3.45.3 unless marked):
  - `DECIMAL` has no fixed-point type: values are stored as `REAL`, or `INTEGER` when whole (for example `orders.total_amount` is 129 REAL and 1 INTEGER). Money aggregates can carry floating-point error. The numeric comparison tolerance is decided in `eval-protocol.md` §4 (relative 1e-9, absolute 1e-6).
  - Integer division truncates (`7/2` is `3`), so ratio and percentage queries need a float operand. This belongs in the semantic layer and gold SQL.
  - `=` is case-sensitive (`'A'='a'` is false) while `LIKE` is case-insensitive for ASCII.
  - `YEAR()`, `DATE_FORMAT()` and `NOW()` do not exist; dates are stored as text such as `2024-10-16 20:23:00` and need `strftime` or string functions. The data spans 2024-03-01 to 2025-06-15, so relative dates need an explicit "today" rule.
  - A bare column with `GROUP BY` is accepted and returns an arbitrary row's value (no `ONLY_FULL_GROUP_BY`), so such SQL can be valid yet wrong; this falls under the "wrong aggregation/grouping" failure category.
  - `PRAGMA foreign_keys` is off by default on each connection.
  - There is no `information_schema`; schema extraction must use `sqlite_master` and `PRAGMA` calls.
  - Not tested here: MySQL `ENUM` columns sort by declared position, whereas these columns are `TEXT` and sort lexically.
- **Read-only and safety mechanics:** Implemented and unit-tested in `text2sql/execution.py`: a `mode=ro` connection (rejects writes and schema changes), `PRAGMA query_only`, a wall-clock timeout through `set_progress_handler` (SQLite has no native statement timeout), and a row limit through `fetchmany(limit + 1)` with a truncation flag. Not implemented: `set_authorizer`; the SQLGlot validator of D-08 provides the allowlist.
- **Environment:** conda environment `text2sql`, Python 3.11.17, SQLite library 3.53.4, PyTorch 2.14.1+cpu (CPU-only; CUDA unavailable). Direct pins are in `requirements.txt` and the full freeze in `requirements-freeze.txt`.
- **Open items:** case-sensitivity (above).
- **Status:** Agreed; the database is built and verified. No downstream component (schema index, retrieval, agent, execution layer, evaluation) is implemented yet.
- **References:** `scripts/build_sqlite_db.py` and its run output. SQLite's own documentation was not consulted; the dialect behaviours above were observed locally on SQLite 3.45.3. The MySQL-side statements about `ENUM` ordering and collation come from the source DDL and general knowledge, not from a source checked in this project.

## D-15 — Schema index and retrieval as implemented: table-level documents, hybrid retrieval, top-3 + FK-parent context

- **Decision (implements D-03, D-04):**
  - **Index** (`scripts/build_schema_index.py`, artifacts in `data/index/`): 12 table-level documents with columns, types, keys, foreign keys, the source DDL comment as description (empty for `restaurants`, `delivery_partners`, `coupons`), and sample values for an explicit allowlist of 13 categorical columns (at most 12 values each). Never indexed: `password_hash`, emails, phone numbers, person names, addresses, free text, dates. `menu_items.created_at` is marked "always NULL". Embeddings: `BAAI/bge-small-en-v1.5` at a pinned revision (384 dimensions, normalized, no passage instruction). `index_meta.json` records the hashes of the database, source SQL, documents and embeddings; `text2sql/retrieval.py` refuses a stale or altered index.
  - **Retrieval** (`text2sql/retrieval.py`): BM25 (`rank-bm25`, default parameters; lowercase tokens, snake_case identifiers split and kept whole, simple plural folding, a short function-word stop list) fused with dense cosine ranking by RRF (k = 60). Only tables with a positive BM25 score are BM25-ranked; ties break by document order, so output is deterministic. Queries use the BGE query instruction by default (untuned).
  - **Prompt context:** the top 3 retrieved tables, then their **direct FK parents only** (no children, no recursion), parents ordered by FK column position. Original ranks are kept, and retrieved and FK-added tables are labelled.
- **Alternatives considered:** column-level indexing (not adopted: every measured miss was the single hub table `restaurants` ranked 6th, and D-03 requires discussion before trying it); top-5 plus parents (8.8 tables on average versus 6.7); one-hop expansion in both directions (10.9 tables, close to the full schema).
- **Evidence (20 gold questions, labels written before retrieval was run):** Recall@3 0.908 (complete on 16/20), Recall@5 0.938 (17/20), FK-expanded required-table coverage 1.000 (20/20) at 6.70 context tables on average (56% of the schema). A random ranking of 12 tables would expect 0.250 at k=3 and 0.417 at k=5. Definitions of the three measures are in `eval-protocol.md` section 3.3; coverage is not Recall.
- **Status:** Implemented; unit-tested (retrieval 23, FK expansion 15); retrieval-only measurement above. Limits: N = 20, questions authored in this project, 12-table schema (Recall@5 is lenient), no confidence intervals.
- **References:** `data/eval/results.jsonl` (runs `retrieval_hybrid_v1`, `retrieval_fk_parents_v1`); model card and revision recorded in `data/index/index_meta.json`.

## D-16 — Generation, agent, safety and evaluation runner as implemented; first full evaluation

- **Decision (implements D-02, D-06, D-07, D-08, D-09):**
  - **Generation** (`text2sql/generation.py`): Qwen through the Ollama client behind a small `ModelClient` interface (a fake in tests); temperature 0, seed 0, `num_ctx` 4096, `num_predict` 400, JSON-schema `format`. Prompt layout for prefix caching: a system message holding the stable instructions and the whole semantic layer (D-05), then a user message with the retrieved and FK-added schema documents (D-15) and the question last. A prompt that does not fit `num_ctx` is refused, never truncated.
  - **Agent** (`text2sql/agent.py`): one LangGraph `StateGraph`: retrieve, generate, route by status, run_sql, assess, repair, finish. Every SQL attempt goes through `execute_readonly` (validation per D-08, then the read-only connection). Repair happens only after a validation or execution failure, at most `max_retries` times (default 1), and the repaired SQL is validated again. `needs_clarification`, `unsupported` and malformed replies end the run without repair. Each attempt keeps its SQL, outcomes and timing so the zero-retry condition can be read from first attempts (`eval-protocol.md` section 5).
  - **Runner** (`text2sql/eval_runner.py`): runs gold questions through that agent and appends one JSONL record per question; EX rule, tolerance and denominators are in `eval-protocol.md` sections 3, 4 and 6.
- **Status, kept distinct:**
  - *Implemented and unit-tested:* the full suite (174 tests) passes, including fake-model coverage of clarification, unsupported, unsafe SQL, repair, failed repair (no third attempt) and malformed replies.
  - *Smoke-tested with the real model:* 3 questions (G04, G13, G16), 3/3 correct; `data/eval/smoke_results.jsonl`, not scored.
  - *Fully evaluated, one run:* all 20 gold questions, run `e2e_top3_fk_retry1_v1`, file `data/eval/e2e_results.jsonl`. Configuration: Qwen as above, top-3 + FK-parent context, `max_retries` 1, executor timeout 10 s and 1000-row limit, semantic layer and database hashes recorded in every record.
- **Result (N = 20, every metric over all 20 questions):** EX 19/20; valid-SQL 20/20; executed-OK 20/20; repair attempted 0/20 (repair success: not applicable, no repaired question); outcomes all `answered`. Recall@3 0.908, Recall@5 0.938, FK-expanded coverage 1.000 at 6.70 mean context tables. Total latency per question: median 48.3 s, mean 44.5 s, maximum 60.9 s, on CPU. Prompt tokens 1,303 to 1,761 per question (total 30,271, mean 1,513.5); generated tokens 38 to 96 (total 1,220, mean 61.0); the largest prompt plus the generation budget used 2,161 of 4,096. The only miss, G12 (aggregation), joined `order_items` unnecessarily, which changed every category average (differences up to 63.4); the generated SQL was valid and executed, so no repair was triggered.
- **Not done or not observed:** the full-schema versus retrieval comparison (Experiment A) has not been run; Experiment B has no data because no question needed repair (its derived zero-retry results equal the one-retry results, 19/20 and 20/20, and its determinism check has nothing to check); the real model produced no recorded clarification, unsupported or unsafe-SQL reply (the 20 questions are all answerable by design); no variance estimate (one run, one seed).
- **Limits on interpretation:** 20 questions written in this project, 5 per difficulty, all answerable by design; the semantic-layer conventions come from the same notes the gold SQL follows, so EX measures conformity to those conventions rather than general business correctness; the 12-table schema makes Recall@5 lenient. Latency varied from 14.5 s to 60.9 s for similar prompt sizes, and the cause was not measured because the runner records wall time per model call, not Ollama's prompt-processing and generation durations. This is a small portfolio evaluation, not evidence of production readiness.
- **References:** `data/eval/e2e_results.jsonl`, `data/eval/smoke_results.jsonl`, `docs/eval-protocol.md`.

## D-17 — Experiment A (full schema vs. retrieval) result on the original 20 questions

- **Decision:** Record the measured outcome of Experiment A and draw no general conclusion from it. Experiment B's configuration is not selected here; this result does not support or rule out either context strategy.
- **What was run:** two sequential runs of G01–G20 with identical settings (Qwen `qwen2.5-coder:7b-instruct-q4_K_M`, temperature 0, seed 0, `num_ctx` 4096, `num_predict` 400, `max_retries` 1, same semantic layer, safety checks, executor limits and equivalence rule). Only the schema context differed: `expA_baseline_top3_fk_retry1_v1` (top-3 + direct FK parents, `data/eval/expA_baseline_results.jsonl`) and `expA_full_schema_retry1_v1` (all 12 table documents in index order, `data/eval/expA_full_results.jsonl`). The full-schema condition reuses the unchanged prompt code (`--context full` in the runner), so its schema block still opens with "Tables most relevant to the question:".
- **Measured (N = 20 each):**
  - EX 19/20 in both; valid-SQL 20/20 in both; executed-OK 20/20 in both; no repairs in either.
  - Paired EX: both correct 18; baseline-only correct 1 (G10); full-schema-only correct 1 (G12); both incorrect 0.
  - Generated SQL identical for 13 of 20 questions; the other 7 are G01, G08, G10, G12, G15, G17 and G19, of which only G10 and G12 changed correctness.
  - Prompt tokens: baseline 30,271 total (mean 1,513.5, range 1,303–1,761, largest prompt plus the 400-token budget 2,161 of 4,096); full schema 40,216 total (mean 2,010.8, range 2,000–2,029, largest 2,429 of 4,096), which is 9,945 more tokens (+32.9%). The full schema fits the configured context on this dataset, and nothing was truncated.
  - The baseline rerun reproduced the earlier `e2e_top3_fk_retry1_v1` run exactly (20 of 20 identical SQL, identical token counts).
  - Total latency per question: baseline median 26.9 s, max 44.1 s; full schema median 8.3 s, max 34.6 s. The earlier baseline run on the same configuration had a median of 48.3 s.
- **Hypothesis, not a finding:** the full-schema prompt prefix is the same for every question, so Ollama may reuse its prompt cache, which would explain the lower latency. This was not measured (Ollama's prompt-processing and generation durations are not recorded). Because the same baseline configuration took 48.3 s and 26.9 s median in two runs, the latency difference is not a generalizable result about either strategy.
- **Not supported:** that either strategy is more accurate. With 20 questions, one flip in each direction, one seed and one run, the comparison cannot establish superiority. The 12-table schema is small enough that full-schema prompting fits easily, which is the case least likely to need retrieval; the result says nothing about larger schemas.
- **Reason for recording:** `eval-protocol.md` requires that results are stated without assuming retrieval is better. The earlier statement in D-16 that Experiment A had not been run is superseded by this entry; D-16's recorded results are unchanged.
- **References:** `data/eval/expA_baseline_results.jsonl`, `data/eval/expA_full_results.jsonl`, `data/eval/e2e_results.jsonl`, `docs/eval-protocol.md` section 5.

## D-18 — Expand the gold set to 40 questions (G21–G40)

- **Decision:** Add 20 harder, answerable questions to `data/eval/gold.jsonl` (appended; G01–G20 byte-identical), within the 20–40 range in CLAUDE.md. The four existing difficulty labels and the record format are unchanged; features such as fan-out, anti-joins and temporal logic are described in each question's `notes`.
- **Alternatives considered:** new difficulty labels or fields (rejected: changes the gold format), fewer than 20 additions, and rewording questions so that retrieval covers them (rejected: gold questions must not be tuned to the system).
- **Validation (before any model output existed for these questions):** each question and its SQL were written first; every SQL passes the read-only validator and executes under the executor limits with a non-empty, non-truncated result; its table label equals the tables parsed from the SQL; and its result was re-derived independently in plain Python from raw table rows and found equivalent under the project's equivalence rule (20 of 20). Candidates that failed validation were replaced, not loosened: an order-total comparison returned 0 (the 21 apparent differences were floating-point noise), a tracking-history question returned 0, a top-N list had tied totals, and a two-decimal rounding had an exact tie (Kolkata 4.125; G33 was changed to one decimal).
- **Known retrieval finding:** G28 ("revenue of each cuisine type") has retrieval coverage 0.50 because `orders` is not in the top-3 + FK context. It is kept as a valid question. Recall@5 for G21–G40 is complete except G26 (0.67) and G28 (0.50). No model has been run on G21–G40.
- **Limits:** the questions were written in this project against the same semantic layer, so they test conformity to its conventions. Several depend on data quirks (for example, 6 coupon orders predate the coupon's valid-from date). The 40-question set is still small; do not read differences of one or two questions as meaningful.
- **References:** `data/eval/gold.jsonl`, `docs/eval-protocol.md` section 2.

## D-19 — First 40-question end-to-end evaluation (retrieval configuration)

- **What was run:** all 40 gold questions (D-18), run `e2e40_top3_fk_retry1_v1`, `data/eval/e2e40_results.jsonl`, started 23:37:57 and finished 23:56:31 on 2026-10-21 (18.6 min wall time). Configuration identical to the Experiment A baseline: Qwen `qwen2.5-coder:7b-instruct-q4_K_M`, temperature 0, seed 0, `num_ctx` 4096, `num_predict` 400, `max_retries` 1, top-3 + direct FK parents, whole semantic layer, executor timeout 10 s and 1000 rows. The command was `python -B -m text2sql.eval_runner e2e40_top3_fk_retry1_v1 --all --out data/eval/e2e40_results.jsonl --context retrieval`. One run, one seed, no variance estimate.
- **Different evaluation set:** this is a different, larger set than the original 20 questions, not a controlled before/after experiment. The first 20 records reproduce the earlier 20-question runs exactly (20 of 20 identical final SQL and EX outcomes).
- **Measured (N = 40, every metric over all 40):**
  - EX 34/40; valid-SQL 40/40; executed-OK 39/40; repair attempted 1/40 and repair succeeded 0/1; outcomes 39 answered, 1 failed (G39), 0 clarification, 0 unsupported at the first attempt, 0 malformed.
  - G01–G20: EX 19/20, executed-OK 20/20, no repairs. G21–G40: EX 15/20, executed-OK 19/20, 1 repair (failed).
  - By label (EX): simple 5/5, join 10/12 (G27, G39), aggregation 9/12 (G12, G22, G31), business_rule 10/11 (G40). Of the 20 new questions: join 5/7, aggregation 5/7, business_rule 5/6.
  - Recall@3 0.917 (complete 32/40), Recall@5 0.948 (35/40), FK-expanded coverage 0.988 (39/40) at 6.83 mean context tables (3 to 9).
  - Latency (total per question): median 27.6 s, mean 27.3 s, max 54.7 s; G01–G20 median 26.7 s, G21–G40 median 29.0 s. Retrieval median 0.029 s. Prompt tokens 63,098 total (mean 1,539.0, range 1,046–1,774); generated 2,814 (mean 68.6, range 38–115); the largest prompt plus the 400-token budget used 2,174 of 4,096. No truncation or context-limit error.
- **Failures (6 EX misses, all with complete retrieval of the required tables):**
  - Valid SQL, wrong result: G12 (unneeded `order_items` join; the same miss as in every earlier baseline run), G27 (join fan-out counts each order's discount once per vegetarian line: 12,429.79 vs 6,259.79), G31 (customers active in either year, `IN ('2024','2025')`, instead of both: 39 vs 20).
  - Valid SQL, correct values, not rounded as asked: G22 and G40 omit the requested two-decimal `ROUND`, so the values differ at the 6th digit or beyond (3.142857 vs 3.14; 1559.92275 vs 1559.92). Under the EX rule these are misses; the rule was not changed.
  - Execution failure: G39 referenced `coupons.valid_from` without joining `coupons` (`no such column`). The repair attempt replied `unsupported` claiming those columns do not exist, although the coupons schema document in the prompt lists them, so repair failed (`repair/repair_not_ready`).
- **Retrieval findings:** G28 retrieved only restaurants, delivery_partners and categories (coverage 0.50, `orders` absent) but was answered correctly; its SQL used `orders` columns that are named in the semantic layer, so the layer, not the schema context, probably supplied them (hypothesis; not tested). G26 has Recall@5 0.67 but complete context coverage and is correct. No miss is attributable to retrieval on this evidence.
- **Hypotheses, not findings:** the model tends to skip a requested rounding when the question states it at the end; a repair message that says "no such column" can lead the model to conclude the column does not exist; the drop from 19/20 to 15/20 on G21–G40 reflects harder questions (fan-out, set logic, joins to a table the model omitted). Candidate fixes (a prompt line about applying requested rounding, join guidance in the repair feedback) were not implemented or tested.
- **Not shown:** that the system is accurate in general; that the harder half is significantly worse (differences of one to three questions at N = 20 per half); Experiment B beyond a single repaired question; real-model clarification, unsupported or unsafe-SQL behavior (none of the questions called for it).
- **References:** `data/eval/e2e40_results.jsonl`, `data/eval/e2e_results.jsonl`, `data/eval/expA_baseline_results.jsonl`, D-16, D-17, D-18.

## Source index

Source IDs (S1–S8) are defined in [`../references/sources.md`](../references/sources.md). The source index identifies references; this file records decisions and rationale. The evaluation protocol owns metric definitions and experiment procedures.
