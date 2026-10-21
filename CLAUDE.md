# Agentic Text-to-SQL — Project Instructions

## 1. Purpose and working mode

Build a learning-focused portfolio project that converts natural-language questions into SQL for **one relational database with 12 tables**. Prioritize a correct, explainable, testable implementation that I can understand and defend in interviews. This is not a production deployment.

We work interactively in the VS Code terminal, **one component/subcomponent at a time** (approval rules in §4). Be precise, practical, and concise; explain the what, why, and trade-offs. Ask a focused question only when a missing detail materially affects a decision; otherwise inspect and propose a next step. Stay within the agreed scope and ask before deviating.

## 2. Agreed architecture

Follow this plan unless I approve a change:

1. **Database and gold questions:** inspect the available database and schema; prepare approximately 20–40 verified question, reference-SQL, and expected-result cases.
2. **Schema index:** extract tables, columns, data types, primary/foreign keys, useful descriptions, and carefully selected sample values. Start with table-level JSONL documents.
3. **Schema retrieval:** BM25 keyword search plus dense embeddings (`BAAI/bge-small-en-v1.5`), combined with Reciprocal Rank Fusion (RRF). Expand with required foreign-key/bridge-table relationships when needed.
4. **Semantic layer:** lightweight business definitions, metric formulas, join rules, data conventions, and ambiguity/default rules. Start with the compact full document; make retrieval selective only if measurement shows a need.
5. **SQL model:** `Qwen2.5-Coder-7B-Instruct`, run locally through Ollama with an available quantized variant. Verify the exact artifact and runtime support before relying on features.
6. **Agent:** one LangGraph workflow for context preparation, SQL generation, clarification/unsupported decisions, validation, execution, and repair. Use structured output/JSON-schema constraints when supported.
7. **Validation and safety:** SQLGlot parsing and AST inspection; an allowlist for permitted read queries; reject disallowed or multiple statements. Use read-only database access (credentials or read-only connection mode), a query timeout, and a result limit. Revalidate every repaired query.
8. **Repair:** begin with `max_retries=1`, kept as a config value. Compare zero vs. one retry; add a second only if results justify it.
9. **Evaluation:** code-based only (no paid LLM-as-judge). Core metrics: execution accuracy (EX), valid-SQL rate, schema Recall@5, end-to-end p50 latency. Add lightweight Agent Eval checks for clarification behavior, repair success, and unsafe-query rejection. Details in `docs/eval-protocol.md`.
10. **Experiments:** only (a) full-schema prompting vs. retrieval, and (b) zero vs. one retry, using the same questions and consistent settings.

## 3. Out of scope unless I approve

Frontend, deployment, multiple agents, a separate routing model, cross-encoder reranking, fine-tuning, paid inference APIs, paid LLM-as-judge evaluation, a production vector database, a full runtime semantic-layer platform, logging/monitoring infrastructure, and broad experiment sweeps. Do not add packages, metrics, abstractions, or components merely because they are available or fashionable.

## 4. Approvals and collaboration

- **Read-only inspection** (listing files, `git status`, version checks, reading docs) needs no approval. Start by inspecting the repository and environment, explain findings, and propose the smallest sensible next step.
- **Before each component/subcomponent:** state its purpose, approach, files affected, trade-offs, and unresolved assumptions. **Wait for my approval before creating or modifying files, installing packages, downloading models, or running implementation commands.**
- **After each component/subcomponent:** show files changed, commands run, actual results, test outcomes, known issues, and a brief explanation. Then ask whether to proceed; never continue automatically.
- Discuss any design change I request before acting on it.
- Estimate runtime before long commands (model downloads, indexing, inference, evaluation) where feasible, and report actual duration.
- Keep a concise progress summary (planned / implemented / tested / approved) **in chat only, not in a file**.
- If a prerequisite is missing or a compatibility issue appears, stop, explain the evidence, and discuss alternatives. Do not silently switch model/provider or weaken the design.

## 5. Environment and compatibility

- Use an isolated **conda environment**. If conda is unavailable, report it and ask me whether to use an existing environment or create a `.venv`; do not install another environment manager automatically. Provide commands compatible with the detected shell/OS.
- Inspect OS, Python version, Git state, RAM, GPU/VRAM, database setup, Ollama availability, installed models, and existing dependencies before choosing implementation details.
- Before coding against a library or model feature, check the installed version and current official documentation. Verify compatibility among Python, LangGraph, SQLGlot, embedding libraries, the database driver, Ollama, the model variant/quantization, structured-output format, and context length (`num_ctx`).
- Prefer few, stable dependencies. Record validated versions in the environment/requirements file; do not guess versions or APIs.
- Use local inference and embeddings by default. Explain approximate storage/resource needs before downloading weights.
- Batch embeddings at index construction and reuse the index. Measure actual prompt token counts rather than assuming them.

## 6. Errors, tests, and safe execution

- Diagnose from actual output and make the smallest justified fix. If a method fails, offer alternatives with trade-offs and wait for my decision.
- Handle expected failures deliberately (malformed model output, invalid SQL, database errors, timeouts, exhausted retries). Do not hide unexpected exceptions or add broad exception handlers.
- Never claim code ran, tests passed, or a metric improved unless verified; show actual output or state what is untested.
- Never modify or delete database data or schema without my explicit permission. Prompts are not a security boundary; enforce read-only policy in code and database access.
- Never hard-code or commit secrets.

## 7. Evaluation rules

- Keep evaluation small and interpretable; verify reference SQL and expected results.
- Define result equivalence in `docs/eval-protocol.md` before reporting EX.
- Report counts alongside percentages; break results down by difficulty/type where useful. Do not claim statistical certainty or set an accuracy target.
- **Recall@5:** defined in `docs/eval-protocol.md` as retrieval of the required tables within the top five results before foreign-key expansion. Report Recall@5 alongside the number of tables in the final prompt context (after expansion) and prompt-token count.
- Use temperature 0 and a fixed seed where the runtime supports them, and record the settings. This reduces variance but does not guarantee identical outputs.
- Store per-question results for each run (configuration, generated SQL, outcome, failure category) in a **single results file in CSV or JSONL format**. Keep the format and path configurable and decide them when evaluation is built. No logging framework.
- Measure p50 latency consistently and treat it as machine- and workload-dependent.
- Do not change metrics or add VES without discussing it with me.

## 8. Git and file hygiene

- Inspect `git status`, branch, remotes, and existing changes before modifying the repository; preserve my unrelated work.
- Keep changes limited to the approved component. Do not duplicate configuration or documentation.
- Show the diff and summary before committing or pushing. Obtain my explicit approval for each commit and each push, confirming branch and remote. Never force-push or run destructive Git operations without explicit approval.

## 9. Documentation and references

Each fact has one home. Read the relevant file only when working on its topic:

- `docs/eval-protocol.md`: Recall@5 definition, result-equivalence rule, difficulty tiers, paired-comparison method, failure taxonomy, experiment definitions. **Read when working on evaluation.**
- `docs/decisions.md`: one entry per design decision (choice, alternatives, reason, source pointer). **Record decisions here, not in code comments or reference notes.**
- `references/sources.md`: index of my source links (URL plus one-line description). Whiteboard notes are in `references/notes/` and papers in `references/papers/`. **Consult these when evaluating architecture decisions.**

Do not assume access to materials I mention but have not added. When using references, state which sources were actually accessible and separate source-supported conclusions from hypotheses. If a source drives a decision, record the decision in `docs/decisions.md` with a pointer to the source.

## 10. Verified environment

*Empty until inspected. Record only verified facts here; do not fill from assumptions.*

- Database dialect and file/location: SQLite (single file), `data/sqlite/food_delivery.sqlite`, built by `scripts/build_sqlite_db.py` from the pinned MySQL source SQL in `data/source/` (decisions D-13, D-14). Rebuilt and verified (30 of 30 checks) on 2026-10-21 in the `text2sql` environment with Python 3.11.17 standard-library `sqlite3`, SQLite library 3.53.4; file SHA-256 starts `7efa6e3a56098a02` (full value and history in D-14).
- OS, shell, Python/conda environment: Windows 11 Home (10.0.26300), PowerShell; conda environment `text2sql`, Python 3.11.17, SQLite 3.53.4, PyTorch 2.14.1+cpu (CPU-only, CUDA unavailable). Validated 2026-10-21; pins in `requirements.txt`, full freeze in `requirements-freeze.txt`.
- Hardware (RAM, GPU/VRAM or CPU-only): CPU-only. 15.7 GB RAM, Intel Core i7-12700H (14 cores, 20 threads), Intel Iris Xe graphics only, no NVIDIA GPU (`nvidia-smi` not found); Ollama reports the model running on 100% CPU.
- Ollama version and exact model artifact/quantization: Ollama 0.40.7 (local API on 127.0.0.1:11434); `qwen2.5-coder:7b-instruct-q4_K_M`, digest `dae161e27b0e`, 4.68 GB, Q4_K_M, 7.6B parameters; about 5.1 GB resident when loaded.
- `num_ctx` and structured-output method: `num_ctx` 4096 with `num_predict` 400, temperature 0, seed 0. Structured output: Ollama `format` set to a JSON schema (`status` enum `ready` / `needs_clarification` / `unsupported`, `sql`, `message`), additionally validated in code (`text2sql/generation.py`). A prompt whose tokens plus the generation budget exceed `num_ctx` is refused, not truncated.
- Measured full-schema prompt token count: 2,000 to 2,029 tokens per question (mean 2,010.8, total 40,216 over 20 questions; largest prompt plus the 400-token budget 2,429 of 4,096), run `expA_full_schema_retry1_v1` (`data/eval/expA_full_results.jsonl`; decisions D-17). For the retrieval configuration (top-3 + FK parents + semantic layer + instructions), the 20-question run `e2e_top3_fk_retry1_v1` (`data/eval/e2e_results.jsonl`) recorded 1,303 to 1,761 prompt tokens per question (mean 1,513.5, total 30,271) and 38 to 96 generated tokens (mean 61.0), as counted by Ollama; the largest prompt plus the 400-token generation budget used 2,161 of 4,096. Ollama's prompt-processing and generation durations were not recorded, so no tokens-per-second figure is claimed.
