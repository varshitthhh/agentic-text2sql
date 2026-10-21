# Reference Sources — Agentic Text-to-SQL

This file is an index of the user's selected sources for project design and interview preparation. Use the original source for detailed claims; a title or short description here is only a navigation cue. Do not assume an external page is accessible unless it has actually been opened/read in the current environment.

## Articles and repositories supplied by the user

### S1. AI Workflow Lab — Text-to-SQL LLM production: schema linking and guardrails (2026)
URL: https://aiworkflowlab.dev/article/text-to-sql-llm-production-schema-linking-guardrails-2026

Use when reviewing: schema linking/retrieval, metadata enrichment, query validation, guardrails, and practical production failure modes.

### S2. llms.blog — Text-to-SQL in production: schema linking, value retrieval, and execution-guided self-correction
URL: https://www.llms.blog/posts/text-to-sql-in-production-schema-linking-value-retrieval-and-execution-guided-self-correction

Use when reviewing: schema linking, value grounding, execution feedback, and bounded SQL self-correction.

### S3. GitHub — `UmitFSD/agentic-text-to-sql`
URL: https://github.com/UmitFSD/agentic-text-to-sql

Use when reviewing: an example agentic Text-to-SQL implementation and its workflow/implementation choices. Inspect the repository itself before asserting specific implementation details.

### S4. Google Cloud — Techniques for improving Text-to-SQL
URL: https://cloud.google.com/blog/products/databases/techniques-for-improving-text-to-sql

Use when reviewing: methods for improving schema context, retrieval, query generation, and overall Text-to-SQL reliability.

### S5. GitHub — `kasivisu4/txt2sql-benchmark`
URL: https://github.com/kasivisu4/txt2sql-benchmark

Use when reviewing: benchmark structure, execution-based evaluation, and metric implementation. Confirm the repository's actual metric formulas before using them.

### S6. Shirokoff — Agentic Analytics: What Changes
URL: https://shirokoff.ca/blog/agentic-analytics-what-changes

Use when reviewing: what agentic analytics changes about workflow, tool use, and analytical task execution. Read the article before attributing specific claims to it.

### S7. NucleusBox — Evaluating Text-to-SQL agents with NucleusIQ
URL: https://www.nucleusbox.com/evaluating-text-to-sql-agents-with-nucleusiq/

Use when reviewing: agent-evaluation concepts, including final outcomes and workflow/trajectory behavior. Treat product-specific claims in the source as such.

### S8. NiteAgent — Text-to-SQL agents in production (2026)
URL: https://niteagent.com/blog/text-to-sql-agents-production-2026/

Use when reviewing: production concerns and practical considerations for Text-to-SQL agents. Verify individual recommendations in the original article.

## User-provided paper

### P1. *Semantic Layers for Reliable LLM-Powered Data Analytics: A Paired Benchmark of Accuracy and Hallucination Across Three Frontier Models* (Cube)
Local project path: `papers/text2sql-paper.pdf` (not included in the public repository because the paper is third-party; obtain it from its original publication)

Use when reviewing: the distinction between raw database schema and business semantics; metric definitions/formulas, table-selection conventions, join paths, data quirks, and disambiguation; context-form versus runtime-form semantic layers; and the paper's paired evaluation design. Its reported accuracy gains belong to its specific dataset/setup and must not be treated as a guaranteed result for this project.

## User-provided whiteboard

### W1. Agentic Text-to-SQL project whiteboard
Image path: [`notes/whiteboard.jpg`](notes/whiteboard.jpg)
Typed extraction: [`notes/whiteboard-notes.md`](notes/whiteboard-notes.md)

Use when reviewing: the user's original scope and brainstorming—Qwen model selection, schema linking/hybrid retrieval, JSON/table chunking, semantic layer, query validation, guardrails, execution feedback with one agent, evaluation, ambiguity handling, schema updates, retrieval/routing failures, inference, throttling, and advanced topics marked for possible interview discussion. Some handwriting is unclear; the typed extraction explicitly preserves uncertainty.

## How to use these sources

- Consult sources relevant to the component being discussed; do not load everything into every coding task.
- Keep architecture decisions and their rationale in `docs/decisions.md`, not here.
- Keep metric definitions, result-equivalence rules, and experiment procedures in `docs/eval-protocol.md`.
- If a source is inaccessible or has not been inspected, say so and do not claim it supports a detail you have not verified.
- The user intends to draw partly from these sources, not copy any one source as the complete architecture.
