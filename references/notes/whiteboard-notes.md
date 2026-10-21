# Whiteboard Notes — Agentic Text-to-SQL Project

This is a typed extraction of the user's whiteboard image. Some handwriting is unclear; uncertain readings are explicitly marked. Refer to [`whiteboard.jpg`](whiteboard.jpg) for the original.

## Core project

- Title: “Agentic Text2SQL Project”.
- LLM selection: “How to select right LLM?”
- Model written: “Qwen 2.5 SQL coder” (likely Qwen2.5-Coder); the user's selected model is `Qwen2.5-Coder-7B-Instruct`.
- Scope: 12 tables, 1 database.

## Main components written on the board

- Schema linking → retrieval / hybrid search.
- JSON and table chunking discussed.
- “Cross Encoder” appears crossed out or marked with an X; treat as a possible future/what-if topic, not an implementation requirement.
- Schema metadata fields: column, sample values, description, type, table.
- Semantic layer / database context; one note about runtime form is partially unclear.
- Query validation (checked).
- Guardrails (checked).
- Execution feedback + retry loop, with one agent.
- LLM agent: Qwen.

## Evaluation ideas on the board

- “EX”, “VES”, and a circled term that resembles “SQL rank” (uncertain).
- LLM-as-judge and code-based evaluation options were sketched. The user's chosen plan is code-based evaluation; do not claim an LLM judge was implemented.
- A circled list appears to mention single-turn/multi-turn, statistical metrics, pass@k, plan, tools, trajectory, output (“o/p”), and one unclear term that may read “role mid”.
- A right-hand grouping may mention “LLM”, “RAG”, and “ETA” or a similar abbreviation; uncertain.

## Additional / interview what-if topics

- Schema update handling.
- Retrieval failures and routing failures.
- Prompting, context construction, and context passing.
- Success criteria and inference.
- Embedding model.
- Disambiguation, naming/routing, natural-language chat.
- Domain fine-tuning for semantic ranking.
- “Retriever = Vector DB, LangGraph, etc.” (handwritten shorthand, not a precise component definition).
- “Routing → no need” (context unclear).
- Throttling and rate limiting.
- Query options.
- Graceful degradation, semantic layer, and agent.

## Current agreed implementation scope

Use the project instructions in `CLAUDE.md` as the authoritative current plan. The whiteboard captures brainstorming; it does not override the more recent agreed scope. Cross-encoder reranking, fine-tuning, separate routing, multiple agents, and production deployment are out of scope unless the user later approves a change.
