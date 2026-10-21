# Evaluation Protocol — Agentic Text-to-SQL

> **Document ownership:** [`CLAUDE.md`](../CLAUDE.md) defines the project's scope and collaboration rules. This file is the single source of truth for evaluation metrics, result comparison, and experiment procedures. [`decisions.md`](decisions.md) records approved design choices and their rationale; it should link here for evaluation-related decisions. [`references/sources.md`](../references/sources.md) indexes external sources and is not a second decision log.

## 1. Purpose and scope

Evaluate the learning-focused Text-to-SQL system using approximately **20–40 verified, answerable questions**. Use code-based evaluation only; a paid LLM-as-judge API is out of scope. Keep the database state and question set unchanged across comparable runs.

Clarification, unsupported-request, and unsafe-query cases are a separate, small set of deterministic **Agent Eval** tests. They do not enter EX because they do not have one expected SQL/result pair.

## 2. Gold-question dataset

Each answerable evaluation case should contain:

- `question_id`
- `question`
- `gold_sql`
- `expected_result`, or a reproducible way to execute `gold_sql` against the same database snapshot
- `required_tables`, manually verified from the gold SQL
- `difficulty`, with exactly one primary label per question (one of `simple`, `join`, `aggregation`, `business_rule`, as used in `data/eval/gold.jsonl`); optional tags may describe secondary features
- `ordered`, true only when row order is part of the expected result (selects ordered versus multiset comparison, section 4)
- Optional notes for required ordering, metric definitions, or important edge cases

Cover single-table queries, joins, filtering, aggregation/grouping, business-semantic rules, and a few complex cases where the schema supports them. Do not force equal counts across categories. Verify the gold SQL and expected result before scoring a case.

**Current set (40 questions).** G01–G20 are the original set (5 questions per difficulty label). G21–G40 (7 `join`, 7 `aggregation`, 6 `business_rule`) were added later and are harder: fan-out and double-counting, anti-joins and missing related records, greatest-per-group, aggregates over aggregates, status-history versus current status, weekday and year logic, and similar-field distinctions. Validation of each added case: the gold SQL passes the read-only validator, executes with a non-empty, non-truncated result, its `required_tables` equal the tables parsed from the SQL, and its result was re-derived independently in plain Python. Questions whose outcome would be degenerate (an empty or zero result), tied in an ordered result, or sensitive to a rounding tie were replaced. `ordered` is true only for G02, G05, G26 and G28. Per-question reasons are in each case's `notes`; the decision is D-18.

Keep a small separate behavior-test set for clarification, unsupported questions, repair, and unsafe-query rejection. A few carefully selected cases are enough for the initial version.

## 3. Metrics

Report **raw counts as well as percentages**. This is a small portfolio evaluation: do not set an accuracy target or claim broad statistical certainty.

### 3.1 Execution Accuracy (EX) — primary metric

`EX = answerable questions whose final predicted SQL executes and returns an equivalent result / total answerable gold questions`.

Invalid SQL, policy rejection, execution errors, timeouts, wrong results, and cases where the agent returns no SQL for an answerable question all count as failures. Use the same database state for gold and predicted queries.

### 3.2 Valid-SQL rate

Report the percentage of answerable questions for which the final SQL candidate—after the configured retry budget—parses successfully and passes the SQL policy checks. A missing SQL candidate counts as invalid. Preserve the initial and final SQL in the per-question result record so repair behavior can be inspected.

### 3.3 Schema Recall@5

This metric is for **table-level retrieval before foreign-key/bridge-table expansion**.

For each answerable question, manually verify `required_tables` from its gold SQL. Take the top five ranked table-level retrieval results (or all results if fewer than five are available). Calculate that question's recall as:

`recall@5(question) = number of required tables found in the top five / total number of required tables`.

Aggregate **Schema Recall@5 as the mean of the per-question recall fractions** across the answerable evaluation questions. It is not the percentage of questions for which every required table was retrieved. Do not silently change the retrieval unit or aggregation rule.

Also report the **number of tables in the final prompt context after relationship expansion** and the prompt-token count for that context. This shows how much schema the model actually receives; the top-five candidate count alone does not show the final context size.

**Implemented context and two separate measures (decided 2026-10-21; see D-15):**
- The prompt context is the retriever's **top 3 tables plus their direct FK parents** (tables referenced by their foreign keys; no children, no recursion). Retrieval ranks are never changed by the expansion.
- **Recall@5** (above) is the protocol metric and is always measured on the top five before expansion. **Recall@3** is the same definition on the top three and is a secondary diagnostic only, not a protocol metric.
- **FK-expanded required-table coverage** ("context coverage") is `|required tables in the final prompt context| / |required tables|`, macro-averaged, reported with the mean number of context tables. It measures the prompt, not retrieval, and is never reported as, or substituted for, Recall@3 or Recall@5.

### 3.4 End-to-end p50 latency

Measure from receipt of the question through retrieval, generation, validation, execution, and any allowed repair, ending at the final result or clarification/unsupported response. Report p50 latency over **answerable evaluation questions**. Run questions sequentially to avoid concurrent-load interference, and exclude model-load/warm-up time (for example, make an untimed warm-up call before measurement).

Record the relevant hardware, model artifact/quantization, context settings, and runtime configuration with the reported measurements. Latency depends on the machine and workload.

Do not include VES in the initial metric set. Do not add or change metrics without discussing it with the user first.

## 4. Result-equivalence rule for EX

Compare the candidate query's result with the gold query's result against the same database state. Use one consistent rule for every run:

- If the question requires ordering (for example, top-N ranking or explicitly ordered output), compare rows in the required order. Otherwise, row order may be ignored.
- Preserve duplicate rows. When row order is not meaningful, compare rows as a **multiset**, not a set.
- Handle `NULL` consistently; never equate it with zero, an empty string, or another value.
- Use type-aware comparison for integer, string, date, and other values. Choose a numeric tolerance before the first scored run only where numeric representation warrants it, record that tolerance with the evaluation configuration, and apply it consistently. Do not choose or adjust a tolerance after seeing results to make a query pass.
- Compare selected columns by position. Ignore aliases unless output-column names are explicitly part of the task. The number and order of output columns must match the gold result; extra, missing, or reordered columns are mismatches.
- Do not normalize away meaningful differences in values, filters, grouping, ordering, or row multiplicity.
- Ensure gold results fit within the configured result limit, or apply an equivalent limit to both gold and candidate queries. Truncated results must not be treated as equivalent evidence.

If equivalence is genuinely unclear for a case, inspect it and document the comparison rule before scoring. Do not change comparison behavior selectively to make a run pass.

**Numeric comparison, decided 2026-10-21 before any scored run** (implemented in `text2sql/equivalence.py`):
- Columns are compared by position. Integers compare exactly. If either value is a float (SQLite stores `DECIMAL` columns as `REAL`; see D-14), the values are equal when `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)`.
- Strings compare exactly and case-sensitively and never equal a number. `NULL` equals only `NULL`.
- Row order matters only when the gold question is flagged `ordered` in `data/eval/gold.jsonl`; otherwise rows compare as a multiset (duplicates count). Truncated results never match. The failure reason is recorded as one of `truncated`, `column_count`, `row_count`, `order`, `values`.
- Basis, measured on the 20 gold queries: result values are int (145), str (421) and float (62) with no NULLs; the worst summation-order noise in the money sums was 2.0e-10 absolute (1.1e-15 relative); omitting a `ROUND(..., 2)` that the question asks for changes a value by up to 0.0045. The tolerance is about 5,000 times the noise and about 3,000 times smaller than that rounding error, so it forgives float noise but not a missing rounding or a one-cent error.
- This was fixed before any generated SQL was scored. Do not change it after seeing results.

## 5. Focused experiments

Run only the following two comparisons initially. Use the same questions, database state, model artifact, semantic-layer content, and generation settings across each comparison; change only the factor being tested.

### Experiment A: full-schema prompting vs. retrieval

- **Baseline:** provide the full schema to the model, using the same schema-document format and the same semantic-layer context as the retrieval configuration.
- **Retrieval configuration:** provide hybrid-retrieved table context, expanded with required foreign-key/bridge-table relationships as designed, plus the same semantic-layer context. Measure Recall@5 before expansion; the model receives the expanded final context.
- Use identical answerable questions for both configurations and set `max_retries=1` (the project default) for both. Experiment B can reuse the selected configuration's one-retry results if that configuration is chosen for further evaluation.
- Use the same `num_ctx` for both configurations. Verify that each prompt fits within the context window while reserving room for generated output and a safety margin; use a compatible tokenizer/runtime estimate where available. Do not run a silently truncated prompt. If the full-schema prompt does not fit, report that as a finding rather than running a truncated baseline.
- Compare EX, valid-SQL rate, p50 latency, and prompt-token count for both configurations. For retrieval, also report Schema Recall@5 and the number of tables in the final prompt context after expansion.
- Report paired outcomes for the same questions: both correct, baseline-only correct, retrieval-only correct, and both incorrect.
- Do not assume retrieval is better. State what the measurements show, including if full-schema prompting performs similarly or better.

### Experiment B: zero retries vs. one retry

- Use the configuration selected after Experiment A; state which configuration was selected and why. Do not assume retrieval must be selected if the results do not support it.
- Compare `max_retries=0` with `max_retries=1`, keeping all other settings and questions identical.
- Compare EX, valid-SQL rate, and p50 latency; inspect whether repair recovered first-attempt failures.
- Start with one retry. Test a second only if observed failure cases provide a concrete reason and the user approves the additional comparison.
- **Decided procedure (2026-10-21, before any run):** the zero-retry condition is read from the first attempts of the one-retry run instead of a second full run. The first generation call is identical in both conditions (same prompt and settings), and repair only starts after the first attempt has been validated and executed. For every question the run must record the first attempt's SQL, validation outcome, execution outcome, result-equivalence outcome and the elapsed time at the end of the first attempt, as well as the final outcome. Zero-retry EX and valid-SQL rate come from the first-attempt fields; its p50 latency is the median of the first-attempt elapsed times and is reported as derived, not separately measured.
- **Validity check:** temperature 0 and a fixed seed do not guarantee identical outputs (section 6). Before reporting Experiment B, re-run the first attempt with `max_retries=0` for at least 5 questions that needed a repair and 5 that did not (all of them if fewer qualify) and confirm the generated SQL equals the recorded first-attempt SQL. If any differ, run the full zero-retry condition separately, with every other setting and question unchanged, and report that run instead.

## 6. Run consistency and result storage

- Use temperature `0` and a fixed seed where supported. Record actual settings. These settings reduce variation but do not guarantee identical outputs.
- Reuse the same database snapshot and retrieval index for paired comparisons.
- Store per-question results for **all evaluation runs in one CSV or JSONL results file**. Choose the format and path when evaluation is implemented. This is an evaluation artifact, not a logging framework.
- **Chosen (2026-10-21):** append-only JSONL, one line per question, each with `run` and `stage` fields; an existing run name is refused. Deliberate deviation from the single-file wording: retrieval-stage runs are in `data/eval/results.jsonl`, the scored end-to-end run in `data/eval/e2e_results.jsonl`, and unscored smoke runs in `data/eval/smoke_results.jsonl`, so scored end-to-end records are not mixed with retrieval-stage or smoke records. The format and fields are identical, and `text2sql/eval_runner.py` reads any of them by run name.
- **Recorded runs (2026-10-21):** `e2e_top3_fk_retry1_v1` (G01–G20; `data/eval/e2e_results.jsonl`), `expA_baseline_top3_fk_retry1_v1` and `expA_full_schema_retry1_v1` (G01–G20; `data/eval/expA_baseline_results.jsonl`, `data/eval/expA_full_results.jsonl`), and `e2e40_top3_fk_retry1_v1` (G01–G40; `data/eval/e2e40_results.jsonl`, 34/40 EX; results and failure analysis in decisions D-17 and D-19). A run on the 40-question set is a different evaluation set from the 20-question runs: compare G01–G20 records within it with earlier runs, but do not present the totals as a controlled before/after comparison. One file per run group is a deliberate deviation from the single-file wording above.
- Each run record should include at least: run/configuration name, question ID, initial and final generated SQL, retrieved top-five table IDs, final prompt-context table count, prompt-token count, validation outcome, execution outcome, result-equivalence outcome, failure category, retry count, and end-to-end latency.
- Use this small failure taxonomy for incorrect cases: wrong table/schema selection, missing or incorrect join, wrong filter/value, wrong aggregation/grouping, wrong output columns, syntax/policy failure, execution failure/timeout, ambiguity/unsupported handling, or other. Assign one primary category where possible; do not invent a cause when evidence is insufficient.
- Report raw counts alongside rates and summarize by the single primary difficulty label where useful. If results are close or surprising, inspect the individual cases before drawing conclusions.

## 7. Lightweight Agent Eval checks

Keep these as small deterministic behavior tests, separate from EX. Status names below are examples; finalize them when the agent output schema is designed.

- **Clarification:** materially ambiguous questions should trigger a clarification rather than silently guessing.
- **Unsupported requests:** questions that cannot be answered using the available schema should receive an explicit unsupported status or a clear safe response.
- **Repair:** a known, fixable validation/execution failure may trigger no more than the configured retry budget; every repaired query must pass validation again.
- **Safety:** disallowed statements must be rejected before database execution. Test this with assertions or mocks; do not execute dangerous statements.

Use assertions over structured statuses, validation decisions, execution calls, and final outcomes. An LLM judge is not required.

### 7.1 Implemented Agent Eval (`tests/test_agent_eval.py`, fake model)

Every scenario drives the public entry point `Text2SQLAgent.run` with a scripted fake model and observes it with spies on `execute_readonly` (executor calls), `validate_select` (validations) and the SQLite `_execute` (real database executions). Each case asserts the outcome (`answered` / `needs_clarification` / `unsupported` / `failed`), model calls, executor calls, database executions, repair count (`len(attempts) - 1`), the stable error code (`stage/code`), that a non-answer never carries a result, that a failure is never reported as a success, that no unused model reply was consumed, and that the database file hash is unchanged.

| # | Scenario | Expected |
|---|---|---|
| 1 | valid SQL | validated, executed once, `answered`, 0 repairs |
| 2 | invalid SQL (execution error, or validation failure) | one repair, revalidated, `answered` |
| 3 | repair also fails / is unsafe | `failed` with the last error code; exactly 1 repair, no third attempt |
| 4 | needs_clarification | specific question passed through; no SQL, no validation, no execution |
| 5 | unsupported | explanation passed through; no SQL, no result, no execution |
| 6 | malformed model output (10 shapes) | `failed` with `generation/<code>`; no validation, execution or repair |
| 7 | unsafe SQL (6 statement kinds) | `failed` with `validation/<code>`; zero database executions, with and without repair budget |
| 8 | valid SQL, wrong against gold | `answered`, `execution_ok`; equivalence reason `values`; not a failure and not repaired |

Also asserted: validation precedes every execution and the executed text is the validator's canonical SQL; only the parsed `sql` field reaches the executor (never the raw JSON, the message, or SQL inside non-`ready` replies; fenced or prefixed JSON is rejected, not salvaged); an exhaustive status × sql × message grid shows a reply is accepted only if `ready` has SQL, or a non-`ready` status has empty SQL and a non-empty message. Retry bounds for `max_retries` 0/1/2 are covered in `tests/test_agent.py`.

**Limitations.** These are fake-model tests of the code paths. They show the agent routes and bounds correctly *given* a reply; they do not show that Qwen produces a clarification for ambiguous questions, declines unsupported ones, or writes repairable SQL. In real-model runs recorded so far, all 23 replies were `ready`; clarification, unsupported, repair and unsafe behavior have not been exercised with real Qwen. Whether a clarification is specific is a prompt-level property checked only by the fake text passing through.
