# Enhancement experiments on question subsets

Each enhancement is shown on the practice questions it targets, plus control
questions it must not break. The approach has the mentors' approval for
subset evaluation. None of the enhancements needs OCR, re-chunking or
re-indexing.

The built enhancements are switches in agent-service (all off by default,
all reported by `/health` and saved with each run), so a step only changes
environment variables and restarts the agent. Candidates marked *not built*
come from the team's no-preprocessing list and need code first.

Questions were chosen from run `full-100-v1` (Claude Opus 5) and
`FAILURE_ANALYSIS_E2E.md`. These experiments run on Gemini, so step 0
measures the Gemini baseline first. Opus results are not a valid "before" for
Gemini runs.

## Setup, once

1. In the repository-root `.env` (gitignored, never commit it):
   - `LLM_PROVIDER=gemini`
   - `GEMINI_API_KEY=` followed by the Google AI Studio key
   - `GEMINI_MODEL=` a model the key can use (list them with the command below)
   - `GEMINI_TEMPERATURE=0` if the model's guidance allows it. Small subsets
     are noisy, and a low temperature makes repeated runs more alike.
   - leave every enhancement switch `false`
2. Restart the agent: `docker compose up -d --build --no-deps agent-service eval-service`
3. Check: `curl -s http://127.0.0.1:8003/health` should report
   `"llm_provider": "gemini"` and every switch `false`.
4. List the models the key can use:
   `docker compose exec agent-service python -c "import os; from google import genai; print([m.name for m in genai.Client(api_key=os.environ['GEMINI_API_KEY']).models.list()])"`
5. Smoke test, one question, to confirm the key and the structured-output
   schemas work before any real step. From `eval-service/`:
   `python scripts/run_answer_eval.py --ids A002 --label gemini-smoke --out-dir results/answer-eval/gemini/smoke --timeout 600`

The runner stops after 3 errors in a row (`--stop-after-errors`). A step that
hits a daily quota can continue later with `--resume` on the same `--out-dir`.

## What free measurements already show

These call no model: they replay the 211 saved searches of full-100-v1
against the index. The replay reproduces every saved result exactly.

| measurement | result | consequence |
|---|---|---|
| **Rank fusion** (`RANK_FUSION_MERGE`) | right document in the grader's five for 57/95 questions with either merge; per search 77 vs 75 of 211; gains A026, A085, loses A011, A013 | no net retrieval gain; its answer-level step is limited to those four questions |
| **Repeated pages in the five** (for candidate diversification) | 19 of 226 searches, 22 of 1,130 passage slots, 15 questions: A011, A012, A019, A025, A034, A036, A037, A053, A060, A063, A075, A083, A086, A099, A100 | small ceiling: about 2% of slots are wasted on a page already present |
| **Decomposition** (Claude, first 44 questions) | right document found for 30 vs 28 of 42 questions; answers unchanged, because the company rule refused the unnamed pages found | measure it after company context, not only alone (steps 8a/8b) |

## Question sets

| set | questions | why |
|---|---|---|
| format targets | A007, A013, A014, A086 | wrong in full-100-v1 from answer format or calculation rules |
| wrong-form answers | A005, A023, A025, A045, A057, A073, A081 | right content, wrong form in full-100-v1 |
| format controls | A016, A031, A033, A034 | correct numbers with and without scale words; must stay correct |
| company targets | A001, A022, A032, A035, A040, A041, A051, A052, A069, A079, A094 | declined for company identity although the right document was retrieved |
| unanswerable controls | A017, A028, A065, A098, A099 | must still be declined |
| cross-document | A001, A018, A020, A022, A024, A026, A048, A059 | need two companies' documents |
| retrieval misses | the 38 answerable questions whose right document never reached the grader in full-100-v1 (95 − 57); 33 of them were declined | targets for retrieval changes |
| table questions | A003, A037, A068 | a table's heading or neighbouring rows were missing from what the grader saw |
| lookup controls | A002, A006 | simple lookups, correct; retrieval changes must not break them |

**Tuning rule.** Enhancements that tune a threshold or parameter are tuned
on the odd-numbered questions only (A001, A003, ...) and reported on the
even-numbered ones. Tuning and reporting on the same questions would measure
how well the settings fit those questions, not a real improvement.

## Evaluation order

### Part 1: free retrieval screens (no model calls)

Each screen replays the saved searches with the change and reports how often
the right document reaches the grader's five. Only a screen that raises that
rate earns an answer-level step (step 11).

| # | enhancement | status | screen | targets |
|---|---|---|---|---|
| R1 | Rank fusion | built (`RANK_FUSION_MERGE`) | **done: no net gain** (above) | A011, A013, A026, A085 |
| R2 | Candidate diversification: collapse repeated pages before taking the top 5 | not built | replay with repeats collapsed | the 15 questions above |
| R3 | Adaptive Top-K: widen retrieval only on a retry after weak evidence | not built | replay retries at top 10 | retrieval misses |
| R4 | Adjacent evidence expansion: add the neighbouring chunks or table parts of a strong hit | not built | does the expanded set contain the gold facts? | table questions |
| R5 | Fusion and reranker tuning: dense/BM25 weights, RRF constant, reranker candidate count | not built (retrieval-api settings) | sweep on odd-numbered questions, report even | all answerable |

### Part 2: answer-level steps (model calls)

Each step is compared with step 0 on the same questions, except where noted.

| # | enhancement | status | questions | compare with |
|---|---|---|---|---|
| **0** | **Baseline** | — | every question in the sets above | — |
| **1** | Answer format fixes | built (`ANSWER_FORMAT_FIXES`) | format targets + format controls | step 0 |
| **2** | Deterministic answer normalization | not built | wrong-form answers + format controls | step 1 |
| **3** | Answer repair after validation: one constrained correction pass when an answer's form or units are wrong | not built | wrong-form answers + format targets + format controls | step 1 |
| **4** | Company context | built (`GRADE_COMPANY_CONTEXT`) | company targets + unanswerable controls + cross-document | step 0 |
| **5** | Abstention / evidence-threshold calibration | not built | same as step 4, tuned on odd-numbered questions | step 4 |
| **6** | Table-aware query rewriting | not built | table questions + retrieval misses that are table questions + lookup controls | step 0 |
| **7** | Query expansion: company, year, metric and synonym variants, fused | not built | retrieval misses + lookup controls | step 0 |
| **8a** | Decomposition alone | built (`QUERY_DECOMPOSITION`) | cross-document + lookup controls | step 0 |
| **8b** | Decomposition with company context | built | cross-document + unanswerable controls | step 4 |
| **9** | Named table | built (`GRADE_REQUIRE_NAMED_TABLE`) | A068 + lookup controls | step 0 |
| **10** | Rank fusion | built (`RANK_FUSION_MERGE`) | A011, A013, A026, A085 + lookup controls | step 0 |
| **11** | Retrieval screens that passed (R2–R5) | after building | each screen's targets + lookup controls | step 0 |
| **12** | Question-type-specific settings: candidate counts, fusion weights and thresholds per answer type | not built | tuned on odd-numbered questions, reported on even | step 13 without it |
| **13** | Everything adopted | — | every question in the sets above | step 0 |

### Why this order

- **Free screens first.** Retrieval changes can be measured without the
  model, so only the ones that find the right document more often cost quota.
- **Format before grading.** Steps 1–3 change only how a found answer is
  written, so their effect is not mixed with what the grader accepts.
- **Company context before decomposition and expansion.** In the Claude run,
  better retrieval was discarded when the grader refused unnamed pages. With
  step 4 in place, step 8b (compared with step 4) shows what decomposition
  itself adds.
- **Calibration and per-type settings late.** They tune earlier
  enhancements, and they carry the most risk of fitting the practice
  questions, so they use the odd/even split.

## Notes on specific candidates

- **Deterministic answer normalization.** The scorer already compares numbers
  by value: commas, `$`, `%`, parentheses for negatives and trailing zeros do
  not change a score. Normalizing the system's own output helps only forms
  the scorer does not already cover, and the rules must be general. Rules
  written to match the gold answers of known failures would tailor the
  result to the benchmark.
- **Abstention calibration.** The grader is a model judgment (sufficient, plus
  a confidence), not a numeric retrieval threshold. Calibration means the
  grader's instructions or a confidence cut-off. Report it together with the
  unanswerable controls, because fewer refusals can mean more answers from
  the wrong company.
- **Adaptive Top-K.** A fixed top 10 was tested on Claude (experiment C in
  `EXPERIMENTS.md`): no gain on the 10-question set, with 47% more input
  tokens. Widening only on a retry keeps that cost to questions that need it.

## Running a step

Set the step's switches in the shell (they override `.env`) and restart only
the agent. Example for step 1, from the repository root:

```bash
ANSWER_FORMAT_FIXES=true docker compose up -d --no-deps agent-service
curl -s http://127.0.0.1:8003/health        # confirm the switches before spending quota
cd eval-service
python scripts/run_answer_eval.py --ids A007,A013,A014,A086,A016,A031,A033,A034 \
    --label step1-format --out-dir results/answer-eval/gemini/step1-format --timeout 600
```

Switch back to the baseline with `docker compose up -d --no-deps agent-service`
and no variables set.

## Size

The built steps (0, 1, 4, 8a, 8b, 9, 10, 13) run about 130 questions in total,
at about 5–6 model calls per question. Each candidate built later adds its
own subset. Check the key's per-minute and per-day limits in Google AI Studio
before starting. The runner waits out per-minute limits and stops on a daily
limit, and `--resume` continues the next day.

## Limits of this design

- Small subsets are noisy: in the Claude replays, two of eleven answers
  changed between identical runs. Read a step's result per question (which
  answers changed, and why, from the trace), not only as a score.
- The target questions came from the Claude run's failures. Gemini may fail
  differently, and step 0 shows how.
- Step 13 shows the combined effect; the single-enhancement steps show which
  enhancement did what.
