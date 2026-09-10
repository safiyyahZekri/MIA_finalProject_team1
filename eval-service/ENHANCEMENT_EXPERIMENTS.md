# Enhancement experiments on question subsets

Each enhancement is shown on the practice questions it targets, plus control
questions it must not break. The approach has the mentors' approval for
subset evaluation. Every enhancement is a switch in agent-service (all off by
default, all reported by `/health` and saved with each run), so a step only
changes environment variables and restarts the agent. No step re-indexes
anything.

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

## Question sets

| set | questions | why |
|---|---|---|
| format targets | A007, A013, A014, A086 | wrong in full-100-v1 from answer format or calculation rules |
| format controls | A016, A031, A033, A034 | correct numbers with and without scale words; must stay correct |
| company targets | A001, A022, A032, A035, A040, A041, A051, A052, A069, A079, A094 | declined for company identity although the right document was retrieved |
| unanswerable controls | A017, A028, A065, A098, A099 | must still be declined; the company rule exists to stop answers from another company's table |
| cross-document | A001, A018, A020, A022, A024, A026, A048, A059 | need two companies' documents: decomposition's target |
| lookup controls | A002, A006 | simple lookups, correct; decomposition and rank fusion must not break them |
| rank-fusion targets | *filled in from the free retrieval replay below* | the right document reaches the merged five only under rank fusion |
| named-table target | A068 | a different table was accepted for "financial highlights" |

## Steps

Each step is compared against step 0 on the same questions, so each shows
one enhancement's effect. The exceptions are step 4b and step 6.

| step | switches on | questions | compare with |
|---|---|---|---|
| **0 Baseline** | none | every question in the sets above | — |
| **1 Answer format** | `ANSWER_FORMAT_FIXES` | format targets + format controls | step 0 |
| **2 Rank fusion** | `RANK_FUSION_MERGE` | rank-fusion targets + lookup controls | step 0 |
| **3 Company context** | `GRADE_COMPANY_CONTEXT` | company targets + unanswerable controls + cross-document | step 0 |
| **4a Decomposition** | `QUERY_DECOMPOSITION` | cross-document + lookup controls | step 0 |
| **4b Decomposition with company context** | `QUERY_DECOMPOSITION`, `GRADE_COMPANY_CONTEXT` | cross-document + unanswerable controls | step 3 |
| **5 Named table** | `GRADE_REQUIRE_NAMED_TABLE` | A068 + lookup controls | step 0 |
| **6 All on** | all five | every question in the sets above | step 0 |

**Why decomposition runs twice.** In the Claude run, decomposition found the
right document slightly more often (30 vs 28 of 42), but the grader refused
the unnamed pages it found, so the answers did not change. Step 4a shows
decomposition alone. Step 4b adds it on top of company context, so its
retrieval gain can reach the answer, and compares against step 3, which
differs from it only by decomposition.

**Step 2 has a free part.** `rank-fusion replay` replays every saved search
from full-100-v1 against the index and merges the results both ways. It
calls no model, measures how often the right document reaches the merged
five, and picks the rank-fusion targets.

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

Steps 0–6 run about 125 questions in total, plus the rank-fusion targets. At
about 5–6 model calls per question, that is roughly 700–800 calls. Check the
key's per-minute and per-day limits in Google AI Studio before starting. The
runner waits out per-minute limits and stops on a daily limit, and `--resume`
continues the next day.

## Limits of this design

- Small subsets are noisy: in the Claude replays, two of eleven answers
  changed between identical runs. Read a step's result per question (which
  answers changed, and why, from the trace), not only as a score.
- The target questions came from the Claude run's failures. Gemini may fail
  differently, and step 0 shows how.
- Step 6 shows the combined effect; the single-switch steps show which
  enhancement did what.
