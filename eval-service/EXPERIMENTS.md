# Answer-level experiments — Claude Opus 5

2026-09-10. agent-service on `claude-opus-5`: effort at the API default
(`high`), `max_tokens` 16000, `fallbacks: "default"`. Index of 2,446 documents
and 14,386 chunks.

Every run used the same 10 practice questions, two per answer type:
A001 and A002 (arithmetic), A005 and A009 (span), A006 and A016 (multi-span),
A036 and A094 (count), A017 and A028 (unanswerable). Before each run the live
agent's `/health` was checked against the intended settings, and a snapshot
was saved to `results/answer-eval/experiments/<run>/run_info.json`.

## Runs

| run | change | why |
|---|---|---|
| R0 baseline | the Opus 5 migration alone | reference |
| A1 format | extraction asks for the exact span: no labels, years, units or commentary | A009 and A016 had the right fact wrapped in extra text |
| A2 repeat | A1 again, unchanged | Opus 5 accepts no `temperature`, so answers vary; this measures that noise |
| B entity | A1 + grading requires evidence tied to the company the question names | A017 was answered from another company's table |
| C top-10 | A1 + `TOP_K_FINAL` 10 instead of 5 | the gold chunk sat just outside the top 5 |
| D1 citations | B + answers cite the passages extraction names | answers cited top-ranked hits, not their source |
| D2 citations + top-10 | D1 + `TOP_K_FINAL` 10 | ten chunks, re-measured on the final setup |

## Results

| run | EM | F1 | C | U | F | A | W | E | fallback citations | cost | input tok/q | latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 baseline | 0.40 | 0.56 | 3 | 1 | 3 | 2 | 1 | 0 | – | $0.510 | 6,971 | 26 s |
| A1 format | 0.50 | 0.60 | 5 | 0 | 1 | 3 | 1 | 0 | – | $0.539 | 7,123 | 26 s |
| A2 repeat | 0.50 | 0.60 | 4 | 1 | 1 | 3 | 1 | 0 | – | $0.499 | 6,560 | 25 s |
| B entity | 0.60 | 0.70 | 6 | 0 | 1 | 3 | 0 | 0 | – | $0.638 | 7,997 | 31 s |
| C top-10 | 0.60 | 0.70 | 5 | 1 | 1 | 2 | 1 | 0 | – | $0.750 | 11,335 | 23 s |
| **D1 citations** | 0.50 | 0.60 | 5 | **0** | 1 | 4 | **0** | 0 | **0** | $0.641 | 7,903 | 32 s |
| D2 citations + top-10 | 0.40 | 0.50 | 4 | 0 | 1 | 5 | 0 | 0 | 0 | $0.798 | 11,655 | 29 s |

**C** correct, cited correctly · **U** correct, but no cited document holds the
gold facts · **F** right content, wrong form · **A** abstained · **W** wrong ·
**E** error. "Cited correctly" means a cited document contains the question's
`gold_facts`. "Fallback citations" counts answers cited from top-ranked hits
because the model named no evidence; it exists only from D1 on.

| question | type | R0 | A1 | A2 | B | C | D1 | D2 |
|---|---|---|---|---|---|---|---|---|
| A001 | arithmetic | A | A | A | A | A | A | A |
| A002 | arithmetic | C | C | U | C | C | C | C |
| A005 | span | F | F | F | F | F | F | F |
| A006 | multi-span | C | C | C | C | C | A | A |
| A009 | span | F | C | C | C | C | A | A |
| A016 | multi-span | F | C | C | C | C | C | C |
| A017 | unanswerable | W | W | W | **C** | W | **C** | **C** |
| A028 | unanswerable | C | C | C | C | C | C | C |
| A036 | count | A | A | A | A | A | A | A |
| A094 | count | U | A | A | A | U | **C** | A |

## Findings

**1. Format fix: adopted.** A009 went from `"0 per cent (2018: 0 per cent) –
per the taxation accounting policy…"` to `"0 per cent"`, and A016 from
`["2019: $2,657 thousand", "2018: $3,035 thousand"]` to `["2,657", "3,035"]`.
Both held in every later run that reached extraction; A009's abstentions in D1
and D2 happened at grading, before extraction ran. A005 is unchanged:
`"Because the Company…"` against gold `"as the Company…"` is a paraphrase,
which F1 (0.91) credits and exact match does not.

**2. Noise is large at this sample size.** A1 and A2, with identical settings,
differ in one outcome. Across all seven runs questions swing on grading and
retrieval variance alone: in B, A006's rewritten query found the debtors note,
while in D1 it went looking for "creditors" and did not. Ten questions separate
large, consistent effects only. Each finding here rests on the trace, not on
one run's score.

**3. Company-match grading: adopted, with a real cost.** A017 asks for
Atlassian's "Microcontrollers and Digital ICs Group" revenue, a segment
Atlassian does not have. It was answered from STMicroelectronics' table in all
four runs without the rule and declined in all three runs with it:

> "this segment structure (ADG, AMS, MDG) belongs to STMicroelectronics, not
> Atlassian. Nothing in the evidence identifies Atlassian, so the figures cannot
> be tied to the named company." — B

The rule also refused A009 in D1 and D2, where the passage holding the fact
never names Intu. D1: "nothing in the evidence identifies the issuer as Intu —
it only refers to 'the Group'". In D2 the grader even noted that "contextual
chunks confirm the source is intu properties plc Annual report 2019", and still
refused. In B, A009 passed only on its third grading attempt. **This corrects
the earlier finding that the rule cost no correct answers.** It trades answers
built on another company's figures for refusals when a company's own excerpt
does not carry its name, which favours faithfulness over coverage. The obvious
refinement to test: accept the company name when it appears in another passage
from the same document.

**4. Ten chunks: not adopted.** Run C gained A094 through retrieval. Run D2,
with the final setup, showed no gain: A094's table was retrieved but the answer
failed at the calculator (finding 6), and A036 and A001 were still missed.
Against D1 it costs 47% more input tokens and 25% more spend.

**5. Citation selection: fixed.** Answers used to cite the top-ranked retrieval
hits (`_top_evidence()` in `agent-service/app/graph.py`), so a citation could
point at a document that did not contain the answer. Every U outcome in R0–C
was a correct answer cited that way, not a lucky guess: in R0 A094, A2 A002 and
C A094 the grader quoted the supporting figures, yet no cited document held
them. This corrects an earlier reading of R0's A094 as right by coincidence.

Extraction now numbers the evidence passages and names the ones its answer
came from, and the answer cites exactly those (`_citations()`). In D1 and D2,
every citation was named by the model, every one was among the passages the
final retrieval returned, and no correct answer was cited wrongly. A094 in D1
is correct and cited to Microsoft's table for the first time. The fix did not
cause the A006 and A009 abstentions: both ended at grading and never reached
extraction.

**6. The calculator cannot express threshold counts (found, not fixed).** In
D2, A094's grader accepted Microsoft's table and extraction wrote
`(508>100)+(151>100)+(397>100)`. The calculator rejected it: "Unsupported
expression element: Compare". Earlier runs wrote `1+1+1`, which the calculator
accepts but which hides the figures the count rests on. Supporting comparisons
(true counts as 1) would let such counts be computed from the cited figures.

## Langfuse traces

| question | before | after |
|---|---|---|
| A009 format | [R0](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/e195be18d906cabe0b06a90d9b0b1cf2) | [A1](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/fc51de4ea5dd7d932869b590dfb8555b) |
| A016 format | [R0](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/7be2ae59b288846b0b2690878fecf1de) | [A1](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/78ad60be6e312494bf48cce507dd8f63) |
| A017 company match | [R0](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/0db5fce0333688b68685d0435e104e52) | [B](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/fb8fcbaa46260692001a179f70df589a) |
| A094 ten chunks | [A1](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/cb06f5e0d3066cdb470158b8caf1b8c4) | [C](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/28dcdfc132f33fbd1fb2fc1aed454f19) |
| A094 citation fix | [C, cited wrongly](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/28dcdfc132f33fbd1fb2fc1aed454f19) | [D1, cited to Microsoft's table](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/096a46b57441a24dd60847bd394b1387) |
| A009 company match cost | [B, passes on third grading](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/fa781fed815af58f121e730f32c72eaa) | [D1, refused: excerpt never names Intu](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/c5a5812b44b241b81e3fb0f95149ab2a) |
| A094 calculator gap | – | [D2, comparison formula rejected](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/4f2a4e460b91f81f1106cd9770617a8d) |

The six experiment runs cost $3.87 in total.

## Reproducing

From the repository root, start agent-service with the run's settings, for
example run D2:

```bash
TOP_K_FINAL=10 GRADE_REQUIRE_ENTITY_MATCH=true docker compose up -d --no-deps agent-service
```

Then, from `eval-service/`:

```bash
python scripts/run_answer_eval.py --sample 10 --label D2-cite-entity-top10 \
    --out-dir results/answer-eval/experiments/D2-cite-entity-top10
python scripts/compare_answer_runs.py path/to/chunks.jsonl \
    R0=results/answer-eval/combined.json \
    D2=results/answer-eval/experiments/D2-cite-entity-top10/combined.json
```
