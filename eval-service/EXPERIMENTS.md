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

## Full practice set: 100 questions

Run `full-100-v1` used the adopted setup, D1: top 5 chunks, company-match
grading, and citations named by the model. It crashed at question 40 on a
console encoding error. It was resumed from the saved batches, after `/health`
confirmed the agent's configuration was unchanged (fixed in `2cc1d20`). All 100
questions were answered, with 0 errors and 100/100 schema-valid responses.

| metric | value |
|---|---|
| exact match | 0.25 |
| F1 | 0.35 |
| numerical accuracy | 0.64: 7 of the 11 answers where both answer and gold are numbers. Over all 42 numeric-gold questions, counting abstentions as wrong: 0.17 |
| retrieval over cited evidence, k=5 | recall 0.337 · precision 0.305 · hit rate 0.337 · MRR 0.326 |
| latency | mean 29.4 s · p50 30.2 s · p95 51.2 s |
| LLM calls per question | 4.97, with 1.26 grading retries on average |
| tokens per question | 7,683 input / 934 output |
| cost | $6.18 total, $0.062 per question |

| outcome | C | U | F | A | W | E | fallback citations |
|---|---|---|---|---|---|---|---|
| all 100 | 26 | 1 | 7 | 52 | 14 | 0 | 0 |

| answer type | n | EM | C | U | F | A | W |
|---|---|---|---|---|---|---|---|
| arithmetic | 40 | 0.12 | 6 | 1 | 0 | 29 | 4 |
| span | 40 | 0.20 | 8 | 0 | 7 | 18 | 7 |
| multi-span | 13 | 0.54 | 7 | 0 | 0 | 3 | 3 |
| count | 2 | 0.00 | 0 | 0 | 0 | 2 | 0 |
| unanswerable | 5 | 1.00 | 5 | 0 | 0 | 0 | 0 |

The weakest scenario is cross-document questions: 8 questions, 0 C, 1 U and
7 A. EM is lower than on the 10-question sample (D1: 0.50) because that sample
was stratified, and 2 of its 10 questions were unanswerable, against 5 of 100
here.

**7. Most declines are retrieval misses.** 95 questions are answerable; the
agent declined 52 and answered 43. The gold document appeared in retrieval, at
any attempt, for 57 of the 95. In the 38 questions where it never appeared,
the agent declined 33 times rather than answer from the wrong document. It
declined 19 questions even though the gold document was retrieved. By a keyword
heuristic over the grader's reasons, which makes the count approximate, 11 of
those 19 cite company identity: A001, A022, A032, A035, A040, A041, A051, A052,
A069, A079 and A094. That is the cost described in finding 3.

The retrieval scores above are computed over cited evidence. A declined answer
cites nothing, so it scores as a miss: the hit rate is 0.34, while gold was
retrieved for 60% of questions. Better retrieval would move this system more
than anything else, for example queries that carry the company name, or
filtering by company.

**8. When the agent answers, it is mostly right.** Of the 43 answerable
questions it answered, 29 have the right content (C 21, U 1, F 7). Reading the
14 wrong answers by hand, 6 are right in substance but scored wrong:

- A096 gives `0.105675` against gold `0.11`. The gold is rounded to two
  decimals, and the scorer allows 1% relative error.
- A014 gives `31.27`, in millions, against `31,252` thousand. The operands came
  from rounded figures in the narrative text.
- A086 gives `["$12.2", "$23.6", "$38.6"]` without "million", although the
  source writes "million" after each figure. The format fix, which asks for no
  units outside the span, was over-applied (`FAILURE_ANALYSIS_E2E.md`, A086).
- A049 answers `IBM` for "International Business Machines Corporation".
- A043 gives a shorter span of the gold sentence (F1 0.72, under the 0.8 used
  for F).
- A013 has the right reason, plus `"p1"` appended as a value because the
  question asks for a page.

The other 8 are genuinely wrong:

- A007 averaged three year-on-year changes instead of two.
- A011 gave an absolute change where the gold is a percentage.
- A038 and A077 used the wrong figures.
- A050 and A067 gave a different explanation.
- A037 and A068 returned incomplete lists.

The scorer was not changed after seeing these results. Rounding to the gold's
precision, and matching company aliases, should be agreed on first and then
applied to every run.

**9. The one U is a judge limitation, not a bad citation.** A018 computed
Jabil's and Advanced Energy's 2018 gross profit from quarterly tables. All
eight operands were confirmed in the two filings it cited. The judge looks for
the annual totals held in the gold documents, which are different filings.
Every citation in the run was named by the model, and none fell back to
top-ranked hits.

| trace | shows |
|---|---|
| [A096](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/da6a4de4c1ba9bca2aebf446c65eb25a) | correct calculation, scored wrong on rounding |
| [A086](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/ed7dacd4a1bce7ed1af4d36f1a47747c) | "million" dropped from the span |
| [A018](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/0ed2c4df2434e528614afae8ea56269f) | correct, cited to quarterly tables the judge does not recognise |
| [A003](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/9854b215415dc06783d8eb0b92962cee) | gold document never retrieved, declined |

To rerun it, from `eval-service/`, with agent-service on the default settings:

```bash
python scripts/run_answer_eval.py --all --timeout 600 --label full-100-v1 \
    --out-dir results/answer-eval/full-100-v1
# after an interruption, continue from the saved batches:
python scripts/run_answer_eval.py --all --resume --timeout 600 --label full-100-v1 \
    --out-dir results/answer-eval/full-100-v1
```

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
