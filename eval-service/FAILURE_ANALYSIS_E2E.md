# Failure analysis: the full system, end to end

These are eleven failed answers, in ten cases, from run `full-100-v1`: all 100 practice questions
sent through the whole pipeline (agent-service on Claude Opus 5, retrieval-api
over 2,446 documents and 14,386 chunks). Each failure is traced to the stage
where it started.

[`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md) is the earlier analysis. It
tested retrieval alone (run `563bbceb`, four retrieval variants, no LLM) on an
earlier index, so it does not cover grading or answer generation. Its data is
consistent: all 800 of its hit flags recheck under the current document-id
aliases. See the last section for where this analysis changes its conclusions.

## How each case was diagnosed

1. **Trace.** Every question's Langfuse trace holds the question, the final
   answer, its citations and its scores. In run `full-100-v1` it did not hold
   the agent's steps; those were saved only in the run's results. From this
   commit, eval-service also logs each agent step as its own numbered span
   (`agent.01.classify`, `agent.02.retrieve`, `agent.03.grade`, …). Those spans
   hold the rewritten queries, the pages retrieved on each attempt, and the
   grader's reasons. The eleven questions below were run again twice after the
   change ("replays"). This shows whether each failure recurs, and gives the
   nine that recurred a trace showing the failure step by step; see
   [Traces](#traces).
2. **Is the fact in the index?** Search the gold page's chunks for the gold
   facts. If a fact is missing or garbled, the failure started at OCR, table
   extraction or chunking.
3. **Where does retrieval rank it?** Replay the agent's exact queries against
   the endpoints it calls: `/search/vector` (dense), `/search/tables` (hybrid,
   tables only) and `/search/bm25` (lexical), all with reranking on. Ranks are
   taken at the agent's top 5 and within the top 50, with reranking on and
   off. This separates retrieval from reranking.
4. **If it was retrieved:** read the grader's reason and the extraction to
   separate grading, numerical reasoning and generation.

`scripts/diagnose_failure.py` prints steps 2–4 for any question id.

## Summary

| case | type | stage | root cause | fix |
|---|---|---|---|---|
| [A003](#a003) | arithmetic | retrieval | Pages are indexed without their company, so "Atlassian" matches nothing on Atlassian's balance sheet | company metadata at ingestion, metadata-aware retrieval |
| [A035](#a035) | span | grading, with reranking | The right passage was retrieved and refused because it never names HC2; the reranker had dropped it from the top 5 twice | company metadata shown to the grader |
| [A001](#a001) | arithmetic | retrieval strategy | One query for two companies' tables; CTS's table never reached the top 5 | query decomposition |
| [A068](#a068) | multi-span | retrieval ranking, then grading | The right table ranked 11th; a different Plexus table was accepted at confidence 0.58 | title-aware table chunks; a grader that requires the named table |
| [A037](#a037) | multi-span | chunking | The table chunk's only context is an unrelated paragraph, and it has no heading | contextual chunk headers |
| [A013](#a013) | span | OCR, generation, grading | OCR text "adoption OfASC 606"; the page number was put into the answer; the grader asked for a page | OCR token repair; the page belongs in citations |
| [A007](#a007) | arithmetic | numerical reasoning | "from 2017–2019" read as three year-on-year changes instead of two | period rule in the calculation prompt |
| [A014](#a014) | arithmetic | numerical reasoning | Used rounded narrative figures in millions over the exact table in thousands; the answer carries no unit | prefer table cells; a scale field on calculated answers |
| [A086](#a086) | multi-span | generation | Our own format rule stripped "million" from the answer span | reword the rule to keep scale words |
| [A077](#a077) | span | question ambiguity, with grading | "HC2's SG&A decrease" has two answers in the report; the company rule picked the passage naming HC2 | flag in the benchmark; grader declines on competing figures |

### Each case in one line

- **A003:** expected 659,439.4 (Atlassian's average total equity), but the
  agent declined because no Atlassian page was retrieved → root cause:
  retrieval failure (pages are indexed without their company) → fix:
  metadata-aware retrieval.
- **A035:** expected "calculated using the two-class method…", but the agent
  declined after retrieving that passage → root cause: grading (the company
  rule refused a page that never names HC2), with the reranker demoting the
  passage → fix: company metadata shown to the grader.
- **A001:** expected 304,811, but the agent declined: CTS's table never reached
  the top 5, and Jabil's was refused as unnamed → root cause: retrieval
  failure (one query for two companies) → fix: query decomposition.
- **A068:** expected 2019–2015, but got 2019–2017 from a different Plexus
  table → root cause: retrieval ranking (the right table ranked 11th, its
  heading extracted as a table row), then grading (the wrong table was
  accepted) → fix: title-aware table chunks, and a grader that requires the
  named table.
- **A037:** expected seven cost lines and segments, but got three segment
  names → root cause: chunking (the table chunk has no heading, only an
  unrelated context line) → fix: contextual chunk headers.
- **A013:** expected "due to the adoption of ASC 606", but got "adjusted due to
  the adoption OfASC 606" plus "p1" → root cause: OCR (merged "OfASC"), then
  generation (the page went into the answer) → fix: OCR token repair, and the
  page taken from citations.
- **A007:** expected 15.16, but got 11.40 → root cause: numerical reasoning
  (three year-on-year changes averaged instead of two) → fix: a period rule
  in the calculation prompt.
- **A014:** expected 31,252 (thousand), but got 31.27 (million) → root cause:
  numerical reasoning (rounded text figures used instead of the exact table,
  and no unit) → fix: prefer table figures, and a scale on calculated answers.
- **A086:** expected "$12.2 million, $23.6 million, $38.6 million", but got
  "$12.2, $23.6, $38.6" → root cause: generation (the format rule stripped
  "million") → fix: keep scale words.
- **A077:** expected "$5.0 million" (Life Sciences), but got −8.6 (the
  Corporate segment) → root cause: an ambiguous question, with grading
  picking the passage that names HC2 → fix: flag the question, and a grader
  that declines on competing figures.
- **A038:** expected 7,473.33, but got 77.33 from IBM's "average recorded
  investment" disclosure → root cause: ambiguous question wording → fix: flag
  the question.

---

## Cases

### A003

**Retrieval: the index does not know which company a page belongs to.**

**Question:** "On average, what was Atlassian Corporation Plc's Total equity
for fiscal years 2015 to 2019?"
**Expected:** 659,439.4 (thousand), the mean of the balance-sheet row `Total
equity | 565,467 | 907,320 | 902,693 | 731,663 | 190,054`.
**Actual:** declined after three retrieval attempts: "No Atlassian total
equity figures appear. The only total equity table belongs to Eros
International Plc, and another to an unrelated company's joint ventures."

**Evidence.**
- The row is in the index, intact, in one chunk of page `ad41cd9a…`.
- None of that page's 4 chunks contains "Atlassian". The indexed filename is
  `ad41cd9a9d25c3a1583098c1d3cf7ac7.pdf`, and the metadata carries no company.
- Replaying all three of the agent's queries, no chunk of the page ranks in
  the top 50 of `/search/vector` or `/search/tables`.

**Root cause.** Retrieval, set up at ingestion. The query's most specific
term, "Atlassian", appears nowhere on the right page, while "total equity"
matches many companies' balance sheets. The grader then correctly refuses
the ones it is shown. The earlier analysis left this case "not yet
diagnosed".

**Fix.**
- **Metadata-aware retrieval:** attach the company, report and year to every
  page at ingestion, then filter or boost by the company the question names.
  Put the company in the chunk text as well, so the grader can see it.
- **Where the mapping comes from:** no file in the repository or the corpus
  folder maps every page to its company. The practice question file does so
  only for its gold pages (`atlassian-corp-plc_2019.pdf`), so the mapping has
  to come from the dataset's source.

### A035

**Grading, with reranking: the right passage was refused for not naming the
company.**

**Question:** "For HC2 Holdings, Inc., how is Earning Per Share (EPS)
calculated?"
**Expected:** "calculated using the two-class method, which allocates earnings
among common stock and participating securities…"
**Actual:** declined: "None of the excerpts can be tied to HC2 Holdings, Inc."

**Evidence.**
- Attempt 2 retrieved the passage: "Earnings per share ("EPS") is calculated
  using the two-class method…". The grader answered: "The fourth passage
  describing the two-class method EPS policy is from an unnamed issuer and
  cannot be tied to HC2 Holdings, so it does not satisfy the
  company-identification requirement."
- The passage does not contain "HC2".
- In attempts 1 and 3, dense search without reranking ranks the passage 2nd
  and 3rd. With reranking, which the agent uses, it falls to 8th and 33rd,
  out of the top 5.

**Root cause.** Grading. The company-match rule (adopted in experiment B to
stop answers built from another company's table) refuses a correct passage,
because the page never names its company. This is the same missing
metadata as A003. The reranker contributed by demoting the passage in two of
three attempts.

**Fix.**
- **Grader:** give it the company metadata from A003's fix, so the rule
  checks the page's recorded company instead of its text.
- **Reranker:** measure it on and off for text questions. It hurt here, but
  it helped in A068 and A038 (below).

### A001

**Retrieval strategy: one query for two companies.**

**Question:** "How far apart were the 2019 finished-goods balances reported by
CTS and Jabil?"
**Expected:** 304,811 = |9,447 − 314,258| (thousand).
**Actual:** declined.

**Evidence.**
- Both inventory tables are in the index, and neither names its company.
- Every query names both companies, so the two tables compete for the same
  five slots.
- CTS's table ranks 9th at best (`/search/tables`), and 12th in dense search.
  It never reached the agent's top 5.
- Jabil's table reached rank 4 on attempt 2, and the grader refused it: "The
  August 31 fiscal year-end inventory table (finished goods $314,258)
  plausibly belongs to Jabil but is unnamed."

**Root cause.** The retrieval strategy. A single query cannot bring back two
unrelated targets. Missing company identity then made the grader refuse the
table that was found.

**Fix.**
- **Query decomposition:** one sub-query per company, retrieved separately,
  then merged. This is the bonus feature.
- **Company metadata:** as in A003.

### A068

**Retrieval ranking, then grading: the same company's wrong table was
accepted.**

**Question:** "Which years does the table cover for Plexus Corp.'s financial
highlights?"
**Expected:** 2019, 2018, 2017, 2016, 2015.
**Actual:** `["2019", "2018", "2017"]`.

**Evidence.**
- The right chunk holds the five-year table, but its heading was extracted as
  the table's first row: `Row 1: ITEM 6. SELECTED FINANCIAL DATA Financial
  Highlights…`. It has no table title, and its context line is the page
  header "Table ofContents".
- The agent retrieved a different Plexus page with a three-year table.
- On replay, the Financial Highlights chunk ranks 11th in `/search/tables`
  (14th without reranking), and is not in dense search's top 50.
- The grader accepted the wrong table with confidence 0.58, while noting "the
  specific 'financial highlights' table is not labeled as such in the
  evidence."

**Root cause.** Retrieval ranking left the right table just outside the top 5,
partly because its heading sits in a row instead of a title. The grader then
accepted evidence it said itself was not the table asked for.

**Fix.**
- **Table extraction and chunking:** use a heading found in the first row as
  the table's title.
- **Grading:** when the question names a table or section and the evidence
  does not match it, reformulate the query instead of answering.

### A037

**Chunking: the table chunk has misleading context and no heading.**

**Question:** "What are Spirent Communications plc's different business
segments considered in the operating costs?"
**Expected:** product development, selling and marketing, administration,
Networks & Security, Lifecycle Service Assurance, Connected Devices, and
Corporate.
**Actual:** the three segment names only.

**Evidence.**
- The "Operating costs" table is in the index, on both copies of the page.
- Its chunk has section `Document` and no table title. The context line that
  chunking prefixed is the paragraph before the table: "near-term
  opportunities for growth, particularly in relation to 400G high-speed
  Ethernet".
- The chunk mentions neither Spirent nor segments.
- On replay, other chunks of the same page rank 1st and 2nd, but this table
  chunk is not in the top 50 of dense or BM25 search.
- The grader answered from segment descriptions elsewhere: "no explicit
  operating-cost-by-segment table is shown, so the answer must be inferred
  from the segment structure".

**Root cause.** Chunking. `build_chunks` prefixes each table with the two
lines before it as context. Here those lines are unrelated, and the table's
own heading appears only as a row.

**Fix.** Contextual chunk headers: the company, the page's section heading and
the table's title in place of the two preceding lines. This is the
contextual-retrieval bonus.

### A013

**OCR, generation and grading.**

**Question:** "For KEMET Corporation, why was the calculation for Other in
2018 adjusted? Which page supports the answer?"
**Expected:** "due to the adoption of ASC 606".
**Actual:** `["adjusted due to the adoption OfASC 606", "p1"]` (EM 0, F1 0.62).

**Evidence.**
- Both KEMET pages carrying the footnote read "adjusted due to the adoption
  OfASC 606" in the index. The phrase "adoption of ASC 606" appears in no
  chunk.
- On attempt 1 the grader found the footnote, but marked the evidence
  insufficient because "the chunk containing the relevant table has no page
  number". That cost two more retrieval rounds.
- Extraction copied the OCR text and added the page marker `p1` as an answer
  value.
- On attempt 1, the gold page's chunk also ranked 4th in BM25 but was dropped
  when the dense and BM25 hits were merged. The merge ranks by raw scores
  from different endpoints and keeps 5. The other KEMET page carried the same
  footnote, so this did not change the outcome.

**Root cause.**
- **OCR:** "of ASC" became "OfASC", so an exact-span answer cannot match.
- **Generation:** the page number went into the answer.
- **Grading:** the page request was treated as missing evidence, although
  every citation already carries its page.

**Recurrence.** The OCR text broke the answer in all 3 runs ("OfASC 606" each
time). The `p1` value appeared only in the original run.

**Fix.**
- **OCR:** repair merged tokens after OCR.
- **Page questions:** tell the grader and extraction that the citation
  supplies the page.
- **Merge:** combine endpoint results by rank (reciprocal rank fusion), not
  by raw scores.

### A007

**Numerical reasoning: the period was misread.**

**Question:** "What was Jabil Inc.'s average year-on-year percentage change in
total net revenue from 2017-2019?"
**Expected:** 15.16, the average of the 2017→2018 (15.91%) and 2018→2019
(14.42%) changes.
**Actual:** 11.40.

**Evidence.**
- Retrieval found another Jabil page with all the figures, and the grader
  listed them.
- The grader said the figures allow "computation of year-on-year percentage
  changes for 2017, 2018 and 2019 and their average".
- The formula averages three changes, starting from 2016→2017.

**Root cause.** Numerical reasoning. "From 2017–2019" was read as the
changes *in* 2017, 2018 and 2019, when it means the changes *between* 2017
and 2019. Retrieval and extraction of the figures were correct.

**Recurrence.** Intermittent: 1 of 3 runs. Both replays averaged the two
changes and answered 15.16 correctly.

**Fix.**
- **Prompt rule:** a change from year X to year Y spans Y−X intervals.
- **Formula:** name the year pairs, so the interpretation shows in the trace.

### A014

**Numerical reasoning: precision and units.**

**Question:** "For Advanced Energy, what was the average total purchase price
amongst the three companies?"
**Expected:** 31,252 (thousand) = (6,072 + 3,000 + 84,684) / 3.
**Actual:** 31.266667, from `(6.1+84.7+3.0)/3`.

**Evidence.**
- The table with the exact figures, in thousands, was retrieved at rank 1 and
  cited.
- The grader and extraction used the rounded narrative figures from a second
  passage instead: "Trek $6.1M, LumaSense $84.7M, and Monroe Electrostatic
  Product Line $3.0M".
- 31.27 million is within 0.05% of the gold value. The calculated answer has
  no unit, so it was scored as 31.27 against 31,252.

**Root cause.** Numerical reasoning and the answer format. Rounded narrative
figures were preferred over exact table values, and the result lost its
scale.

**Fix.**
- **Figures:** prefer table cells when a figure appears in both a table and
  the text.
- **Scale:** add a scale to calculated answers, so the scorer can convert.

### A086

**Generation: our own format rule removed part of the answer.**

**Question:** "What rent expense did Atlassian Corporation Plc incur on the
Group's operating leases in fiscal years 2017, 2018, and 2019, respectively?"
**Expected:** `["$12.2 million", "$23.6 million", "$38.6 million"]`.
**Actual:** `["$12.2", "$23.6", "$38.6"]`.

**Evidence.**
- Attempt 2 retrieved both copies of the right page. The grader identified
  Atlassian from context ("June 30 fiscal year end, offices in Amsterdam…,
  Sydney…"), which is the company rule working as intended.
- The source reads "$38.6 million, $23.6 million, and $12.2 million", with a
  line break only after "$38.6".
- Extraction dropped "million" from all three values, including the two
  written on one line.

**Root cause.** Generation. The format instruction added in experiment A1
("Do not add labels, years, units, currency symbols or explanation that are
not part of that span") was over-applied, although "million" is part of the
span. This corrects `EXPERIMENTS.md`, which partly blamed the line break.

**Recurrence.** Intermittent: 1 of 3 runs. Both replays kept "million" and
were correct. The rule's wording still allows the error, so the fix stands,
at lower priority.

**Fix.** Reword the rule to keep scale words that follow a number in the
source, and add a test for it.

### A077

**Question ambiguity, with grading. A038 is the same kind of case.**

**Question:** "What was HC2 Holdings, Inc.'s decrease in the selling, general
and administrative expenses?"
**Expected:** "$5.0 million", for the Life Sciences segment.
**Actual:** −8.6, the Non-operating Corporate segment (24.9 − 33.5).

**Evidence.**
- The gold page was retrieved (3rd in dense search): "Selling, general and
  administrative expenses from our Life Sciences segment… decreased $5.0
  million". It never names HC2.
- The grader chose between them: "One excerpt explicitly references HC2… SG&A
  decreased $8.6 million… Some ambiguity remains since another excerpt cites
  a $5.0 million decrease for a Life Sciences segment…, but the HC2-named
  passage supplies the likely intended number."

**A038**, the same pattern:
- **Question:** "What is IBM's average recorded investment?"
- **Expected:** 7,473.33, the mean of the `Recorded investment` row (5,567,
  3,831, 13,022).
- **What happened:** the table ranked 1st in `/search/tables`. The model used
  a different IBM disclosure literally titled "average recorded investment",
  for impaired loans (138, 49, 45).

**Root cause.** The questions' wording. Both were derived from questions
asked about a single page: "What was the decrease in the selling, general
and administrative expenses?" and "What is the average of Recorded
investment?". Across the whole report, each now has more than one valid
answer. A038's rewording turned "average of Recorded investment" into the
name of a different disclosure. In A077 the company rule broke the tie
toward the passage that names HC2.

**Fix.**
- **Benchmark:** these are not pipeline defects; flag both as ambiguous in
  the benchmark report.
- **Grader:** when it finds competing figures ("ambiguity remains"), decline
  or name both, instead of choosing one.

---

## What this adds up to

- **Missing company identity is the most common cause.** It causes A003, A035
  and A001, and it shapes A077. In the full run, 34 of the 52 declined
  answers gave a company-identity reason (a keyword count over the grader's
  reasons, so approximate). The highest-leverage fix is to record each
  page's company at ingestion and retrieve with it. That is metadata-aware
  retrieval, the fix the project brief itself gives as an example.
- **Retrieval strategy and chunk context come next:** one query for two
  targets (A001), and table chunks without their headings (A068, A037).
- **Generation and numerical reasoning are the cheapest fixes:** prompt and
  schema changes for A007, A014, A086 and A013. A014 and A013 failed in all
  three runs. A007 and A086 failed in one of three: Opus 5 accepts no
  temperature setting, so answers vary between runs.
- **The reranker is mixed.** It demoted the right passage in A035, but raised
  it in A068 (14th to 11th), A038 (dense: 21st to 5th) and A013 (BM25: 15th
  to 6th). It needs a measured on/off run, not removal.
- **No table-extraction error caused any of these ten failures, and OCR
  caused one (A013).** OCR noise does appear elsewhere, for example "Recordedi
  investment" and "Fiscal Yearl Ended", but those questions failed for other
  reasons.

## Where this changes the earlier analysis

- **A003** was "not yet diagnosed". It is a company-identity retrieval miss
  (above).
- **A013** was attributed to the reranker, measured on `/search` hybrid. The
  agent does not call that endpoint. On the endpoints it does call, reranking
  raised the passage, and the failure came from OCR, the page request and
  generation.
- **A001:** the agent's rewritten query did reach Jabil's table (rank 4). The
  diagnosis stands: one query cannot reach both tables, and query
  decomposition is the fix.
- **A005, A011 and A021** were not re-examined here.

## Traces

Each failed question has run three times: the original run `full-100-v1`,
then two replays of the same eleven questions after the tracing change
(`failure-replay` and `failure-replay-2`, $0.59 and $0.62). The original
trace shows the question, the answer, the citations and the scores. The
replay-2 trace also shows every agent step, numbered in order. Replay 1's
spans are unnumbered, so they are not linked.

| case | outcome: original · replay 1 · replay 2 | answer and scores | with agent steps |
|---|---|---|---|
| A003 | declined · declined · declined | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/9854b215415dc06783d8eb0b92962cee) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/ace4495363217f9d56d6ea54c4b791f8) |
| A035 | declined · declined · declined | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/560af909245b80fc3f2733619ef7837f) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/a6d566f6fcf4a083f6bdf763b066e88d) |
| A001 | declined · declined · declined | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/22ca7da981d8904aae59f44c68e60647) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/eef708b865bdf4974d73af84ae42c995) |
| A068 | wrong · wrong · wrong | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/d38d1019b677939b634ced487f458dd0) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/3516b1fc896e31624d95af25b3194eb0) |
| A037 | wrong · wrong · wrong | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/ec11f956215e651dcb6c68d0110401b4) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/834d77a131ceacbfda055385d86fbb21) |
| A013 | wrong · wrong · wrong | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/490beab3daae09120dc083d863960b34) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/a39f98f67d48e41705de8ed0a5aaf213) |
| A007 | wrong · **correct** · **correct** | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/5f13f59d703883a7e9a864bb85671778) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/85b5633169c3a097cd1e4e247136ca3e) |
| A014 | wrong · wrong · wrong | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/acc8c6a5cf937bf2349a2556c92ce3e7) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/a79b3015e1f40de9bdafb715dbf4472c) |
| A086 | wrong · **correct** · **correct** | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/ed7dacd4a1bce7ed1af4d36f1a47747c) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/42f7f531cc20af7607a533cf7fed09e4) |
| A077 | wrong · wrong · wrong | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/2fbf678bc2e3aee424c6df1f2ebbdfb1) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/5c2e1cf28c37aa75557648a4cef2dad8) |
| A038 | wrong · wrong · wrong | [original](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/92edbc93d1e30942144a5f6b4d5cd844) | [replay 2](https://cloud.langfuse.com/project/cmtv69p2f070wad0dt27on2ut/traces/dcf84d5b3afb5b17ddabf0cde4de5bd4) |

Nine of the eleven failed the same way in all three runs, so their replay-2
traces show the failure step by step. A007 and A086 were correct in both
replays, so no trace with agent steps shows their failure. For those two, the
evidence quoted comes from the steps saved in the original run's results
(`results/answer-eval/full-100-v1/combined.json`), and the replay traces show
the corrected runs.

The step evidence quoted in the other cases also comes from the original
run. Where the replays differ in detail, the failure is the same:
- **A035:** the replays never retrieved the two-class passage, and the grader
  refused for the same reason (no evidence tied to HC2).
- **A001:** replay 2 again retrieved Jabil's table on attempt 2 and refused it:
  "Jabil is never named".
