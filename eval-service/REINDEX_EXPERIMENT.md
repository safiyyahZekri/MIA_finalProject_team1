# Re-index experiment: contextual text, company metadata, table titles

2026-09-11. Features from the failure analysis that need the index rebuilt but
no OCR (FAILURE_ANALYSIS_E2E.md: A003, A035, A001, A022, A037, A068).

## What was built

`retrieval-api/app/index_format.py`. An index records its format in
`manifest.json`; retrieval-api builds, embeds and searches new documents the
way the index already was. An index with no recorded format is the legacy
one, unchanged. Three switches, each tried on top of the one before:

| index | change | what a CTS table chunk is searched with |
|---|---|---|
| live (legacy) | none | `a0b0….pdf a0b0….pdf train\a0b0….pdf Document Row 1: …` |
| contextual | `contextual_text`: no hash filenames, uids or source paths; section first | `Row 1: …` |
| + company | `company_metadata`: the company the document prints in its own headers | `CTS CORPORATION Row 1: …` |
| + titles | `table_titles`: a first row that is only a heading becomes the title | `CTS CORPORATION Table: Operating costs …` |

The legacy text spends a median 83 of 255 tokens per chunk on the hash
filename three times. Chunks over the embedder's 512 tokens: 563 → 196.

The agent gets `EVIDENCE_COMPANY_LABEL` (off by default): each passage label
the grader and extractor see carries `company=<name>`.

### Company detection (`retrieval-api/app/company.py`)

The name counts only where a filing prints its own name: a line with page
furniture on it ("CTS CORPORATION 40", "KEMET CORPORATION AND SUBSIDIARIES",
"Plexus Corp. Notes to Consolidated Financial Statements"), or a bare name
standing alone or beside a furniture line ("Table of Contents" / "VMware,
Inc." / "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS"). Names inside prose,
in table rows with figures, or beside prose (an acquisition heading) are
ignored. The practice questions were used only to check the result, never
to set a value.

- 593 of 2,446 documents get a company (24%). Most pages never print one,
  including A003's Atlassian and A007's Jabil pages.
- On the 33 gold documents with a detected company, 33 match the company the
  question names.
- Two rounds of rules removed the wrong names found in samples: acquired
  companies and subsidiaries in note headings, wrapped sentences ending in a
  company name.

### Table titles (`retrieval-api/app/tables.py`)

543 tables get a title from their first row (A037 "Operating costs", A068
"… Financial Highlights …"). A sample of 45 read about 70–80% correct; the
rest are mostly column headers alone in the first row ("Fair Value").

## Method

No OCR. `retrieval-api/scripts/rebuild_index.py` rebuilds every document from
the stored chunks with the chunker's own code:

1. **verify**: rebuilding all 2,446 documents in the legacy format reproduces
   the stored 14,386 chunks field for field (this caught a parser bug on ten
   tables, fixed before any build).
2. **build**: rebuild in the new format, reuse a vector wherever the embedded
   text is unchanged, embed the rest, write to a new directory, reload it.
3. Each index served by a second retrieval-api; the live index was only read.

| build | embedded | reused | time |
|---|---:|---:|---:|
| contextual | 14,386 | 0 | 11.5 min |
| + company | 3,335 | 11,051 | 2.7 min |
| + titles | 563 | 13,823 | 35 s |

Tests: 137 retrieval-api tests (including rebuild == upload for all eight
switch combinations), 187 agent-service tests.

## Retrieval results

Four-variant ablation, all 100 practice questions (95 scored), `candidate_k=30`,
`top_k=10`, unscoped. The live control is the same index's run from the chunk
budget experiment; retrieval is deterministic.

| variant | metric | live | contextual | + company | + titles |
|---|---|---:|---:|---:|---:|
| dense | recall@5 | **0.342** | 0.268 | 0.321 | 0.321 |
| dense | MRR | 0.221 | 0.229 | **0.239** | 0.239 |
| bm25 | recall@5 | 0.468 | 0.511 | **0.526** | 0.526 |
| bm25 | MRR | 0.356 | 0.392 | **0.441** | 0.438 |
| hybrid | recall@5 | 0.421 | 0.374 | **0.453** | 0.453 |
| hybrid | MRR | 0.304 | 0.264 | **0.347** | 0.347 |
| hybrid + reranker | recall@1 | 0.242 | 0.211 | **0.311** | 0.311 |
| hybrid + reranker | recall@5 | 0.411 | 0.474 | **0.553** | 0.553 |
| hybrid + reranker | recall@10 | 0.495 | 0.526 | **0.584** | 0.584 |
| hybrid + reranker | hit rate@5 | 0.411 | 0.484 | **0.568** | 0.568 |
| hybrid + reranker | MRR | 0.323 | 0.313 | **0.420** | 0.420 |

Questions whose recall@5 changed (hybrid + reranker):

- live → contextual: gained A016, A026, A029, A035, A040, A048, A073, A075,
  A094, A097; lost A043, A052, A086
- contextual → + company: gained A001, A010, A021, A022, A058, A072, A076,
  A089, A090, A100; lost A088, A097
- + company → + titles: no change

**What got worse.** Contextual text alone lowered dense recall@5
(0.342 → 0.268: 12 lost, 5 gained) and every hybrid metric, and hybrid +
reranker MRR. With company metadata, dense recall@5 is still below the live
index (0.321). Table titles changed no question's recall; MRR moved by less
than 0.004.

**What helped.** Company metadata raised every metric in every variant over
the build before it; for the variant the agent uses, recall@5 0.411 → 0.553
and MRR 0.323 → 0.420 against the live index.

## Answer results

Agent: Gemini 3.5 Flash Lite with the switches of run `full100-g4` plus
`EVIDENCE_COMPANY_LABEL`, retrieving from the + company index. Reference:
`full100-g4`, same switches, live index, no label.

### 20-question subset

| metric | live index | company index |
|---|---:|---:|
| exact match | 9/20 | 9/20 |
| F1 | 0.45 | 0.50 |
| declined | 11 | 8 |

Gained A022, lost A018. Gold documents found by the agent's own searches
improved on A001 (0/2 → 1/2), A022 (1/2 → 2/2) and A035 (0/1 → 1/1), and
fell on none. A018 was lost although retrieval was identical: the live run
had summed Advanced Energy's quarterly figures from a non-gold page.

### All 100 questions

Run `full100-g5-company`, 42 minutes, no errors.

| metric | Gemini baseline | full100-g4 (live index) | full100-g5-company |
|---|---:|---:|---:|
| exact match | 19/100 | 26/100 | **30/100** |
| F1 | 0.222 | 0.351 | **0.402** |
| numerical accuracy (numeric gold, answered) | 0.71 over 7 | 0.53 over 17 | **0.78 over 18** |
| arithmetic exact | 3/40 | 5/40 | **8/40** |
| span exact | 7/40 | 11/40 | 11/40 |
| multi-span exact | 4/13 | 5/13 | **6/13** |
| unanswerable exact | 5/5 | 5/5 | 5/5 |
| declined an answerable question | 69 | 46 | **39** |
| answered wrong | 12 | 27 | 31 |
| errors | 0 | 1 | 0 |
| gold documents all found by the agent's searches (of 95) | 50 | 55 | **58** |
| gold document never found (of 95) | 41 | 36 | **33** |
| average latency | 28.1 s | 29.8 s | 24.2 s |
| LLM calls per question | 5.36 | 4.87 | 4.73 |
| tokens per question | 5,690 | 10,396 | 9,957 |

Against `full100-g4`: gained A002, A010, A021, A022, A031, A076, A090; lost
A006, A018, A081. For comparison, the first evaluation (Claude Opus 5, no
enhancements) scored 25/100, F1 0.35.

**What got worse.** Wrong answers rose from 27 to 31 as declines fell from 46
to 39: the label lets the grader accept more evidence, and some of it leads to
a wrong figure. Latency is not a clean comparison (the live runs sent traces
and shared the CPU).

**Reading it.** One run of each; Gemini varies by a few questions between runs,
so +4 is a measured improvement in the direction the retrieval ablation
predicts rather than a precise size. Five of the seven gained questions
(A010, A021, A022, A076, A090) are ones whose gold document entered the
hybrid + reranker top 5 when company metadata was added in the retrieval
ablation.

## Reproducing

Inside the retrieval-api image, with the live index volume read-only at
`/data`, the model cache mounted, and `retrieval-api/` at `/work`:

```bash
python scripts/rebuild_index.py --data-dir /data verify
python scripts/rebuild_index.py --data-dir /data companies --output /exp/companies.json
python scripts/rebuild_index.py --data-dir /data build \
    --format contextual_text,company_metadata --output-dir /exp/idx-company
```

Serve a built directory with a second retrieval-api
(`-e LEDGER_RETRIEVAL_DATA_DIR=/data -v <dir>:/data`), run the ablation with
`app.retrieval_benchmark.run_ablation`, and point the agent at it with
`RETRIEVAL_API_URL`.
