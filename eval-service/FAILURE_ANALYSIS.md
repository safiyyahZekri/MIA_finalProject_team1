# Failure analysis

Baseline run `563bbceb`, full corpus (2,386 unique documents, 101/101 gold
documents present), all 100 practice questions, four retrieval variants
(dense, bm25, hybrid, hybrid+reranker). Evidence for every case below is
`failure_cases.json` — real retrieved-vs-gold document ids, real hit/miss
flags, cross-referenced against each question's own metadata in
`questions_setA_practice.json` (task_family, scenario, retrieval_tier,
hard negatives). This is a retrieval-stage analysis: it does not cover
generation, since a full-corpus run through agent-service's LLM wasn't
completed today. Where the data doesn't support a specific claim, that's
stated rather than guessed.

## 6 real cases, root-caused

### A001 — retrieval: cross-document questions get one query for two lookups
**Q:** "How far apart were the 2019 finished-goods balances reported by CTS and Jabil?"
**Expected:** both gold documents (`045aaea8...`, `7d631ffe...`) in the
top 10. **Actual:** neither appeared, in any of the 4 variants.
**Root cause:** `task_family: derived_cross_document`, `retrieval_tier: 3`.
The question is embedded/tokenized as a single query, but it needs two
independent document lookups (CTS's filing, Jabil's filing). One query
vector can't represent two unrelated retrieval targets well — this is a
retrieval-strategy gap, not a ranking weakness; no variant would have
fixed it without changing the query itself.
**Fix:** query decomposition for cross-document questions — split into
one sub-query per anchor/company, retrieve each independently, merge.

### A003 — anomaly: an easy-tier question every variant still missed
**Q:** "On average, what was Atlassian Corporation Plc's Total equity for fiscal years 2015 to 2019?"
**Expected:** `ad41cd9a...` in top 10. **Actual:** missed by all 4
variants — and none of the declared `hard_negative_doc_uids` showed up
in the top 3 either, which is unusual for a `retrieval_tier: 1` question
(direct retrieval, not expected to be hard).
**Root cause:** can't be stated as fact from this data alone — the
retrieval_tier rating implies this shouldn't be hard, so the honest
conclusion is "not yet diagnosed," not "chunking bug." This needs the
actual PDF open next to the top-3 wrong hits to see whether Atlassian's
filing was chunked/embedded at all, which isn't possible from this
machine (no corpus access here).
**Fix (of the process, not the bug):** flag `retrieval_tier: 1` misses
specifically for manual review — a tier-1 miss is a stronger signal of a
pipeline defect than a tier-3 miss, and should be triaged first.

### A021 — retrieval (fusion): hybrid loses a signal bm25 already had
**Q:** "What were Black Knight's net trade receivables as reported in 2017?"
**Expected:** `0f425496...` in top 5. **Actual:** bm25 hit@5 = **true**,
but hybrid hit@5 = **false** — hybrid demoted a document its own bm25
component had correctly ranked.
**Root cause:** fusion weighting. This is a direct, concrete instance of
the pattern `BASELINE.md` flagged in aggregate (hybrid scoring below its
best single component, 0.4211 vs bm25's 0.4684) — here it's not a
statistic, it's one specific question where the fusion step actively
threw away a correct answer.
**Fix:** re-weight or replace the fusion formula in retrieval-api so it
can't score below its strongest input — out of scope to change today
(retrieval-api internals), but this case is the concrete justification.

### A005 — retrieval (dense): embeddings underperform on causal/narrative text
**Q:** "Why does Sykes Enterprises expect to recognize the majority of the customer arrangements with termination rights into revenue?"
**Expected:** `80cb9e7e...` in top 5. **Actual:** dense hit@5 = **false**;
bm25, hybrid, and hybrid+reranker all hit@5 = **true**.
**Root cause:** `primary_scenario: narrative_causal`, `company_reference_style: grounded_alias`.
No table, no distinctive number to anchor on — just company name and
filing-specific vocabulary, which is exactly what bm25's exact-token
matching is good at and dense embeddings tend to blur across similar
narrative passages.
**Fix:** for narrative/causal questions specifically, don't rely on dense
alone — either weight bm25 higher for this scenario type, or use it as
the sole retriever when the question has no numeric anchor.

### A011 — retrieval (fusion): the reverse of A021, still a fusion problem
**Q:** "By how much did Cogeco Inc.'s revenue in fourth-quarter 2019 increase?"
**Expected:** `1bfcc33a...` in top 5. **Actual:** dense hit@5 = **true**,
but hybrid hit@5 = **false** — again hybrid underperforms one of its own
inputs, this time the dense side instead of the lexical side.
**Root cause:** same fusion-weighting issue as A021, evidenced from the
opposite direction — confirms this isn't "dense is weak" or "bm25 is
weak" in isolation, it's specifically that combining them loses whichever
one was right.
**Fix:** same as A021 — this is the second concrete case for that fix,
not a separate problem.

### A013 — reranking: correct document demoted out of top 5
**Q:** "For KEMET Corporation, why was the calculation for Other in 2018 adjusted? Which page supports the answer?"
**Expected:** `ff072e53...` in top 5. **Actual:** hybrid hit@5 = **true**,
hybrid+reranker hit@5 = **false** (still hit@10 = true, so the reranker
pushed it from inside the top 5 to somewhere in 6–10, not out entirely).
**Root cause:** the cross-encoder reranker actively demoted a correct,
already-ranked result. Consistent with `BASELINE.md`'s aggregate finding
(reranker recall@5 0.4105 vs hybrid's 0.4211, only recall@10 improves).
**Fix:** skip the reranker for recall@5-sensitive use, or only apply it
when the caller actually wants k=10 depth.

## What this adds up to

Two of six cases (A021, A011) point at the same root cause from opposite
directions — worth weighting accordingly rather than treating as two
separate findings. The weak layer is fusion, not either retriever alone;
the reranker actively hurts the top-5 window it's meant to improve; and
cross-document questions are a retrieval-*strategy* gap (no ranking
change fixes them). A003 is flagged honestly as unresolved rather than
assigned a root cause the data doesn't support.

---

## Reference: how to pull more cases

`failure_cases.json` holds, for all 100 questions, what was asked, which
documents were gold, and what each of the four variants actually retrieved
(top 10) in baseline run `563bbceb`. Every uid is a corpus filename stem —
a case is inspected by opening `<uid>.pdf`, no service required.

Regenerate after a new run with:

```bash
python scripts/export_failure_cases.py --comparison results/retrieval-ablation-<run-id>.json
```

`buckets` groups the remaining cases:

| bucket | count | what it shows |
|---|---|---|
| `missed_by_all_variants` | 34 | no variant found the gold document at all |
| `bm25_hit_dense_miss` | 23 | lexical matching worked where embeddings failed |
| `dense_hit_bm25_miss` | 10 | the reverse — semantic matching earned its place |
| `reranker_dropped_from_top5` | 8 | hybrid had it in the top 5, the reranker pushed it out |

Working a case: find it by `question_id`, open the gold PDF(s), locate the
passage, then open the top 2-3 wrongly-returned documents and ask what
they share with the query that the gold document doesn't. Write the
mechanism, not the symptom.

