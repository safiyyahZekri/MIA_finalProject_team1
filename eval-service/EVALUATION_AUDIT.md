# full-100-v1 evaluation audit — 2026-09-10

The mixed document ID hypothesis does **not** explain this run's low answer
accuracy. With the current `identity_aliases.json`, all 649 hash-based hits
in the 95 answerable questions' retrieval traces resolve. Recomputing final
citation hit rates changes none of the 100 saved results. No corpus rebuild,
ID migration, OCR, or inference rerun was needed.

Inputs were the supplied `ledger-eval-full-100-v1/combined.json` and the
repository's practice questions and identity aliases, after inspecting the
latest evaluation update on `origin/feature/retrieval-integration` (`fa5afbd`).
The audit command records SHA-256 hashes of all three inputs for reproducibility.

## The reporting issue and fix

The old answer benchmark calls citation-based scores `retrieval`. A refusal
contains no citations, so it scores zero even when retrieval found a gold
document. New fields separate citations from initial and final retrieval;
old fields remain explicitly labelled as legacy citation metrics.

Measured from the saved ranked traces, K = 5, 95 questions with gold documents:

| Evidence source | Recall@5 | Precision@5 | Hit rate@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Final answer citations | 0.3368 | 0.0716 | 0.3368 | 0.3263 |
| Initial agent retrieval | 0.4684 | 0.1095 | 0.4737 | 0.3368 |
| Final agent retrieval | 0.5158 | 0.1221 | 0.5263 | 0.3528 |

Precision uses K as the denominator, including unfilled slots as misses.
The original 0.305 citation precision divided by the number of citations
actually returned and must not be compared directly with these Precision@5
values. Ranks refer to returned hits; document recall counts unique gold
documents. A document match alone does not establish passage-level relevance.

At least one gold document was found in some attempt for 57/95 questions
(60%). This combines retries and is **not** Recall@5 of one ranked search.
The last attempt found a gold document for 50/95 questions; the first did
for 45/95. The evaluator must not cherry-pick the best attempt as its final
retrieval result.

## What remains a real answer-quality limitation

Exact Match remains **0.2500**, F1 **0.3498**. Numerical accuracy is **7/11**
among comparable numeric predictions, or **7/42** across numeric-gold
questions with refusals counted wrong. The audit does not loosen scoring
after seeing errors or silently label paraphrases/rounding/unit omissions correct.

Of 52 answerable questions declined, 33 never retrieved a gold document;
19 retrieved one in at least one attempt but still declined. The saved
grading reasons include missing company attribution and insufficient facts.
Those 19 are not automatically grader errors: retrieving the right document
does not guarantee that the chunk contains all required evidence, especially
for multi-document questions.

Next quality experiments should measure document/company context available
to grading and passage-level retrieval coverage, using source-derived
metadata and held-out questions. Do not scope searches using gold IDs, remove
the evidence checks, or infer company labels from benchmark answers.

## Reproduce

From the repository root, with evaluation dependencies installed:

```bash
python eval-service/scripts/audit_answer_eval.py \
  --report ../ledger-eval-full-100-v1/combined.json \
  --questions eval-service/questions_setA_practice.json \
  --identity-aliases eval-service/identity_aliases.json \
  --k 5 \
  --output-dir eval-service/results/full-100-v1-audit
```

Read `audit.md` for the summary and `audit.json` for per-question identities,
ranked metrics, unresolved IDs, trace issues, and input hashes. The command
does not call any service or modify the supplied results or index.
