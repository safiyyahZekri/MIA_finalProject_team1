# Saved evaluation audit

Offline recomputation. The original answers, scores and corpus are unchanged.

Questions: 100; errors: 0.
Exact Match: 0.2500; F1: 0.3498.
Numerical accuracy: 0.6364 over 11 comparable answers; 0.1667 over all 42 numeric-gold questions.

| Evidence source | Questions | Recall@K | Precision@K | Hit rate | MRR@K |
|---|---:|---:|---:|---:|---:|
| Final answer citations | 95 | 0.3368 | 0.0716 | 0.3368 | 0.3263 |
| Initial agent retrieval | 95 | 0.4684 | 0.1095 | 0.4737 | 0.3368 |
| Final agent retrieval | 95 | 0.5158 | 0.1221 | 0.5263 | 0.3528 |

Gold-document hit in any attempt: 0.6000 over 95 questions. This is retry coverage, not single-search Recall@K.
Unresolved hash IDs: 0.
Malformed trace questions: 0.
Saved citation hit-rate changes after identity resolution: 0.
Answerable questions declined: 52.
Gold document found, then declined: 19 (A001, A009, A022, A032, A035, A040, A041, A042, A044, A048, A051, A052, A060, A063, A069, A079, A089, A094, A097).
Gold document never found, then declined: 33.

Precision uses K as the denominator; older reports divided by the number of returned citations.
Document-level matches do not establish that the retrieved passage contains the answer or company attribution.
Strict EM is unchanged. Paraphrases, rounding and omitted units are not automatically relabelled correct.
