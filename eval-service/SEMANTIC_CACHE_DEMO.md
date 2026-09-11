# Semantic caching — measured effect

Bonus feature: `SemanticCache` (`app/semantic_cache.py`), wired into
`run_benchmark` as an opt-in `use_cache` flag. Caches `(question,
document_id) -> system response` within a run — exact match first,
falling back to token-set Jaccard similarity (`cache_similarity_threshold`,
default 0.85) for near-duplicate phrasing of the same question. Scoped by
`document_id` so the same question about two different documents is
correctly treated as two different cache entries.

Chosen over extending the retrieval ablation runner because every other
candidate this data points at (re-weighting hybrid fusion, making the
reranker optional) requires changing retrieval-api internals — out of
scope for today. This is fully contained in eval-service.

## Correctness (exact numbers, not estimated)

`tests/test_semantic_cache.py` (8 tests) covers the cache in isolation:
exact match, case/punctuation normalization, fuzzy match above threshold,
miss below threshold, miss on an unrelated question, and correct
per-`document_id` scoping.

`tests/test_benchmark_cache.py` proves it end-to-end with a real call
counter, not inferred behavior: 4 questions (1 unique question repeated
exactly, paraphrased, and repeated again under a different document)
against a fake system that counts every real call it receives.

| | real calls to the system |
|---|---|
| `use_cache=False` | 4 / 4 (every question) |
| `use_cache=True` | 2 / 4 (only the genuinely new question, once per document) |

## Measured latency effect

20 questions (5 unique questions × 4: original + exact repeat + paraphrase
+ exact repeat again), same document, run against a fake system with a
fixed 250ms simulated latency per real call — enough to make the effect
measurable in wall-clock time, not just call counts.

| | wall-clock time | cache hits | real calls made |
|---|---|---|---|
| without cache | 5.009s | — | 20 |
| with cache | 1.504s | 14 | 6 |

**3.33x speedup**, measured, on this run (reproducible via
`python -m pytest tests/test_benchmark_cache.py -v` for the correctness
proof, or by re-running the timing script inline in this repo's
`run_benchmark` with `use_cache` toggled).

## Honest limits

- Cache is in-memory, per-run — doesn't persist across separate benchmark
  invocations (a process restart starts cold). That's a reasonable scope
  for "don't re-hit the system within one benchmark pass," not a claim of
  a production-grade cache layer.
- Token-Jaccard is a real but coarse similarity measure — it will treat
  some genuinely different questions that happen to share most words as
  duplicates. `cache_similarity_threshold` is exposed precisely so this
  can be tuned or disabled (`use_cache=False`, the default) per use case.
