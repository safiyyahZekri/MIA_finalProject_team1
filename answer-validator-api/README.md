# answer-validator-api

Service #7 in the LEDGER architecture. Single source of truth for what counts
as a valid, grounded answer. Strictly validates the four required
`answer_type`s (`direct`, `calculated`, `multi_span`, `insufficient_evidence`)
against the Strict Answer Schema from the project spec, logs a
`[ANSWER-VALIDATOR-SUCCESS]` / `[ANSWER-VALIDATOR-ERROR]` line for every
request, and returns `{"valid": bool, "reason": str|null}`.

This matches the request/response shape Safiyyah confirmed against for
`orchestrator-api`: she sends the answer object produced by `agent-service`
unchanged, and expects `{"valid": true, "reason": null}` back.

## Run it

```bash
docker build -t ledger-answer-validator-api .
docker run -p 8002:8002 ledger-answer-validator-api
```

Or without Docker:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

## Try it

```bash
curl -X POST http://127.0.0.1:8002/validate_answer \
  -H "Content-Type: application/json" \
  -d '{
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017", "page": 1, "section": "Income Statement"}],
        "params": {"value": "$142.5M"}
      }'
# -> {"valid": true, "reason": null}
# console also logs:
# [ANSWER-VALIDATOR-SUCCESS] Received and validated answer of type 'direct' with evidence {'document_id': 'doc_017', 'page': 1}.
```

## What is strictly enforced

- `answer_type` must be one of the four allowed values.
- No unexpected keys at the top level or inside `params` (`extra="forbid"`
  on every pydantic model) — deviations are rejected, not silently accepted.
- `evidence` must be a non-empty list of `{document_id, page[, section]}` for
  every type except `insufficient_evidence`, where it may be empty.
- Per-type required `params` keys:
  - `direct`: `value` (string or number)
  - `calculated`: `value` (number) and `formula` (string)
  - `multi_span`: `values` (non-empty array)
  - `insufficient_evidence`: `reason` (non-empty string)

### Known, documented limitation

The spec says `calculated` answers need "one citation per operand used in
the formula." Matching that exactly from a raw formula string is ambiguous
(e.g. a `*100` for a percentage isn't itself "cited"). We enforce: evidence
must be non-empty, and if the formula clearly combines 2+ numeric literals,
at least 2 citations are required. If your team wants a stricter or looser
rule, it's isolated in `validator.py::_count_numeric_literals` and the block
right after it in `validate_answer`.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

13 tests cover all four schema examples from the spec plus targeted failure
cases (missing key, missing evidence, unknown type, extra key, etc.).
