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
docker run -p 8004:8004 ledger-answer-validator-api
```

Or without Docker:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8004
```

## Try it

```bash
curl -X POST http://127.0.0.1:8004/validate_answer \
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

### Resolved: what "one citation per operand" actually means

The spec text — "one citation per operand used in the formula" — is
ambiguous read literally, since formulas reuse operands (e.g. `x` appearing
twice) and contain constants that were never "cited" from anywhere (a `/100`
for a percentage). Rather than guess, this was checked against the real
`questions_setA_practice.json` arithmetic examples (40 total: 32
single-document, 8 cross-document), comparing each record's evidence-entry
count against two candidate signals:

| Candidate signal | Matches evidence-entry count |
|---|---|
| Number of numeric literals in the `derivation` formula string | **0 / 32** single-document cases |
| Number of distinct `(source_doc_uid, source_page)` pairs behind the operands | **32 / 32** cases (all 40, including cross-document) |

Concretely: a single-document calculation averaging six comma-formatted
figures from one table (six numeric literals) is cited with exactly **one**
evidence entry, because all six values live on the same page. A
cross-document calculation with only two literals is cited with **two**
entries, because the two operands come from two different filings. Evidence
count tracks *distinct source location*, not *formula complexity* — a
formula-string heuristic (counting numbers in `formula`) is actively wrong
0/32 times it was checked and was removed.

**What the validator enforces as a result:** `evidence` must be non-empty
for `calculated` answers (same floor as `direct`/`multi_span`) and nothing
more. The validator only ever sees the evidence array the agent declares —
it has no access to gold data — so it cannot check that the declared
entries actually match the true number of distinct source pages behind the
formula; that per-operand attribution has to happen where the operands are
actually looked up (the calculator tool / retrieval step in agent-service),
not be reconstructed after the fact from a formula string. If agent-service
wants this checked more strictly, the enforceable version of the real rule
is "one evidence entry per distinct (document_id, page) an operand came
from" — that's a fact only the agent can know at generation time, not
something inferable from `formula` alone.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

19 tests: `test_validator.py` covers all four schema examples plus targeted
failure cases (missing key, missing evidence, unknown type, extra key, the
`calculated` evidence-count dataset findings above); `test_api_contract.py`
exercises the live FastAPI endpoint to confirm the exact orchestrator
contract — a raw, unwrapped answer object in, `{"valid": bool, "reason":
string|null}` out.
