"""Agent citations keyed by content hash must still score against gold uids.

1,343 of the 2,446 documents in the live index are keyed `sha256-...`, and
agent-service cites evidence by `document_id` alone, so without these
aliases every correct retrieval of those documents scored as a miss.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import _extract_retrieved_doc_ids  # noqa: E402
from scripts.build_identity_aliases import index_aliases  # noqa: E402

DUPLICATES = {"twin-uid": "canonical-uid"}
DOCUMENTS = [
    {"document_id": "sha256-aaa", "source_doc_uid": None, "source_filename": "solo-uid.pdf"},
    {"document_id": "sha256-bbb", "source_doc_uid": None, "source_filename": "twin-uid.pdf"},
    {"document_id": "canonical-uid", "source_doc_uid": "canonical-uid", "source_filename": "canonical-uid.pdf"},
]


def test_hash_keyed_document_maps_to_its_uid():
    assert index_aliases(DOCUMENTS, DUPLICATES)["sha256-aaa"] == "solo-uid"


def test_hash_keyed_twin_resolves_to_the_canonical_uid():
    assert index_aliases(DOCUMENTS, DUPLICATES)["sha256-bbb"] == "canonical-uid"


def test_uid_keyed_document_needs_no_alias():
    assert "canonical-uid" not in index_aliases(DOCUMENTS, DUPLICATES)


def test_agent_citation_by_hash_scores_against_gold_uid():
    answer = {"evidence": [{"document_id": "sha256-aaa", "page": 1}]}
    aliases = {**DUPLICATES, **index_aliases(DOCUMENTS, DUPLICATES)}

    assert _extract_retrieved_doc_ids(answer, aliases) == ["solo-uid"]
    # Guards the fix itself: without the index aliases it is still a miss.
    assert _extract_retrieved_doc_ids(answer, DUPLICATES) == ["sha256-aaa"]


def test_eval_service_image_ships_the_alias_file():
    """The evaluators load identity_aliases.json from the service root, but the
    image once copied only app/. Inside the container the file was missing, so
    load_identity_aliases() quietly returned {} and every containerised run
    scored hash-keyed and duplicated documents as misses -- 54 of the 101 gold
    documents -- with no error anywhere."""
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")
    assert any(line.split()[:2] == ["COPY", "identity_aliases.json"] for line in dockerfile.splitlines())
