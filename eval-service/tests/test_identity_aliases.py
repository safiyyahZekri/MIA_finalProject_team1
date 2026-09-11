from __future__ import annotations

import json

from app.retrieval_benchmark import (
    _gold_identity,
    _hit_identity,
    load_identity_aliases,
)
from scripts.build_identity_aliases import build_aliases


ALIASES = {"twin-uid": "canonical-uid"}


def test_gold_and_hit_agree_when_documents_are_byte_identical() -> None:
    """The corpus stores one copy of duplicated content, under whichever uid
    was ingested first, so gold naming a different uid for the same bytes
    must still count as a hit rather than a miss."""
    gold = _gold_identity({"source_doc_uid": "twin-uid"}, "document", ALIASES)
    hit = _hit_identity(
        {"filename": "canonical-uid.pdf", "document_id": "sha256-abc"},
        "document",
        ALIASES,
    )

    assert gold == hit == "canonical-uid"


def test_identities_outside_the_map_are_untouched() -> None:
    assert _gold_identity({"source_doc_uid": "solo-uid"}, "document", ALIASES) == "solo-uid"
    assert (
        _hit_identity({"filename": "solo-uid.pdf"}, "document", ALIASES) == "solo-uid"
    )


def test_without_aliases_the_two_names_still_disagree() -> None:
    """Guards the fix itself: with no map the mismatch that produced the
    all-zero metrics must reappear, so the test cannot pass vacuously."""
    gold = _gold_identity({"source_doc_uid": "twin-uid"}, "document", None)
    hit = _hit_identity({"filename": "canonical-uid.pdf"}, "document", None)

    assert gold != hit


def test_missing_alias_file_is_not_fatal(tmp_path) -> None:
    assert load_identity_aliases(tmp_path / "absent.json") == {}


def test_alias_file_round_trips(tmp_path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({"a": "b"}), encoding="utf-8")

    assert load_identity_aliases(path) == {"a": "b"}


def test_builder_groups_identical_bytes_only(tmp_path) -> None:
    (tmp_path / "aaa.pdf").write_bytes(b"same bytes")
    (tmp_path / "bbb.pdf").write_bytes(b"same bytes")
    (tmp_path / "ccc.pdf").write_bytes(b"different bytes")

    aliases = build_aliases(tmp_path)

    # Lowest uid is the representative, and unique content is never aliased.
    assert aliases == {"bbb": "aaa"}


def test_builder_ignores_jupyter_checkpoints(tmp_path) -> None:
    (tmp_path / "aaa.pdf").write_bytes(b"same bytes")
    checkpoints = tmp_path / ".ipynb_checkpoints"
    checkpoints.mkdir()
    (checkpoints / "aaa-checkpoint.pdf").write_bytes(b"same bytes")

    assert build_aliases(tmp_path) == {}
