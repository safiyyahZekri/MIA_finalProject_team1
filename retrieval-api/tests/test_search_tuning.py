import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models import SearchMode


@pytest.mark.parametrize("weight", [0, 0.35, 0.55, 1])
@pytest.mark.parametrize("constant", [1, 60, 200])
def test_tuning_scores_are_bounded_and_raw_fusion_retained(
    engine, sample_document, weight, constant
):
    client = TestClient(create_app(engine))
    client.post("/documents", json=sample_document)
    response = client.post(
        "/search",
        json={
            "query": "finished goods 2019",
            "dense_weight": weight,
            "rrf_k": constant,
            "rerank": False,
        },
    )
    assert response.status_code == 200
    for hit in response.json()["hits"]:
        assert 0 <= hit["score"] <= 1
        assert 0 <= hit["scores"]["fusion"] <= 1 / (constant + 1)


@pytest.mark.parametrize(
    "payload", [{"dense_weight": -1}, {"dense_weight": 2}, {"rrf_k": 0}]
)
def test_invalid_search_tuning_is_rejected(engine, payload):
    client = TestClient(create_app(engine))
    assert (
        client.post("/search", json={"query": "revenue", **payload}).status_code == 422
    )


def test_weight_changes_ordering_signal_without_index_rebuild(engine):
    def dense_good(weight):
        return engine._final_confidence(SearchMode.hybrid, 0.9, 0.1, 0.01, None, weight)

    def lexical_good(weight):
        return engine._final_confidence(SearchMode.hybrid, 0.1, 0.9, 0.01, None, weight)

    assert dense_good(1) > lexical_good(1)
    assert dense_good(0) < lexical_good(0)


def test_default_confidence_calibration_remains_unchanged(engine):
    dense = engine._dense_confidence(0.7)
    expected = round(0.55 * dense + 0.35 * 0.8 + 0.1 * 0.61, 6)
    assert engine._final_confidence(SearchMode.hybrid, 0.7, 0.8, 0.01, None) == expected
