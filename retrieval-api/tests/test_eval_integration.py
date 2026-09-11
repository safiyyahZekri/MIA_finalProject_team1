from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "eval-service/app/retrieval_benchmark.py"
    name = "ledger_retrieval_benchmark_integration"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RecordingClient:
    def __init__(self, client: TestClient):
        self.client = client
        self.payloads: list[dict] = []

    def post(self, url: str, json: dict):
        self.payloads.append(json)
        path = url.removeprefix("http://retrieval")
        return self.client.post(path, json=json)


def test_eval_ablation_searches_actual_retrieval_api_without_gold_scope(
    engine, sample_document: dict
) -> None:
    first = deepcopy(sample_document)
    first["source_doc_uid"] = "gold-cts"
    second = deepcopy(sample_document)
    second["source_doc_uid"] = "other-jabil"
    second["source_filename"] = "jabil.pdf"
    second["document"]["document_id"] = "random-processor-id"
    second["document"]["pages"][0]["blocks"][2]["uuid"] = "jabil-table"
    api = TestClient(create_app(engine))
    indexed = api.post("/documents/batch", json={"documents": [first, second]})
    assert indexed.status_code == 201

    benchmark = _load_benchmark_module()
    client = RecordingClient(api)
    comparison = benchmark.run_ablation(
        "http://retrieval",
        [
            {
                "question_id": "q1",
                "question_text": "CTS finished goods inventory 2019",
                "source_doc_uid": "gold-cts",
                "gold_evidence": [{"source_doc_uid": "gold-cts"}],
            }
        ],
        candidate_k=2,
        top_k=2,
        client=client,
        save_results=False,
    )

    assert set(comparison["variants"]) == {
        "dense",
        "bm25",
        "hybrid",
        "hybrid_reranker",
    }
    assert all("document_id" not in payload for payload in client.payloads)
    assert all(summary["errors"] == 0 for summary in comparison["variants"].values())
    assert comparison["variants"]["hybrid"]["recall_at_5"] == 1.0
