from __future__ import annotations

from pathlib import Path

import pytest

from app.chunking import ChunkingConfig
from app.engine import RetrievalEngine

from .fakes import FakeEmbedder, FakeReranker


@pytest.fixture
def engine(tmp_path: Path) -> RetrievalEngine:
    return RetrievalEngine(
        data_dir=tmp_path,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        chunking=ChunkingConfig(max_chars=180, overlap_blocks=1),
    )


@pytest.fixture
def sample_document() -> dict:
    return {
        "document": {
            "document_id": "doc-cts-2019",
            "pages": [
                {
                    "bbox": [0, 0, 1200, 1600],
                    "page_number": 1,
                    "blocks": [
                        {
                            "bbox": [10, 10, 500, 40],
                            "uuid": "heading-1",
                            "text": "Inventory",
                            "order": 1,
                            "content_type": "paragraph",
                        },
                        {
                            "bbox": [10, 50, 700, 100],
                            "uuid": "paragraph-1",
                            "text": "Inventory balances are reported in thousands of dollars.",
                            "order": 2,
                            "content_type": "paragraph",
                        },
                        {
                            "bbox": [10, 120, 900, 400],
                            "uuid": "table-1",
                            "text": "",
                            "order": 3,
                            "content_type": "table",
                            "cells": [
                                {
                                    "bbox": [10, 120, 200, 150],
                                    "text": "Item",
                                    "row_span": [0, 1],
                                    "col_span": [0, 1],
                                },
                                {
                                    "bbox": [210, 120, 300, 150],
                                    "text": "2019",
                                    "row_span": [0, 1],
                                    "col_span": [1, 2],
                                },
                                {
                                    "bbox": [310, 120, 400, 150],
                                    "text": "2018",
                                    "row_span": [0, 1],
                                    "col_span": [2, 3],
                                },
                                {
                                    "bbox": [10, 160, 200, 190],
                                    "text": "Finished goods",
                                    "row_span": [1, 2],
                                    "col_span": [0, 1],
                                },
                                {
                                    "bbox": [210, 160, 300, 190],
                                    "text": "9,447",
                                    "row_span": [1, 2],
                                    "col_span": [1, 2],
                                },
                                {
                                    "bbox": [310, 160, 400, 190],
                                    "text": "8,100",
                                    "row_span": [1, 2],
                                    "col_span": [2, 3],
                                },
                            ],
                        },
                    ],
                }
            ],
        },
        "source_filename": "cts-corporation_2019.pdf",
        "metadata": {"company": "CTS Corporation", "year": 2019},
    }
