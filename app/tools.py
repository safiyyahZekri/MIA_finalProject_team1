"""
The four deterministic tools required by the spec:
  search_documents(query)   -> corpus-wide semantic (vector) retrieval
  search_tables(query)      -> table-aware retrieval
  calculate(expression)     -> deterministic arithmetic (never the LLM)
  filter_documents(metadata)-> direct non-vector metadata lookup

Exposed both as plain async functions (used directly by the deterministic
LangGraph nodes in app.graph) and as LangChain @tool objects (TOOLS list),
so this module also plugs into a tool-calling agent executor if one is
wired up in front of / alongside the graph.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from app.calculator import CalculatorError, calculate as _calculate
from app.retrieval_client import retrieval_client


async def search_documents(query: str, top_k: Optional[int] = None, document_id: Optional[str] = None) -> list[dict]:
    """Corpus-wide semantic (embedding) search over all indexed financial documents."""
    return await retrieval_client.search_documents(query, top_k=top_k, document_id=document_id)


async def search_bm25(query: str, top_k: Optional[int] = None, document_id: Optional[str] = None) -> list[dict]:
    """Non-vector lexical (BM25) search -- the required non-embedding retrieval path."""
    return await retrieval_client.search_bm25(query, top_k=top_k, document_id=document_id)


async def search_tables(query: str, top_k: Optional[int] = None, document_id: Optional[str] = None) -> list[dict]:
    """Table-aware search for line items / numeric figures inside financial tables."""
    return await retrieval_client.search_tables(query, top_k=top_k, document_id=document_id)


async def filter_documents(metadata: dict, top_k: Optional[int] = None) -> list[dict]:
    """Direct metadata filter lookup (e.g. by document_id, page, section) -- non-vector."""
    return await retrieval_client.filter_documents(metadata, top_k=top_k)


def calculate(expression: str) -> float:
    """Evaluate a pure arithmetic expression deterministically. Raises CalculatorError on invalid input."""
    return _calculate(expression)


# ------------------------------------------------------- LangChain @tool ---
# Available for a tool-calling agent variant; not required by the
# deterministic graph in app.graph, which calls the functions above directly.

@tool
async def search_documents_tool(query: str) -> list[dict]:
    """Corpus-wide semantic (embedding) search over all indexed financial documents."""
    return await search_documents(query)


@tool
async def search_tables_tool(query: str) -> list[dict]:
    """Table-aware search for line items / numeric figures inside financial tables."""
    return await search_tables(query)


@tool
def calculate_tool(expression: str) -> float:
    """Evaluate a pure arithmetic expression, e.g. '(150-120)/120*100'. Never guess arithmetic yourself."""
    try:
        return calculate(expression)
    except CalculatorError as e:
        raise ValueError(str(e)) from e


@tool
async def filter_documents_tool(document_id: str = None, page: int = None, section: str = None) -> list[dict]:
    """Direct metadata filter lookup by document_id/page/section (non-vector retrieval)."""
    metadata = {k: v for k, v in {"document_id": document_id, "page": page, "section": section}.items() if v is not None}
    return await filter_documents(metadata)


TOOLS = [search_documents_tool, search_tables_tool, calculate_tool, filter_documents_tool]
