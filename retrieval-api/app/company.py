"""The issuer a financial document names in its own page furniture.

Filenames in this corpus are hashes and most pages never print their company,
so retrieval cannot tell one company's "Inventories" table from another's. A
filing does print its own name in running headers and footers, subsidiaries
banners and notes titles:

    "CTS CORPORATION 40"                      "172 Spirax-Sarco Engineering plc"
    "Lam Research Corporation 2019 10-K 61"   "KEMET CORPORATION AND SUBSIDIARIES"
    "Plexus Corp. Notes to Consolidated Financial Statements"
    "VMware, Inc." between "Table of Contents" and "NOTES TO CONSOLIDATED ..."

A name counts only on such a line: with page furniture on the line itself, or
as a bare name standing alone or beside a furniture line. Beside prose, a bare
name is a heading or a wrapped sentence about another company ("Energid
Technologies Corporation" above Teradyne's acquisition note, "Acquisition of
AOL Inc."), and names inside sentences or table rows with figures are ignored.
A document whose headers name two companies equally often gets no company.
Detection reads only the document's own chunks, so an upload and a rebuild of
the same document always agree.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from .models import Chunk
from .tables import first_row_title

# Legal-form words a company name ends with.
_SUFFIXES = {
    "ag", "asa", "bancorp", "co", "company", "corp", "corporation", "inc",
    "incorporated", "limited", "llc", "llp", "lp", "ltd", "nv", "plc", "sa", "se", "spa",
}
_SAME_SUFFIX = {"company": "co", "corporation": "corp", "incorporated": "inc", "limited": "ltd"}
_CONNECTORS = {"&", "and", "de", "der", "des", "du", "la", "le", "of", "the", "van", "von"}
# Words that make a heading rather than a name: "The Company", "Parent Company".
_GENERIC = {
    "affiliated", "annual", "associated", "business", "combined", "companies", "company",
    "consolidated", "corporate", "financial", "group", "holding", "its", "joint", "note",
    "notes", "operating", "other", "our", "parent", "related", "report", "reporting",
    "segment", "statement", "statements", "subsidiaries", "subsidiary", "this", "total",
    "venture", "ventures",
}
# Page furniture, tolerant of the OCR damage in the corpus ("FINANCIAL:
# STATEMENTS", "CONSOLIDATEDI", "AnnualReport: 2019", "Table ofContents").
_FURNITURE_END = (
    r"[-–—]?\s*\(\s*continued\s*\)",
    r"[-–—]?\s*continued",
    r"notes?\s+to\s+(?:the\s+)?(?:consolidated\w?\s+|combined\s+)?financial\w?:?\s*statements",
    r"(?:consolidated\w?\s+|combined\s+)?financial\w?:?\s*statements",
    r"and\s+(?:its\s+)?(?:consolidated\s+)?subsidiaries",
    r"annual\s*report:?(?:\s*and\s*accounts)?(?:\s*(?:19|20)\d{2}(?:[-/–]\d{2,4})?)?",
    r"(?:(?:19|20)\d{2}\s*)?(?:form\s*)?(?:10-?k|20-?f|40-?f)",
    r"part\s+[ivx]+",
    r"table\s*of\s*contents",
    r"\(?(?:u\.?\s?s\.?\s+)?dollars\s+in\s+(?:thousands|millions|billions)\)?",
    r"\d{1,3}",
    r"[|:\-–—]",
)
_FURNITURE_START = (r"table\s*of\s*contents", r"\d{1,3}", r"[|:\-–—]")
_END = re.compile(r"(?:^|\s+)(?:" + "|".join(_FURNITURE_END) + r")\s*$", re.IGNORECASE)
_START = re.compile(r"^(?:" + "|".join(_FURNITURE_START) + r")(?:\s+|$)", re.IGNORECASE)
_TOKEN = re.compile(r"&|[A-Za-z0-9][A-Za-z0-9&'.\-/]*,?")


def _strip_furniture(line: str) -> str:
    text = " ".join(line.split())
    while True:
        stripped = _START.sub("", _END.sub("", text)).strip()
        if stripped == text:
            return text
        text = stripped


def _is_furniture(line: str) -> bool:
    text = " ".join(line.split())
    return bool(text) and not _strip_furniture(text)


def issuer_name(line: str) -> tuple[str, str] | None:
    """(comparison key, name as printed) when a line is a bare company name
    once its page furniture is removed."""
    text = re.sub(r"\.{2,}", ".", _strip_furniture(line)).rstrip(",").strip()
    tokens = text.split()
    if not 2 <= len(tokens) <= 8 or not all(_TOKEN.fullmatch(token) for token in tokens):
        return None
    words = [re.sub(r"[.,]", "", token).casefold() for token in tokens]
    if words[-1] not in _SUFFIXES or words[0] in _CONNECTORS - {"the"}:
        return None
    name_tokens, name_words = tokens[:-1], words[:-1]
    if any(character.isupper() for character in text):
        # Cased names capitalise every word but connectors ("Jabil Inc.",
        # "Procter & Gamble"), which rules out sentences that end in a name.
        if not all(
            word in _CONNECTORS or any(character.isupper() for character in token)
            for token, word in zip(name_tokens, name_words)
        ):
            return None
    elif len(name_tokens) > 3:
        # All lower case only as a short brand, "intu properties plc".
        return None
    if not any(
        word not in _GENERIC and word not in _CONNECTORS and re.search(r"[a-z]", word)
        for word in name_words
    ):
        return None
    key = " ".join(
        [*(re.sub(r"[^a-z0-9&]", "", word) for word in name_words), _SAME_SUFFIX.get(words[-1], words[-1])]
    )
    return key, text


def _header_lines(chunk: Chunk) -> list[str]:
    """The lines of a chunk that name a company the way page furniture does."""
    if chunk.content_type == "table":
        title = first_row_title(chunk.table_cells)
        return [title] if title and issuer_name(title) else []
    lines = chunk.text.split("\n")
    found = []
    for index, line in enumerate(lines):
        if issuer_name(line) is None:
            continue
        if " ".join(line.split()) != _strip_furniture(line):
            found.append(line)
            continue
        neighbours = [lines[i] for i in (index - 1, index + 1) if 0 <= i < len(lines)]
        if not neighbours or any(_is_furniture(neighbour) for neighbour in neighbours):
            found.append(line)
    return found


def detect_company(chunks: Iterable[Chunk]) -> str | None:
    """The company the document's own headers name most often, if one does."""
    lines: set[tuple[int, str]] = set()
    for chunk in chunks:
        lines.update((chunk.page, line) for line in _header_lines(chunk))

    counts: Counter[str] = Counter()
    printed: dict[str, Counter[str]] = defaultdict(Counter)
    for _, line in sorted(lines):
        key, name = issuer_name(line)
        counts[key] += 1
        printed[key][name] += 1
    if not counts:
        return None
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return min(printed[ranked[0][0]].items(), key=lambda item: (-item[1], item[0]))[0]
