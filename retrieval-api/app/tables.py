"""Grid view of extracted table cells, shared by chunking and company detection."""

from __future__ import annotations

import re

from .models import Cell

UNIT_PATTERN = re.compile(
    r"(?i)(?:amounts?\s+)?(?:in|expressed in)\s+"
    r"(?:u\.s\.\s+)?(?:dollars?|thousands?|millions?|billions?|percent(?:ages?)?)"
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_PERIOD_LABEL = re.compile(
    r"(?i)\b(?:years?|ended|ending|months?|quarters?|fiscal|weeks?|period|as\s+of)\b"
)
# Column headers, page furniture and footnotes that sit in a table's first row
# but do not name the table: "Year Ended", "Yearl Ended May 31", "Fiscal
# year-end", "Three Months Ended", "December 31", "Fair Values as of",
# "Table ofContents", "Notes to Consolidated Financial Statements",
# "Notes: () Based on ...".
_NOT_A_TITLE_START = re.compile(
    r"(?i)^\W*(?:"
    r"(?:year\w?|years|fiscal|quarters?|months?|weeks?|periods?|as\s+of|for\s+the"
    r"|table\s*of\s*contents|notes?\s*:)(?:\b|\W|$)"
    r"|notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?(?:financial\s+)?statements\W*(?:continued\W*)?$"
    r"|.*\b(?:months?|weeks?|years?|quarters?)\s+end(?:ed|ing)\b"
    r"|.*\bas\s*of\W*$"
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2}\b"
    r")"
)
# A figure means a data row whose cells were merged: "... 2,590 (27,066) $".
_FIGURE = re.compile(r"\d,\d{3}|\$\s*\(?\d|\(\d")


def clean(value: str) -> str:
    return " ".join(value.split())


def table_grid(cells: list[Cell]) -> list[list[str]]:
    """Cell text by row and column; a spanning cell fills every slot it covers."""
    row_count = max(cell.row_span[1] for cell in cells)
    column_count = max(cell.col_span[1] for cell in cells)
    grid: list[list[str]] = [["" for _ in range(column_count)] for _ in range(row_count)]
    for cell in sorted(
        cells,
        key=lambda value: (value.row_span[0], value.col_span[0], value.row_span[1], value.col_span[1]),
    ):
        value = clean(cell.text)
        if not value:
            continue
        for row in range(cell.row_span[0], cell.row_span[1]):
            for column in range(cell.col_span[0], cell.col_span[1]):
                grid[row][column] = value
    return grid


def _units_like(value: str) -> bool:
    """A whole value that only states units: "(in thousands)", "Amounts in millions"."""
    return (
        value.startswith("(")
        or bool(UNIT_PATTERN.fullmatch(value.strip(" ().,")))
        or bool(re.fullmatch(r"[$£€%]+", value))
    )


def _ends_like_a_sentence(value: str) -> bool:
    if value.endswith((":", ";", ",")):
        return True
    # "amounts." ends a sentence; "Corp." and "S.A." end a name.
    words = value.rstrip(".").split()
    return value.endswith(".") and bool(words) and len(words[-1]) > 4


def first_row_title(cells: list[Cell]) -> str | None:
    """The heading a table carries in its own first row, if that row is one.

    The first filled row must hold a heading in its leftmost filled cell and
    otherwise only units or period labels:
        "Operating costs | [empty] | [empty]"                           (A037)
        "ITEM 6. SELECTED FINANCIAL DATA Financial Highlights (dollars in
         thousands, ...) | Fiscal Years Ended"                          (A068)
    A first row of years, figures, column headers ("Year Ended"), page
    furniture ("Table of Contents"), a footnote, a single word or a prose
    fragment is not.
    """
    if not cells:
        return None
    rows = [row for row in table_grid(cells) if any(row)]
    if len(rows) < 2 or len(rows[0]) < 2:
        return None
    title = next(value for value in rows[0] if value)
    letters = [character for character in title if character.isalpha()]
    if (
        len(letters) < 3
        or not letters[0].isupper()
        or not 2 <= len(title.split()) <= 20
        or len(title) > 160
        or _YEAR.search(title)
        or _FIGURE.search(title)
        or _NOT_A_TITLE_START.match(title)
        or _units_like(title)
        or _ends_like_a_sentence(title)
    ):
        return None
    for other in {value for value in rows[0] if value and value != title}:
        period_label = not re.search(r"\d", other) and bool(_PERIOD_LABEL.search(other))
        if not (_units_like(other) or period_label):
            return None
    return title
