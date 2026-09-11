from __future__ import annotations

import pytest

from app.company import detect_company, issuer_name
from app.models import Cell, Chunk
from app.tables import first_row_title


def _cells(rows: list[list[str]]) -> list[Cell]:
    return [
        Cell(bbox=[0, 0, 1, 1], text=value, row_span=[r, r + 1], col_span=[c, c + 1])
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
        if value
    ]


def _paragraph(text: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"{page}-{text[:16]}",
        document_id="doc",
        source_filename="doc.pdf",
        page=page,
        section="Document",
        content_type="paragraph",
        text=text,
        parent_text=text,
        bbox=(0, 0, 1, 1),
        source_block_ids=["block"],
    )


def _table(rows: list[list[str]], page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"{page}-table",
        document_id="doc",
        source_filename="doc.pdf",
        page=page,
        section="Document",
        content_type="table",
        text="table",
        parent_text="table",
        bbox=(0, 0, 1, 1),
        source_block_ids=["table"],
        table_cells=_cells(rows),
    )


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("CTS CORPORATION 40", "CTS CORPORATION"),
        ("172 Spirax-Sarco Engineering plc", "Spirax-Sarco Engineering plc"),
        ("Spirent Communications plc Annual Report 2019", "Spirent Communications plc"),
        ("Lam Research Corporation 2019 10-K 61", "Lam Research Corporation"),
        ("intu properties plc Annual report2019", "intu properties plc"),
        ("BLACK KNIGHT, INC.", "BLACK KNIGHT, INC."),
        ("KEMET CORPORATION AND SUBSIDIARIES", "KEMET CORPORATION"),
        ("Plexus Corp. Notes to Consolidated Financial Statements", "Plexus Corp."),
        (
            "ADVANCED ENERGY INDUSTRIES, INC. NOTES TO CONSOLIDATED FINANCIAL STATEMENTS - (continued)",
            "ADVANCED ENERGY INDUSTRIES, INC.",
        ),
        ("The Boeing Company", "The Boeing Company"),
        ("AT&T Inc.", "AT&T Inc."),
    ],
)
def test_a_line_holding_only_a_company_name_and_page_furniture_names_the_issuer(line, name):
    found = issuer_name(line)
    assert found is not None
    assert found[1] == name


@pytest.mark.parametrize(
    "line",
    [
        "The Company",
        "The Group",
        "Parent Company",
        "Net income attributable to Jabil Inc.",
        "Our sales to Applied Materials, Inc., LAM Research, and Nidec Corporation include precision power products",
        'General Electric Company ("GE")',
        "Row 3: Applied Materials, Inc. | 36,849 | 14.9 %",
        "Slaughter V. Sykes Enterprises, Inc., Case No. 17 Civ. 2038.",
        "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS - (Continued)",
        "SYKES",
        "",
    ],
)
def test_names_inside_sentences_rows_or_generic_headings_are_not_an_issuer(line):
    assert issuer_name(line) is None


def test_case_punctuation_and_legal_form_spelling_do_not_split_one_company():
    assert issuer_name("CTS CORPORATION 53")[0] == issuer_name("CTS Corp.")[0]


def test_the_issuer_comes_from_headers_not_from_companies_named_in_the_text():
    chunks = [
        _paragraph("Our sales to Applied Materials, Inc. and Nidec Corporation grew.\nCTS CORPORATION 53"),
        _paragraph("CTS CORPORATION 54", page=2),
        _table([["Customer", "2019"], ["General Electric Company", "12.4%"]]),
    ]
    assert detect_company(chunks) == "CTS CORPORATION"


def test_a_table_whose_first_row_is_the_company_banner_counts():
    chunks = [
        _table([["TEEKAY CORPORATION AND SUBSIDIARIES", "", ""], ["", "2019", "2018"], ["Revenue", "10", "9"]])
    ]
    assert detect_company(chunks) == "TEEKAY CORPORATION"


def test_two_companies_named_equally_often_give_no_company():
    assert detect_company([_paragraph("Plexus Corp. 12"), _paragraph("Jabil Inc. 40", page=2)]) is None


def test_a_line_repeated_on_one_page_counts_once():
    chunks = [_paragraph("Plexus Corp. 12"), _paragraph("Plexus Corp. 12"), _paragraph("Jabil Inc. 40", page=2)]
    assert detect_company(chunks) is None


def test_the_company_named_most_often_wins():
    chunks = [_paragraph("Plexus Corp. 12"), _paragraph("Jabil Inc."), _paragraph("Plexus Corp. 13", page=2)]
    assert detect_company(chunks) == "Plexus Corp."


def test_a_document_without_a_header_name_gets_no_company():
    assert detect_company([_paragraph("The Company recognizes revenue when control transfers.")]) is None


@pytest.mark.parametrize(
    ("text", "company"),
    [
        ("Vodafone Group Plc\n170", "Vodafone Group Plc"),
        ("224\nTencent Holdings Limited", "Tencent Holdings Limited"),
        (
            (
                "Table of Contents\nVMware, Inc.\nNOTES TO CONSOLIDATED FINANCIAL: STATEMENTS -(Continued)\n"
                "The following table summarizes our leases."
            ),
            "VMware, Inc.",
        ),
        ("TATA CONSULTANCY SERVICES LIMITED\nAnnual Report 2018-19", "TATA CONSULTANCY SERVICES LIMITED"),
        ("BTGroup plc\nAnnualReport: 2019", "BTGroup plc"),
        ("Form 10-K Part II\nCincinnati Bell Inc.", "Cincinnati Bell Inc."),
        ("BRAINSTORM CELL THERAPEUTICS INC.\nU.S. dollars in thousands", "BRAINSTORM CELL THERAPEUTICS INC."),
        ("LEIDOS HOLDINGS, INC.", "LEIDOS HOLDINGS, INC."),
        ("JACK IN THE BOX INC.. AND SUBSIDIARIES", "JACK IN THE BOX INC."),
    ],
)
def test_a_bare_name_counts_alone_or_beside_page_furniture(text, company):
    assert detect_company([_paragraph(text)]) == company


@pytest.mark.parametrize(
    "text",
    [
        "Energid Technologies Corporation\nOn February 26, 2018, Teradyne acquired all of the issued shares.",
        "5.3\nGimi MS Corporation\nIn April 2019, Gimi MS Corporation entered into a Subscription Agreement.",
        "loss. Such gain or loss, if any, may be material.\nAcquisition of AOL Inc.\nIn May 2015, we entered into",
        "Fiscal 2019 Acquisitions\nCatalyst Repository Systems Inc.\nOn January 31, 2019, we acquired it.",
        "network\nTotal Accountants NV\nnetwork",
        "In May 2015, we completed the Acquisition of AOL Inc. 2015\nand paid cash.",
    ],
)
def test_a_bare_name_beside_prose_is_not_the_issuer(text):
    assert detect_company([_paragraph(text)]) is None


def test_a_first_row_holding_only_a_heading_titles_the_table():
    rows = [["Operating costs", "", ""], ["million", "2019", "2018"], ["Product development", "96.5", "96.9"]]
    assert first_row_title(_cells(rows)) == "Operating costs"


def test_a_heading_beside_a_period_label_titles_the_table():
    heading = (
        "ITEM 6. SELECTED FINANCIAL DATA Financial Highlights "
        "(dollars in thousands, except per share amounts)"
    )
    rows = [
        [heading, "", "Fiscal Years Ended", ""],
        ["Income Statement Data", "September 28, 2019", "September 29, 2018", "September 30, 2017"],
        ["Net sales", "3,164,434", "2,873,508", "2,528,052"],
    ]
    assert first_row_title(_cells(rows)) == heading


@pytest.mark.parametrize(
    "rows",
    [
        [["2019", "%"], ["232.3", "72.6"]],
        [["Item", "2019", "2018"], ["Finished goods", "9,447", "8,100"]],
        [["Name", "Position", "Age"], ["A. Smith", "Director", "61"]],
        [["Operating costs"], ["Product development"]],
        [["Operating costs", "", ""]],
        [["other relevant bodies; adopting a low risk appetite", ""], ["a", "b"]],
        [["Assets:", "", ""], ["Cash", "1", "2"]],
        [["(in thousands)", "", ""], ["Cash", "1", "2"]],
        [["Financial Highlights 2019", "", ""], ["Cash", "1", "2"]],
        [["Year Ended", "", ""], ["Cash", "1", "2"]],
        [["Yearl Ended May 31, Percent Change", ""], ["Cash", "1"]],
        [["Fiscal year-end", "", ""], ["Cash", "1", "2"]],
        [["Table ofContents", "", ""], ["Cash", "1", "2"]],
        [["Notes: () Based on the Group's direct equity interest", ""], ["Cash", "1"]],
        [["Interest cost capitalized - project assets 2,590 (27,066) $ Interest expense", ""], ["Cash", "1"]],
        [["Balance at December 31, $ 2,870", ""], ["Cash", "1"]],
        [["Goodwill", "", ""], ["Cash", "1", "2"]],
        [["Three Months Ended", "", ""], ["Cash", "1", "2"]],
        [["Revenue and Receipts For the Year Ended", "", ""], ["Cash", "1", "2"]],
        [["Notes to Consolidated Financial Statements (Continued)", ""], ["Cash", "1"]],
        [["NOTES TO THE CONSOLIDATED STATEMENTS", ""], ["Cash", "1"]],
        [["December 31", "", ""], ["Cash", "1", "2"]],
        [["Fair Values as of", "", ""], ["Cash", "1", "2"]],
    ],
)
def test_first_rows_that_are_not_headings_give_no_title(rows):
    assert first_row_title(_cells(rows)) is None


def test_a_company_banner_before_the_notes_heading_still_titles_the_table():
    rows = [["Plexus Corp. Notes to Consolidated Financial Statements", "", ""], ["", "2019", "2018"], ["Cash", "1", "2"]]
    assert first_row_title(_cells(rows)) == "Plexus Corp. Notes to Consolidated Financial Statements"
