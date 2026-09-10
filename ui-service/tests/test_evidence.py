import pymupdf

from evidence import render_citation_page


def _one_page_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=100, height=100)
    page.insert_text((10, 20), "Revenue 2022: $42 million")
    value = document.tobytes()
    document.close()
    return value


def test_citation_box_is_drawn_at_ocr_scale() -> None:
    image = render_citation_page(_one_page_pdf(), 1, [(20, 20, 120, 60)])

    assert image.size == (200, 200)
    red, green, blue = image.getpixel((20, 20))
    assert red > green
    assert red > blue


def test_out_of_range_page_is_rejected() -> None:
    try:
        render_citation_page(_one_page_pdf(), 2, [(0, 0, 10, 10)])
    except ValueError as exc:
        assert "outside a 1-page PDF" in str(exc)
    else:
        raise AssertionError("expected an invalid page to fail")
