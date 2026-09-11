"""Render retrieval bounding boxes on their cited source-PDF pages."""

from __future__ import annotations

from io import BytesIO

import pymupdf
from PIL import Image, ImageDraw


def render_citation_page(
    pdf_bytes: bytes,
    page_number: int,
    boxes: list[list[int] | tuple[int, int, int, int]],
) -> Image.Image:
    """Render a page at the OCR service's 2x scale and overlay its boxes."""
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_index = page_number - 1 if page_number > 0 else 0
        if page_index >= document.page_count:
            raise ValueError(
                f"citation page {page_number} is outside a {document.page_count}-page PDF"
            )
        pixmap = document[page_index].get_pixmap(
            matrix=pymupdf.Matrix(2, 2), alpha=False
        )
        image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGBA")
    finally:
        document.close()

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        x0, y0, x1, y1 = (int(value) for value in box)
        x0, x1 = sorted((max(0, x0), min(image.width, x1)))
        y0, y1 = sorted((max(0, y0), min(image.height, y1)))
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle((x0, y0, x1, y1), fill=(255, 221, 0, 72), outline=(220, 38, 38, 255), width=4)
    return Image.alpha_composite(image, overlay).convert("RGB")
