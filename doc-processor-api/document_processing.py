

from doctr.models import ocr_predictor
import uuid
from typing import List, Union, Literal, Dict, Tuple
from pydantic import BaseModel
import numpy as np
import pymupdf
import torch
import hashlib
import re

from img2table.document import Image as Img2TableImage
from img2table.ocr import DocTR as Img2TableDocTR


def get_document_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


class Word(BaseModel):
    word_list: List[str]
    bbox_list: List[List[int]]


class Block(BaseModel):
    bbox: List[int]
    uuid: str
    words: Word
    text: str
    order: int
    content_type: Literal["paragraph"]


class Cell(BaseModel):
    bbox: List[int]
    text: str
    row_span: List[int]
    col_span: List[int]


class Table_Block(BaseModel):
    bbox: List[int]
    uuid: str
    text: str
    order: int
    content_type: Literal["table"]
    cells: List[Cell]


class Page(BaseModel):
    bbox: List[int]
    blocks: List[Union[Block, Table_Block]]
    page_number: int


class DocumentJSON(BaseModel):
    pages: List[Page]
    document_id: str


class PDF_to_Image:
    def convert(self, pdf_bytes):
        images = []
        png_bytes_list = []
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count
        for page in range(page_count):
            pix = doc[page].get_pixmap(
                matrix=pymupdf.Matrix(2, 2), alpha=False, colorspace=pymupdf.csRGB
            )
            image_matrix = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            images.append(image_matrix)
            png_bytes_list.append(pix.tobytes("png"))
        return images, png_bytes_list


class OCR_Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ocr_predictor(pretrained=True, preserve_aspect_ratio=True)
        self.model = self.model.to(self.device)

    def __call__(self, pdf_to_image):
        return self.model(pdf_to_image)


class Table_Model:
    def __init__(self):
        self.ocr = Img2TableDocTR(detect_language=False)

    def __call__(self, page_png_bytes):
        img = Img2TableImage(src=page_png_bytes, detect_rotation=False)
        return img.extract_tables(
            ocr=self.ocr,
            implicit_rows=True,
            implicit_columns=True,
            borderless_tables=True,
            min_confidence=30,
        )


def _bbox_overlap_ratio(inner, outer):
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    x_overlap = max(0, min(ix1, ox1) - max(ix0, ox0))
    y_overlap = max(0, min(iy1, oy1) - max(iy0, oy0))
    inter_area = x_overlap * y_overlap
    inner_area = max(1, (ix1 - ix0) * (iy1 - iy0))
    return inter_area / inner_area


def _normalize_cell_text(text: str) -> str:
 
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _fix_dollar_signs(cells: List[Cell], mode: str = "strip") -> None:

    if mode == "strip":
        for c in cells:
            tokens = [t for t in c.text.split() if t != "$"]
            c.text = " ".join(tokens)
        return

    if mode == "reattach":
        by_row: Dict[Tuple[int, int], List[Cell]] = {}
        for c in cells:
            by_row.setdefault(tuple(c.row_span), []).append(c)
        for row_cells in by_row.values():
            row_cells.sort(key=lambda c: (c.col_span[0], c.bbox[0]))
            for i in range(len(row_cells) - 1):
                cur, nxt = row_cells[i], row_cells[i + 1]
                tokens = cur.text.split()
                if tokens and tokens[-1] == "$":
                    cur.text = " ".join(tokens[:-1])
                    if not nxt.text.lstrip().startswith("$"):
                        nxt.text = ("$ " + nxt.text).strip()
        return

    raise ValueError(f"Unknown mode: {mode!r}")


def _band_overlap_span(x0, x1, edges, min_overlap_fraction=0.2):

    width = max(1e-6, x1 - x0)
    overlaps = []
    for i in range(len(edges) - 1):
        band_lo, band_hi = edges[i], edges[i + 1]
        ov = min(x1, band_hi) - max(x0, band_lo)
        overlaps.append(max(0.0, ov))

    covered = [i for i, ov in enumerate(overlaps) if ov / width >= min_overlap_fraction]
    if not covered:
        center = (x0 + x1) / 2
        covered = [
            min(
                range(len(edges) - 1),
                key=lambda i: abs(center - (edges[i] + edges[i + 1]) / 2),
            )
        ]
    return min(covered), max(covered)



def _build_cells_from_extracted_table(extracted_table):
 
    raw_cells = []
    seen = set()
    for row in extracted_table.content.values():
        for c in row:
            key = (c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2, c.value)
            if key in seen:
                continue
            seen.add(key)
            raw_cells.append(c)

    if not raw_cells:
        return [], [], []

    row_edges = sorted({c.bbox.y1 for c in raw_cells} | {c.bbox.y2 for c in raw_cells})
    col_edges = sorted({c.bbox.x1 for c in raw_cells} | {c.bbox.x2 for c in raw_cells})

    cells = []
    for c in raw_cells:
        row_start, row_end = _band_overlap_span(c.bbox.y1, c.bbox.y2, row_edges)
        col_start, col_end = _band_overlap_span(c.bbox.x1, c.bbox.x2, col_edges)
        cells.append(
            Cell(
                bbox=[int(c.bbox.x1), int(c.bbox.y1), int(c.bbox.x2), int(c.bbox.y2)],
                text=_normalize_cell_text(c.value or ""),
                row_span=[row_start, row_end],
                col_span=[col_start, col_end],
            )
        )
    return cells, row_edges, col_edges


def _looks_like_header_fragment(line_bbox, table_bbox):
    x0, y0, x1, y1 = line_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    horizontally_within = (x0 >= tx0 - 20) and (x1 <= tx1 + 20)
    narrow_enough = (x1 - x0) < 0.45 * (tx1 - tx0)
    return horizontally_within and narrow_enough


def _column_band(line_bbox, col_edges):
    xc = (line_bbox[0] + line_bbox[2]) / 2
    for i in range(len(col_edges) - 1):
        if col_edges[i] <= xc <= col_edges[i + 1]:
            return i
    return len(col_edges) - 2


def _stitch_header_row(table_block: Table_Block, col_edges, all_lines, header_search_margin=110):
    """Absorb short, column-aligned text sitting directly above the table
    into a single synthesized header row (row 0), shifting existing body
    rows down by one. Returns the set of line uuids that were absorbed,
    so the caller can exclude them from the normal paragraph pass."""
    tx0, ty0, tx1, ty1 = table_block.bbox

    candidates = [
        line
        for line in all_lines
        if ty0 - header_search_margin <= line["bbox"][3] <= ty0 + 2
        and _looks_like_header_fragment(line["bbox"], table_block.bbox)
    ]
    if not candidates:
        return set()

    per_column: Dict[int, List[dict]] = {}
    for line in candidates:
        col = _column_band(line["bbox"], col_edges)
        per_column.setdefault(col, []).append(line)

    for c in table_block.cells:
        c.row_span = [c.row_span[0] + 1, c.row_span[1] + 1]

    used_uuids = set()
    new_cells = []
    min_x, min_y, max_x = tx0, ty0, tx1
    for col, lines in per_column.items():
        lines_sorted = sorted(lines, key=lambda l: l["bbox"][1])  # top -> bottom
        text = " ".join(_normalize_cell_text(l["text"]) for l in lines_sorted)
        xs = [l["bbox"][0] for l in lines_sorted] + [l["bbox"][2] for l in lines_sorted]
        ys = [l["bbox"][1] for l in lines_sorted] + [l["bbox"][3] for l in lines_sorted]
        new_cells.append(
            Cell(
                bbox=[min(xs), min(ys), max(xs), max(ys)],
                text=text,
                row_span=[0, 1],
                col_span=[col, col+1],
            )
        )
        min_x = min(min_x, min(xs))
        min_y = min(min_y, min(ys))
        max_x = max(max_x, max(xs))
        for l in lines_sorted:
            used_uuids.add(l["uuid"])

    table_block.cells = new_cells + table_block.cells
    table_block.bbox = [min_x, min_y, max_x, ty1]
    return used_uuids


def JSON_Processing(output, pdf_to_image, png_bytes_list, pdf_bytes, table_model):
    if len(output.pages) != len(pdf_to_image):
        raise ValueError("OCR model generated unequal amount of pages ")

    pages_list = []
    for page_number, page in enumerate(output.pages):
        page_no = page_number + 1
        height, width = pdf_to_image[page_number].shape[:2]


        all_lines = []
        for block in page.blocks:
            for line in block.lines:
                if not line.words:
                    continue
                x2 = y2 = 100000
                x3 = y3 = -1
                words_list, bbox_list = [], []
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    x0 *= int(width)
                    x1 *= int(width)
                    y0 *= int(height)
                    y1 *= int(height)
                    x2 = min(x2, x0)
                    y2 = min(y2, y0)
                    x3 = max(x3, x1)
                    y3 = max(y3, y1)
                    words_list.append(word.value)
                    bbox_list.append([int(x0), int(y0), int(x1), int(y1)])
                all_lines.append(
                    {
                        "uuid": str(uuid.uuid4()),
                        "bbox": [int(x2), int(y2), int(x3), int(y3)],
                        "text": " ".join(words_list),
                        "words": Word(word_list=words_list, bbox_list=bbox_list),
                    }
                )

        extracted_tables = table_model(png_bytes_list[page_number])
        table_blocks = []
        absorbed_line_uuids = set()
        for et in extracted_tables:
            bbox = [int(et.bbox.x1), int(et.bbox.y1), int(et.bbox.x2), int(et.bbox.y2)]
            cells, _row_edges, col_edges = _build_cells_from_extracted_table(et)
            if not cells:
                continue
            _fix_dollar_signs(cells, mode="strip")  

            table_block = Table_Block(
                bbox=bbox,
                uuid=str(uuid.uuid4()),
                text="",
                order=0,
                content_type="table",
                cells=cells,
            )
            absorbed_line_uuids |= _stitch_header_row(table_block, col_edges, all_lines)
            table_block.text = " ".join(c.text for c in table_block.cells if c.text)
            table_blocks.append(table_block)

        table_bboxes = [tb.bbox for tb in table_blocks]

    
        blocks_list = []
        for line in all_lines:
            if line["uuid"] in absorbed_line_uuids:
                continue
            if any(_bbox_overlap_ratio(line["bbox"], tb) > 0.5 for tb in table_bboxes):
                continue
            blocks_list.append(
                Block(
                    bbox=line["bbox"],
                    uuid=line["uuid"],
                    words=line["words"],
                    text=line["text"],
                    order=len(blocks_list) + 1,
                    content_type="paragraph",
                )
            )

        blocks_list.extend(table_blocks)
        blocks_list = sorted(blocks_list, key=lambda b: (b.bbox[1], b.bbox[0]))
        for idx, b in enumerate(blocks_list):
            b.order = idx + 1

        pages_list.append(
            Page(
                bbox=[0, 0, int(width), int(height)],
                blocks=blocks_list,
                page_number=page_no,
            )
        )

    return DocumentJSON(pages=pages_list, document_id=get_document_id(pdf_bytes))


