"""Layout/segmentation via PyMuPDF4LLM's high-precision vector layout engine.

PyMuPDF4LLM performs vector layout analysis (font hierarchies, text blocks, table rectangles,
image bounds) directly from PDF documents with high accuracy and zero GPU overhead.
"""

from __future__ import annotations

from doc2md.classify import PP_DOCLAYOUT_LABEL_TO_BUCKET
from doc2md.config import Settings
from doc2md.layout_engines.base import (
    LayoutEngine,
    finalize_regions,
    make_region,
    resolve_source_pdf,
    safe_detect,
)
from doc2md.models import PageImage, Region

PYMUPDF4LLM_LABEL_MAP: dict[str, str] = {
    "section-header": "doc_title",
    "title": "doc_title",
    "header": "doc_title",
    "heading": "doc_title",
    "text": "text",
    "paragraph": "text",
    "list": "content",
    "table": "table",
    "picture": "image",
    "figure": "image",
    "page-header": "header",
    "page-footer": "footer",
}


class PyMuPDF4LLMEngine(LayoutEngine):
    """Vector-based layout engine powered by PyMuPDF4LLM."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def detect(self, page: PageImage) -> list[Region]:
        def _run() -> list[Region]:
            regions: list[Region] = []
            possible_pdf = resolve_source_pdf(page)
            if possible_pdf is None:
                return regions

            import fitz
            import pymupdf4llm

            with fitz.open(possible_pdf) as doc:
                if not (0 <= (page.page_no - 1) < len(doc)):
                    return regions

                pdf_page = doc[page.page_no - 1]
                pw, ph = float(pdf_page.rect.width), float(pdf_page.rect.height)
                scale_x = page.width / pw if pw > 0 else 1.0
                scale_y = page.height / ph if ph > 0 else 1.0

                # use_ocr=False: pymupdf4llm's newer "layout" mode defaults to
                # True and will opportunistically use ANY OCR backend it finds
                # importable (e.g. rapidocr-onnxruntime, installed for the
                # unrelated --ocr-fast-path feature) - which changes this
                # engine's table/region classification output and defeats the
                # whole point of this being the zero-GPU-overhead vector engine.
                chunks = pymupdf4llm.to_markdown(str(possible_pdf), page_chunks=True, use_ocr=False)
                page_idx = page.page_no - 1
                if not (0 <= page_idx < len(chunks)):
                    return regions

                p_chunk = chunks[page_idx]
                page_boxes = p_chunk.get("page_boxes", [])
                for box_info in page_boxes:
                    cls_name = box_info.get("class", "text").lower()
                    raw_bbox = box_info.get("bbox", (0, 0, pw, ph))
                    x0, y0, x1, y1 = raw_bbox
                    bbox = (
                        int(x0 * scale_x),
                        int(y0 * scale_y),
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                    )

                    mapped_label = PYMUPDF4LLM_LABEL_MAP.get(cls_name, "text")
                    region = make_region(
                        page.page_no,
                        mapped_label,
                        PP_DOCLAYOUT_LABEL_TO_BUCKET,
                        self._settings.skip_labels,
                        bbox,
                        1.0,
                    )
                    if region is not None:
                        regions.append(region)

            return finalize_regions(regions, self._settings)

        return safe_detect("pymupdf4llm", page, self._settings, PP_DOCLAYOUT_LABEL_TO_BUCKET, _run)
