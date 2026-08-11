"""Layout/segmentation via PyMuPDF4LLM's high-precision vector layout engine.

PyMuPDF4LLM performs vector layout analysis (font hierarchies, text blocks, table rectangles,
image bounds) directly from PDF documents with high accuracy and zero GPU overhead.
"""

from __future__ import annotations

from doc2md.classify import PP_DOCLAYOUT_LABEL_TO_BUCKET, bucket_for_label
from doc2md.config import Settings
from doc2md.layout_engines.base import LayoutEngine, full_page_fallback_region, resolve_source_pdf
from doc2md.layout_engines.dedup import suppress_contained_duplicates
from doc2md.layout_engines.reading_order import assign_reading_order
from doc2md.models import Bucket, PageImage, Region

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
        regions: list[Region] = []

        possible_pdf = resolve_source_pdf(page)

        if possible_pdf is not None:
            try:
                import fitz
                import pymupdf4llm

                doc = fitz.open(possible_pdf)
                if 0 <= (page.page_no - 1) < len(doc):
                    pdf_page = doc[page.page_no - 1]
                    pw, ph = float(pdf_page.rect.width), float(pdf_page.rect.height)
                    scale_x = page.width / pw if pw > 0 else 1.0
                    scale_y = page.height / ph if ph > 0 else 1.0

                    chunks = pymupdf4llm.to_markdown(str(possible_pdf), page_chunks=True)
                    page_idx = page.page_no - 1
                    if 0 <= page_idx < len(chunks):
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
                            bucket = bucket_for_label(
                                mapped_label, PP_DOCLAYOUT_LABEL_TO_BUCKET, self._settings.skip_labels
                            )
                            if bucket is not Bucket.SKIP:
                                regions.append(
                                    Region(
                                        page_no=page.page_no,
                                        label=mapped_label,
                                        bucket=bucket,
                                        bbox=bbox,
                                        score=1.0,
                                        order_index=0,
                                    )
                                )
                doc.close()
            except Exception:
                pass

        if not regions:
            regions.extend(full_page_fallback_region(page, self._settings, PP_DOCLAYOUT_LABEL_TO_BUCKET))

        return assign_reading_order(suppress_contained_duplicates(regions))
