"""Layout/segmentation via Microsoft MarkItDown document text & layout analyzer.

MarkItDown is a unified document conversion tool from Microsoft. For PDF and document
pages, it extracts text blocks, tables, and images/figures using rich document & layout
analyzers (such as pdfplumber, pdfminer, and MarkItDown converters).

Like the Docling engine, regions are ordered using doc2md's heuristic reading
order pass (doc2md.layout_engines.reading_order).
"""

from __future__ import annotations

import io
from PIL import Image

from doc2md.classify import MARKITDOWN_LABEL_TO_BUCKET, bucket_for_label
from doc2md.config import Settings
from doc2md.layout_engines.base import LayoutEngine, resolve_source_pdf
from doc2md.layout_engines.dedup import suppress_contained_duplicates
from doc2md.layout_engines.reading_order import assign_reading_order
from doc2md.models import Bucket, PageImage, Region


class MarkItDownEngine(LayoutEngine):
    """Layout detection engine powered by Microsoft MarkItDown & pdfplumber."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            from markitdown import MarkItDown

            self._converter = MarkItDown()
        return self._converter

    def detect(self, page: PageImage) -> list[Region]:
        regions: list[Region] = []
        _converter = self._get_converter()

        # Check if source_name points to an existing PDF or document file
        possible_pdf = resolve_source_pdf(page)

        if possible_pdf is not None:
            try:
                import pdfplumber

                with pdfplumber.open(possible_pdf) as pdf:
                    if 0 <= (page.page_no - 1) < len(pdf.pages):
                        pdf_page = pdf.pages[page.page_no - 1]
                        pw, ph = float(pdf_page.width), float(pdf_page.height)
                        scale_x = page.width / pw if pw > 0 else 1.0
                        scale_y = page.height / ph if ph > 0 else 1.0

                        # 1. Detect tables via pdfplumber (used by MarkItDown for rich PDF extraction)
                        tables = pdf_page.find_tables()
                        table_bboxes = []
                        for tbl in tables:
                            x0, top, x1, bottom = tbl.bbox
                            bbox = (
                                int(x0 * scale_x),
                                int(top * scale_y),
                                int(x1 * scale_x),
                                int(bottom * scale_y),
                            )
                            table_bboxes.append(bbox)
                            bucket = bucket_for_label(
                                "Table", MARKITDOWN_LABEL_TO_BUCKET, self._settings.skip_labels
                            )
                            if bucket is not Bucket.SKIP:
                                regions.append(
                                    Region(
                                        page_no=page.page_no,
                                        label="Table",
                                        bucket=bucket,
                                        bbox=bbox,
                                        score=1.0,
                                        order_index=0,
                                    )
                                )

                        # 2. Detect figures / images on the page
                        for img_obj in getattr(pdf_page, "images", []):
                            x0 = float(img_obj.get("x0", 0))
                            top = float(img_obj.get("top", 0))
                            x1 = float(img_obj.get("x1", x0 + float(img_obj.get("width", 0))))
                            bottom = float(img_obj.get("bottom", top + float(img_obj.get("height", 0))))
                            if (x1 - x0) > 10 and (bottom - top) > 10:
                                bbox = (
                                    int(x0 * scale_x),
                                    int(top * scale_y),
                                    int(x1 * scale_x),
                                    int(bottom * scale_y),
                                )
                                bucket = bucket_for_label(
                                    "Picture", MARKITDOWN_LABEL_TO_BUCKET, self._settings.skip_labels
                                )
                                if bucket is not Bucket.SKIP:
                                    regions.append(
                                        Region(
                                            page_no=page.page_no,
                                            label="Picture",
                                            bucket=bucket,
                                            bbox=bbox,
                                            score=0.9,
                                            order_index=0,
                                        )
                                    )

                        # 3. Detect text blocks / words
                        words = pdf_page.extract_words()
                        if words:
                            # Group words into blocks / lines
                            words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
                            current_block = []
                            blocks = []
                            for word in words_sorted:
                                if not current_block:
                                    current_block.append(word)
                                else:
                                    last_word = current_block[-1]
                                    # If on same line or close vertical gap (<= 15 pt), add to block
                                    if abs(word["top"] - last_word["top"]) < 15:
                                        current_block.append(word)
                                    else:
                                        blocks.append(current_block)
                                        current_block = [word]
                            if current_block:
                                blocks.append(current_block)

                            for block in blocks:
                                min_x0 = min(w["x0"] for w in block)
                                min_top = min(w["top"] for w in block)
                                max_x1 = max(w["x1"] for w in block)
                                max_bottom = max(w["bottom"] for w in block)
                                bbox = (
                                    int(min_x0 * scale_x),
                                    int(min_top * scale_y),
                                    int(max_x1 * scale_x),
                                    int(max_bottom * scale_y),
                                )

                                # Skip if block is inside a detected table
                                inside_table = False
                                for tb_box in table_bboxes:
                                    if (
                                        bbox[0] >= tb_box[0] - 2
                                        and bbox[1] >= tb_box[1] - 2
                                        and bbox[2] <= tb_box[2] + 2
                                        and bbox[3] <= tb_box[3] + 2
                                    ):
                                        inside_table = True
                                        break
                                if inside_table:
                                    continue

                                label = "Title" if (max_bottom - min_top) > 24 else "Text"
                                bucket = bucket_for_label(
                                    label, MARKITDOWN_LABEL_TO_BUCKET, self._settings.skip_labels
                                )
                                if bucket is not Bucket.SKIP:
                                    regions.append(
                                        Region(
                                            page_no=page.page_no,
                                            label=label,
                                            bucket=bucket,
                                            bbox=bbox,
                                            score=0.95,
                                            order_index=0,
                                        )
                                    )
            except Exception:
                pass

        # Fallback if no PDF regions detected or for direct image pages:
        # run MarkItDown on page image stream or treat whole page as region
        if not regions:
            buf = io.BytesIO()
            page.image.save(buf, format="PNG")
            buf.seek(0)
            try:
                res = _converter.convert_stream(buf, file_extension=".png")
                extracted_text = res.text_content if res else ""
            except Exception:
                extracted_text = ""

            bucket = bucket_for_label("Text", MARKITDOWN_LABEL_TO_BUCKET, self._settings.skip_labels)
            if bucket is not Bucket.SKIP:
                regions.append(
                    Region(
                        page_no=page.page_no,
                        label="Text",
                        bucket=bucket,
                        bbox=(0, 0, page.width, page.height),
                        score=1.0,
                        order_index=0,
                    )
                )

        return assign_reading_order(suppress_contained_duplicates(regions))
