"""Layout/segmentation via Microsoft MarkItDown-style document layout analysis.

Rich PDF extraction (table/figure/text-block detection) is done directly via
pdfplumber, the same library MarkItDown itself uses internally for PDFs. For
inputs with no backing PDF (e.g. a directory of page images), or a PDF page
pdfplumber can't parse, this falls back to a single full-page text region
like every other engine (`doc2md.layout_engines.base.full_page_fallback_region`)
rather than a real MarkItDown conversion call - the `markitdown` package's
`convert_stream()` on a scanned page image has no text layer to work from
here anyway, and doc2md's own VLM already handles that region.

Regions are ordered using doc2md's heuristic reading-order pass
(doc2md.layout_engines.reading_order), same as the Docling engine.
"""

from __future__ import annotations

from doc2md.classify import MARKITDOWN_LABEL_TO_BUCKET
from doc2md.config import Settings
from doc2md.layout_engines.base import (
    LayoutEngine,
    finalize_regions,
    make_region,
    resolve_source_pdf,
    safe_detect,
)
from doc2md.models import PageImage, Region


class MarkItDownEngine(LayoutEngine):
    """Layout detection engine powered by pdfplumber (MarkItDown's own PDF backend)."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def detect(self, page: PageImage) -> list[Region]:
        def _run() -> list[Region]:
            regions: list[Region] = []

            # Check if source_name points to an existing PDF or document file
            possible_pdf = resolve_source_pdf(page)
            if possible_pdf is None:
                return regions

            import pdfplumber

            with pdfplumber.open(possible_pdf) as pdf:
                if not (0 <= (page.page_no - 1) < len(pdf.pages)):
                    return regions

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
                    region = make_region(
                        page.page_no, "Table", MARKITDOWN_LABEL_TO_BUCKET, self._settings.skip_labels, bbox, 1.0
                    )
                    if region is not None:
                        regions.append(region)

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
                        region = make_region(
                            page.page_no,
                            "Picture",
                            MARKITDOWN_LABEL_TO_BUCKET,
                            self._settings.skip_labels,
                            bbox,
                            0.9,
                        )
                        if region is not None:
                            regions.append(region)

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
                        region = make_region(
                            page.page_no, label, MARKITDOWN_LABEL_TO_BUCKET, self._settings.skip_labels, bbox, 0.95
                        )
                        if region is not None:
                            regions.append(region)

            return finalize_regions(regions, self._settings)

        return safe_detect("markitdown", page, self._settings, MARKITDOWN_LABEL_TO_BUCKET, _run)
