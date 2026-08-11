"""Pluggable OCR text-recognition engines - the opt-in fast path for plain
TEXT_LIKE regions (Settings.ocr_fast_path), used instead of a VLM call.

Each engine wraps a *layout* engine's own bundled OCR model, keyed by
`settings.layout_engine`, so the OCR backend used always matches the active
layout engine rather than being an unrelated fourth choice. Not every layout
engine bundles OCR weights - `doclayout_yolo`, `markitdown`, and
`pymupdf4llm` have none, so `get_ocr_engine` returns None for those and the
pipeline falls back to the VLM, same as when the fast path is off.
"""

from __future__ import annotations

from doc2md.config import Settings
from doc2md.ocr_engines.base import OCREngine


def get_ocr_engine(settings: Settings) -> OCREngine | None:
    if settings.layout_engine == "mineru":
        from doc2md.ocr_engines.mineru_ocr import MineruOCREngine

        return MineruOCREngine()
    if settings.layout_engine == "paddleocr":
        from doc2md.ocr_engines.paddleocr_ocr import PaddleOCRTextEngine

        return PaddleOCRTextEngine()
    if settings.layout_engine == "docling":
        from doc2md.ocr_engines.docling_ocr import DoclingOCREngine

        return DoclingOCREngine()
    # doclayout_yolo / markitdown / pymupdf4llm bundle no OCR model - no fast
    # path available for these, the pipeline falls back to the VLM.
    return None
