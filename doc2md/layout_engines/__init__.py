"""Pluggable layout/segmentation engines."""

from __future__ import annotations

from doc2md.config import Settings
from doc2md.layout_engines.base import LayoutEngine


def get_layout_engine(settings: Settings) -> LayoutEngine:
    if settings.layout_engine == "mineru":
        from doc2md.layout_engines.mineru_engine import MinerULayoutEngine

        return MinerULayoutEngine(settings)
    if settings.layout_engine == "docling":
        from doc2md.layout_engines.docling_engine import DoclingLayoutEngine

        return DoclingLayoutEngine(settings)
    if settings.layout_engine == "doclayout_yolo":
        from doc2md.layout_engines.doclayout_yolo_engine import DocLayoutYOLOEngine

        return DocLayoutYOLOEngine(settings)
    if settings.layout_engine == "markitdown":
        from doc2md.layout_engines.markitdown_engine import MarkItDownEngine

        return MarkItDownEngine(settings)
    if settings.layout_engine == "pymupdf4llm":
        from doc2md.layout_engines.pymupdf4llm_engine import PyMuPDF4LLMEngine

        return PyMuPDF4LLMEngine(settings)
    if settings.layout_engine == "paddleocr":
        from doc2md.layout_engines.paddleocr_engine import PaddleOCREngine

        return PaddleOCREngine(settings)
    raise ValueError(
        f"Unknown layout engine {settings.layout_engine!r} (expected 'mineru', 'docling', 'doclayout_yolo', 'markitdown', 'pymupdf4llm', or 'paddleocr')"
    )


