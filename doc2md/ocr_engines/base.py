"""OCR engines: a fast, non-VLM text-recognition path for plain TEXT_LIKE
regions. Each engine wraps a layout engine's own bundled OCR text-recognition
model - not layout detection, that's doc2md.layout_engines.

Only used when Settings.ocr_fast_path is on; the caller (pipeline.py) falls
back to the VLM on any recognize() failure or empty result, so an engine
here doesn't need its own fallback logic of its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class OCREngine(ABC):
    """Recognizes the text in a single cropped region image, top-to-bottom."""

    @abstractmethod
    def recognize(self, image: Image.Image) -> str: ...
