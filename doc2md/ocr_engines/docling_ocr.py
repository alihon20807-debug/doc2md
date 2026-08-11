"""OCR text recognition for the Docling layout engine via RapidOCR (ONNX,
no torch requirement, small first-use model download).

Docling's own layout-only integration (`docling-ibm-models`) - the one
doc2md actually uses, see doc2md.layout_engines.docling_engine - has no
bundled OCR; only the full `docling` package doc2md deliberately avoids
does. `rapidocr-onnxruntime` is the lightweight substitute.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from doc2md.ocr_engines.base import OCREngine


class DoclingOCREngine(OCREngine):
    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        return self._ocr

    def recognize(self, image: Image.Image) -> str:
        ocr = self._get_ocr()
        img_np = np.array(image.convert("RGB"))
        result, _elapse = ocr(img_np)
        if not result:
            return ""
        lines = [entry[1] for entry in result if len(entry) > 1 and entry[1]]
        return "\n".join(lines)
