"""OCR text recognition via MinerU's bundled PytorchPaddleOCR (a PyTorch
reimplementation of PP-OCR det+rec, already part of the installed
mineru[core] package - no new dependency, only its own weights need
downloading on first use).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from doc2md.ocr_engines.base import OCREngine


class MineruOCREngine(OCREngine):
    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR

            self._ocr = PytorchPaddleOCR()
        return self._ocr

    def recognize(self, image: Image.Image) -> str:
        ocr = self._get_ocr()
        img_np = np.array(image.convert("RGB"))[:, :, ::-1]  # PP-OCR checkpoints expect BGR
        results = ocr.ocr(img_np)

        page_result = results[0] if results else None
        if not page_result:
            return ""
        lines = [text for _box, (text, _score) in page_result if text]
        return "\n".join(lines)
