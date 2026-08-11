"""OCR text recognition via Baidu PaddleOCR's own OCR pipeline (PP-OCR det+rec).

Distinct from doc2md.layout_engines.paddleocr_engine, which only uses
PaddleOCR's *layout*-detection model - this uses its dedicated `PaddleOCR`
text-recognition pipeline class. `paddleocr`/`paddlepaddle` are already
installed dependencies; only the OCR det/rec weights need downloading on
first use (small, PP-OCR mobile-class models).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from doc2md.ocr_engines.base import OCREngine


class PaddleOCRTextEngine(OCREngine):
    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._ocr

    def recognize(self, image: Image.Image) -> str:
        ocr = self._get_ocr()
        img_np = np.array(image.convert("RGB"))[:, :, ::-1]  # PP-OCR checkpoints expect BGR
        results = ocr.predict(img_np)

        lines: list[str] = []
        for result in results:
            texts = result.get("rec_texts") if hasattr(result, "get") else None
            if texts:
                lines.extend(t for t in texts if t)
        return "\n".join(lines)
