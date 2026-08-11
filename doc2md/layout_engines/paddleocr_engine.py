"""Layout/segmentation via Baidu PaddleOCR (PP-StructureV3 / LayoutDetection).

PaddleOCR's LayoutDetection engine detects document layout regions (Title, Text, Table,
Figure, Header, Footer) using Baidu's vision models.
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image

from doc2md.classify import PP_DOCLAYOUT_LABEL_TO_BUCKET, bucket_for_label
from doc2md.config import Settings
from doc2md.layout_engines.base import LayoutEngine, full_page_fallback_region
from doc2md.layout_engines.dedup import suppress_contained_duplicates
from doc2md.layout_engines.reading_order import assign_reading_order
from doc2md.models import Bucket, PageImage, Region

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

PADDLE_LABEL_MAP: dict[str, str] = {
    "title": "doc_title",
    "text": "text",
    "table": "table",
    "figure": "image",
    "header": "header",
    "footer": "footer",
    "reference": "reference",
    "equation": "display_formula",
}


class PaddleOCREngine(LayoutEngine):
    """Layout detection engine powered by Baidu PaddleOCR."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from paddleocr import LayoutDetection

                self._engine = LayoutDetection()
            except Exception:
                try:
                    from paddleocr import PPStructureV3

                    self._engine = PPStructureV3()
                except Exception:
                    self._engine = False
        return self._engine if self._engine is not False else None

    def detect(self, page: PageImage) -> list[Region]:
        regions: list[Region] = []
        engine = self._get_engine()

        if engine is not None:
            img_np = np.array(page.image.convert("RGB"))[:, :, ::-1]
            try:
                if hasattr(engine, "predict"):
                    results = engine.predict(img_np)
                else:
                    results = engine(img_np)

                for item in results:
                    if isinstance(item, dict):
                        raw_type = item.get("type", item.get("label", "text")).lower()
                        bbox_list = item.get("bbox", [0, 0, page.width, page.height])
                    else:
                        raw_type = getattr(item, "type", getattr(item, "label", "text")).lower()
                        bbox_list = getattr(item, "bbox", [0, 0, page.width, page.height])

                    x0, y0, x1, y1 = bbox_list[:4]

                    mapped_label = PADDLE_LABEL_MAP.get(raw_type, "text")
                    bucket = bucket_for_label(
                        mapped_label, PP_DOCLAYOUT_LABEL_TO_BUCKET, self._settings.skip_labels
                    )
                    if bucket is not Bucket.SKIP:
                        regions.append(
                            Region(
                                page_no=page.page_no,
                                label=mapped_label,
                                bucket=bucket,
                                bbox=(int(x0), int(y0), int(x1), int(y1)),
                                score=0.9,
                                order_index=0,
                            )
                        )
            except Exception:
                pass

        if not regions:
            regions.extend(full_page_fallback_region(page, self._settings, PP_DOCLAYOUT_LABEL_TO_BUCKET))

        return assign_reading_order(suppress_contained_duplicates(regions))
