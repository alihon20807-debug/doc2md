"""Layout/segmentation via OpenDataLab's DocLayout-YOLO model.

DocLayout-YOLO is a high-speed YOLOv10-based detector fine-tuned on DocSynth-300K / DocStructBench.
It detects titles, plain text, headers, footers, figures, tables, formulas, captions, etc.
at sub-10ms latency per page.
"""

from __future__ import annotations

import numpy as np

from doc2md.classify import DOCLAYOUT_YOLO_LABEL_TO_BUCKET, bucket_for_label
from doc2md.config import Settings
from doc2md.layout_engines.base import LayoutEngine
from doc2md.layout_engines.dedup import suppress_contained_duplicates
from doc2md.layout_engines.reading_order import assign_reading_order
from doc2md.models import Bucket, PageImage, Region


class DocLayoutYOLOEngine(LayoutEngine):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="opendatalab/PDF-Extract-Kit-1.0",
            filename="models/Layout/YOLO/doclayout_yolo_ft.pt",
        )
        self._model = YOLOv10(model_path)
        return self._model

    def detect(self, page: PageImage) -> list[Region]:
        model = self._load()
        img_np = np.array(page.image)

        results = model.predict(img_np, imgsz=1024, conf=self._settings.layout_confidence, verbose=False)

        regions: list[Region] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                label = model.names[cls_id] if 0 <= cls_id < len(model.names) else str(cls_id)
                bucket = bucket_for_label(label, DOCLAYOUT_YOLO_LABEL_TO_BUCKET, self._settings.skip_labels)
                if bucket is Bucket.SKIP:
                    continue
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                regions.append(
                    Region(
                        page_no=page.page_no,
                        label=label,
                        bucket=bucket,
                        bbox=(int(xmin), int(ymin), int(xmax), int(ymax)),
                        score=score,
                        order_index=0,
                    )
                )

        return assign_reading_order(suppress_contained_duplicates(regions))
