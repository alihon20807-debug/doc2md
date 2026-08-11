"""Layout/segmentation via OpenDataLab's DocLayout-YOLO model.

DocLayout-YOLO is a high-speed YOLOv10-based detector fine-tuned on DocSynth-300K / DocStructBench.
It detects titles, plain text, headers, footers, figures, tables, formulas, captions, etc.
at sub-10ms latency per page.
"""

from __future__ import annotations

import numpy as np

from doc2md.classify import DOCLAYOUT_YOLO_LABEL_TO_BUCKET
from doc2md.config import Settings
from doc2md.layout_engines.base import LayoutEngine, finalize_regions, make_region, safe_detect
from doc2md.models import PageImage, Region


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
        def _run() -> list[Region]:
            model = self._load()
            img_np = np.array(page.image)

            results = model.predict(img_np, imgsz=1024, conf=self._settings.layout_confidence, verbose=False)

            regions: list[Region] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id] if 0 <= cls_id < len(model.names) else str(cls_id)
                    xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                    region = make_region(
                        page.page_no,
                        label,
                        DOCLAYOUT_YOLO_LABEL_TO_BUCKET,
                        self._settings.skip_labels,
                        (int(xmin), int(ymin), int(xmax), int(ymax)),
                        float(box.conf[0]),
                    )
                    if region is not None:
                        regions.append(region)

            return finalize_regions(regions, self._settings)

        return safe_detect("doclayout_yolo", page, self._settings, DOCLAYOUT_YOLO_LABEL_TO_BUCKET, _run)
