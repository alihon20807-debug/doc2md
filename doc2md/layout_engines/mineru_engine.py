"""Layout/segmentation via MinerU's PP-DocLayoutV2 model.

PP-DocLayoutV2 is a heavyweight RT-DETR-based detector. It ships with a
built-in reading-order head, so the boxes it returns are already ranked in
reading order - no separate reading-order heuristic is needed.
"""

from __future__ import annotations

import os

from doc2md.classify import PP_DOCLAYOUT_LABEL_TO_BUCKET
from doc2md.config import Settings
from doc2md.layout_engines.base import LayoutEngine, make_region, safe_detect
from doc2md.models import PageImage, Region


class MinerULayoutEngine(LayoutEngine):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        # Imported lazily: these pull in torch/transformers and trigger the
        # (potentially multi-GB, first-run-only) model download.
        from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
        from mineru.utils.config_reader import get_device
        from mineru.utils.enum_class import ModelPath
        from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

        device = self._settings.layout_device or get_device()
        model_root = os.path.join(
            auto_download_and_get_model_root_path(ModelPath.pp_doclayout_v2),
            ModelPath.pp_doclayout_v2,
        )
        self._model = PPDocLayoutV2LayoutModel(
            weight=model_root,
            device=device,
            conf=self._settings.layout_confidence,
        )
        return self._model

    def detect(self, page: PageImage) -> list[Region]:
        def _run() -> list[Region]:
            model = self._load()
            predictions = model.predict(page.image)

            regions: list[Region] = []
            for pred in predictions:
                xmin, ymin, xmax, ymax = pred["bbox"]
                region = make_region(
                    page.page_no,
                    pred["label"],
                    PP_DOCLAYOUT_LABEL_TO_BUCKET,
                    self._settings.skip_labels,
                    (int(xmin), int(ymin), int(xmax), int(ymax)),
                    float(pred["score"]),
                    order_index=int(pred["index"]),
                )
                if region is not None:
                    regions.append(region)
            # predictions are already reading-order-sorted by the model, but the
            # skip filter above can only remove entries, not reorder them.
            regions.sort(key=lambda r: r.order_index)
            return regions

        return safe_detect("mineru", page, self._settings, PP_DOCLAYOUT_LABEL_TO_BUCKET, _run)
