"""Layout/segmentation via IBM Docling's layout model (RT-DETR, DocLayNet classes).

docling-layout-heron is the heavyweight RT-DETR-based detector Docling itself
uses by default, trained on DocLayNet-style categories (Text, Title,
Section-header, List-item, Caption, Footnote, Formula, Code, Picture, Table,
Page-header/footer, etc.). We call docling_ibm_models directly rather than
depending on the full `docling` package, so weights are fetched straight from
the model's HF repo on first use.

It has no built-in reading-order head, so a heuristic sort
(doc2md.layout_engines.reading_order) is applied.
"""

from __future__ import annotations

from doc2md.classify import DOCLING_LABEL_TO_BUCKET
from doc2md.config import Settings
from doc2md.layout_engines.base import LayoutEngine, finalize_regions, make_region, safe_detect
from doc2md.models import PageImage, Region

DOCLING_LAYOUT_REPO_ID = "docling-project/docling-layout-heron"
DOCLING_LAYOUT_REVISION = "main"


class DoclingLayoutEngine(LayoutEngine):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor

        # Imported lazily: these pull in torch/transformers and trigger the
        # (first-run-only) model download from Hugging Face Hub.
        import torch
        from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
        from huggingface_hub import snapshot_download

        weights_dir = self._settings.docling_weights_dir
        if not (weights_dir / "model.safetensors").exists():
            weights_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=DOCLING_LAYOUT_REPO_ID,
                revision=DOCLING_LAYOUT_REVISION,
                local_dir=str(weights_dir),
            )

        device = self._settings.layout_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._predictor = LayoutPredictor(
            artifact_path=str(weights_dir),
            device=device,
            base_threshold=self._settings.layout_confidence,
        )
        return self._predictor

    def detect(self, page: PageImage) -> list[Region]:
        def _run() -> list[Region]:
            predictor = self._load()
            predictions = predictor.predict(page.image)

            regions: list[Region] = []
            for pred in predictions:
                region = make_region(
                    page.page_no,
                    pred["label"],
                    DOCLING_LABEL_TO_BUCKET,
                    self._settings.skip_labels,
                    (int(pred["l"]), int(pred["t"]), int(pred["r"]), int(pred["b"])),
                    float(pred["confidence"]),
                )
                if region is not None:
                    regions.append(region)
            return finalize_regions(regions, self._settings)

        return safe_detect("docling", page, self._settings, DOCLING_LABEL_TO_BUCKET, _run)
