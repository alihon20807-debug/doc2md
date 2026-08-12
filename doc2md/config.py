"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# PP-DocLayoutV2 labels we drop entirely rather than sending to the VLM -
# running header/footer *text* and page numbers add noise to a clean markdown
# document. header_image/footer_image are deliberately NOT here: they map to
# Bucket.PICTURE in classify.PP_DOCLAYOUT_LABEL_TO_BUCKET (a logo or figure
# living in a header/footer band is still worth extracting) - skipping them
# here would make that mapping entry unreachable.
#
# This vocabulary is MinerU/PP-DocLayoutV2's native label spelling. It's
# passed as-is to every layout engine's bucket_for_label() call, but only
# meaningfully applies to engines routed through PP_DOCLAYOUT_LABEL_TO_BUCKET
# (mineru, paddleocr, pymupdf4llm) - docling/markitdown/doclayout_yolo use
# their own differently-spelled label vocabularies and already bake their own
# header/footer skip decisions into their own per-engine *_LABEL_TO_BUCKET
# dicts in classify.py, so this setting is a no-op for those three by design.
DEFAULT_SKIP_LABELS: frozenset[str] = frozenset(
    {
        "header",
        "footer",
        "number",
    }
)


@dataclass
class Settings:
    # --- VLM backend ---
    vlm_backend: str = "vllm"  # "vllm" (local vLLM), "llama_server" (legacy local), or "openrouter" (hosted)
    vlm_temperature: float = 0.1
    vlm_repetition_penalty: float = 1.15  # vLLM-only; guards against decoding-loop repetition on ambiguous crops
    vlm_max_tokens: int = 2048  # ceiling for TABLE/PICTURE regions - tables and Mermaid diagrams can legitimately run long
    vlm_max_tokens_text_like: int = 768  # real TEXT_LIKE regions (titles, paragraphs, formulas) rarely need more than ~150-200 tokens; a tighter cap makes a decoding loop hit the ceiling (and get caught by the degenerate-retry check) sooner instead of running most of the way to vlm_max_tokens first
    vlm_timeout_s: float = 180.0
    vlm_max_retries: int = 5
    vlm_max_concurrency: int = 32  # parallel in-flight VLM requests to maximize vLLM continuous batching
    vlm_requests_per_minute: float = 15  # per-model cap when using openrouter; 0 = unlimited

    # --- vLLM server ---
    vllm_url: str = "http://127.0.0.1:8000"
    vllm_model: str = "cyankiwi/gemma-4-12B-it-qat-AWQ-INT4"

    # --- llama.cpp server (local Gemma vision model) ---
    llama_server_url: str = "http://127.0.0.1:8080"

    # --- OpenRouter (hosted, free-tier models) ---
    openrouter_api_key: str | None = None  # falls back to the OPENROUTER_API_KEY env var
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_text_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    openrouter_diagram_model: str = "google/gemma-4-31b-it:free"

    # --- page rendering ---
    render_dpi: int = 200

    # --- region cropping ---
    crop_padding_px: int = 6

    # --- layout detection ---
    layout_engine: str = "mineru"  # "mineru", "docling", "doclayout_yolo", "markitdown", "pymupdf4llm", or "paddleocr"
    layout_device: str | None = None  # None -> auto-detect (cuda if available)
    layout_confidence: float = 0.45
    skip_labels: frozenset[str] = field(default_factory=lambda: DEFAULT_SKIP_LABELS)
    docling_weights_dir: Path = field(default_factory=lambda: _REPO_ROOT / "models" / "docling-layout-heron")
    region_containment_threshold: float = 0.8  # dedup: intersection-over-smaller-area to suppress a duplicate box
    reading_order_overlap_frac: float = 0.5  # row-bucket heuristic: vertical overlap fraction to join the same row
    # --- line-region merging ---
    # Detectors (verified on mineru) often emit one region per visual line
    # rather than per paragraph; each region pays a full separate VLM
    # round-trip regardless of how little text it holds. Merging
    # reading-order-consecutive same-label TEXT_LIKE lines that are stacked
    # closely together loses no content (unlike a size/confidence filter,
    # rejected - see docs/OPTIMIZATION_PLAN.md) - see layout_engines/merge.py.
    region_merge_enabled: bool = False  # opt-in for now - see docs/OPTIMIZATION_PLAN.md "line-region merging" section for why
    region_merge_max_gap_frac: float = 0.75  # vertical gap tolerance, as a fraction of the preceding line's own height
    region_merge_x_align_px: int = 20  # horizontal start-alignment tolerance, in pixels at render_dpi

    # --- OCR fast path (opt-in) ---
    # Uses the active layout engine's own bundled OCR text-recognition model
    # (mineru/paddleocr/docling only - see doc2md.ocr_engines) instead of a
    # VLM call for plain TEXT_LIKE regions. Titles and formulas always go to
    # the VLM regardless of this setting, since OCR can't produce Markdown
    # headings or LaTeX. Off by default: the VLM path is the well-verified
    # one; this trades some accuracy for speed once a user opts in.
    ocr_fast_path: bool = False

    # --- output ---
    assets_dirname_suffix: str = "_assets"
    resume: bool = True  # pick up from a prior run's progress checkpoint, if one matches

