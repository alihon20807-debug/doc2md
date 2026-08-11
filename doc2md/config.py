"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# PP-DocLayoutV2 labels we drop entirely rather than sending to the VLM -
# running headers/footers/page numbers add noise to a clean markdown document.
DEFAULT_SKIP_LABELS: frozenset[str] = frozenset(
    {
        "header",
        "header_image",
        "footer",
        "footer_image",
        "number",
    }
)


@dataclass
class Settings:
    # --- VLM backend ---
    vlm_backend: str = "vllm"  # "vllm" (local vLLM), "llama_server" (legacy local), or "openrouter" (hosted)
    vlm_temperature: float = 0.1
    vlm_max_tokens: int = 2048
    vlm_timeout_s: float = 180.0
    vlm_max_retries: int = 5
    vlm_max_concurrency: int = 32  # parallel in-flight VLM requests to maximize vLLM continuous batching
    vlm_requests_per_minute: float = 15  # per-model cap when using openrouter; 0 = unlimited

    # --- vLLM server ---
    vllm_url: str = "http://127.0.0.1:8000"
    vllm_model: str = "google/gemma-3-4b-it"

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
    layout_engine: str = "mineru"  # "mineru", "docling", "doclayout_yolo", or "markitdown"
    layout_device: str | None = None  # None -> auto-detect (cuda if available)
    layout_confidence: float = 0.45
    skip_labels: frozenset[str] = field(default_factory=lambda: DEFAULT_SKIP_LABELS)
    docling_weights_dir: Path = field(default_factory=lambda: _REPO_ROOT / "models" / "docling-layout-heron")

    # --- output ---
    assets_dirname_suffix: str = "_assets"
    resume: bool = True  # pick up from a prior run's progress checkpoint, if one matches

