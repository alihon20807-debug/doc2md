from __future__ import annotations

import enum
from pathlib import Path

import typer

from doc2md.config import Settings
from doc2md.pipeline import convert_document

app = typer.Typer(add_completion=False, no_args_is_help=True)


class LayoutEngineName(str, enum.Enum):
    mineru = "mineru"
    docling = "docling"
    doclayout_yolo = "doclayout_yolo"
    markitdown = "markitdown"
    pymupdf4llm = "pymupdf4llm"
    paddleocr = "paddleocr"


class VLMBackendName(str, enum.Enum):
    vllm = "vllm"
    llama_server = "llama_server"
    openrouter = "openrouter"


@app.command()
def convert(
    input_path: Path = typer.Argument(
        ..., exists=True, help="A PDF file, a PPT/PPTX presentation, an image file, or a directory of page images"
    ),
    output_dir: Path = typer.Option(
        Path("out"), "--output", "-o", help="Directory to write the .md file and asset images into"
    ),
    vlm_backend: VLMBackendName = typer.Option(
        VLMBackendName.vllm,
        "--vlm-backend",
        help=(
            "'vllm' (default): an async local vLLM instance. 'llama_server': legacy llama-server. "
            "'openrouter': OpenRouter hosted API."
        ),
    ),
    vllm_url: str = typer.Option(
        "http://127.0.0.1:8000",
        "--vllm-url",
        "--server",
        help="Base URL of the running vLLM OpenAI-compatible server.",
    ),
    vllm_model: str = typer.Option(
        "cyankiwi/gemma-4-12B-it-qat-AWQ-INT4",
        "--vllm-model",
        help="Model name served by vLLM instance (e.g. cyankiwi/gemma-4-12B-it-qat-AWQ-INT4).",
    ),
    llama_server_url: str = typer.Option(
        "http://127.0.0.1:8080",
        "--llama-server-url",
        help="Base URL of the running llama-server instance, used only with --vlm-backend llama_server.",
    ),
    openrouter_api_key: str | None = typer.Option(
        None,
        "--openrouter-api-key",
        help="OpenRouter API key. Falls back to the OPENROUTER_API_KEY env var. Required with --vlm-backend openrouter",
    ),
    openrouter_text_model: str = typer.Option(
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "--openrouter-text-model",
        help="OpenRouter model for plain text/title/list/footnote regions",
    ),
    openrouter_diagram_model: str = typer.Option(
        "google/gemma-4-31b-it:free",
        "--openrouter-diagram-model",
        help="OpenRouter model for tables and pictures",
    ),
    concurrency: int = typer.Option(
        32, "--concurrency", help="Max parallel in-flight VLM requests to maximize vLLM continuous batching"
    ),
    rate_limit: float = typer.Option(
        15.0,
        "--rate-limit",
        help="Max requests per minute per model when using --vlm-backend openrouter (0 = unlimited).",
    ),
    dpi: int = typer.Option(200, "--dpi", help="DPI used to rasterize PDF pages"),
    layout_engine: LayoutEngineName = typer.Option(
        LayoutEngineName.mineru,
        "--layout-engine",
        help=(
            "Layout/segmentation engine: 'mineru' (PP-DocLayoutV2), 'docling' (DocLayNet), "
            "'markitdown' (Microsoft MarkItDown), 'pymupdf4llm' (PyMuPDF4LLM), 'paddleocr' (Baidu PaddleOCR)"
        ),
    ),

    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help=(
            "Pick up from a previous interrupted run's progress checkpoint in --output "
            "(same input file, layout engine, dpi, and page count). --no-resume ignores "
            "any existing checkpoint and starts over."
        ),
    ),
    ocr_fast_path: bool = typer.Option(
        False,
        "--ocr-fast-path/--no-ocr-fast-path",
        help=(
            "Use the active --layout-engine's own bundled OCR model for plain text regions "
            "instead of a VLM call (faster, less accurate). Only available for "
            "'mineru'/'paddleocr'/'docling'; other engines fall back to the VLM regardless. "
            "Titles and formulas always use the VLM."
        ),
    ),
) -> None:
    """Convert a document (PDF, PPT, or images) into a Markdown file via local layout + async vLLM."""
    settings = Settings(
        vlm_backend=vlm_backend.value,
        vllm_url=vllm_url,
        vllm_model=vllm_model,
        llama_server_url=llama_server_url,
        openrouter_api_key=openrouter_api_key,
        openrouter_text_model=openrouter_text_model,
        openrouter_diagram_model=openrouter_diagram_model,
        vlm_max_concurrency=concurrency,
        vlm_requests_per_minute=rate_limit,
        render_dpi=dpi,
        layout_engine=layout_engine.value,
        resume=resume,
        ocr_fast_path=ocr_fast_path,
    )
    result = convert_document(input_path, output_dir, settings)
    md_path = output_dir / f"{Path(result.source_path).stem}.md"
    typer.echo(f"Wrote {md_path}")
    if result.asset_dir:
        typer.echo(f"Assets: {result.asset_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

