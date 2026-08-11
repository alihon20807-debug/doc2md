# doc2md

Convert documents (PDFs or page images) into clean Markdown using a fully
local pipeline: a heavyweight layout-detection model finds every region on
each page, and a local vision-language model (VLM) reads each region and
writes its Markdown. No cloud APIs, no OCR-only guesswork — text is
transcribed, tables become real Markdown tables, and diagrams are redrawn as
Mermaid instead of being dumped in as flat images.

## Purpose

Scanned reports, papers, and slide decks are easy to read but hard to reuse.
doc2md turns a PDF (or a folder of page images) into a single `.md` file that
preserves the document's structure — headings, body text, tables, formulas,
and diagrams — so it can be searched, diffed, fed into other tools, or edited
like any other text file.

## Capabilities

- **Input**: PDF files, PPT/PPTX presentations, a single image, or a
  directory of page images (PNG/JPG/TIFF/BMP/WebP).
- **Layout-aware**: pluggable layout-detection engine, defaulting to MinerU's
  PP-DocLayoutV2 model (RT-DETR-based), which detects every region on a
  page — titles, body text, lists, footnotes, formulas, tables,
  figures/charts — and returns them already sorted in reading order, not just
  top-to-bottom. Five alternative engines are also available (see
  "Layout engines" below).
- **Per-region content extraction** via a local vision-language model —
  **vLLM is the default backend** (an async local vLLM server, see "Setup"
  below); `llama-server` is available as a legacy alternative, and
  OpenRouter's hosted API as a third, cloud fallback:
  - Titles/headings → Markdown headings (`#`, `##`)
  - Body text, lists, footnotes, references → transcribed Markdown text
  - Formulas → LaTeX (`$...$` / `$$...$$`)
  - Tables → GitHub-flavored Markdown tables
  - Figures → the VLM decides per-image whether it's a **structured diagram**
    (flowchart, sequence diagram, state machine, org chart, simple graph) —
    if so, it's redrawn as a fenced ` ```mermaid ` block; otherwise the
    region is cropped and embedded as a normal image.
- **Optional OCR fast path** (`--ocr-fast-path`, off by default): for plain
  text regions, use the active layout engine's own bundled OCR model instead
  of a VLM call — faster, somewhat less accurate. Only available when
  `--layout-engine` is `mineru`, `paddleocr`, or `docling` (the only three
  that bundle an OCR model); other engines fall back to the VLM regardless.
  Titles and formulas always go through the VLM even with this on, since OCR
  can't produce a Markdown heading or LaTeX.
- **Runs fully offline** once the layout model, VLM weights, and (optionally)
  OCR weights are downloaded — no data leaves your machine.
- **Clean output**: a single `<name>.md` plus a `<name>_assets/` folder
  containing only the images that were actually embedded (nothing is written
  for diagrams that became Mermaid).

## How it works

```
PDF / PPT / images
    │  render.py (PyMuPDF / LibreOffice / python-pptx)
    ▼
per-page images
    │  layout_engines/ (default: MinerU PP-DocLayoutV2)
    │  (also available: docling, doclayout_yolo, markitdown, pymupdf4llm, paddleocr)
    ▼
bounding boxes + labels, in reading order
    │  classify.py + crop.py
    ▼
cropped regions, bucketed as text / table / picture
    │  vlm_client.py + prompts.py  (→ vLLM (default) / llama-server / OpenRouter)
    │  ocr_engines/ (optional fast path for plain text, see above)
    ▼
per-region Markdown / Mermaid / image references
    │  markdown_builder.py
    ▼
<name>.md + <name>_assets/
```

## Layout engines

doc2md's layout detection is pluggable (`--layout-engine`):

- **`mineru`** (default) — MinerU's PP-DocLayoutV2 (Shanghai AI Lab / OpenDataLab). Installs cleanly via `uv sync`; no extra setup needed. Bundles its own OCR model (usable via `--ocr-fast-path`).
- **`docling`** — IBM Docling's layout model (`docling-layout-heron`, RT-DETRv2-based), trained on DocLayNet's 17-class label set (Section-header, List-item, Caption, Footnote, Formula, Code, Picture, Table, Page-header/footer, etc.). Called directly through `docling-ibm-models` rather than the full `docling` package, so it's a pure pip dependency — no native build toolchain needed. Weights (~660 MB) download automatically from `docling-project/docling-layout-heron` on Hugging Face the first time this engine runs. Unlike PP-DocLayoutV2 it has no built-in reading-order head, so the row-bucket heuristic in `reading_order.py` is applied instead. OCR fast path uses `rapidocr-onnxruntime` (a separate, lightweight dependency).
- **`doclayout_yolo`** — OpenDataLab's DocLayout-YOLO model (`opendatalab/PDF-Extract-Kit-1.0`, YOLOv10-based, fine-tuned on DocSynth-300K). Sub-10ms real-time bounding box detection per page. No bundled OCR — `--ocr-fast-path` has no effect with this engine.
- **`markitdown`** — PDF-native extraction via pdfplumber (the same library MarkItDown itself uses internally for PDFs): real text/table/figure positions read directly from the PDF's text layer, not OCR. Falls back to a single full-page region for inputs with no PDF backing. No bundled OCR.
- **`pymupdf4llm`** — PyMuPDF4LLM's high-precision vector layout engine. Font hierarchy analysis and exact vector bounding box segmentation directly from PDF objects, zero GPU overhead. No bundled OCR.
- **`paddleocr`** — Baidu PaddleOCR (PP-Structure / LayoutDetection) layout engine. Bundles its own OCR model (usable via `--ocr-fast-path`).

## Requirements

- Windows with an NVIDIA GPU (developed against an RTX 5080, 16 GB VRAM;
  a smaller GPU can still work with a smaller quant, see below).
- [`uv`](https://docs.astral.sh/uv/) for Python environment management.
- A running local inference server for the VLM — see "Setup" below.

## Setup

```
uv sync
```

This creates a `.venv` pinned to Python 3.12 and installs a CUDA-enabled
`torch`/`torchvision` build from PyTorch's `cu128` index — the default PyPI
wheels are CPU-only, which won't work here.

The layout model (PP-DocLayoutV2, or whichever `--layout-engine` you choose)
downloads its weights automatically from Hugging Face the first time you run
doc2md; so does any OCR weights needed by `--ocr-fast-path`.

### VLM server (default: vLLM)

`--vlm-backend vllm` is the default and expects an OpenAI-compatible vLLM
server at `http://127.0.0.1:8000` (override with `--server`/`--vllm-url`).
Getting vLLM running for this model stack (Gemma 4, AWQ quantization, WSL2)
needed several non-obvious fixes — see `CLAUDE.md`'s "Running vLLM under
WSL2" section for the short version, and `handoffs/vllm_guide_handoff.md`
for the full reference (exact launch script, every bug hit and its fix,
throughput data). The verified launch script is checked in at
`configs/vllm_launch.sh`.

### Alternative: llama-server (legacy)

`--vlm-backend llama_server` talks to a `llama-server` instance instead
(default `http://127.0.0.1:8080`, override with `--llama-server-url`) — see
[`scripts/setup_llama_server.md`](scripts/setup_llama_server.md) for how
that was set up (binaries, model download, launch command):

```
llama-server.exe ^
  --model gemma-4-12B-it-Q4_0.gguf ^
  --mmproj mmproj-gemma-4-12B-it-Q8_0.gguf ^
  -ngl 999 --ctx-size 8192 --host 127.0.0.1 --port 8080
```

### Alternative: OpenRouter (hosted)

`--vlm-backend openrouter` uses OpenRouter's hosted API instead of a local
server — requires `--openrouter-api-key` or the `OPENROUTER_API_KEY` env var.

## Usage

```
uv run doc2md path/to/document.pdf -o out/
uv run doc2md path/to/presentation.pptx -o out/
uv run doc2md path/to/page_images_dir/ -o out/
uv run doc2md path/to/single_page.png -o out/

# legacy llama-server backend instead of the default vLLM
uv run doc2md path/to/document.pdf -o out/ --vlm-backend llama_server

# a different layout engine, with the OCR fast path enabled
uv run doc2md path/to/document.pdf -o out/ --layout-engine docling --ocr-fast-path
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `-o`, `--output` | `out` | Directory to write the `.md` file and asset images into |
| `--vlm-backend` | `vllm` | VLM backend: `vllm` (default), `llama_server` (legacy), or `openrouter` (hosted) |
| `--vllm-url`, `--server` | `http://127.0.0.1:8000` | Base URL of the running vLLM OpenAI-compatible server |
| `--vllm-model` | `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` | Model name served by the vLLM instance |
| `--llama-server-url` | `http://127.0.0.1:8080` | Base URL of the running llama-server instance (only used with `--vlm-backend llama_server`) |
| `--openrouter-api-key` | none (falls back to `OPENROUTER_API_KEY` env var) | OpenRouter API key, required with `--vlm-backend openrouter` |
| `--openrouter-text-model` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | OpenRouter model for plain text/title/list/footnote regions |
| `--openrouter-diagram-model` | `google/gemma-4-31b-it:free` | OpenRouter model for tables and pictures |
| `--concurrency` | `32` | Max parallel in-flight VLM requests, to exploit vLLM's continuous batching |
| `--rate-limit` | `15.0` | Max requests per minute per model when using `--vlm-backend openrouter` (`0` = unlimited) |
| `--dpi` | `200` | DPI used to rasterize PDF pages — higher improves legibility of small text/diagrams at the cost of slower rendering and larger crops |
| `--layout-engine` | `mineru` | Layout/segmentation engine: `mineru`, `docling`, `doclayout_yolo`, `markitdown`, `pymupdf4llm`, or `paddleocr` |
| `--resume` / `--no-resume` | `--resume` | Pick up from a previous interrupted run's progress checkpoint in `--output` (same input file, layout engine, dpi, and page count) |
| `--ocr-fast-path` / `--no-ocr-fast-path` | `--no-ocr-fast-path` | Use the active layout engine's own bundled OCR model for plain text regions instead of a VLM call (only `mineru`/`paddleocr`/`docling`; titles/formulas always use the VLM) |

Run `uv run doc2md --help` for the full option list.

### Output

```
out/
  document.md
  document_assets/
    p003_004_image.png   # only created for figures the VLM decided are NOT diagrams
```

## Project layout

```
doc2md/
  models.py               # Region, PageImage, RegionResult, DocumentResult
  config.py                # Settings (VLM backend, dpi, padding, skip labels, ...)
  render.py                 # PDF/PPT/image -> list[PageImage]
  layout_engines/             # pluggable layout detection
    base.py                     # LayoutEngine ABC + shared helpers (make_region, finalize_regions, safe_detect, ...)
    mineru_engine.py              # default: MinerU PP-DocLayoutV2
    docling_engine.py               # IBM Docling docling-layout-heron
    doclayout_yolo_engine.py          # OpenDataLab DocLayout-YOLO
    markitdown_engine.py                # pdfplumber-based PDF extraction
    pymupdf4llm_engine.py                 # PyMuPDF4LLM vector extraction
    paddleocr_engine.py                     # Baidu PaddleOCR layout detection
    reading_order.py                          # row-bucket heuristic for engines with no native ordering
    dedup.py                                    # containment-based duplicate-box suppression
  ocr_engines/               # pluggable OCR fast path (--ocr-fast-path)
    base.py                     # OCREngine ABC
    mineru_ocr.py, paddleocr_ocr.py, docling_ocr.py
  classify.py               # label -> bucket (text/table/picture/skip), per layout engine
  crop.py                    # region cropping + asset saving
  vlm_client.py               # vLLM / llama-server / OpenRouter HTTP clients
  prompts.py                    # prompt templates per bucket, title/formula label sets
  pipeline.py                     # orchestration (async, checkpoint/resume)
  markdown_builder.py                # final Markdown assembly
  cli.py                               # `doc2md <input> [options]`
configs/
  vllm_launch.sh                # verified-working vLLM launch script (see CLAUDE.md)
  llamacpp_launch.ps1             # tuned llama-server launch script
scripts/
  setup_llama_server.md        # how the local llama-server (legacy backend) was set up
  make_sample_pdf.py            # generates a synthetic test PDF for smoke-testing
  fast_download.py               # bandwidth-conscious Hugging Face downloader
benchmark/
  ...                          # separate, self-contained llama.cpp/vLLM throughput harness - see benchmark/PROJECT.md
reports/, handoffs/, docs/    # benchmark output, session handoff docs, and supporting assets
```

## Known limitations

- Reading order comes from each engine's own head (MinerU has a native one)
  or the row-bucket heuristic (`reading_order.py`) for the others; generally
  solid but not perfect on unusual layouts (e.g. dense multi-column academic
  papers with side notes).
- Each region is one VLM call with only its own cropped image as context, so
  very small or ambiguous crops can occasionally be mis-transcribed. Increasing
  `--dpi` improves crop legibility at the cost of slower rendering.
- Mermaid conversion is best-effort: complex or unusual diagrams may be
  simplified or, if the VLM can't confidently structure them, fall back to
  being embedded as a plain image instead.
- `--ocr-fast-path` trades accuracy for speed and is only available for 3 of
  6 layout engines (see above); it has never been formally quality-compared
  against the VLM path on real documents.
