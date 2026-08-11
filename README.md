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

- **Input**: PDF files, a single image, or a directory of page images
  (PNG/JPG/TIFF/BMP/WebP).
- **Layout-aware**: pluggable layout-detection engine, defaulting to MinerU's
  PP-DocLayoutV2 model (RT-DETR-based), which detects every region on a
  page — titles, body text, lists, footnotes, formulas, tables,
  figures/charts — and returns them already sorted in reading order, not just
  top-to-bottom. An alternative engine is also available and verified
  working end-to-end: IBM Docling's layout model (`--layout-engine docling`,
  DocLayNet classes, also RT-DETR-based, works out of the box).
- **Per-region content extraction** via a local VLM (Gemma, served through
  `llama-server`, GPU-accelerated):
  - Titles/headings → Markdown headings (`#`, `##`)
  - Body text, lists, footnotes, references → transcribed Markdown text
  - Formulas → LaTeX (`$...$` / `$$...$$`)
  - Tables → GitHub-flavored Markdown tables
  - Figures → the VLM decides per-image whether it's a **structured diagram**
    (flowchart, sequence diagram, state machine, org chart, simple graph) —
    if so, it's redrawn as a fenced ` ```mermaid ` block; otherwise the
    region is cropped and embedded as a normal image.
- **Runs fully offline** once the layout model and VLM weights are downloaded
  — no data leaves your machine.
- **Clean output**: a single `<name>.md` plus a `<name>_assets/` folder
  containing only the images that were actually embedded (nothing is written
  for diagrams that became Mermaid).

## How it works

```
PDF / images
    │  render.py (PyMuPDF)
    ▼
per-page images
    │  layout_engines/ (default: MinerU PP-DocLayoutV2)
    │  (also available: docling, markitdown, etc.)
    ▼
bounding boxes + labels, in reading order
    │  classify.py + crop.py
    ▼
cropped regions, bucketed as text / table / picture
    │  vlm_client.py + prompts.py  (→ llama-server → Gemma)
    ▼
per-region Markdown / Mermaid / image references
    │  markdown_builder.py
    ▼
<name>.md + <name>_assets/
```

## Layout engines

doc2md's layout detection is pluggable (`--layout-engine`):

- **`doclayout_yolo`** — OpenDataLab's DocLayout-YOLO model (`opendatalab/PDF-Extract-Kit-1.0`, YOLOv10-based, fine-tuned on DocSynth-300K). Delivers sub-10ms real-time bounding box detection per page with high recall for embedded text shapes and slide visual containers.
- **`mineru`** (default) — MinerU's PP-DocLayoutV2 (Shanghai AI Lab / OpenDataLab). Installs cleanly via `uv sync`; no extra setup needed.
- **`markitdown`** — Microsoft MarkItDown layout analyzer & document text extractor (`markitdown`). Performs page layout segmentation and table/text/figure detection across document formats (PDF, DOCX, PPTX, XLSX, HTML, images).
- **`pymupdf4llm`** — PyMuPDF4LLM high-precision vector layout engine. Performs font hierarchy analysis and exact vector bounding box segmentation directly from PDF objects with zero GPU overhead.
- **`paddleocr`** — Baidu PaddleOCR (PP-Structure / LayoutDetection) layout engine. Performs document layout parsing and multi-class region detection.
- **`docling`** — IBM Docling's layout model (`docling-layout-heron`, RT-DETRv2-based, trained on DocLayNet's 17-class label set).
  Section-header, List-item, Caption, Footnote, Formula, Code, Picture,
  Table, Page-header/footer, etc.). Called directly through
  `docling-ibm-models` rather than the full `docling` package, so it's a pure
  pip dependency — no native build toolchain needed. Weights (~660 MB)
  download automatically from `docling-project/docling-layout-heron` on
  Hugging Face the first time this engine runs. Unlike PP-DocLayoutV2 it has
  no built-in reading-order head, so the row-bucket heuristic in
  `reading_order.py` is applied instead. Verified working end-to-end against
  `sample_docs/test.pdf`.

## Requirements

- Windows with an NVIDIA GPU (developed against an RTX 5080, 16 GB VRAM;
  a smaller GPU can still work with a smaller quant, see below).
- [`uv`](https://docs.astral.sh/uv/) for Python environment management.
- `llama-server` (from [llama.cpp](https://github.com/ggml-org/llama.cpp))
  running with a vision-capable Gemma GGUF loaded.

## Setup

```
uv sync
```

This creates a `.venv` pinned to Python 3.12 and installs a CUDA-enabled
`torch`/`torchvision` build from PyTorch's `cu128` index — the default PyPI
wheels are CPU-only, which won't work here.

The layout model (PP-DocLayoutV2) downloads its weights automatically from
Hugging Face (falling back to ModelScope) the first time you run doc2md.

You also need a running `llama-server` with a vision-capable Gemma model
loaded — see [`scripts/setup_llama_server.md`](scripts/setup_llama_server.md)
for exactly how this was set up (binaries, model download, launch command).
In short:

```
llama-server.exe ^
  --model gemma-4-12B-it-Q4_0.gguf ^
  --mmproj mmproj-gemma-4-12B-it-Q8_0.gguf ^
  -ngl 999 --ctx-size 8192 --host 127.0.0.1 --port 8080
```

## Usage

```
uv run doc2md path/to/document.pdf -o out/
uv run doc2md path/to/page_images_dir/ -o out/
uv run doc2md path/to/single_page.png -o out/
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `-o`, `--output` | `out` | Directory to write the `.md` file and asset images into |
| `--server` | `http://127.0.0.1:8080` | Base URL of the running `llama-server` |
| `--dpi` | `200` | DPI used to rasterize PDF pages — higher improves legibility of small text/diagrams at the cost of slower rendering and larger crops |
| `--layout-engine` | `mineru` | Layout/segmentation engine: `mineru` (default, works out of the box), `docling`, `markitdown`, `pymupdf4llm`, or `paddleocr` |

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
  models.py            # Region, PageImage, RegionResult, DocumentResult
  config.py             # Settings (server url, dpi, padding, skip labels, ...)
  render.py             # PDF/image -> list[PageImage]
  layout_engines/        # pluggable layout detection
    base.py                # LayoutEngine ABC
    mineru_engine.py         # default: MinerU PP-DocLayoutV2
    docling_engine.py         # alternative: IBM Docling docling-layout-heron
    reading_order.py              # row-bucket heuristic for engines with no native ordering
    dedup.py                        # containment-based duplicate-box suppression (docling)
  classify.py            # label -> bucket (text/table/picture/skip)
  crop.py                 # region cropping + asset saving
  vlm_client.py            # llama-server HTTP client
  prompts.py               # prompt templates per bucket
  pipeline.py               # orchestration
  markdown_builder.py        # final Markdown assembly
  cli.py                      # `doc2md <input> [options]`
scripts/
  setup_llama_server.md        # how the local llama-server was set up
  make_sample_pdf.py            # generates a synthetic test PDF for smoke-testing
```

## Known limitations

- Reading order comes from PP-DocLayoutV2's built-in reading-order head; it's
  generally solid but not perfect on unusual layouts (e.g. dense multi-column
  academic papers with side notes).
- Each region is one VLM call with only its own cropped image as context, so
  very small or ambiguous crops can occasionally be mis-transcribed. Increasing
  `--dpi` improves crop legibility at the cost of slower rendering.
- Mermaid conversion is best-effort: complex or unusual diagrams may be
  simplified or, if the VLM can't confidently structure them, fall back to
  being embedded as a plain image instead.
