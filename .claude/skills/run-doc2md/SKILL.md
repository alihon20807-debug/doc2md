---
name: run-doc2md
description: Build, run, and smoke-test doc2md (the local PDF/image-to-Markdown pipeline in this repo). Use when asked to start doc2md, run its tests, build it, verify it's working, or convert a document end-to-end.
---

doc2md is a CLI tool (`uv run doc2md <input> -o <output>`), not a GUI or web
app — there's no window to screenshot. It's driven via
`.claude/skills/run-doc2md/smoke.sh`, a committed smoke-test script that
builds/imports the CLI, runs layout detection standalone, and (if a VLM
server is reachable) does a real end-to-end conversion and checks the
output actually has the structure doc2md is supposed to produce.

**Platform note**: unlike a typical headless-Linux-container target, this
project genuinely requires Windows + an NVIDIA GPU (see the repo's own
`README.md` "Requirements") and, for the default vLLM backend, WSL2. This
skill was authored and verified on that real target platform, not adapted
from a Linux assumption — there's no `apt-get` step because there's nothing
to work around.

## Prerequisites

- Windows with an NVIDIA GPU.
- [`uv`](https://docs.astral.sh/uv/) on `PATH`.
- For the default `--vlm-backend vllm`: a running vLLM server (see
  "VLM server" below). Without one, the smoke test still verifies the CLI
  builds and layout detection works, but skips the real conversion check.

## Setup

```bash
uv sync
```

Creates `.venv` (pinned Python 3.12) and installs a CUDA `torch`/`torchvision`
build. The layout model and any OCR weights download automatically on first
use (several hundred MB to a few GB depending on `--layout-engine`).

No separate build step — `uv sync` is it.

## VLM server

The default backend (`--vlm-backend vllm`) needs a live server. Check first:

```bash
curl -s -m 3 http://127.0.0.1:8000/v1/models
```

If unreachable, it runs inside WSL2 and needs bringing up:

```bash
MSYS_NO_PATHCONV=1 wsl -- bash -lc 'bash ~/launch_vllm.sh'
```

then wait ~2-3 minutes (weight load + compile + CUDA graph capture) and
re-poll the curl above. `MSYS_NO_PATHCONV=1` is required when driving WSL
from a Windows-hosted shell — Git Bash/MSYS silently mangles the leading
`/` in `~/launch_vllm.sh` otherwise. `configs/vllm_launch.sh` in this repo
is a git-tracked backup of that script's real contents (the live copy is
WSL-side, untracked, machine-specific). Full detail:
`handoffs/vllm_guide_handoff.md`.

## Run (agent path)

```bash
bash .claude/skills/run-doc2md/smoke.sh
```

Runs from the repo root regardless of cwd (resolves its own path). Four
checks, each independently pass/fail:

| Check | What it verifies | Needs a VLM server? |
|---|---|---|
| 1. `uv run doc2md --help` | CLI is installed/importable | No |
| 2. `curl .../v1/models` | VLM server reachability | No (reports, doesn't require) |
| 3. Layout detection | `mineru` engine finds real regions in `sample_docs/test.pdf` | No |
| 4. Real conversion | Runs `doc2md` on `sample_docs/test.pdf`, checks output has a heading, a table, and a Mermaid block | Yes — skipped if no server |

Verified this session: with the vLLM server up, all 4 checks pass,
including a real conversion whose output was manually inspected and
confirmed correct (title → `#` heading, table → real GFM table with
accurate cell values, flowchart → an actual ` ```mermaid ` block).

## Run (human path)

```bash
uv run doc2md path/to/document.pdf -o out/
```

Writes `out/document.md` (+ `out/document_assets/` if any figure was
embedded as an image rather than redrawn as Mermaid). See the repo's
`README.md` for the full flag reference.

## Test

```bash
uv run python -m unittest discover -s benchmark -v
```

17 tests, all pass. This covers `benchmark/` only (a separate, self-contained
throughput-measurement harness unrelated to the conversion pipeline) —
`doc2md/` itself has no automated test suite, which is exactly why this
skill's smoke script exists as the way to verify it's actually working.

---

## Gotchas

- **`doc2md` is not on `PATH`.** It's a `uv`-managed entry point, not a
  global install — always `uv run doc2md ...`, or `uv run --project
  <repo-path> doc2md ...` from outside the repo. There's no PowerShell
  profile alias like some of the user's other CLI tools have.
- **`--layout-engine paddleocr` has a real, pre-existing environment issue
  on this machine**: PaddlePaddle's oneDNN backend raises
  `NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
  not support ...` for both its layout-detection and OCR-fast-path calls.
  This isn't a doc2md bug — it's caught and logged
  (`doc2md: paddleocr failed on page N: ...`), and the pipeline falls back
  to a single full-page region rather than crashing. Don't mistake `mineru`
  detecting 4 real regions vs. `paddleocr` detecting 1 full-page region on
  the same test document for a doc2md defect.
- **`--layout-engine markitdown`/`pymupdf4llm` need the actual input file
  path resolvable, not just an image.** These two do real PDF-native
  extraction (pdfplumber / PyMuPDF4LLM) keyed off the original PDF path
  (`PageImage.source_path`, set by `render.py`); pass the real PDF path as
  input, not a pre-rendered image of it, or they'll silently fall back to
  a single full-page region and rely entirely on the VLM.

## Troubleshooting

- **`curl: (7) Failed to connect ... :8000`**: no vLLM server running. See
  "VLM server" above.
- **Smoke test check 4 is skipped**: same cause — check 2's curl failed.
  Checks 1 and 3 passing without check 4 just means the CLI and layout
  detection work; it says nothing about the VLM path.
- **A stale `*.doc2md_progress.json` file next to expected output**: a
  previous run was interrupted. `doc2md` auto-resumes from it by default
  (`--resume`, the default); pass `--no-resume` to force a clean restart.
