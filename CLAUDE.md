# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two things live in the same repo:

1. **`doc2md`** (the actual product) — a local pipeline that converts PDFs / page
   images into clean Markdown: a layout-detection model finds regions on each
   page, then a local vision-language model (VLM) transcribes each region
   (text, tables → GFM tables, formulas → LaTeX, diagrams → Mermaid).
2. **`benchmark/`** — a separate, self-contained harness (see `PROJECT.md`) for
   benchmarking llama.cpp vs vLLM decode throughput/bandwidth on the dev
   machine's RTX 5080. It doesn't depend on `doc2md/` and vice versa.

## Commands

```bash
uv sync                                  # install deps incl. CUDA torch (cu128 index)
uv run doc2md path/to/file.pdf -o out/   # run the converter
uv run doc2md --help                     # full CLI option list
uv run python -m unittest discover -s benchmark -v   # run benchmark unit tests
```

There is no linter/formatter config and no test suite for `doc2md/` itself
(the only tests in the repo are `benchmark/test_bandwidth_calculator.py`,
`unittest`-based, not pytest — pytest isn't a dependency). `test/` and
`sample_docs/` hold sample input documents for manual conversion runs, not
automated tests — `reports/` holds benchmark output only, not test reports.
Two files previously in `reports/` (`benchmark_results.json`,
`benchmark_results_vllm.json`) were deleted — they claimed >100% of
theoretical GPU memory bandwidth, which is physically impossible; they were
either copy-pasted example/placeholder values or came from an unreviewed,
stalled milestone in `.agents/`, not a real benchmark run. Don't recreate
files like that from `PROJECT.md`'s illustrative JSON schema — it's for
shape only, not real numbers.

`doc2md` requires a running local inference server before conversion works:
- **vLLM** (default backend) at `http://127.0.0.1:8000`, OpenAI-compatible
  `/v1/chat/completions`, serving a Gemma vision model, run inside WSL2 (see
  "Running vLLM under WSL2" below for the current, verified-working setup —
  `configs/vllm_launch.ps1`/`.sh` are stale and reference `VLLM_USE_V1`,
  an env var that no longer exists in current vLLM).
- **llama-server** (`--vlm-backend llama_server`) as a legacy alternative —
  see `scripts/setup_llama_server.md` and `configs/llamacpp_launch.ps1`.
- **OpenRouter** (`--vlm-backend openrouter`) as a hosted fallback requiring
  `OPENROUTER_API_KEY`.

Note: `README.md` predates the vLLM backend and still describes llama-server
as the primary/required server — `doc2md/cli.py` is the source of truth;
`--vlm-backend vllm` is the current default.

## Running vLLM under WSL2 (Gemma 4 / AWQ) — verified gotchas

See also `handoffs/vllm_guide_handoff.md` for the full reference — exact launch
script contents, every bug's precise error text, complete throughput data,
and a prioritized list of what to try next — and
`docs/VLLM_SESSION_JOURNAL.md` for the narrative account of how each bug
was actually found, in order, including the dead ends.

Getting `vllm serve` to actually start for this model stack (Gemma 4,
`compressed-tensors`/AWQ quantization, WSL2) took several non-obvious fixes,
found by working through real crash tracebacks rather than guessing. All of
them live in the WSL-side launch script (`~/launch_vllm.sh` inside the WSL
filesystem — not checked into this git repo). If that script goes missing,
or vLLM/transformers get upgraded, expect these to resurface:

- **`transformers` must be pinned to `5.14.1`, not latest.** `transformers`
  5.15.0 introduced a stricter guard (`HeterogeneousConfigMixin`) on
  per-layer config attributes. Gemma 4 mixes sliding-attention layers
  (`head_dim=256`) with full-attention layers (`head_dim=512`), and vLLM
  0.27.x's config-conversion code reads `head_dim` via plain
  `getattr`/`hasattr` in many places that predate this guard, so it raises
  `AmbiguousGlobalPerLayerAttributeError` instead of just returning a value.
  Confirmed upstream bug: https://github.com/vllm-project/vllm/issues/51744.
  Downgrading transformers is the real fix — this is architecture-level, not
  specific to any one checkpoint or quantization format; it hits any
  `gemma4`/`gemma4_unified` model on vLLM 0.27.x with transformers >=5.15.
  `scripts/patch_vllm_gemma4_head_dim.py` and
  `scripts/patch_vllm_heterogeneous_config.py` in this repo are narrower
  monkeypatches for the same root cause, superseded by the transformers
  pin — safe to delete next time this area is touched.
- **`VLLM_WSL2_ENABLE_PIN_MEMORY=1` is required**, or CUDA pinned/unified
  memory (UVA) is unavailable and the engine fails at device init with
  `RuntimeError: UVA is not available`.
- **`CUDA_HOME` isn't set by default** in the WSL Python environment, and
  FlashInfer's sampling kernel needs it (plus `ninja` on `PATH`) to
  JIT-compile. Even with both set, the JIT build can still fail with "CUDA
  compiler and CUDA toolkit headers are incompatible" — FlashInfer's bundled
  CCCL headers don't match the pip-installed `nvcc` version in this
  environment. Not worth chasing a toolchain match for an optional fast
  path: set `VLLM_USE_FLASHINFER_SAMPLER=0` to fall back to vLLM's native
  sampler instead.
- The model actually verified working is
  `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` (community AWQ-INT4 requant of
  Google's QAT checkpoint) — pass
  `--vllm-model cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` to `doc2md`, since
  `cli.py`'s hardcoded default (`google/gemma-3-4b-it`) is stale. Google's
  own official release, `google/gemma-4-12B-it-qat-w4a16-ct` (symmetric
  quant, no zero-point tensor), is a plausible alternative but was not
  fully verified end-to-end — download stalled on a slow connection during
  testing. For any future large Hugging Face download, use
  `scripts/fast_download.py <repo_id> <cache_dir>` — it sets
  `HF_HUB_ENABLE_HF_TRANSFER=1` (the Rust `hf_transfer` downloader) and
  uses `snapshot_download` with 16 parallel workers; it's the one
  downloader script kept after a cleanup that removed four other,
  redundant one-off download scripts from `scripts/`.
- **A stale `VLLM::EngineCore` subprocess survives a plain
  `pkill -f "vllm serve"`** — vLLM renames that subprocess's title, so it
  keeps running and holding VRAM after a normal kill, and the next launch
  starts a second engine alongside it (~15GB double VRAM usage from two
  coexisting engines). `launch_vllm.sh` kills both the `"vllm serve"` and
  `"VLLM::EngineCore"` process-name patterns before relaunching.
- **Driving WSL from Windows Git Bash mangles paths unless
  `MSYS_NO_PATHCONV=1` is set** (or absolute `/home/...` paths are used) —
  without it, paths passed to `wsl.exe` get silently rewritten into Windows
  form and commands fail in confusing ways.

### Known gaps (vLLM backend, as of this writing)

- **doc2md has never actually been run end-to-end against the live vLLM
  server** — all verification so far is synthetic load-testing
  (`scripts/vllm_throughput_test.py`) against real page-region crops and
  real prompts, not a full `uv run doc2md ...` conversion. `cli.py`'s
  `--vllm-model` default is still the unrelated `google/gemma-3-4b-it`, so
  it must be passed explicitly.
- The `transformers==5.14.1` pin lives only in the untracked WSL launch
  script — there's no requirements file capturing it, so a fresh WSL
  environment setup would need to rediscover this.
- doc2md's own `--concurrency` (default 32) has not been tested against the
  server's real `--max-num-seqs 128` capacity at real-document scale — only
  the standalone throughput test script has been.
- No process supervision/auto-restart exists for the vLLM server — it does
  not survive a WSL2 restart, reboot, or crash.

### Measured throughput (RTX 5080 Laptop, 16GB, `--max-model-len 4096`)

Real doc2md-style request (real page-region crop, actual text-transcription
prompt, ~80 completion tokens/request). Aggregate tokens/sec scales with
concurrent connections until the GPU KV cache genuinely saturates:

| `--max-num-seqs` / concurrent connections | Aggregate tok/s | Notes |
|---|---|---|
| 1 | 61 | single-stream, bandwidth-bound |
| 8 | 435 | |
| 32 | 954 | |
| 64 | 1160 | GPU KV cache usage ~40% — still headroom |
| 128 | 1258 (peak internal 1511) | GPU KV cache usage ~71% — still no queueing |
| 192 | 1286 (peak internal 1462) | GPU KV cache hits ~100%, requests start queueing (real cap) — throughput barely improves over 128 while avg latency nearly doubles |

**`--max-num-seqs 128` is the sweet spot** and what `launch_vllm.sh` is set
to: peak throughput with zero queueing. This workload is entirely
GPU-bound — CPU utilization measured at ~4-5% (essentially one core) across
all 24 available cores during a saturating load test, so there is no CPU
tuning lever here.

KV cache dtype was deliberately left at the `bf16` default rather than
tried at `fp8`: vLLM's own FP8 KV cache blog
(https://vllm.ai/blog/2026-04-22-fp8-kvcache) names contexts under ~7k
tokens, `head_dim=256`, and "many small sliding-window attention layers" as
cases to avoid FP8 — this workload matches all three (doc2md's real prompts
run ~300-400 tokens total, and Gemma 4's sliding-attention layers use
`head_dim=256`). Separately, FP8 KV cache was never evaluated on
vision/multimodal models in that post at all, and multimodal-specific
research (AKVQ-VL, KVCapsule) finds that generic/uniform KV quantization
"overlooks attention saliency differences of multimodal tokens" — a
specialized asymmetric scheme would likely be needed to quantize KV cache
safely here, and vLLM doesn't ship one.

Gemma 4 is "encoder-free" in the sense that it has no separate heavyweight
vision tower (a 550M-parameter ViT is replaced by a single 35M-parameter
projection matmul over raw image patches) — confirmed directly in this repo's
downloaded checkpoint, where the entire `vision_embedder` + `embed_vision`
submodule totals well under 100MB against ~11GB for the shared transformer
trunk. It's also token-efficient: an ~862×892px region crop measured at
only ~167 vision tokens (295 total prompt tokens minus ~128 for the system
+ user text prompt and chat-template overhead). But image tokens still flow
through the same causal self-attention as text tokens and consume the same
KV cache per token — "encoder-free" doesn't mean free of KV cache cost, it
means no separate encoder *and* fewer tokens per image than typical ViT-tokenizer
VLM architectures (e.g. LLaVA-style ~576 tokens per tile), which is real
but distinct from the misconception that images bypass KV cache entirely.

## llama.cpp tuning — verified gotchas (legacy backend)

vLLM is the primary/production VLM backend for `doc2md` going forward;
`llama-server` (`--vlm-backend llama_server`) is kept as a legacy fallback.
The tuning below was verified on this machine and is what
`configs/llamacpp_launch.ps1` encodes, in case that backend is ever revived:

- An orphaned `llama-server.exe` process from a previous run silently holds
  VRAM in the background, forcing the next run to swap and stutter — kill
  stale processes before relaunching.
- Full GPU offload needs `-ngl 99` explicitly; without it, some transformer
  layers run on CPU.
- Windows WDDM downclocks the GPU's P-state during the gaps between tokens,
  which flash attention and steady batching (below) help avoid by keeping
  the GPU consistently busy.
- The hybrid Intel Core Ultra 9 275HX's E-cores cause OpenMP thread
  contention if all cores are used — pinning to P-cores only (`-t 8`) beats
  using every core.
- The default multi-slot context splitting (`-np 4`) fragments KV-cache
  bandwidth across slots; single-stream use wants one slot (`-np 1`).
- Flash attention is not on by default and must be passed explicitly
  (`-fa on`).

With `-ngl 99 -fa on -t 8 -tb 16 -np 1 -ub 1024 -c 4096`, decode throughput
went from a baseline **5.2–9.8 t/s** (~4% bandwidth efficiency) to a
verified **74.0–74.6 t/s** (~534–539 GB/s, ~59.6–60.1% of the 896 GB/s
theoretical bus bandwidth) — see `reports/performance_report.md` for the
full diagnosis. Note the model's real physical decode ceiling on this GPU is
**124.10 t/s** (896 GB/s ÷ 7.22 GB model size); the ">=150 t/s" target
`PROJECT.md` originally set was never physically reachable on this
hardware/model combination.

## Architecture (`doc2md/`)

Pipeline stages, each in its own module, orchestrated by `pipeline.py`'s
`convert_document_async`:

```
input (PDF / images / PPTX)
    │  render.py            — PyMuPDF rasterization → list[PageImage]
    ▼
layout_engines/get_layout_engine(settings)   — pluggable, chosen by --layout-engine
    │  returns list[Region] in reading order (bbox, label, score, order_index)
    ▼
classify.py                — label -> Bucket (TEXT_LIKE / TABLE / PICTURE / SKIP)
    ▼
crop.py                    — crop_region() per Region
    ▼
vlm_client.py               — async HTTP call per region to the VLM backend,
    │                          prompt selected by bucket (prompts.py)
    ▼
markdown_builder.py         — assembles per-region markdown into the final .md,
                               in page/order_index order
```

Key points for working in this codebase:

- **Everything is async and concurrency-bounded.** `pipeline.py` fires one
  VLM request per region concurrently, bounded by `asyncio.Semaphore(settings.vlm_max_concurrency)`
  (`--concurrency`, default 32) to exploit vLLM's continuous batching. A
  single region's VLM failure is caught in `_process_region_safe_async` and
  becomes an inline `> **doc2md: failed to extract...**` note rather than
  aborting the whole document.
- **Layout engines are pluggable via `LayoutEngine` ABC** (`layout_engines/base.py`,
  one `detect(page) -> list[Region]` method). `layout_engines/__init__.py`'s
  `get_layout_engine()` is the only place new engines need registering.
  Six engines are implemented, selected via `--layout-engine`: `mineru`
  (default, PP-DocLayoutV2), `docling` (`docling-layout-heron`, DocLayNet
  classes, verified end-to-end), `doclayout_yolo`, `markitdown`,
  `pymupdf4llm` (pure vector, zero GPU overhead), `paddleocr`. Engines that
  lack a native reading-order head (e.g. `docling`) fall back to the
  row-bucket heuristic in `reading_order.py`; `dedup.py` suppresses
  containment-based duplicate boxes for engines that need it.
- **Picture regions get a diagram/image decision at VLM time**, not layout
  time: the VLM is asked to redraw structured diagrams as Mermaid; if it
  declines (`NOT_DIAGRAM_SENTINEL` in `prompts.py`), `pipeline.py` crops and
  saves the region as a real asset file instead (`crop.save_region_crop`),
  referenced via a relative Markdown image link. Nothing is written to
  `<name>_assets/` for regions that became Mermaid.
- **Resumability**: `pipeline.py` checkpoints per-page results to
  `<output>/<name>.doc2md_progress.json` after each batch of tasks. The
  checkpoint is keyed by a fingerprint (resolved input path, layout engine,
  DPI, total page count) — if any of those change, the checkpoint is
  considered stale and ignored (`--no-resume` also skips it). The checkpoint
  file is deleted on successful completion.
- **`Settings` (`config.py`) is the single config object** threaded through
  every stage — CLI flags in `cli.py` just populate it. When adding a new
  pipeline knob, add it to `Settings` first.
- Labels in `config.DEFAULT_SKIP_LABELS` (headers/footers/page numbers) are
  dropped before ever reaching the VLM.
