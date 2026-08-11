# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two things live in the same repo:

1. **`doc2md`** (the actual product) — a local pipeline that converts PDFs / page
   images into clean Markdown: a layout-detection model finds regions on each
   page, then a local vision-language model (VLM) transcribes each region
   (text, tables → GFM tables, formulas → LaTeX, diagrams → Mermaid).
2. **`benchmark/`** — a separate, self-contained harness (see `benchmark/PROJECT.md`)
   for benchmarking llama.cpp vs vLLM decode throughput/bandwidth on the dev
   machine's RTX 5080. It doesn't depend on `doc2md/` and vice versa.
   `PROJECT.md` lives under `benchmark/`, not the repo root, since it only
   documents that harness, not doc2md itself.

## Commands

```bash
uv sync                                  # install deps incl. CUDA torch (cu128 index)
uv run doc2md path/to/file.pdf -o out/   # run the converter
uv run doc2md --help                     # full CLI option list
uv run python -m unittest discover -s benchmark -v   # run benchmark unit tests
```

There is no linter/formatter config and no test suite for `doc2md/` itself
(the only tests in the repo are `benchmark/test_bandwidth_calculator.py`,
`unittest`-based, not pytest — pytest isn't a dependency). `sample_docs/`
holds sample input documents for manual conversion runs, not automated
tests. **`test/` does not actually hold sample input documents** despite
the name — it currently contains two unrelated files (a `.pptx`/`.pdf` pair
from what looks like personal university coursework, `FALLSEM2026-27_VL_
BACSE203...`), noticed but deliberately not touched during the 2026-08-11
consolidation pass since they might be the user's own files worth keeping;
confirm with the user before assuming they're safe to delete or move.
`reports/` holds benchmark output only, not test reports.
Two files previously in `reports/` (`benchmark_results.json`,
`benchmark_results_vllm.json`) were deleted — they claimed >100% of
theoretical GPU memory bandwidth, which is physically impossible; they were
either copy-pasted example/placeholder values or came from an unreviewed,
stalled milestone in `.agents/`, not a real benchmark run. Don't recreate
files like that from `benchmark/PROJECT.md`'s illustrative JSON schema — it's
for shape only, not real numbers.

`doc2md` requires a running local inference server before conversion works:
- **vLLM** (default backend) at `http://127.0.0.1:8000`, OpenAI-compatible
  `/v1/chat/completions`, serving a Gemma vision model, run inside WSL2 (see
  "Running vLLM under WSL2" below, and `configs/vllm_launch.sh` for the
  actual verified-working launch script — a prior `configs/vllm_launch.ps1`
  and an earlier, different `configs/vllm_launch.sh` were deleted in a
  consolidation pass; both referenced `VLLM_USE_V1`, an env var that no
  longer exists in current vLLM, among other stale settings).
- **llama-server** (`--vlm-backend llama_server`) as a legacy alternative —
  see `scripts/setup_llama_server.md` and `configs/llamacpp_launch.ps1`.
- **OpenRouter** (`--vlm-backend openrouter`) as a hosted fallback requiring
  `OPENROUTER_API_KEY`.

`README.md` was updated in the same consolidation pass that fixed the bugs
below to reflect vLLM as the default backend — `doc2md/cli.py` remains the
ultimate source of truth for current flags/defaults if the two ever drift
again.

## Running vLLM under WSL2 (Gemma 4 / AWQ) — verified gotchas

See also `handoffs/vllm_guide_handoff.md` for the full reference — exact launch
script contents, every bug's precise error text, complete throughput data,
a prioritized list of what to try next, and (in its final section) the
narrative account of how each bug was actually found, in order, including
the dead ends.

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
  Google's QAT checkpoint) — this is now `cli.py`'s and `config.py`'s
  `--vllm-model` default (fixed in a later consolidation pass; it used to
  default to the unrelated `google/gemma-3-4b-it`), so no flag is needed for
  the common case. Google's own official release,
  `google/gemma-4-12B-it-qat-w4a16-ct` (symmetric quant, no zero-point
  tensor), is a plausible alternative but was not fully verified end-to-end
  — download stalled on a slow connection during testing. For any future
  large Hugging Face download, use
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
  real prompts, not a full `uv run doc2md ...` conversion. This session's
  consolidation pass fixed several real pipeline/config bugs (see
  "Consolidation pass" below) and verified them with fakes/mocks standing in
  for the VLM server, but still never exercised a live vLLM server directly
  — that gap is unchanged.
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
`benchmark/PROJECT.md` originally set was never physically reachable on this
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
  `pymupdf4llm` (pure vector, zero GPU overhead), `paddleocr`. All six share
  the same contract via `layout_engines/base.py` helpers: `make_region()`
  (label→bucket→Region, skipping SKIP-bucket labels), `finalize_regions()`
  (dedup + reading order — every engine but `mineru`, which has its own
  native reading-order head), and `safe_detect()` (one consistent
  failure policy: a detection exception or zero-region result is logged and
  falls back to a synthesized full-page region, instead of the pre-2026-08-11-
  consolidation mix of some engines silently swallowing errors and others
  aborting the whole document). `dedup.py` suppresses containment-based
  duplicate boxes; `reading_order.py` is the row-bucket heuristic used by
  every engine except `mineru`.
- **OCR fast path is pluggable the same way**, via `ocr_engines/` (`OCREngine`
  ABC, `get_ocr_engine()`). Opt-in with `--ocr-fast-path`; only `mineru`,
  `paddleocr`, and `docling` bundle an OCR model (mineru: its own
  `PytorchPaddleOCR`; paddleocr: PaddleOCR's own `PaddleOCR` pipeline class,
  distinct from its layout-detection model; docling: `rapidocr-onnxruntime`,
  since Docling's layout-only integration has no bundled OCR of its own).
  For plain `TEXT_LIKE` regions only — titles and formulas (see
  `prompts.TITLE_LABELS`/`FORMULA_LABELS`) always go through the VLM
  regardless, since OCR can't produce a Markdown heading or LaTeX. Falls
  back to the VLM on any OCR exception or empty result.
- **Picture regions get a diagram/image decision at VLM time**, not layout
  time: the VLM is asked to redraw structured diagrams as Mermaid; if it
  declines (`NOT_DIAGRAM_SENTINEL` in `prompts.py`), `pipeline.py` crops and
  saves the region as a real asset file instead (`crop.save_region_crop`),
  referenced via a relative Markdown image link. Nothing is written to
  `<name>_assets/` for regions that became Mermaid.
- **Resumability**: `pipeline.py` checkpoints per-page results to
  `<output>/<name>.doc2md_progress.json` incrementally, as each page's
  regions finish (`asyncio.as_completed` over per-page task groups, all
  sharing one semaphore so cross-page concurrency is unaffected) — a crash
  mid-run only loses whichever pages hadn't finished yet, not the whole
  document. On resume, layout detection itself is also skipped for
  already-checkpointed pages, not just the VLM calls. The checkpoint is
  keyed by a fingerprint (resolved input path, layout engine, DPI, total
  page count) — if any of those change, the checkpoint is considered stale
  and ignored (`--no-resume` also skips it). The checkpoint file is deleted
  on successful completion. (Both the incremental-checkpointing and
  skip-detection-on-resume behaviors were bugs before the 2026-08-11
  consolidation pass — checkpointing used to happen only once, after every
  region in the document finished, which defeated the point of resuming a
  crashed run.)
- **`Settings` (`config.py`) is the single config object** threaded through
  every stage — CLI flags in `cli.py` just populate it. When adding a new
  pipeline knob, add it to `Settings` first.
- Labels in `config.DEFAULT_SKIP_LABELS` (headers/footers/page numbers) are
  dropped before ever reaching the VLM. `header_image`/`footer_image` are
  deliberately NOT in this set even though `header`/`footer` are — they map
  to `Bucket.PICTURE` in `classify.PP_DOCLAYOUT_LABEL_TO_BUCKET` (a
  logo/figure in a header/footer band is still worth extracting), and being
  in the skip set would make that mapping entry unreachable (this was a real
  bug, fixed in the 2026-08-11 consolidation pass below).

## Consolidation pass (2026-08-11) — real bugs found and fixed

A deep audit of the whole `doc2md/` package (not just docs) found several
genuinely broken things that had been running unnoticed. Recorded here so
they don't get silently reintroduced or rediscovered from scratch:

- **`resolve_source_pdf()` only worked if the input PDF happened to be in
  the process's current working directory.** `PageImage.source_name` only
  ever stored the file's stem (e.g. `"test"`, from `path.stem`), never the
  actual path — `layout_engines/base.py`'s `resolve_source_pdf()` tried to
  find the PDF by checking `Path(page.source_name)` relative to cwd, which
  is essentially never where the input file actually lives for a real
  invocation (`doc2md some/other/dir/file.pdf`). This silently disabled the
  entire "rich PDF extraction" code path for the `markitdown` and
  `pymupdf4llm` engines — they'd detect zero real regions and always fall
  back to a single full-page region, with no error or warning, relying
  entirely on the VLM to do vector-extraction-quality engines' whole job.
  Fixed by adding `PageImage.source_path` (an actual resolved `Path`, set by
  `render.py` wherever a real PDF is opened) and having `resolve_source_pdf`
  prefer it over the stem-based cwd guess.
- **`--vlm-backend llama_server` silently pointed at vLLM's port.** `cli.py`
  had `--server`/`--vllm-url` (for vLLM) double as the llama-server URL too
  (`llama_server_url=vllm_url`), so using the llama-server backend without
  also passing `--server` explicitly hit `127.0.0.1:8000` (vLLM's default)
  instead of `127.0.0.1:8080` (llama-server's real default, from
  `config.py`). Fixed with a dedicated `--llama-server-url` flag, defaulting
  independently to `8080`.
- **`--rate-limit`/`vlm_requests_per_minute` was a documented flag that did
  nothing.** Stored on `Settings`, read nowhere — no proactive OpenRouter
  rate limiting existed, only reactive 429 retry. Fixed with a real
  token-bucket-style `_RateLimiter` in `vlm_client.py`, applied in
  `AsyncOpenRouterClient.ask()`.
- **`DOCLAYOUT_YOLO_LABEL_TO_BUCKET` routed formulas to `Bucket.PICTURE`**,
  while every other engine routes them to `Bucket.TEXT_LIKE` for LaTeX
  transcription — under `--layout-engine doclayout_yolo`, formulas got the
  diagram/Mermaid treatment instead of LaTeX. Fixed to match every other
  engine.
- **`prompts.TITLE_LABELS`/`FORMULA_LABELS` only listed MinerU's native
  label spellings**, but `text_prompt()` is called generically for every
  engine's `TEXT_LIKE` regions — under `docling`/`markitdown`/`doclayout_yolo`,
  titles silently got the generic body-text prompt (no Markdown heading) and
  Docling formulas got generic text instead of LaTeX. Fixed by extending
  both sets to cover every engine's real label spellings.
- **Checkpointing/resume bugs** — see the "Resumability" bullet above.
- **Per-region CPU thread pool was sized off `vlm_max_concurrency`**
  (an I/O-bound VLM-request-fan-out setting), letting it scale arbitrarily
  high for CPU-bound crop/detect work. Now capped at
  `min(vlm_max_concurrency, os.cpu_count())`.
- **`--vllm-model` default was stale** (`google/gemma-3-4b-it`) — see above.
- **Adding the new `rapidocr-onnxruntime` dependency (for the `docling` OCR
  fast path) silently broke the unrelated `pymupdf4llm` engine.**
  `pymupdf4llm`'s newer "layout" extraction mode defaults to `use_ocr=True`
  and opportunistically uses *any* OCR backend it finds importable in the
  environment — once `rapidocr-onnxruntime` became available, `pymupdf4llm`
  started routing this engine's table detection through an OCR-assisted
  path instead of its pure-vector one, and a real table region silently
  turned into three fragmented `image` regions with no `table` region at
  all (verified by running the *original*, pre-consolidation code against
  the same test document — same broken result, proving this wasn't a
  regression in the refactor itself, only in the new dependency). Fixed
  with an explicit `use_ocr=False` in `pymupdf4llm_engine.py`'s
  `to_markdown()` call, restoring this engine's documented "zero GPU
  overhead, pure vector" behavior regardless of what other OCR packages are
  installed. **General lesson for future sessions**: installing a new
  optional-dependency-detecting package (OCR backends are especially prone
  to this) can silently change behavior in unrelated code that never
  imports it directly — re-verify all layout engines' actual output after
  adding any new dependency, not just the one you added it for.

Also added: the `--ocr-fast-path` capability described above, and shared
`layout_engines/base.py` helpers (`make_region`/`finalize_regions`/
`safe_detect`) that eliminated a lot of near-identical boilerplate that had
been copy-pasted across all 6 engines with subtly inconsistent
error-handling and fallback behavior between them.
