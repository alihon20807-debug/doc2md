# HANDOFF: Documentation consolidation & repo cleanup (doc2md)

**Date:** 2026-08-11
**Goal:** the repo had accumulated a lot of stale/fabricated documentation
and leftover files from the vLLM/llama.cpp benchmarking saga (see
`handoffs/vllm_guide_handoff.md`) and an earlier, incomplete multi-agent
benchmarking run. This session found and fixed the specific inaccuracies,
removed what was safe to remove, and consolidated duplicated code.
**Outcome:** done. Every change below was verified (test suite, import
checks, CLI smoke test) before being considered complete.

## 1. Current state (verify before trusting anything below)

```
git status --porcelain -uall   # everything is still untracked as of this
                                # handoff — see section 6, "what's not done"
```

Directories touched by this session: `reports/`, `scripts/`, `llama.cpp-bin/`,
`.agents/` (deleted), `handoffs/` (new), plus `CLAUDE.md`, `PROJECT.md`,
`docs/VLLM_SESSION_JOURNAL.md`, `benchmark/run_benchmark.py`,
`doc2md/vlm_client.py`, `doc2md/layout_engines/{base,markitdown_engine,
pymupdf4llm_engine,paddleocr_engine}.py`.

## 2. Fabricated data found and removed

Two files in `reports/` claimed >100% of the RTX 5080's theoretical memory
bandwidth (896 GB/s) — physically impossible, since decode throughput is
bandwidth-bound and can't exceed `bandwidth / model_size`:

- `reports/benchmark_results.json` (150.1 t/s, "120.9% efficiency") — its
  values were near-identical to the *illustrative example* JSON schema in
  `PROJECT.md`'s "Interface Contracts" section, i.e. documentation-example
  values that got saved as if they were a real result.
- `reports/benchmark_results_vllm.json` (155.0 t/s, "124.9% efficiency") —
  traced back to `.agents/teamwork_preview_worker_m4`, a milestone in a
  now-deleted multi-agent orchestration trace whose own `progress.md`
  showed it stalled at task 3/7. No reviewer or auditor ever validated its
  output, unlike every other milestone in that trace.

Both were deleted. `reports/llamacpp_benchmark.json` and
`reports/llamacpp_benchmark_final.json` were kept — both are under the
124.10 t/s physical ceiling, mutually consistent (74.6 / 74.0 t/s), and
independently confirmed by `reports/performance_report.md`'s diagnosis.

`PROJECT.md`'s illustrative JSON schema example was also fixed — it
previously used the same kind of impossible values (140.8% efficiency),
which is exactly what got copy-pasted into a fake result file in the first
place. It now uses real, physically-consistent example numbers.

## 3. Mislabeled status corrected

`PROJECT.md`'s milestone table marked M4 (vLLM Setup & Optimization) and M5
(Audit, Verification & Report) as `DONE`. The (now-deleted) `.agents/`
trace showed M4 never finished and M5 never started via that pipeline — the
vLLM setup that *did* eventually work was done later, in a separate manual
debugging session, documented in `handoffs/vllm_guide_handoff.md` and
`docs/VLLM_SESSION_JOURNAL.md`. `PROJECT.md` now reflects this and points
to the real source of truth.

`reports/performance_report.md` mixed real, cross-confirmed llama.cpp
results with a stale/incorrect vLLM fix description (it claimed
`VLLM_USE_V1=0` was the fix for `RuntimeError: UVA is not available` — the
actual fix, found later, is `VLLM_WSL2_ENABLE_PIN_MEMORY=1` plus a
`transformers==5.14.1` pin; `VLLM_USE_V1` doesn't exist in current vLLM at
all). It was rewritten: retitled to llama.cpp-only, the vLLM section now
explains it was superseded and points elsewhere, the embedded JSON
"snapshot" was fixed to match the real file (it previously showed
`criteria_passed: true` when the real file says `false`, and a fabricated
`prompt_tps: 3221.85` vs. the real file's `0.9`), and the "Antigravity AI
Assistant" footer was dropped.

## 4. Files deleted as leftover/redundant

- **`.agents/`** (52 files) — the stalled multi-agent orchestration trace
  from section 2/3 above. Fully superseded; findings already folded into
  `CLAUDE.md`/`PROJECT.md`.
- **12 of 16 scripts in `scripts/`**: `download_model.py`, `download_awq.py`,
  `download_with_progress.py`, `multi_thread_download.py` (four separate,
  overlapping reimplementations of `snapshot_download`/chunked downloading
  — redundant with the fifth, `fast_download.py`), and `build_shim.py`,
  `check_hf_gemma4.py`, `check_model_size.py`, `check_tokenizer_size.py`,
  `check_vllm_support.py`, `patch_vllm_config.py`, `read_gguf_meta.py`,
  `test_vllm_import.py` (one-off debugging scratch scripts from the
  WSL2/GGUF saga, zero doc references, job already done). Kept:
  `fast_download.py` (now documented in `CLAUDE.md` as the canonical
  downloader for future large HF downloads), `make_sample_pdf.py`,
  `vllm_throughput_test.py`, `setup_llama_server.md`.
- **`llama.cpp-bin/server.log`, `server_run.log`, `cudart133.zip`,
  `llama-cuda133.zip`** (~514MB) — run logs and already-extracted archives
  (confirmed via file timestamps that the DLLs/EXEs inside were already
  extracted separately in the same directory).
- **`scripts/patch_vllm_gemma4_head_dim.py`,
  `patch_vllm_heterogeneous_config.py`** — deleted in an earlier pass of
  this same session; `CLAUDE.md` already documented these as superseded by
  the `transformers==5.14.1` pin. (`docs/VLLM_SESSION_JOURNAL.md` still had
  a stale line claiming these were "still in `scripts/`" — fixed to point
  here instead.)

**Deliberately kept**: `vendor/installers/` (~3.15GB CUDA 12.8.1 + VS Build
Tools installers — no doc references, but deleting them means a 3.15GB
re-download if ever needed again) and `models/` (7.1GB, actively-used model
weights). Both already gitignored.

## 5. Code consolidation

All changes below were verified behavior-preserving: `uv run python -m
unittest discover -s benchmark -v` passed 17/17 after every step, `uv run
doc2md --help` ran cleanly, and every touched module was confirmed to
import without error.

- **`benchmark/run_benchmark.py`**: `run_llamacpp_benchmark`'s try block,
  its except-fallback block, and `run_vllm_benchmark` each independently
  computed `decode_tps`/`prompt_tps` and built an identical 9-key result
  dict. Extracted into one `_build_benchmark_result(res, prompt_tokens,
  tokens_generated)` helper. Also deleted the dead `make_http_request`
  function (unused — both benchmark functions actually use
  `make_streaming_http_request`), and removed the now-unused `Tuple` and
  `urllib.error` imports.
- **`DEFAULT_MODEL_SIZE_BYTES`** was defined identically in both
  `benchmark/bandwidth_calculator.py` and `benchmark/run_benchmark.py` (the
  test suite even asserted the two copies were equal). `run_benchmark.py`
  now imports the single definition from `bandwidth_calculator.py` instead.
- **`doc2md/vlm_client.py`**: `AsyncVLLMClient.ask`, `AsyncOpenRouterClient.ask`,
  and `VLMClient.ask` each rebuilt an identical chat-messages payload
  structure inline, differing only in whether/what `model` key gets added.
  Extracted into `_build_chat_payload(settings, image, prompt,
  system_prompt, *, model=None)` — omitting `model` reproduces the legacy
  llama-server client's exact original payload shape (no `model` key at
  all). **This is the vLLM code path** — treated carefully given how much
  debugging went into getting vLLM working; verified via a plain import
  check immediately after editing, then again via the full test suite.
- **`doc2md/layout_engines/`**: `markitdown_engine.py` and
  `pymupdf4llm_engine.py` had an identical block resolving a page image
  back to its source PDF path; `paddleocr_engine.py` and
  `pymupdf4llm_engine.py` had an identical "no regions detected → synthesize
  one full-page region" fallback block. Added `resolve_source_pdf(page)`
  and `full_page_fallback_region(page, settings, label_map, label="text")`
  to `doc2md/layout_engines/base.py` and updated all three engines to use
  them.

## 6. What's NOT done

- **Nothing in this repo was committed to git before this session** — every
  deletion above is permanent, there was no history to fall back on. Local
  git tracking is being set up as the very next step after this handoff
  (see the conversation this handoff came from, or just run `git log` to
  check whether that happened).
- This was a documentation/dead-code cleanup pass, not a functional
  verification of the doc2md pipeline itself — the code-consolidation
  changes were checked for import correctness and payload-shape equivalence,
  not by actually running a live conversion against a vLLM server. The
  "doc2md has never been run end-to-end against the live vLLM server" gap
  noted in `CLAUDE.md`'s "Known gaps" section is still open.
- `configs/vllm_launch.ps1` / `.sh` were left untouched — already correctly
  flagged in `CLAUDE.md` as stale/don't-use, and they disagree with each
  other on `--tokenizer` plus `.sh` has a likely line-continuation bug. Not
  worth fixing scripts nobody should run.
- `scripts/setup_llama_server.md`'s framing (written as if llama-server
  were the only backend) wasn't rewritten — pre-existing, already-known
  staleness, same as `README.md`'s.
