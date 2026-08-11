# HANDOFF: vLLM + Gemma 4 12B AWQ serving for doc2md

**Status as of 2026-08-11:** vLLM is running, verified working end-to-end
(real chat completions confirmed), throughput-tested and tuned to a
measured, evidence-based setting. **doc2md itself has NOT been run against
this server yet** — that's the single biggest remaining gap. Read
"What's NOT done" near the bottom before assuming this is fully wired up.

This file is the single source of truth for continuing this work. §10 at the
bottom is the narrative account of how each bug was found, in order,
including the dead ends - read that if you want the "why did this take so
long" story (it used to be a separate file, `docs/VLLM_SESSION_JOURNAL.md`,
merged in here to keep the vLLM story in one place). `CLAUDE.md` has the
condensed version of the gotchas below, meant for quick orientation at the
start of a session, not full detail.

---

## 1. Current live state (verify before trusting anything below)

The server may or may not still be running by the time you read this — WSL2
and the underlying Windows session can both restart independently of this
repo. Check first:

```bash
# From Windows (Git Bash / this session's Bash tool):
MSYS_NO_PATHCONV=1 wsl -- bash -lc 'ps aux | grep "vllm serve" | grep -v grep'
MSYS_NO_PATHCONV=1 wsl -- bash -lc 'curl -s -m 5 http://127.0.0.1:8000/v1/models'
```

If nothing's running, bring it back up:

```bash
MSYS_NO_PATHCONV=1 wsl -- bash -lc 'bash ~/launch_vllm.sh'
```

Then wait ~2-3 minutes (weight load + `torch.compile` + CUDA graph capture)
and poll `curl http://127.0.0.1:8000/v1/models` until it responds. Watch
`~/vllm_serve.log` for progress or errors:

```bash
MSYS_NO_PATHCONV=1 wsl -- bash -lc 'tail -f ~/vllm_serve.log'
```

**IMPORTANT — the `MSYS_NO_PATHCONV=1` prefix is not optional.** This
session runs Git Bash (MSYS) on Windows. Any `wsl -- bash -lc '...'` call
whose inner command contains an absolute POSIX-looking path (anything
starting with `/`) gets silently mangled by MSYS *before* `wsl.exe` ever
sees it — `/home/aliho/...` becomes `C:/Program Files/Git/home/aliho/...`.
The resulting error looks nothing like a path problem
(`bash: line 1: C:/Program: No such file or directory`). Every WSL command
in this doc and in `launch_vllm.sh`'s own invocation needs this prefix if
you're driving it from a Windows-hosted shell the same way this session did.
If you're running commands directly inside a WSL terminal (not proxied
through Windows), this doesn't apply.

## 2. The launch script

Lives at `~/launch_vllm.sh` **inside the WSL filesystem** — it is
deliberately **not checked into this git repo**, because it's specific to
this machine's WSL environment (hardcoded paths under `/home/aliho/...`).
Full current contents, reproduced here so this fact survives even if the
WSL filesystem is ever wiped:

```bash
#!/usr/bin/env bash
set -x
pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 2
: > ~/vllm_serve.log
export HF_HOME=/mnt/c/Users/aliho/.cache/huggingface
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME=/home/aliho/.python312/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:/home/aliho/.python312/bin:$PATH"
# FlashInfer's JIT-compiled sampler fails to build in this env (its bundled
# cccl headers don't match the pip-installed nvcc version) - not worth
# chasing a toolchain match for an optional fast-path; the native sampler
# fallback works fine.
export VLLM_USE_FLASHINFER_SAMPLER=0
setsid nohup /home/aliho/.python312/bin/vllm serve cyankiwi/gemma-4-12B-it-qat-AWQ-INT4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --host 0.0.0.0 \
  --port 8000 > ~/vllm_serve.log 2>&1 < /dev/null &
disown
sleep 3
ps aux | grep vllm | grep -v grep
```

Note it kills **both** `vllm serve` and `VLLM::EngineCore` process patterns
before relaunching — see §4 for why that second pattern matters.

**Environment this runs in:**
- WSL2 distro: Ubuntu (`wsl -l -v` to confirm), kernel `6.18.33.2-microsoft-standard-WSL2`
- Python: a hand-built interpreter at `/home/aliho/.python312/` (**not** the
  system `python3`, **not** a `uv`-managed venv — it's a standalone
  install with its own `bin/`, `lib/python3.12/site-packages/`, etc. This
  is unrelated to and separate from this repo's own `uv`-managed
  `doc2md` environment on the Windows side.)
- `vllm==0.27.1`, `transformers==5.14.1` (pinned down from 5.15.0 — see §5)
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU, 16303 MiB VRAM, ~896 GB/s memory bandwidth
- 24 CPU cores available to WSL2, 15GB RAM allocated to WSL2

## 3. The model

**Currently serving:** `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` — a
community AWQ-INT4 requantization of Google's Gemma 4 12B QAT checkpoint,
`compressed-tensors` format, asymmetric quantization (has a `zero_point`
tensor per quantized weight), group_size=32, 4-bit.

Weights live at `C:\Users\aliho\.cache\huggingface\hub\models--cyankiwi--gemma-4-12B-it-qat-AWQ-INT4\`
(Windows-side HF cache) — `launch_vllm.sh` points `HF_HOME` there so vLLM
reads directly from the already-downloaded files. **This is a slower disk
than WSL's native filesystem** (9P/DrvFs mount, not ext4) — weight loading
measured at ~40 seconds every restart because of this. See §7 for the
tradeoff and why it wasn't fixed.

**Alternative considered and abandoned (for now):**
`google/gemma-4-12B-it-qat-w4a16-ct` — Google's own official release,
symmetric quantization (no `zero_point` tensor, `observer: memoryless_minmax`
vs. the AWQ requant's `observer: mse`). This was tried specifically because
the AWQ checkpoint was hitting what looked like a deep architecture bug
(`AssertionError: Attempted to load weight (torch.Size([512])) into
parameter (torch.Size([256]))`), on the theory that the asymmetric
zero-point tensor's extra packing dimension was confusing vLLM's QKV
shard-loading logic. **The download stalled at ~1MB/s** (confirmed via a
direct `curl -r 0-52428800` range-request speed test against HF's Xet CDN,
not a client-side issue) and was abandoned once the `transformers==5.14.1`
downgrade turned out to fix the *original* AWQ checkpoint's crash too —
so the theory that it was AWQ-specific was wrong; it was the transformers
version all along (see §5's Chapter 2 in the journal for the full story).

**Should you revisit Google's checkpoint?** Possibly, if:
- Output quality from the AWQ requant is ever in question (no formal
  quality comparison was done this session — only a functional smoke test
  confirming it generates coherent text).
- Bandwidth to HF's CDN is better than it was during this session (worth
  re-testing the download speed before committing to a multi-hour wait).
- The file is single-shard (`model.safetensors`, ~10.26GB, `content-length:
  10264229896` confirmed via `curl -I`) — no sharding, so no chunked-load
  parallelism benefit either way.

To try it: `--vllm-model google/gemma-4-12B-it-qat-w4a16-ct` in
`launch_vllm.sh`'s `vllm serve` line, and don't set `HF_HOME` to the
Windows cache (let it download fresh into WSL's native `~/.cache/huggingface`,
which is much faster to read from than the Windows-mounted cache — see §7).

## 4. Every bug hit, in the order they were found, with root cause and fix

This is the reference list — see `docs/VLLM_SESSION_JOURNAL.md` for the
narrative version with more "why did this look reasonable at the time"
context.

### 4.1 MSYS path mangling (tooling, not vLLM)
**Symptom:** `bash: line 1: C:/Program: No such file or directory` when
running any `wsl -- bash -lc '/home/aliho/...'` command.
**Cause:** Git Bash/MSYS on Windows auto-translates leading-`/` arguments
before handing them to a native executable like `wsl.exe`.
**Fix:** Prefix every such command with `MSYS_NO_PATHCONV=1`.

### 4.2 Stale `EngineCore` subprocess survives `pkill`
**Symptom:** GPU memory usage after a "clean" relaunch is roughly double
what one model instance should need (~15GB instead of ~9GB); two engines
end up racing for VRAM.
**Cause:** vLLM's `EngineCore` subprocess renames its own process title —
`ps` shows it as `VLLM::EngineCore`, not `vllm serve ...` — so a
`pkill -f "vllm serve"` doesn't match it.
**Fix:** `launch_vllm.sh` kills both `"vllm serve"` and `"VLLM::EngineCore"`
patterns before every launch.

### 4.3 `RuntimeError: UVA is not available`
**Symptom:** Engine core fails at device init (`worker.init_device()`).
**Cause:** CUDA pinned/unified memory is disabled by default under WSL2 in
this vLLM version, gated behind `VLLM_WSL2_ENABLE_PIN_MEMORY` (see
`vllm/platforms/cuda.py::is_pin_memory_available()`). The *old* repo config
(`configs/vllm_launch.ps1`) sets `VLLM_USE_V1=0` as a workaround for this —
**that variable no longer exists** in this vLLM version (confirmed via
`WARNING: Unknown vLLM environment variable detected: VLLM_USE_V1` in the
logs; the V0 engine was removed entirely upstream at some point). Setting
it is a no-op now.
**Fix:** `export VLLM_WSL2_ENABLE_PIN_MEMORY=1`.

### 4.4 `AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute...`
**Symptom:** Crashes during config conversion, in
`vllm/transformers_utils/model_arch_config_convertor.py` and later (after a
narrow patch) again in `vllm/config/utils.py::getattr_iter`.
**Cause:** `transformers` 5.15.0 introduced `HeterogeneousConfigMixin` — a
stricter guard that raises this custom exception instead of just returning
a value when code does plain `getattr`/`hasattr` on a per-layer attribute of
a heterogeneous config. Gemma 4 genuinely has heterogeneous
`head_dim`/`num_key_value_heads` (sliding-attention layers use
`head_dim=256`, full-attention layers use `head_dim=512` — see §6 for the
full architecture breakdown). vLLM 0.27.x's config code predates this
transformers change and reads these attributes in many places via plain
`getattr`, so it raises everywhere, one callsite at a time as you patch
around it.
**Confirmed upstream bug:** [vllm-project/vllm#51744](https://github.com/vllm-project/vllm/issues/51744)
(open at time of writing, no owner assigned). Confirmed fix per that issue:
downgrade `transformers`.
**Fix applied:** `pip install "transformers==5.14.1"` in the WSL venv
(`/home/aliho/.python312/bin/pip install "transformers==5.14.1"`). This
fixes **every** callsite at once, unlike patching individual functions.
**Residual risk:** this pin is **not durable** — it's a bare `pip install`
in a hand-managed venv, recorded nowhere except this document and the
launch script's implicit dependency on it. If anyone runs a bare
`pip install --upgrade` in that venv, or the venv gets rebuilt, transformers
jumps back to whatever's latest and this entire failure mode returns. There
is no `requirements.txt` for this WSL venv to pin it in. **This is the
single most likely thing to silently break this setup in the future** —
see §8 for what to do about it.
**Narrower monkeypatches that predate finding the real fix** (superseded,
not needed if the transformers pin holds, kept in the repo in case the pin
ever needs bypassing again for some reason): `scripts/patch_vllm_gemma4_head_dim.py`
and `scripts/patch_vllm_heterogeneous_config.py`. Both directly edit the
installed vLLM source under `/home/aliho/.python312/lib/python3.12/site-packages/vllm/`
and are idempotent (safe to re-run). **Consider deleting both** next time
this area is touched — they add confusion about which fix is "the" fix
without adding safety margin, since the transformers pin already covers
everything they cover and more.

### 4.5 `AssertionError: Attempted to load weight (torch.Size([512])) into parameter (torch.Size([256]))`
**Symptom:** Crash during weight loading (`model.load_weights`), specific
to `q_norm`/`k_norm` parameters.
**Cause:** Not root-caused precisely in the moment — but it stopped
occurring entirely once `transformers==5.14.1` was applied, and it's
consistent with the same per-layer `head_dim` confusion as §4.4 (some code
path resolving a norm parameter's shape from a different, inconsistent
`head_dim` value than the path that loads its weight). This is what
originally triggered the detour into trying Google's checkpoint (§3) before
the transformers downgrade was found to fix it on the *original* checkpoint
too.
**Fix:** same as §4.4 — the transformers downgrade resolved this too.
**If this reappears** on a transformers version where 4.4's exact error
message isn't showing: suspect the same root cause anyway and try the
transformers pin before assuming it's a new, separate bug.

### 4.6 FlashInfer JIT sampler: three sequential toolchain failures
All three block `compile_or_warm_up_model()` → `warmup_kernels()` →
FlashInfer's `top_k_top_p_sampling_from_logits`, which JIT-compiles a CUDA
kernel on first use.

1. **`assert cuda_home is not None`** (in `vllm/third_party/deep_gemm` —
   actually just a *warning*, not fatal, this specific one, but the same
   underlying missing-`CUDA_HOME` issue recurs fatally below) —
   `CUDA_HOME` isn't set by default in this Python environment.
   **Fix:** the pip-installed `nvidia-cuda-nvcc` package puts a real `nvcc`
   binary at `.../site-packages/nvidia/cu13/bin/nvcc` — point `CUDA_HOME`
   at `.../site-packages/nvidia/cu13` (the parent of that `bin/`).
2. **`FileNotFoundError: ... 'ninja'`** — `ninja` was `pip`-installed
   (`ninja==1.13.0` confirmed present) but its binary at
   `/home/aliho/.python312/bin/ninja` isn't on `PATH` when the process is
   launched via an absolute interpreter path (`/home/aliho/.python312/bin/vllm`)
   rather than an activated venv. **Fix:** add `/home/aliho/.python312/bin`
   to `PATH` explicitly.
3. **`error: "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"`**
   — a genuine version mismatch: FlashInfer bundles its own CCCL
   (`cuda/std/__cccl/...`) headers under
   `flashinfer/data/cccl/libcudacxx/include/`, and they don't agree with
   the pip-installed `nvcc` (`nvidia-cuda-nvcc==13.3.73`) version's own
   toolkit headers. **Not fixed** — decided not worth chasing a toolchain
   version match for what's an optional fast-path kernel. **Fix applied
   instead:** `export VLLM_USE_FLASHINFER_SAMPLER=0`, falling back to
   vLLM's native (non-JIT) sampler implementation. No further crashes after
   this; server starts and serves normally.

**If you want the FlashInfer fast sampler working properly later:** the
real fix would be finding a `nvidia-cuda-nvcc` pip package version whose
bundled headers match what FlashInfer's vendored CCCL expects, or building
FlashInfer's kernels against a system-installed CUDA toolkit instead of the
pip-distributed one. Not attempted — the native sampler fallback works and
throughput testing (§6) showed no obvious sampler-side bottleneck (this is
entirely GPU-bound already; see the CPU-utilization finding).

## 5. Throughput: full data and how it was measured

**Test methodology:** `scripts/vllm_throughput_test.py` (canonical copy in
this repo; also exists as `~/throughput_test.py` in WSL, which may drift —
prefer the repo copy going forward and copy it into WSL to run it). Sends
concurrent `/v1/chat/completions` requests directly to the running server
(bypassing doc2md's own pipeline entirely) using:
- A **real** cropped region from `sample_docs/test.pdf` (`docs/vllm_throughput_test_region.png`,
  862×892px, generated via `doc2md.render.render_pdf` + `doc2md.crop.crop_region`)
- doc2md's **actual** system prompt and text-transcription prompt
  (`doc2md/prompts.py`'s `SYSTEM_PROMPT` and `_DEFAULT_TEXT_PROMPT`,
  duplicated into the test script rather than imported so the script has no
  dependency on the doc2md package)
- `max_tokens=300` per request (doc2md's own default is 2048; 300 was
  chosen to keep test cycles short while still exercising real decode-length
  behavior — measured completions averaged ~80 tokens, well under the cap)
- Fires N concurrent workers continuously for a fixed wall-clock duration
  (30s in all runs below), computing aggregate throughput as
  `total_completion_tokens / wall_clock_seconds`

**Important methodology note:** the *first* throughput measurement in this
session used doc2md's `PICTURE_PROMPT` (the diagram-vs-not classification
prompt) against a text-region image. That's a **meaningless** throughput
number — the correct response to that prompt against non-diagram content is
the ~5-token `NOT_DIAGRAM` sentinel, so the measurement was dominated by
prefill/network overhead, not decode. It measured 195 tok/s and was
discarded. **If you rerun this test, make sure you're using a prompt that
actually produces substantial output** (the current script's
`TEXT_PROMPT` does, at ~80 tokens/response against this test image).

**Results**, `--max-model-len 4096` fixed throughout, only `--max-num-seqs`
varied:

| `--max-num-seqs` = concurrency | Aggregate tok/s (client-measured) | Peak internal tok/s (vLLM's own `loggers.py` line) | GPU KV cache usage | Queueing? |
|---|---|---|---|---|
| 1 | 61.2 | — | — | n/a |
| 8 | 434.9 | — | — | no |
| 32 | 953.7 | 1129.3 | 14.3% | no |
| 64 | 1160.5 | 1399.0 | 40.7% | no (`Waiting: 0` throughout) |
| 128 | 1258.1 | 1510.8 | 71.1% | no (`Waiting: 0` throughout) |
| 192 | 1286.4 | 1461.6 | **99.6-99.9%** | **yes** — 14-30 requests waiting |

**The gap between "aggregate" and "peak internal" numbers** is measurement
methodology, not a real discrepancy: the client-measured aggregate averages
over the *whole* test window including startup ramp-up (first ~5-10s where
concurrency hasn't reached steady state yet), while vLLM's own periodic log
line reports throughput over just the preceding ~10s interval, so it
captures the true steady-state peak. Both are real; the internal number is
the more meaningful "sustained decode throughput" figure.

**Why 192 is the real cap and 128 was chosen as the running setting:** at
192, GPU KV cache usage genuinely saturates (~100%) and vLLM's scheduler
starts queueing new requests instead of running them immediately — the
textbook signature of hitting a hard resource limit. Throughput barely
improved over 128 (2%) while average per-request latency nearly doubled
(7.85s → 11.60s) — pure queueing delay with no compute benefit. 128 is the
last setting with **zero queueing** and already-near-peak throughput, so
it's the better operating point unless you specifically need to squeeze out
that last ~2% and don't mind the latency cost.

**CPU utilization check:** during the saturating 128-concurrency test,
`top -bn1` sampled 3 times mid-run showed **95.1-95.9% idle** across all 24
available WSL2 CPU cores (~4% average utilization = roughly one core's
worth of total work). This workload is unambiguously GPU-bound. There is no
CPU/thread-count lever to pull here — don't spend time on
`OMP_NUM_THREADS`, core pinning, or similar; it was checked, not assumed.

**Not tried / explicitly decided against:**
- **`--kv-cache-dtype fp8`** — deliberately not tested. vLLM's own FP8 KV
  cache blog post ([vllm.ai/blog/2026-04-22-fp8-kvcache](https://vllm.ai/blog/2026-04-22-fp8-kvcache))
  names four conditions under which to avoid FP8 KV cache, and **three
  apply directly to this workload**: contexts under ~7k tokens (doc2md's
  real requests run ~300-400 tokens total), `head_dim=256` (Gemma 4's
  sliding-attention layers use exactly this), and "many small
  sliding-window attention layers" (40 of Gemma 4's 48 layers are
  sliding-window). That same blog post never evaluated any vision/multimodal
  model at all. Separately, multimodal-specific literature (AKVQ-VL,
  KVCapsule — see §6) finds that *generic/uniform* KV quantization
  "overlooks attention saliency differences of multimodal tokens" and
  underperforms specialized asymmetric schemes, which vLLM doesn't ship. If
  you want to revisit this: it would need an actual output-quality A/B
  comparison against the bf16 baseline before trusting it for production
  use, not just a throughput number — the whole point of the caveats above
  is that FP8 KV cache's accuracy risk is workload-dependent and this
  workload trips several of the known-risky patterns.
- **Shorter `--max-model-len`** — currently 4096, but real requests measured
  at only ~300-400 total tokens (see §6 for the exact prompt/vision token
  breakdown). Lowering this (e.g. to 1024 or 1536) would free more KV cache
  budget per concurrent slot and could push the real cap (§5's 192→queueing
  point) higher. **Not tried this session** — a good next experiment.
- **Concurrency values between 128 and 192** (e.g. 144, 160, 176) to
  pinpoint more precisely where queueing starts. **Not tried** — 128 and
  192 were enough to establish the shape of the curve and pick an operating
  point; finer resolution wasn't judged worth the ~2-3 minutes per relaunch
  cycle given the user's explicit "don't min-max" instruction.
- **Multiple concurrent doc2md conversions** — all testing so far hit the
  server directly with synthetic load, not through doc2md's own pipeline
  and its `--concurrency` semaphore. See §9 for why this matters.

## 6. Architecture facts learned about this specific model

Gathered from the checkpoint's actual `config.json`, `model.safetensors.index.json`
tensor shapes, and public research — not assumptions:

- **Architecture class:** `Gemma4UnifiedForConditionalGeneration`
  (`model_type: gemma4_unified`) — a single shared transformer trunk
  processing text, image, and audio tokens together (early fusion), not a
  separate-encoder-plus-cross-attention design.
- **Text config:** `hidden_size=3840`, `num_hidden_layers=48`,
  `num_attention_heads=16`, `num_key_value_heads=8`,
  `intermediate_size=15360`, `vocab_size=262144`,
  `max_position_embeddings=131072` (architectural max; we run
  `--max-model-len 4096`, ~32x headroom below the real requirement — see
  §5's "not tried" list for why lowering it further is a live option).
- **Heterogeneous attention:** `layer_types` mixes `sliding_attention`
  (most layers, `head_dim=256`, `sliding_window=1024`) and `full_attention`
  (roughly 1 in 6 layers — a `global_head_dim=512` config field is used for
  these). This heterogeneity is the direct cause of §4.4's bug.
- **"Encoder-free" vision, quantitatively confirmed from this checkpoint's
  actual files:** the `model.vision_embedder.*` + `model.embed_vision.*`
  submodule (patch-dense projection, layernorms, positional embedding) is
  only ~59MB in the first safetensors shard, against ~5.3GB for that same
  shard's `language_model` weights — i.e., under 1% of total weight mass.
  Public research (Gemma 4 Technical Report,
  [arxiv.org/html/2607.02770v1](https://arxiv.org/html/2607.02770v1))
  confirms this is by design: a 550M-parameter ViT is replaced with "a
  single large matmul (35M parameters)" over raw image patches
  (48×3 RGB patches), with 2D positional embeddings added directly before
  a final LayerNorm — no separate frozen vision tower at all.
- **Vision token cost, measured directly (not from a spec sheet):** sent a
  real 862×892px region crop with the full system+text prompt, got back
  `prompt_tokens: 295`. Independently tokenizing just the system prompt
  (46 tokens) and text prompt (67 tokens) with this model's own tokenizer
  leaves ~295 - 113 - (~15 tokens of chat-template overhead) ≈ **~167
  vision tokens** for that image. That's notably cheap compared to
  ViT-tokenizer VLM architectures like LLaVA (~576 tokens per 336×336 tile) —
  consistent with the "encoder-free" design being genuinely token-efficient,
  not just weight-efficient.
- **Important nuance — this does NOT mean images are free of KV cache
  cost.** Because image tokens flow through the *same* causal
  self-attention as text tokens (no separate cross-attention mechanism with
  its own fixed-size memory), every vision token still occupies a KV cache
  slot exactly like a text token would. "Encoder-free" here means "no
  separate encoder weights" and "fewer tokens per image than typical
  ViT-based VLMs" — both real, valuable properties — but not "images don't
  consume KV cache." Don't let a future conversation about this
  architecture drift into assuming the latter.
- **KV cache architectural optimizations already baked into the model**
  (i.e., already reflected in the numbers §5 measured, not something to go
  chase separately): per the technical report, later layers reuse K/V
  projections from earlier layers of the same attention type ("KV cache
  sharing"), and p-RoPE (p=0.25) on global-attention layers reduces global
  KV cache by ~37.5%. These are why the 12B model's KV cache footprint is
  smaller than a naive per-layer-independent design would produce — but
  they're fixed properties of the checkpoint, not a runtime lever.

## 7. Known inefficiencies not fixed (deliberately, or by oversight)

- **Weight loading reads from the Windows-mounted HF cache
  (`/mnt/c/Users/aliho/.cache/huggingface`, a 9P/DrvFs filesystem), not
  WSL2's native ext4 filesystem.** Every restart pays ~40 seconds for this
  (`Loading weights took 39-44 seconds` consistently across every relaunch
  in this session) that a native-filesystem copy wouldn't. This is a
  **one-time cost per restart**, not a decode-throughput issue (once loaded,
  everything runs from GPU VRAM) — it wasn't fixed because it doesn't
  affect the throughput numbers in §5 and the server isn't expected to
  restart often. If restart frequency ever becomes a real cost (e.g.
  CI/automated testing that restarts vLLM per-run), copying the checkpoint
  to `~/.cache/huggingface` (WSL-native) and dropping `HF_HOME` from
  `launch_vllm.sh` would fix this — budget ~10GB of WSL2's disk (there's
  926GB free on the WSL virtual disk, so this is not a constraint).
- **`transformers==5.14.1` pin is not durable** — see §4.4's "residual
  risk" note. This is the most likely thing to silently break the whole
  setup later. See §8.
- **The two now-superseded monkeypatch scripts** (`scripts/patch_vllm_gemma4_head_dim.py`,
  `scripts/patch_vllm_heterogeneous_config.py`) are still in the repo,
  unused by the current setup (the transformers pin covers everything they
  cover). Not deleted this session — low priority, but worth cleaning up
  next time this area is touched so a future reader doesn't wonder which
  fix is "the real one."
- **`vllm_serve.log` grows unbounded** with no rotation. Not a problem yet
  at the scale this has been tested, but worth knowing if this server ends
  up staying up for a long time.
- **No process supervision / auto-restart.** If `vllm serve` crashes (OOM,
  driver reset, etc.) or the WSL2 VM restarts, nothing brings it back up
  automatically — you have to notice it's down and rerun `launch_vllm.sh`
  manually. Fine for interactive development, not fine for anything
  resembling production use.

## 8. Concrete next things to try

Roughly in order of how much value they'd add relative to effort, but not
a strict priority order — pick based on what you actually need next:

1. **Run doc2md against this server for real** (see §9 — this is the
   biggest remaining gap). `doc2md/cli.py`'s `--vllm-model` default has
   since been fixed to `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` (a later
   consolidation pass), so a plain `uv run doc2md sample_docs/test.pdf -o out/`
   now points at the right model without extra flags - but running that
   command and confirming the output actually looks right has still never
   been done. All throughput testing so far bypassed doc2md's own pipeline
   entirely.
2. **Pin `transformers==5.14.1` durably.** There's no requirements file for
   the WSL venv today. Consider: a small `requirements-wsl-vllm.txt` (or
   similar) checked into this repo documenting the exact pinned versions
   (`vllm==0.27.1`, `transformers==5.14.1`, plus whatever else matters), even
   though nothing currently installs *from* it automatically — at minimum
   it stops the pin from being tribal knowledge that only exists in this
   HANDOFF file.
3. **Run doc2md's own `--concurrency` at something matching the server's
   real capacity.** doc2md's default is `--concurrency 32` (bounds a
   client-side `asyncio.Semaphore` in `pipeline.py`); the server is
   configured for `--max-num-seqs 128`. A real document conversion with
   many regions would currently only ever have 32 requests in flight at
   once, well under what the server can actually absorb per §5's data.
   Worth testing `--concurrency 96` or `128` on a real multi-page/multi-region
   document to see if real-world (not synthetic) throughput actually scales
   the way the synthetic test predicts — image sizes, prompt lengths, and
   output lengths will all differ per-region in a real document, unlike the
   single fixed test image used in §5.
4. **Try `--max-model-len` values below 4096** (1024, 1536, 2048) to see if
   the real concurrency ceiling (currently ~150-190 before queueing) moves
   higher — per §5's "not tried" list, real requests only need ~300-400
   tokens of context, so 4096 is generously oversized and eating into KV
   cache budget that could otherwise support more concurrent slots.
5. **Formal output-quality comparison** between the current AWQ checkpoint
   and Google's official `w4a16-ct` checkpoint (§3), and separately between
   `bf16` and `fp8` KV cache if that's ever revisited (§5) — nothing this
   session validated output *quality*, only that the server functions and
   how fast it runs. A side-by-side comparison on a handful of real
   doc2md region crops (text, table, diagram) covering all three bucket
   types would close this gap.
6. **Investigate whether FlashInfer's JIT sampler can actually be made to
   work** (§4.6's third failure) — not because it's currently a bottleneck
   (it isn't; this workload is GPU-bound per §5's CPU-idle finding, and the
   native sampler fallback works correctly), but because if a future
   workload profile ever becomes sampler-bound (e.g. very high top-k/top-p
   diversity settings, or many parallel low-latency small requests where
   sampling overhead matters more relative to decode), having the faster
   path available would matter. Would need a `nvidia-cuda-nvcc` pip package
   version whose headers match what FlashInfer's vendored CCCL expects, or
   a system CUDA toolkit install instead of the pip-distributed one.
7. **Clean up the two superseded patch scripts** (§4.4, §7) — low effort,
   removes future confusion.
8. **Consider process supervision** for `vllm serve` if this ever needs to
   stay reliably up unattended (§7) — e.g. a WSL-side systemd unit, or a
   simple watchdog loop that reruns `launch_vllm.sh` on process exit.
9. **Finer-grained concurrency sweep** (144, 160, 176) if you want the
   exact queueing threshold rather than the current bracketing between 128
   (no queueing) and 192 (queueing) — see §5.
10. **Re-attempt Google's official checkpoint download** if/when bandwidth
    to HF's CDN is confirmed better than the ~1MB/s measured this session
    (test first with a `curl -r 0-52428800 <url> -o /dev/null -w
    "%{speed_download}\n"` range request before committing to the full
    ~10GB download) — see §3 for the exact repo id and why it might be
    worth it (symmetric quant, avoids the zero-point tensor entirely).

## 9. What's NOT done — read this before assuming the job is finished

- **doc2md has never actually converted a real document against this
  server.** Every measurement in §5 talks to `/v1/chat/completions`
  directly via `scripts/vllm_throughput_test.py`, bypassing
  `doc2md/pipeline.py`, `doc2md/vlm_client.py`, and the whole layout
  detection → crop → classify pipeline entirely. The server being fast and
  correct in isolation doesn't guarantee the full pipeline works end to end
  — different image sizes per region, the `--layout-engine`'s actual
  detected regions, `doc2md`'s specific prompt selection per bucket
  (`text_prompt()` picks title/formula/default prompts differently), and
  error-handling paths (`_process_region_safe_async`'s failure fallback)
  are all unexercised.
- **`doc2md/cli.py`'s `--vllm-model` default has been fixed** (a later
  consolidation pass changed it from `google/gemma-3-4b-it` to
  `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4`), but that fix was made without
  ever running a real end-to-end conversion against the live server - see
  the item above. The default now being correct doesn't by itself confirm
  the full pipeline produces good output.
- **No output-quality validation was done**, only functional (does it
  respond, does it not crash) and throughput (how fast) testing. See §8.5.
- **The server is not persistent/supervised.** It will not survive a WSL2
  restart, a machine reboot, or a crash without manual intervention. See
  §7.

## 10. How this was found (session narrative)

The sections above are the reference facts. This section is different: it's
the *order* things were discovered in, and *why* each wrong turn looked
reasonable at the time. If a new, different-looking vLLM crash shows up on
this stack later, the pattern-matching here (each fix exposing the next
layer of the onion) is probably more useful than the finished checklist
above.

**Outcome, for context:** working, verified end-to-end, sustaining ~1500
tok/s peak generation throughput at `--max-num-seqs 128`. Took six distinct
crash signatures and one dead end to get there.

### Chapter 1: the environment itself fights back

Before any vLLM-specific problem showed up, two purely mechanical issues
had to be solved:

1. **MSYS path mangling.** The Bash tool used in this session runs Git Bash
   (MSYS) on Windows. Any command that shells out to `wsl.exe` with an
   argument that *looks* like an absolute POSIX path — `/home/aliho/...` —
   gets silently rewritten by MSYS into a Windows path before `wsl.exe`
   ever sees it, because MSYS assumes it's translating a path for *itself*,
   not for the Linux side of a WSL call. `/home/aliho/.python312/bin/vllm`
   became `C:/Program Files/Git/home/aliho/.python312/bin/vllm`, and the
   resulting error (`bash: line 1: C:/Program: No such file or directory`)
   looks nothing like a path problem at first glance — it looks like the
   shell forgot how to parse its own command line. The fix is
   `MSYS_NO_PATHCONV=1` prefixed on every `wsl -- bash -lc '...'` call.
   Every command in this session that touches WSL uses this prefix for a
   reason — don't drop it.

2. **Leftover subprocess titles evade `pkill`.** vLLM's `EngineCore`
   subprocess renames itself away from the `vllm serve ...` command line
   that launched it (shows up in `ps` as `VLLM::EngineCore`). A `pkill -f
   "vllm serve"` between launches leaves it running, silently holding
   several GB of GPU memory from the *previous* attempt while a new attempt
   starts. This wasn't obvious until `nvidia-smi` showed ~15GB used right
   after a "clean" relaunch that should have used ~9GB — two engines were
   coexisting. `launch_vllm.sh` now kills both patterns explicitly.

Neither of these is a vLLM bug. They're friction specific to doing WSL2
GPU work through a Windows-hosted tool session, and they cost real time
before the actual investigation could start.

### Chapter 2: three different crashes that all traced back to one cause

The first real vLLM crash was `RuntimeError: UVA is not available` at
device init. The existing repo config (`configs/vllm_launch.ps1`) set
`VLLM_USE_V1=0` as a workaround for exactly this — a comment there
explained WSL2 doesn't support the UVA (unified virtual addressing) the V1
engine's `UvaBuffer` needs. Except: `VLLM_USE_V1` doesn't exist as a
recognized env var in this vLLM version anymore (`WARNING: Unknown vLLM
environment variable detected: VLLM_USE_V1`) — the V0 engine was removed
entirely at some point after that config was written. Setting it does
nothing. Chasing the wrong lever burned one full launch-and-wait cycle
(~3 minutes) before checking the actual cause: pinned/unified memory under
WSL2 is gated behind a *different*, still-current flag,
`VLLM_WSL2_ENABLE_PIN_MEMORY=1`, found by reading vLLM's own
`platforms/cuda.py::is_pin_memory_available()` source. Once set, the UVA
error disappeared for good.

The next crash was new: `AmbiguousGlobalPerLayerAttributeError: 'head_dim'
is a per-layer attribute...`, deep in vLLM's config-conversion code. First
instinct was to patch it locally — and that did work, twice, once for the
metadata read that crashes first (`Gemma4ModelArchConfigConvertor.get_head_size()`)
and then again for a *second* crash at a different callsite
(`getattr_iter` in `vllm/config/utils.py`) once the first patch was in.
Patching individual callsites one at a time as they surfaced is a losing
game against a codebase-wide assumption change — there was no way to know
how many more callsites existed without hitting each one. A web search
before writing a third patch found the actual answer:
[vllm-project/vllm#51744](https://github.com/vllm-project/vllm/issues/51744),
an open, already-diagnosed issue — `transformers` 5.15.0 introduced a
stricter per-layer-attribute guard that vLLM 0.27.x's code (written before
that guard existed) trips on everywhere it reads `head_dim` for a
heterogeneous model like Gemma 4 (which genuinely does mix `head_dim=256`
sliding-attention layers with `head_dim=512` full-attention layers).
Downgrading `transformers` to `5.14.1` fixed every instance of this at
once — the two source patches already applied became redundant, not wrong,
just unnecessary. (They've since been deleted from `scripts/` in a later
cleanup pass — see `handoffs/repo_cleanup_handoff.md`.)

That single `pip install transformers==5.14.1` also retroactively explained
a *third*, seemingly unrelated crash from earlier in the session: a plain
`AssertionError: Attempted to load weight (torch.Size([512])) into
parameter (torch.Size([256]))` during weight loading, which at the time
looked like a completely separate, deeper architecture bug in vLLM's
`Gemma4Attention` module — different enough that it triggered a detour into
downloading Google's official checkpoint (`google/gemma-4-12B-it-qat-w4a16-ct`)
on the theory that the community AWQ requant's asymmetric quantization
scheme was the actual cause. That download stalled at ~1MB/s (confirmed via
a direct `curl` range-request speed test) and was abandoned after
discovering the transformers downgrade fixed the *original* checkpoint too.
In hindsight the size-mismatch assertion was consistent with the same
per-layer `head_dim` confusion — some code path was sizing a norm parameter
from a different (wrong) resolved value than another path used to load its
weight — but this wasn't diagnosed cleanly in the moment; the downgrade
just made the problem stop occurring, which is a slightly less satisfying
but perfectly serviceable resolution.

**Lesson for next time:** when a config-metadata error and a completely
different-looking weight-loading error both show up on the same brand-new
model architecture close together in time, check whether they share an
upstream dependency version before assuming they're two separate bugs.

### Chapter 3: the last mile is always toolchain problems

With the engine actually starting and loading weights, the very last crash
was in FlashInfer's JIT-compiled sampling kernel — three sequential
failures, each one solved and immediately replaced by the next:

1. `assert cuda_home is not None` — `CUDA_HOME` wasn't set. Found the
   pip-installed `nvcc` binary under
   `nvidia/cu13/bin/` inside the venv's site-packages and pointed
   `CUDA_HOME` there.
2. `FileNotFoundError: ... 'ninja'` — `ninja` was `pip`-installed but its
   binary lives in the venv's `bin/` directory, which wasn't on `PATH` for
   a script launched via an absolute interpreter path rather than an
   activated venv. Added it to `PATH`.
3. `error: "CUDA compiler and CUDA toolkit headers are incompatible"` — a
   genuine version mismatch between FlashInfer's *bundled* CCCL headers and
   the pip-installed `nvcc`. This one didn't have a clean fix available in
   a reasonable amount of time, so the decision was to stop chasing a
   toolchain match for what's ultimately an optional fast path and just
   disable it: `VLLM_USE_FLASHINFER_SAMPLER=0`, falling back to vLLM's
   native (slower-to-JIT but always-available) sampler implementation.

That was the last blocker. The server started clean, served a real chat
completion, and the rest of the session was tuning and measurement rather
than debugging.

### Chapter 4: what "push it as far as it goes" actually looked like

The user's ask after it worked was explicitly *not* to fine-tune — just
confirm a safe upper bound. `--max-num-seqs` went 8 → 32 → 64 → 128 → 192,
each step re-launched (full ~2-3 minute reload each time: weight load from
the slower Windows-mounted cache, `torch.compile`, then CUDA graph capture
across a growing list of batch sizes) and load-tested with a *realistic*
request — a real cropped region from `sample_docs/test.pdf`, doc2md's real
system+text prompts, not a synthetic one-word probe. That distinction
mattered: the very first throughput measurement used doc2md's *picture*
prompt against a *text* region crop, which correctly triggered the
`NOT_DIAGRAM` sentinel response — five tokens, dominated by network/prefill
overhead, not decode. The number that came back (195 tok/s) was real but
meaningless for a decode-throughput question. Switching to the
text-transcription prompt against the same image produced ~80-token
responses and a completely different, much more informative set of
numbers.

The actual cap showed up unambiguously at 192: GPU KV cache usage hit
99.6-99.9% and requests started queueing (`Waiting: 14-30 reqs` in vLLM's
own periodic log line) for the first time in the whole sweep. Every
concurrency level below that had `Waiting: 0` the entire test. 128 was the
last setting with zero queueing and had already reached peak-adjacent
throughput (1258 tok/s aggregate, 1511 tok/s internal peak) — so that's
where it was left running, rather than the marginally-higher-but-queued
192.

CPU-core tuning, floated as a possible lever, was checked empirically
rather than argued about: `top` sampled during the saturating 128-concurrent
test showed 95%+ idle across all 24 available cores. Confirmed
GPU-bound, not worth pursuing.

### What would have gone faster with foresight

- Checking `pkill` actually killed everything (via `ps aux | grep -i
  vllm`, not just checking the launcher script's own exit code) after
  *every* relaunch, not just when memory numbers looked suspicious, would
  have caught the stale-`EngineCore` issue on the first occurrence instead
  of the third.
- Searching GitHub issues for the exact exception class name
  (`AmbiguousGlobalPerLayerAttributeError`) before writing the first
  monkeypatch would have found the transformers-version root cause
  immediately, skipping two patch-and-relaunch cycles.
- Testing the realistic (text-transcription) prompt from the very first
  throughput measurement, rather than the picture-classification prompt,
  would have skipped one wasted measurement.

None of these were unreasonable calls in the moment — they're only obvious
in hindsight, which is the whole point of writing this down.
