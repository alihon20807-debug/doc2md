# Getting vLLM working with Gemma 4 12B AWQ on WSL2 — a session journal

**Date:** 2026-08-11
**Goal:** get vLLM actually serving `gemma-4-12B` (AWQ INT4, QAT) for doc2md's
image-description workload, aiming for maximum aggregate throughput.
**Outcome:** working, verified end-to-end, sustaining ~1500 tok/s peak
generation throughput at `--max-num-seqs 128`. Took six distinct crash
signatures and one dead end to get there. This is the story of how, in
order, for whoever reads this next (including a future instance of me).

## Why this is worth reading, not just the handoff doc

`handoffs/vllm_guide_handoff.md` has the reference material — exact commands,
exact env vars, the full bug list. This file is different: it's the
*order* things were discovered in, and *why* each wrong turn looked
reasonable at the time. If you hit a new, different-looking vLLM crash on
this stack later, the pattern-matching here (each fix exposing the next
layer of the onion) is probably more useful than the finished checklist.

## Chapter 1: the environment itself fights back

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

## Chapter 2: three different crashes that all traced back to one cause

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

## Chapter 3: the last mile is always toolchain problems

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

## Chapter 4: what "push it as far as it goes" actually looked like

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

## What would have gone faster with foresight

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
