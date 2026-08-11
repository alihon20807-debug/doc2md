#!/usr/bin/env bash
# Verified-working vLLM launch script for Gemma 4 12B AWQ on WSL2 (RTX 5080).
#
# This is a git-tracked backup of the actual, live launch script
# (`~/launch_vllm.sh` inside the WSL filesystem, NOT this repo) - checked in
# here specifically so it survives a WSL filesystem wipe or reinstall, per
# CLAUDE.md's "Running vLLM under WSL2" section. Run it from inside WSL, or
# from Windows via `MSYS_NO_PATHCONV=1 wsl -- bash -lc 'bash ~/launch_vllm.sh'`
# (the MSYS_NO_PATHCONV=1 prefix is required - see
# handoffs/vllm_guide_handoff.md section 1 for why).
#
# Paths below (`/home/aliho/...`) are specific to this machine's WSL2
# environment - a hand-built Python 3.12 interpreter at
# ~/.python312/, unrelated to this repo's own uv-managed venv. Adjust them
# if running on a different machine. Every non-obvious flag/env var here is
# explained in full in handoffs/vllm_guide_handoff.md (sections 2-4); this
# file is the "what to run," that file is the "why."
#
# This supersedes the previous configs/vllm_launch.ps1 and .sh (deleted in a
# consolidation pass): those loaded a local GGUF file directly via
# --load-format gguf, disagreed with each other on --tokenizer, set the
# dead VLLM_USE_V1 env var (removed upstream, confirmed via
# "WARNING: Unknown vLLM environment variable detected: VLLM_USE_V1"), used
# --max-num-seqs 1 instead of the measured sweet spot of 128, and the .sh
# variant had a bash line-continuation bug (a trailing comment after the
# line-continuing backslash silently truncated the command). None of that
# reflects the actually-working setup below.

set -x

pkill -9 -f "vllm serve" 2>/dev/null
# vLLM's EngineCore subprocess renames its own process title (shows up in
# `ps` as "VLLM::EngineCore", not "vllm serve ...") so it survives a plain
# `pkill -f "vllm serve"` and silently holds VRAM from the previous run,
# causing two engines to coexist on the next launch. Kill both patterns.
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 2

: > ~/vllm_serve.log

export HF_HOME=/mnt/c/Users/aliho/.cache/huggingface
# CUDA pinned/unified memory (UVA) is unavailable under WSL2 by default;
# without this, device init fails with "RuntimeError: UVA is not available".
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME=/home/aliho/.python312/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:/home/aliho/.python312/bin:$PATH"
# FlashInfer's JIT-compiled sampler fails to build in this env (its bundled
# CCCL headers don't match the pip-installed nvcc version) - not worth
# chasing a toolchain match for an optional fast path; the native sampler
# fallback works fine.
export VLLM_USE_FLASHINFER_SAMPLER=0

# --max-num-seqs 128: measured sweet spot - peak throughput with zero request
# queueing (192 saturates the GPU KV cache and starts queueing for only ~2%
# more throughput at nearly double the latency). See
# handoffs/vllm_guide_handoff.md section 5 for the full sweep data.
setsid nohup /home/aliho/.python312/bin/vllm serve cyankiwi/gemma-4-12B-it-qat-AWQ-INT4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --host 0.0.0.0 \
  --port 8000 > ~/vllm_serve.log 2>&1 < /dev/null &
disown

sleep 3
ps aux | grep vllm | grep -v grep
