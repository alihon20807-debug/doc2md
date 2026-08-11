#!/usr/bin/env bash
# vLLM single-batch launch script for RTX 5080 (Milestone 4)
# Fix: VLLM_USE_V1=0 forces V0 engine — V1 engine uses UvaBuffer which requires
# Unified Virtual Addressing (UVA), unavailable in WSL2's virtualized CUDA layer.
export VLLM_USE_V1=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
/home/aliho/.python312/bin/vllm serve /mnt/c/Users/aliho/Documents/Codes/doc2md/models/gemma-4-12B-it-Q4_0.gguf \
  --load-format gguf \
  --tokenizer google/gemma-4-12b-it \  # corrected: GGUF metadata = gemma4, repo = google/gemma-4-12b-it
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
