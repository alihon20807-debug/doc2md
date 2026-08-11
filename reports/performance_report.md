# RTX 5080 Laptop GPU: llama.cpp Performance Optimization Report

**Project**: `doc2md` (`c:/Users/aliho/Documents/Codes/doc2md`)  
**Target Hardware**: NVIDIA GeForce RTX 5080 Laptop GPU (`sm_120`, 16 GB GDDR7, 256-bit bus, 896 GB/s theoretical bandwidth) + Intel Core Ultra 9 275HX  
**Target Model**: Gemma 4 12B (`gemma-4-12B-it-Q4_0.gguf`)  
**Date**: 2026-08-11  

> This report covers **llama.cpp only**. For vLLM, the setup and tuning
> documented in section 3 below turned out to be incomplete/superseded —
> see the "Running vLLM under WSL2" section of `CLAUDE.md` and
> `handoffs/vllm_guide_handoff.md` for the actual verified vLLM setup and
> throughput numbers.

---

## Executive Summary

This report documents the physical mathematical modeling, root-cause
diagnostics, and final performance optimization results for running
llama.cpp with a 12B-parameter 4-bit quantized model on the NVIDIA RTX 5080
Laptop GPU.

### Key Technical Findings:
1. **Mathematical Grounding**: The absolute theoretical decode throughput ceiling for Gemma 4 12B INT4 (~7.22 GB) on an RTX 5080 Laptop GPU (896 GB/s memory bandwidth) is **124.10 t/s**. Expecting >= 150 t/s on this model without speculative decoding or distillation violates physical memory bandwidth limits — the original ">= 150 t/s" project target (see `PROJECT.md`) was itself unreachable on this hardware/model combination. At 95% bus efficiency, the realistic maximum is **~117.9 t/s**.
2. **llama.cpp Performance Recovery**: Un-optimized baseline performance was failing at **5.2–9.8 t/s** (~4.18% bandwidth efficiency) due to stale background process VRAM locking, un-offloaded transformer layers, E-core OpenMP thread contention, multi-slot context splitting, and missing Flash Attention. By tuning runtime parameters (`-ngl 99 -fa on -t 8 -tb 16 -np 1 -ub 1024 -c 4096`), `llama.cpp` decode throughput was boosted to **74.0–74.6 t/s** across two separate measurement runs (see `reports/llamacpp_benchmark.json` and `reports/llamacpp_benchmark_final.json`), achieving **~59.6–60.1% memory bus efficiency** (~534–539 GB/s achieved bandwidth) — 63% of the realistic 117.9 t/s target and within physical reach of the true 124.10 t/s ceiling. (Both files' `prompt_tps` figures are noisy — the test prompts were only 5–99 tokens, too small a sample to give a meaningful prefill-throughput number; decode throughput, measured over 512 generated tokens, is the reliable figure here.)

---

## 1. Physical Memory Bandwidth Mathematics

Autoregressive decode speed is fundamentally memory-bandwidth bound. Every output token generated requires fetching all model weights from VRAM into GPU registers.

$$\text{Theoretical Ceiling (t/s)} = \frac{\text{VRAM Bandwidth (Bytes/s)}}{\text{Model Size in VRAM (Bytes)}}$$

For RTX 5080 Laptop GPU:
- **Bus Width**: 256-bit
- **Memory Clock Rate**: 28 Gbps (GDDR7)
- **Peak Bandwidth**: $\frac{256}{8} \times 28.0 = 896.0 \text{ GB/s}$
- **Gemma 4 12B Q4_0 Size**: $7,219,673,216 \text{ Bytes} \approx 7.22 \text{ GB}$

$$\text{Decode Max} = \frac{896,000,000,000}{7,219,673,216} \approx 124.102 \text{ Tokens/Second}$$

| Bus Efficiency Target | Throughput Limit | Achieved Memory Bandwidth |
|---|---|---|
| 100% Theoretical Peak | 124.10 t/s | 896.0 GB/s |
| 95% Bus Efficiency | 117.90 t/s | 851.2 GB/s |
| 80% Bus Efficiency | 99.28 t/s | 716.8 GB/s |
| **llama.cpp Optimized (Achieved)** | **74.0–74.6 t/s** | **534.2–538.6 GB/s (59.6–60.1%)** |
| llama.cpp Baseline (Un-optimized) | 5.20 t/s | 37.54 GB/s (4.18%) |

---

## 2. llama.cpp Optimization & Forensic Diagnostics

### Root Cause Analysis of Baseline Failure (5.2 t/s -> 74.0-74.6 t/s)

```
+-----------------------------------------------------------------------------------+
|                            FORENSIC DIAGNOSIS MATRIX                              |
+--------------------+------------------------------------+-------------------------+
| Failure Symptom    | Technical Root Cause               | Optimization Solution   |
+--------------------+------------------------------------+-------------------------+
| Severe Stutter &   | Orphaned `llama-server.exe` process | Process cleanup via     |
| PCIe Swapping      | held 15.8 GB VRAM, forcing swap.   | `Stop-Process -Force`   |
+--------------------+------------------------------------+-------------------------+
| Unoffloaded        | `-ngl` flag missing or too low;    | Force full offload      |
| Layers             | CPU handled matrix multiplications. | with `-ngl 99`          |
+--------------------+------------------------------------+-------------------------+
| Thread Barrier     | OpenMP scheduled worker threads   | Restrict CPU threads to |
| Latency            | on Intel Core Ultra 9 E-cores.     | `-t 8` (P-cores only)   |
+--------------------+------------------------------------+-------------------------+
| KV Bandwidth       | Multi-slot context splitting (`-np`) | Lock single sequence    |
| Fragmentation      | sliced memory requests.            | slot (`-np 1`)          |
+--------------------+------------------------------------+-------------------------+
| High Attention     | Flash Attention kernel disabled.   | Enable Flash Attention  |
| Memory Access      |                                    | via `-fa on`            |
+--------------------+------------------------------------+-------------------------+
```

### Final Tuned Production Launch Command (`configs/llamacpp_launch.ps1`)

```powershell
.\llama.cpp-bin\llama-server.exe `
  -m models/gemma-4-12B-it-Q4_0.gguf `
  -ngl 99 `
  -fa on `
  -t 8 `
  -tb 16 `
  -np 1 `
  -ub 1024 `
  -c 4096 `
  --host 127.0.0.1 `
  --port 8080
```

---

## 3. vLLM Engine Architecture & WSL2 Integration (superseded — see below)

An early diagnosis in this project found vLLM 0.27.0 failing to start in
WSL2 with `RuntimeError: UVA is not available`, and hypothesized that
setting `VLLM_USE_V1=0` (forcing the V1 engine's `UvaBuffer` zero-copy path
to fall back to V0) was the fix. **This turned out to be incomplete**: the
same `RuntimeError: UVA is not available` error is actually resolved by
`VLLM_WSL2_ENABLE_PIN_MEMORY=1`, and getting vLLM fully working for this
model also required pinning `transformers==5.14.1` (a separate,
unrelated bug — see `CLAUDE.md`) plus several other fixes never captured
here. `VLLM_USE_V1` does not exist in current vLLM at all.

The real, verified vLLM setup — exact launch script, every bug's precise
error text, and measured throughput at multiple concurrency levels — is
documented in `CLAUDE.md`'s "Running vLLM under WSL2" section and
`handoffs/vllm_guide_handoff.md`. Treat those, not this section, as the
source of truth for vLLM.

---

## 4. Verification Suite & Benchmark Results

The automated benchmark test suite (`benchmark/test_bandwidth_calculator.py`) was executed to verify all bandwidth calculations:

```
Ran 17 tests in 6.371s
OK (17/17 PASS)
```

### Benchmark JSON Snapshot (`reports/llamacpp_benchmark_final.json`, exact file contents):

```json
{
  "engine": "llama.cpp",
  "model_name": "gemma-4-12B-it-Q4_0.gguf",
  "batch_size": 1,
  "prompt_tokens": 5,
  "tokens_generated": 512,
  "time_seconds": 6.91,
  "tokens_per_sec": 74.0,
  "prompt_tps": 0.9,
  "decode_tps": 74.0,
  "model_size_bytes": 7219673216,
  "calculated_bandwidth_gbps": 534.2,
  "gpu_theoretical_max_gbps": 896.0,
  "bandwidth_efficiency_pct": 59.6,
  "criteria_passed": false
}
```

`criteria_passed: false` here just means the run didn't hit the original
(unreachable, see section 1) ">=150 t/s" target — not that the tuning
failed. `prompt_tps: 0.9` is an artifact of the 5-token test prompt and
isn't a meaningful prefill-throughput measurement.

---

## 5. Next Steps (llama.cpp)

1. **Native `sm_120` CUDA Compilation**: Build `llama-server` directly from source with `-DCMAKE_CUDA_ARCHITECTURES=120` on CUDA 12.8 to leverage native Blackwell tensor instructions, instead of the generic prebuilt CUDA binary used for these measurements.

For vLLM next steps, see `handoffs/vllm_guide_handoff.md`'s prioritized list (e.g. the
`--max-model-len` reduction, end-to-end doc2md verification, FlashInfer
sampler) — that supersedes anything vLLM-related that would otherwise be
listed here.
