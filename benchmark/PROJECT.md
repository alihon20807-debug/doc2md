# Project: RTX 5080 vLLM & llama.cpp High-Performance Benchmarking

## Architecture
- Target GPU: NVIDIA GeForce RTX 5080 Laptop GPU (16 GB GDDR7, 256-bit bus width, `sm_120`)
- Inference Engines: llama.cpp (CUDA backend), vLLM (WSL2 PyTorch/CUDA backend)
- Model Scope: 12B parameter 4-bit quantized model (`gemma-4-12B-it-Q4_0.gguf`, 6.71 GiB / 7.22 GB)
- Benchmarking Harness: Automated script harness measuring single-batch decode throughput (tokens/sec), memory bandwidth utilization (GB/s), and percentage of theoretical maximum memory bandwidth for RTX 5080.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration & Diagnostics | Assess GPU environment, drivers, CUDA setup, inspect existing llama.cpp/vLLM state, identify bottlenecks & misconfigurations | None | DONE |
| 2 | Benchmark Harness & Math Engine | Create automated benchmark suite, mathematical GPU bandwidth calculator, and test harness | M1 | DONE |
| 3 | llama.cpp Performance Optimization | Configure/tune llama.cpp (CUDA backend, flash attention, full layer offload, optimal thread count, batching) to achieve >=150 t/s | M1, M2 | DONE |
| 4 | vLLM Setup & Optimization | Configure/tune vLLM (WSL2, CUDA backend, flash attention, tensor parallelism, memory fraction) to achieve max hardware bandwidth | M1, M2 | SUPERSEDED — this milestone's own run stalled (see below) |
| 5 | Audit, Verification & Report | Run forensic audit, execute full verification benchmark, write comprehensive final markdown report | M3, M4 | NOT DONE via this milestone |

M4/M5 as tracked by this harness were never actually finished — the agent
run driving them (`.agents/teamwork_preview_worker_m4`) stalled partway
through and no reviewer/auditor ever validated its output. A benchmark file
it produced (`reports/benchmark_results_vllm.json`, claiming a physically
impossible >100% bandwidth efficiency) has since been deleted for that
reason. The vLLM setup that *did* end up working was done later, in a
separate debugging session unrelated to this milestone tracker — see
`CLAUDE.md`'s "Running vLLM under WSL2" section and
`handoffs/vllm_guide_handoff.md` for the
real, verified setup, gotchas, and throughput numbers. No formal audit/report
(M5) was ever produced for vLLM; `reports/performance_report.md` covers
llama.cpp (M3) only.

## Interface Contracts
### Benchmark Harness ↔ Engines
- Command line execution & HTTP API benchmarking for both engines.
- Output JSON Schema — illustrative only, values below are made up for
  shape, not a real result (a previous version of this example used values
  that exceeded 100% of theoretical bandwidth and ended up copy-pasted into
  an actual `reports/*.json` file as if it were a real measurement; that
  file has since been deleted — don't reuse fabricated-looking numbers like
  that in a real report):
```json
{
  "engine": "llama.cpp | vLLM",
  "model_name": "gemma-4-12B-it-Q4_0.gguf",
  "batch_size": 1,
  "prompt_tokens": 128,
  "tokens_generated": 512,
  "time_seconds": 6.91,
  "tokens_per_sec": 74.0,
  "model_size_bytes": 7219673216,
  "calculated_bandwidth_gbps": 534.2,
  "gpu_theoretical_max_gbps": 896.0,
  "bandwidth_efficiency_pct": 59.6
}
```

## Code Layout
- `benchmark/`: Benchmark runner scripts, HTTP benchmarker, and bandwidth calculation modules
- `configs/`: Engine launch parameter configurations (`configs/llamacpp_launch.ps1`, `configs/vllm_launch.sh`)
- `reports/`: Benchmark output JSONs and final optimization markdown report
