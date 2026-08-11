"""Programmatic & CLI Benchmark Runner for llama.cpp and vLLM HTTP Servers.

Executes single-batch LLM inference benchmarks, measures exact generation time,
calculates decode throughput (tokens/sec), evaluates achieved GPU memory bandwidth
and efficiency using BandwidthCalculator, and writes results to JSON format matching
the PROJECT.md schema specification.
"""

import os
import sys
import time
import json
import argparse
import urllib.request
from typing import Dict, Any, Optional

from benchmark.bandwidth_calculator import (
    BandwidthCalculator,
    get_theoretical_max_bandwidth,
    DEFAULT_MODEL_SIZE_BYTES,
)

# Default configurations
DEFAULT_LLAMA_CPP_URL = "http://127.0.0.1:8080"
DEFAULT_VLLM_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL_NAME = "gemma-4-12B-it-Q4_0.gguf"
DEFAULT_MODEL_PATH = os.path.join("models", "gemma-4-12B-it-Q4_0.gguf")
DEFAULT_OUTPUT_PATH = os.path.join("reports", "benchmark_result_latest.json")

# Standard benchmark prompt calibrated for ~128 prompt tokens
BENCHMARK_PROMPT_128 = (
    "You are an advanced AI assistant performing system benchmarking on GPU hardware. "
    "Analyze the computational efficiency of autoregressive language model decoding on modern GPU architectures "
    "such as NVIDIA Blackwell. Focus specifically on memory bandwidth utilization, tensor core occupancy, "
    "and latency breakdown between prompt processing phase (prefill) and token generation phase (decode). "
    "Provide a structured technical response discussing memory bus width, GDDR7 transfer rates, cache hierarchy "
    "optimizations, and flash attention implementations for single-batch inference workloads. "
    "Begin your technical breakdown now."
)


def get_model_file_size(model_path: str = DEFAULT_MODEL_PATH) -> int:
    """Returns exact model weight size in bytes from disk or fallback baseline."""
    if os.path.isfile(model_path):
        try:
            size = os.path.getsize(model_path)
            if size > 0:
                return size
        except Exception:
            pass

    # Check relative to working directory or fallback
    alt_path = os.path.join(os.getcwd(), model_path)
    if os.path.isfile(alt_path):
        try:
            size = os.path.getsize(alt_path)
            if size > 0:
                return size
        except Exception:
            pass

    return DEFAULT_MODEL_SIZE_BYTES


def make_streaming_http_request(
    url: str, endpoint: str, payload: dict, timeout: float = 120.0
) -> Dict[str, Any]:
    """Sends HTTP POST request with stream=True and parses SSE chunks.

    Records exact timestamp when the first chunk arrives (t_first_chunk / TTFT)
    and when the final chunk arrives (t_final).

    Returns:
        Dictionary with timing metrics, token counts, usage/timings, and raw chunks.
    """
    full_url = f"{url.rstrip('/')}/{endpoint.lstrip('/')}"
    stream_payload = dict(payload)
    stream_payload["stream"] = True

    data_bytes = json.dumps(stream_payload).encode("utf-8")
    req = urllib.request.Request(
        full_url,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    t_first_chunk: Optional[float] = None
    t_final: Optional[float] = None

    raw_chunks = []
    text_chunks_count = 0
    usage_info: Dict[str, Any] = {}
    timings_info: Dict[str, Any] = {}

    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue

            if line.startswith("data:"):
                now = time.perf_counter()
                data_content = line[5:].strip()

                if data_content == "[DONE]":
                    t_final = now
                    break

                try:
                    chunk_json = json.loads(data_content)
                    raw_chunks.append(chunk_json)

                    if t_first_chunk is None:
                        t_first_chunk = now
                    t_final = now

                    # Count text content choices (vLLM / OpenAI format)
                    if "choices" in chunk_json:
                        choices = chunk_json["choices"]
                        if isinstance(choices, list) and len(choices) > 0:
                            text_chunks_count += 1

                    # Count content key (llama.cpp native format)
                    if "content" in chunk_json and "choices" not in chunk_json:
                        text_chunks_count += 1

                    # Extract usage metadata if present
                    if "usage" in chunk_json and isinstance(chunk_json["usage"], dict):
                        usage_info.update(chunk_json["usage"])

                    # Extract timings metadata if present (llama.cpp native)
                    if "timings" in chunk_json and isinstance(chunk_json["timings"], dict):
                        timings_info.update(chunk_json["timings"])

                except json.JSONDecodeError:
                    pass

    t_end_total = time.perf_counter()

    if t_first_chunk is None:
        t_first_chunk = t_end_total
    if t_final is None:
        t_final = t_end_total

    ttft_seconds = t_first_chunk - t0
    decode_time_seconds = t_final - t_first_chunk

    return {
        "t0": t0,
        "t_first_chunk": t_first_chunk,
        "t_final": t_final,
        "ttft_seconds": ttft_seconds,
        "decode_time_seconds": decode_time_seconds,
        "wall_time": t_end_total - t0,
        "text_chunks_count": text_chunks_count,
        "usage": usage_info,
        "timings": timings_info,
        "raw_chunks": raw_chunks,
    }


def _build_benchmark_result(
    res: Dict[str, Any], prompt_tokens: int, tokens_generated: int
) -> Dict[str, Any]:
    """Computes prompt/decode throughput from a streaming request's timing
    data and assembles the common result dict shared by every benchmark
    runner (llama.cpp native, llama.cpp OpenAI-compatible fallback, vLLM).
    """
    ttft = res["ttft_seconds"]
    decode_time = res["decode_time_seconds"]

    prompt_tps = (prompt_tokens / ttft) if ttft > 0 else 0.0

    if decode_time > 0 and tokens_generated > 1:
        decode_tps = (tokens_generated - 1) / decode_time
    elif decode_time > 0 and tokens_generated == 1:
        decode_tps = 1.0 / decode_time
    else:
        decode_tps = 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "tokens_generated": tokens_generated,
        "ttft_seconds": ttft,
        "decode_time_seconds": decode_time,
        "time_seconds": decode_time,
        "prompt_tps": prompt_tps,
        "decode_tps": decode_tps,
        "tokens_per_sec": decode_tps,
        "raw_response": res["raw_chunks"],
    }


def run_llamacpp_benchmark(
    base_url: str = DEFAULT_LLAMA_CPP_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_tokens_target: int = 128,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Runs single-batch benchmark against llama.cpp HTTP server using SSE streaming."""
    native_payload = {
        "prompt": BENCHMARK_PROMPT_128,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    try:
        res = make_streaming_http_request(
            base_url, "/completion", native_payload, timeout=timeout
        )

        timings = res["timings"]
        prompt_tokens = timings.get("prompt_n", prompt_tokens_target)
        tokens_generated = timings.get("predicted_n", res["text_chunks_count"])

        return _build_benchmark_result(res, prompt_tokens, tokens_generated)

    except Exception:
        # Fallback to OpenAI compatible endpoint /v1/completions
        v1_payload = {
            "model": model_name,
            "prompt": BENCHMARK_PROMPT_128,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        res = make_streaming_http_request(
            base_url, "/v1/completions", v1_payload, timeout=timeout
        )

        usage = res["usage"]
        prompt_tokens = usage.get("prompt_tokens", prompt_tokens_target)
        tokens_generated = usage.get("completion_tokens", res["text_chunks_count"])

        return _build_benchmark_result(res, prompt_tokens, tokens_generated)


def run_vllm_benchmark(
    base_url: str = DEFAULT_VLLM_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_tokens_target: int = 128,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Runs single-batch benchmark against vLLM HTTP server using SSE streaming."""
    payload = {
        "model": model_name,
        "prompt": BENCHMARK_PROMPT_128,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    res = make_streaming_http_request(
        base_url, "/v1/completions", payload, timeout=timeout
    )

    usage = res["usage"]
    prompt_tokens = usage.get("prompt_tokens", prompt_tokens_target)
    tokens_generated = usage.get("completion_tokens", res["text_chunks_count"])

    return _build_benchmark_result(res, prompt_tokens, tokens_generated)


def run_mock_benchmark(
    engine: str = "llama.cpp",
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_tokens: int = 128,
    max_tokens: int = 512,
    simulated_tps: float = 100.0,  # arbitrary synthetic value for dry-run testing, not a real measurement
    model_size_bytes: int = DEFAULT_MODEL_SIZE_BYTES,
) -> Dict[str, Any]:
    """Generates mock benchmark execution metrics for testing/verification."""
    decode_time_seconds = round((max_tokens - 1) / simulated_tps, 4) if simulated_tps > 0 else 1.0
    ttft_seconds = 0.05
    prompt_tps = round(prompt_tokens / ttft_seconds, 1)

    return {
        "prompt_tokens": prompt_tokens,
        "tokens_generated": max_tokens,
        "ttft_seconds": ttft_seconds,
        "decode_time_seconds": decode_time_seconds,
        "time_seconds": decode_time_seconds,
        "prompt_tps": prompt_tps,
        "decode_tps": simulated_tps,
        "tokens_per_sec": simulated_tps,
        "mock": True,
    }


def run_benchmark(
    engine: str = "llama.cpp",
    url: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    model_path: str = DEFAULT_MODEL_PATH,
    batch_size: int = 1,
    prompt_tokens_target: int = 128,
    max_tokens: int = 512,
    gpu_config: str = "auto",
    output_path: str = DEFAULT_OUTPUT_PATH,
    dry_run: bool = False,
    simulated_tps: Optional[float] = None,
) -> Dict[str, Any]:
    """Main benchmark entry point supporting llama.cpp and vLLM HTTP benchmarking.

    Args:
        engine: 'llama.cpp' or 'vLLM' / 'vllm'
        url: Server HTTP base URL (defaults to http://127.0.0.1:8080 or :8000)
        model_name: Name of model for reporting
        model_path: Path to model file for weight byte size calculation
        batch_size: Inference batch size (default 1)
        prompt_tokens_target: Target prompt token count (default 128)
        max_tokens: Target decode token count (default 512)
        gpu_config: RTX 5080 GPU preset ('auto', 'desktop', 'laptop_896', 'laptop_768')
        output_path: Destination JSON report file path
        dry_run: If True, uses mock execution metrics without calling live HTTP server
        simulated_tps: Optional t/s override for mock dry-run

    Returns:
        Structured result dictionary formatted per PROJECT.md schema.
    """
    engine_norm = engine.strip().lower()
    if "llama" in engine_norm:
        engine_label = "llama.cpp"
        server_url = url or DEFAULT_LLAMA_CPP_URL
    elif "vllm" in engine_norm:
        engine_label = "vLLM"
        server_url = url or DEFAULT_VLLM_URL
    else:
        engine_label = engine
        server_url = url or DEFAULT_LLAMA_CPP_URL

    model_bytes = get_model_file_size(model_path)

    if dry_run:
        tps = simulated_tps if simulated_tps is not None else 100.0  # arbitrary synthetic default, not a real measurement
        exec_metrics = run_mock_benchmark(
            engine=engine_label,
            model_name=model_name,
            prompt_tokens=prompt_tokens_target,
            max_tokens=max_tokens,
            simulated_tps=tps,
            model_size_bytes=model_bytes,
        )
    else:
        if "llama" in engine_norm:
            exec_metrics = run_llamacpp_benchmark(
                base_url=server_url,
                model_name=model_name,
                prompt_tokens_target=prompt_tokens_target,
                max_tokens=max_tokens,
            )
        else:
            exec_metrics = run_vllm_benchmark(
                base_url=server_url,
                model_name=model_name,
                prompt_tokens_target=prompt_tokens_target,
                max_tokens=max_tokens,
            )

    # Compute bandwidth metrics using BandwidthCalculator
    calculator = BandwidthCalculator(gpu_config=gpu_config)
    calc_res = calculator.calculate(
        tokens_per_sec=exec_metrics["tokens_per_sec"],
        model_weight_bytes=model_bytes,
    )

    # Build report dict matching PROJECT.md JSON schema
    report = {
        "engine": engine_label,
        "model_name": model_name,
        "batch_size": batch_size,
        "prompt_tokens": exec_metrics["prompt_tokens"],
        "tokens_generated": exec_metrics["tokens_generated"],
        "time_seconds": round(exec_metrics["time_seconds"], 2),
        "tokens_per_sec": round(exec_metrics["tokens_per_sec"], 1),
        "prompt_tps": round(exec_metrics.get("prompt_tps", 0.0), 1),
        "decode_tps": round(exec_metrics.get("decode_tps", exec_metrics["tokens_per_sec"]), 1),
        "model_size_bytes": model_bytes,
        "calculated_bandwidth_gbps": round(calc_res["calculated_bandwidth_gbps"], 1),
        "gpu_theoretical_max_gbps": round(calc_res["gpu_theoretical_max_gbps"], 1),
        "bandwidth_efficiency_pct": round(calc_res["bandwidth_efficiency_pct"], 1),
        "criteria_passed": calc_res["criteria_passed"],
    }

    # Ensure target output directory exists and save JSON report
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Automated Benchmark Harness & Bandwidth Calculation Engine for doc2md"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="llama.cpp",
        choices=["llama.cpp", "llamacpp", "vllm", "vLLM"],
        help="Target inference engine (llama.cpp or vLLM)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="HTTP server base URL (default: http://127.0.0.1:8080 for llama.cpp, :8000 for vllm)",
    )
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model filename / identifier",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to model weights file on disk",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Inference batch size (default: 1)"
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=128,
        help="Target prompt token length (default: 128)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max decode tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--gpu-config",
        type=str,
        default="auto",
        help="RTX 5080 GPU spec preset ('auto', 'desktop', 'laptop_896', 'laptop_768')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON report output path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run mock benchmark without connecting to live server",
    )
    parser.add_argument(
        "--simulated-tps",
        type=float,
        default=None,
        help="Simulated tokens/sec for dry-run testing",
    )

    args = parser.parse_args()

    try:
        report = run_benchmark(
            engine=args.engine,
            url=args.url,
            model_name=args.model_name,
            model_path=args.model_path,
            batch_size=args.batch_size,
            prompt_tokens_target=args.prompt_tokens,
            max_tokens=args.max_tokens,
            gpu_config=args.gpu_config,
            output_path=args.output,
            dry_run=args.dry_run,
            simulated_tps=args.simulated_tps,
        )
        print(f"Benchmark completed successfully for engine: {report['engine']}")
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"Benchmark execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
