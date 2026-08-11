"""Unit Tests for RTX 5080 Bandwidth Calculator & Benchmark Harness.

Executes verification of mathematical calculations, GPU spec presets, auto-detection,
validation threshold checks, BandwidthCalculator class, and benchmark runner output schema.
"""

import os
import json
import tempfile
import unittest
import unittest.mock
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from benchmark.bandwidth_calculator import (
    calculate_achieved_bandwidth,
    calculate_bandwidth_efficiency,
    get_theoretical_max_bandwidth,
    detect_gpu_config,
    validate_criteria,
    BandwidthCalculator,
    RTX5080_SPECS,
)
from benchmark.run_benchmark import run_benchmark, get_model_file_size


class TestBandwidthCalculator(unittest.TestCase):
    """Test suite for memory bandwidth math engine and RTX 5080 spec logic."""

    def test_calculate_achieved_bandwidth_basic(self):
        """Verify achieved bandwidth calculation formula: (t/s * bytes) / 1e9."""
        # 150 t/s with 6.0 GB weights = 900.0 GB/s
        self.assertAlmostEqual(
            calculate_achieved_bandwidth(150.0, 6_000_000_000), 900.0, places=4
        )

        # 150.1 t/s with 7205759404 bytes
        expected_gbps = (150.1 * 7205759404) / 1e9
        self.assertAlmostEqual(
            calculate_achieved_bandwidth(150.1, 7205759404), expected_gbps, places=4
        )

        # Zero throughput
        self.assertEqual(calculate_achieved_bandwidth(0.0, 6_000_000_000), 0.0)

    def test_calculate_achieved_bandwidth_invalid_inputs(self):
        """Verify ValueError raises on negative inputs."""
        with self.assertRaises(ValueError):
            calculate_achieved_bandwidth(-1.0, 6_000_000_000)

        with self.assertRaises(ValueError):
            calculate_achieved_bandwidth(150.0, -100)

    def test_calculate_bandwidth_efficiency_basic(self):
        """Verify bandwidth efficiency percentage formula: (achieved / max) * 100."""
        # 100% efficiency
        self.assertAlmostEqual(
            calculate_bandwidth_efficiency(896.0, 896.0), 100.0, places=4
        )

        # Desktop 960 GB/s with 900 GB/s achieved
        self.assertAlmostEqual(
            calculate_bandwidth_efficiency(900.0, 960.0), 93.75, places=4
        )

        # Laptop 768 GB/s with 1081.5 GB/s achieved (over 100% via optimization/caching)
        expected_eff = (1081.5 / 768.0) * 100.0
        self.assertAlmostEqual(
            calculate_bandwidth_efficiency(1081.5, 768.0), expected_eff, places=4
        )

    def test_calculate_bandwidth_efficiency_invalid_inputs(self):
        """Verify error handling for invalid theoretical max or negative bandwidth."""
        with self.assertRaises(ValueError):
            calculate_bandwidth_efficiency(-10.0, 896.0)

        with self.assertRaises(ValueError):
            calculate_bandwidth_efficiency(500.0, 0.0)

        with self.assertRaises(ValueError):
            calculate_bandwidth_efficiency(500.0, -896.0)

    def test_rtx5080_specs_resolution(self):
        """Verify RTX 5080 specification presets resolution."""
        self.assertEqual(get_theoretical_max_bandwidth("desktop"), 960.0)
        self.assertEqual(get_theoretical_max_bandwidth("laptop"), 896.0)
        self.assertEqual(get_theoretical_max_bandwidth("laptop_896"), 896.0)
        self.assertEqual(get_theoretical_max_bandwidth("laptop_768"), 768.0)

        # Numerical float input resolution
        self.assertEqual(get_theoretical_max_bandwidth(1024.0), 1024.0)
        self.assertEqual(get_theoretical_max_bandwidth("1024.0"), 1024.0)

        # Invalid spec name
        with self.assertRaises(ValueError):
            get_theoretical_max_bandwidth("non_existent_gpu")

    def test_gpu_auto_detection(self):
        """Verify auto-detection returns a valid positive theoretical bandwidth."""
        config_key = detect_gpu_config()
        self.assertIn(config_key, RTX5080_SPECS)

        auto_max = get_theoretical_max_bandwidth("auto")
        self.assertGreater(auto_max, 0.0)
        self.assertIn(auto_max, [960.0, 896.0, 768.0])

    def test_validate_criteria(self):
        """Verify performance criteria validation (t/s >= 150 and efficiency >= 95%)."""
        # Both passed
        res1 = validate_criteria(150.1, 95.5)
        self.assertTrue(res1["passed"])
        self.assertTrue(res1["tps_passed"])
        self.assertTrue(res1["efficiency_passed"])

        # TPS failed
        res2 = validate_criteria(149.0, 98.0)
        self.assertFalse(res2["passed"])
        self.assertFalse(res2["tps_passed"])
        self.assertTrue(res2["efficiency_passed"])

        # Efficiency failed
        res3 = validate_criteria(160.0, 94.0)
        self.assertFalse(res3["passed"])
        self.assertTrue(res3["tps_passed"])
        self.assertFalse(res3["efficiency_passed"])

    def test_bandwidth_calculator_class(self):
        """Verify BandwidthCalculator class calculation and structure."""
        calc = BandwidthCalculator(gpu_config="laptop_896")
        self.assertEqual(calc.theoretical_max_gbps, 896.0)

        result = calc.calculate(tokens_per_sec=150.0, model_weight_bytes=6_000_000_000)

        self.assertEqual(result["tokens_per_sec"], 150.0)
        self.assertEqual(result["model_size_bytes"], 6_000_000_000)
        self.assertEqual(result["calculated_bandwidth_gbps"], 900.0)
        self.assertEqual(result["gpu_theoretical_max_gbps"], 896.0)
        self.assertAlmostEqual(result["bandwidth_efficiency_pct"], 100.4464, places=2)
        self.assertTrue(result["criteria_passed"])

    def test_custom_theoretical_max_override(self):
        """Verify custom theoretical_max_gbps override in BandwidthCalculator."""
        calc = BandwidthCalculator(theoretical_max_gbps=1000.0)
        self.assertEqual(calc.theoretical_max_gbps, 1000.0)
        result = calc.calculate(tokens_per_sec=100.0, model_weight_bytes=5_000_000_000)
        self.assertEqual(result["calculated_bandwidth_gbps"], 500.0)
        self.assertEqual(result["bandwidth_efficiency_pct"], 50.0)

    def test_run_benchmark_dry_run_output_schema(self):
        """Verify run_benchmark outputs JSON report matching PROJECT.md schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "benchmark_results.json")

            report = run_benchmark(
                engine="llama.cpp",
                model_name="gemma-4-12B-it-Q4_0.gguf",
                batch_size=1,
                prompt_tokens_target=128,
                max_tokens=512,
                gpu_config="laptop_896",
                output_path=out_file,
                dry_run=True,
                simulated_tps=150.1,
            )

            # Check return dict
            self.assertEqual(report["engine"], "llama.cpp")
            self.assertEqual(report["model_name"], "gemma-4-12B-it-Q4_0.gguf")
            self.assertEqual(report["batch_size"], 1)
            self.assertEqual(report["prompt_tokens"], 128)
            self.assertEqual(report["tokens_generated"], 512)
            self.assertEqual(report["tokens_per_sec"], 150.1)
            self.assertIn("calculated_bandwidth_gbps", report)
            self.assertIn("gpu_theoretical_max_gbps", report)
            self.assertIn("bandwidth_efficiency_pct", report)
            self.assertIn("criteria_passed", report)

            # Verify file contents on disk
            self.assertTrue(os.path.isfile(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                disk_report = json.load(f)

class MockResponse:
    """Mock HTTP response object for simulating line-by-line SSE streams in urllib.request."""

    def __init__(self, lines):
        self.lines = [line.encode("utf-8") if isinstance(line, str) else line for line in lines]
        self._iter = iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return self._iter

    def readline(self):
        try:
            return next(self._iter)
        except StopIteration:
            return b""


class TestStreamingBenchmarkHarness(unittest.TestCase):
    """Test suite for live and mocked HTTP benchmark functions and SSE streaming parser."""

    def test_default_fallback_model_size_bytes(self):
        """Verify default fallback model size bytes is exact size of gemma-4-12B-it-Q4_0.gguf (7,219,673,216)."""
        from benchmark.bandwidth_calculator import DEFAULT_MODEL_SIZE_BYTES as BANDWIDTH_DEFAULT
        from benchmark.run_benchmark import DEFAULT_MODEL_SIZE_BYTES as RUNNER_DEFAULT

        self.assertEqual(BANDWIDTH_DEFAULT, 7219673216)
        self.assertEqual(RUNNER_DEFAULT, 7219673216)
        self.assertEqual(get_model_file_size("non_existent_file.gguf"), 7219673216)

    def test_decode_tps_calculation_formula(self):
        """Verify decode_tps formula: (tokens_generated - 1) / (t_final - t_first_chunk)."""
        tokens = 512
        t_first = 0.5
        t_final = 3.9
        decode_duration = t_final - t_first
        expected_tps = (tokens - 1) / decode_duration

        self.assertAlmostEqual(expected_tps, 511 / 3.4, places=4)

    @unittest.mock.patch("urllib.request.urlopen")
    def test_make_streaming_http_request_llamacpp_native(self, mock_urlopen):
        """Verify make_streaming_http_request parses llama.cpp native SSE stream."""
        from benchmark.run_benchmark import make_streaming_http_request

        lines = [
            'data: {"content": "Hello", "multimodal": false, "slot_id": 0, "stop": false}\n',
            'data: {"content": " world", "multimodal": false, "slot_id": 0, "stop": false}\n',
            'data: {"content": "!", "multimodal": false, "slot_id": 0, "stop": true, "timings": {"prompt_n": 128, "predicted_n": 512}}\n',
        ]
        mock_urlopen.return_value = MockResponse(lines)

        payload = {"prompt": "test", "n_predict": 512}
        res = make_streaming_http_request("http://127.0.0.1:8080", "/completion", payload)

        self.assertIsNotNone(res["t_first_chunk"])
        self.assertIsNotNone(res["t_final"])
        self.assertGreaterEqual(res["t_final"], res["t_first_chunk"])
        self.assertEqual(res["timings"]["prompt_n"], 128)
        self.assertEqual(res["timings"]["predicted_n"], 512)
        self.assertEqual(res["text_chunks_count"], 3)

    @unittest.mock.patch("urllib.request.urlopen")
    def test_make_streaming_http_request_vllm_openai(self, mock_urlopen):
        """Verify make_streaming_http_request parses vLLM / OpenAI SSE stream."""
        from benchmark.run_benchmark import make_streaming_http_request

        lines = [
            'data: {"id": "cmpl-1", "choices": [{"text": "Hello", "index": 0}]}\n',
            'data: {"id": "cmpl-1", "choices": [{"text": " world", "index": 0}]}\n',
            'data: {"id": "cmpl-1", "choices": [{"text": "!", "index": 0}], "usage": {"prompt_tokens": 128, "completion_tokens": 512}}\n',
            "data: [DONE]\n",
        ]
        mock_urlopen.return_value = MockResponse(lines)

        payload = {"model": "gemma-4-12B-it-Q4_0.gguf", "prompt": "test", "max_tokens": 512}
        res = make_streaming_http_request("http://127.0.0.1:8000", "/v1/completions", payload)

        self.assertIsNotNone(res["t_first_chunk"])
        self.assertIsNotNone(res["t_final"])
        self.assertEqual(res["usage"]["prompt_tokens"], 128)
        self.assertEqual(res["usage"]["completion_tokens"], 512)
        self.assertEqual(res["text_chunks_count"], 3)

    @unittest.mock.patch("urllib.request.urlopen")
    def test_run_llamacpp_benchmark_streaming(self, mock_urlopen):
        """Verify run_llamacpp_benchmark uses SSE streaming and calculates TTFT/decode_tps."""
        from benchmark.run_benchmark import run_llamacpp_benchmark

        lines = [
            'data: {"content": "Hello", "stop": false}\n',
            'data: {"content": " world", "stop": true, "timings": {"prompt_n": 128, "predicted_n": 512}}\n',
        ]
        mock_urlopen.return_value = MockResponse(lines)

        res = run_llamacpp_benchmark(base_url="http://127.0.0.1:8080", max_tokens=512)

        self.assertEqual(res["prompt_tokens"], 128)
        self.assertEqual(res["tokens_generated"], 512)
        self.assertIn("prompt_tps", res)
        self.assertIn("decode_tps", res)
        self.assertIn("ttft_seconds", res)
        self.assertIn("decode_time_seconds", res)
        self.assertEqual(res["tokens_per_sec"], res["decode_tps"])

    @unittest.mock.patch("urllib.request.urlopen")
    def test_run_vllm_benchmark_streaming(self, mock_urlopen):
        """Verify run_vllm_benchmark sends stream=True and calculates TTFT/decode_tps."""
        from benchmark.run_benchmark import run_vllm_benchmark, DEFAULT_MODEL_NAME

        lines = [
            'data: {"id": "1", "choices": [{"text": "Chunk 1"}]}\n',
            'data: {"id": "1", "choices": [{"text": "Chunk 2"}], "usage": {"prompt_tokens": 128, "completion_tokens": 512}}\n',
            "data: [DONE]\n",
        ]
        mock_urlopen.return_value = MockResponse(lines)

        res = run_vllm_benchmark(base_url="http://127.0.0.1:8000", max_tokens=512)

        # Check stream=True was sent in HTTP payload
        args, _ = mock_urlopen.call_args
        req = args[0]
        req_body = json.loads(req.data.decode("utf-8"))
        self.assertTrue(req_body["stream"])

        self.assertEqual(res["prompt_tokens"], 128)
        self.assertEqual(res["tokens_generated"], 512)
        self.assertIn("prompt_tps", res)
        self.assertIn("decode_tps", res)
        self.assertEqual(res["tokens_per_sec"], res["decode_tps"])

    @unittest.mock.patch("urllib.request.urlopen")
    def test_run_benchmark_live_mocked_http(self, mock_urlopen):
        """Verify run_benchmark with dry_run=False executes HTTP streaming request and outputs valid JSON report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "vllm_benchmark.json")

            lines = [
                'data: {"id": "1", "choices": [{"text": "Part 1"}]}\n',
                'data: {"id": "1", "choices": [{"text": "Part 2"}], "usage": {"prompt_tokens": 128, "completion_tokens": 512}}\n',
                "data: [DONE]\n",
            ]
            mock_urlopen.return_value = MockResponse(lines)

            report = run_benchmark(
                engine="vLLM",
                url="http://127.0.0.1:8000",
                gpu_config="laptop_896",
                output_path=out_file,
                dry_run=False,
            )

            self.assertEqual(report["engine"], "vLLM")
            self.assertEqual(report["prompt_tokens"], 128)
            self.assertEqual(report["tokens_generated"], 512)
            self.assertIn("prompt_tps", report)
            self.assertIn("decode_tps", report)
            self.assertEqual(report["tokens_per_sec"], report["decode_tps"])
            self.assertEqual(report["model_size_bytes"], 7219673216)
            self.assertTrue(os.path.exists(out_file))


if __name__ == "__main__":
    unittest.main()
