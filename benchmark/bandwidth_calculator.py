"""RTX 5080 Memory Bandwidth & Efficiency Calculation Engine.

Provides mathematical calculation of achieved GPU memory bandwidth, memory bandwidth
efficiency percentage, GPU specification lookup/auto-detection for NVIDIA RTX 5080 variants,
and performance criteria validation for LLM inference benchmarks.
"""

import sys
import ctypes
import subprocess
from typing import Dict, Any, Union, Optional

# Theoretical memory bandwidth presets for NVIDIA GeForce RTX 5080 (GB/s)
# - Desktop RTX 5080: 256-bit GDDR7 @ 30 Gbps = 960.0 GB/s
# - Laptop RTX 5080 (High/Standard): 256-bit GDDR7 @ 28 Gbps = 896.0 GB/s
# - Laptop RTX 5080 (Low clock): 256-bit GDDR7 @ 24 Gbps = 768.0 GB/s
RTX5080_SPECS: Dict[str, float] = {
    "desktop": 960.0,
    "rtx5080_desktop": 960.0,
    "laptop": 896.0,
    "laptop_896": 896.0,
    "rtx5080_laptop": 896.0,
    "laptop_768": 768.0,
    "rtx5080_laptop_768": 768.0,
}

DEFAULT_FALLBACK_BANDWIDTH_GBPS = 896.0  # Laptop RTX 5080 default
DEFAULT_MODEL_SIZE_BYTES = 7_219_673_216  # gemma-4-12B-it-Q4_0.gguf exact size in bytes


def calculate_achieved_bandwidth(
    tokens_per_sec: float, model_weight_bytes: Union[int, float]
) -> float:
    """Computes achieved memory bandwidth in GB/s.

    Formula: (tokens_per_sec * model_weight_bytes) / 1e9

    Args:
        tokens_per_sec: Decode speed in tokens per second.
        model_weight_bytes: Total model weight size in bytes.

    Returns:
        Achieved memory bandwidth in GB/s.
    """
    if tokens_per_sec < 0:
        raise ValueError(f"tokens_per_sec must be non-negative, got {tokens_per_sec}")
    if model_weight_bytes < 0:
        raise ValueError(f"model_weight_bytes must be non-negative, got {model_weight_bytes}")

    return (tokens_per_sec * float(model_weight_bytes)) / 1e9


def calculate_bandwidth_efficiency(
    achieved_bandwidth_gbps: float, theoretical_max_bandwidth_gbps: float
) -> float:
    """Computes memory bandwidth efficiency as a percentage (%).

    Formula: (achieved_bandwidth / theoretical_max_bandwidth) * 100

    Args:
        achieved_bandwidth_gbps: Achieved memory bandwidth in GB/s.
        theoretical_max_bandwidth_gbps: Theoretical maximum memory bandwidth in GB/s.

    Returns:
        Bandwidth efficiency percentage (%).
    """
    if achieved_bandwidth_gbps < 0:
        raise ValueError(f"achieved_bandwidth_gbps must be non-negative, got {achieved_bandwidth_gbps}")
    if theoretical_max_bandwidth_gbps <= 0:
        raise ValueError(
            f"theoretical_max_bandwidth_gbps must be greater than 0, got {theoretical_max_bandwidth_gbps}"
        )

    return (achieved_bandwidth_gbps / theoretical_max_bandwidth_gbps) * 100.0


def detect_gpu_device_name() -> Optional[str]:
    """Attempts to detect the installed NVIDIA GPU device name using PyTorch, NVML, or nvidia-smi."""
    # 1. Try PyTorch if imported / available
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if device_name:
                return device_name
    except Exception:
        pass

    # 2. Try NVML via ctypes (Windows nvml.dll or Linux libnvidia-ml.so)
    try:
        lib_name = "nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so"
        nvml = ctypes.CDLL(lib_name)
        nvml.nvmlInit_v2()
        handle = ctypes.c_void_p()
        nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
        name_buf = ctypes.create_string_buffer(96)
        nvml.nvmlDeviceGetName(handle, name_buf, 96)
        nvml.nvmlShutdown()
        name_str = name_buf.value.decode("utf-8", errors="ignore")
        if name_str:
            return name_str
    except Exception:
        pass

    # 3. Try nvidia-smi command line
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return None


def detect_gpu_config() -> str:
    """Detects GPU hardware configuration variant ('desktop', 'laptop_896', 'laptop_768').

    Returns:
        Config string key for RTX5080_SPECS.
    """
    device_name = detect_gpu_device_name()
    if device_name:
        name_upper = device_name.upper()
        if "5080" in name_upper:
            if "LAPTOP" in name_upper or "MOBILE" in name_upper:
                return "laptop_896"
            else:
                return "desktop"
    # Default fallback when detection is indeterminate
    return "laptop_896"


def get_theoretical_max_bandwidth(config: Union[str, float, int] = "auto") -> float:
    """Resolves the theoretical maximum memory bandwidth in GB/s for a given config or GPU.

    Args:
        config: Config name string ('auto', 'desktop', 'laptop', 'laptop_896', 'laptop_768')
                or explicit numerical GB/s float/int.

    Returns:
        Theoretical max memory bandwidth in GB/s.
    """
    if isinstance(config, (int, float)):
        val = float(config)
        if val <= 0:
            raise ValueError(f"Theoretical bandwidth must be positive, got {val}")
        return val

    config_str = str(config).strip().lower()

    if config_str == "auto":
        detected_key = detect_gpu_config()
        return RTX5080_SPECS.get(detected_key, DEFAULT_FALLBACK_BANDWIDTH_GBPS)

    if config_str in RTX5080_SPECS:
        return RTX5080_SPECS[config_str]

    # Try numeric conversion from string
    try:
        val = float(config_str)
        if val <= 0:
            raise ValueError(f"Theoretical bandwidth must be positive, got {val}")
        return val
    except ValueError:
        valid_keys = list(RTX5080_SPECS.keys()) + ["auto"]
        raise ValueError(
            f"Unknown GPU config '{config}'. Must be one of {valid_keys} or a positive float GB/s value."
        )


def validate_criteria(
    tokens_per_sec: float,
    efficiency_pct: float,
    target_tps: float = 150.0,
    target_efficiency_pct: float = 95.0,
) -> Dict[str, Any]:
    """Validates whether achieved benchmark metrics meet performance criteria.

    Default Criteria:
        - Decode speed >= 150.0 tokens/sec
        - Memory bandwidth efficiency >= 95.0%

    Args:
        tokens_per_sec: Achieved decode tokens per second.
        efficiency_pct: Achieved memory bandwidth efficiency percentage.
        target_tps: Minimum required tokens/sec (default: 150.0).
        target_efficiency_pct: Minimum required efficiency percentage (default: 95.0).

    Returns:
        Dictionary containing pass/fail flags and target comparison.
    """
    tps_passed = tokens_per_sec >= target_tps
    efficiency_passed = efficiency_pct >= target_efficiency_pct
    overall_passed = tps_passed and efficiency_passed

    return {
        "passed": overall_passed,
        "tps_passed": tps_passed,
        "efficiency_passed": efficiency_passed,
        "tokens_per_sec": float(tokens_per_sec),
        "target_tps": float(target_tps),
        "efficiency_pct": float(efficiency_pct),
        "target_efficiency_pct": float(target_efficiency_pct),
    }


class BandwidthCalculator:
    """RTX 5080 Memory Bandwidth Calculator & Evaluation Engine."""

    def __init__(
        self,
        gpu_config: Union[str, float, int] = "auto",
        theoretical_max_gbps: Optional[float] = None,
    ):
        """Initialize calculator with GPU spec or custom theoretical max.

        Args:
            gpu_config: Predefined RTX 5080 config name ('auto', 'desktop', 'laptop', 'laptop_896', 'laptop_768')
                        or numerical bandwidth float.
            theoretical_max_gbps: Optional explicit override for theoretical max bandwidth GB/s.
        """
        if theoretical_max_gbps is not None:
            if theoretical_max_gbps <= 0:
                raise ValueError(
                    f"theoretical_max_gbps must be positive, got {theoretical_max_gbps}"
                )
            self.theoretical_max_gbps = float(theoretical_max_gbps)
            self.gpu_config = f"custom_{self.theoretical_max_gbps}GBs"
        else:
            self.gpu_config = str(gpu_config)
            self.theoretical_max_gbps = get_theoretical_max_bandwidth(gpu_config)

    def calculate_achieved_bandwidth(
        self, tokens_per_sec: float, model_weight_bytes: Union[int, float]
    ) -> float:
        """Computes achieved memory bandwidth in GB/s."""
        return calculate_achieved_bandwidth(tokens_per_sec, model_weight_bytes)

    def calculate_efficiency(self, achieved_bandwidth_gbps: float) -> float:
        """Computes memory bandwidth efficiency percentage."""
        return calculate_bandwidth_efficiency(
            achieved_bandwidth_gbps, self.theoretical_max_gbps
        )

    def validate(
        self,
        tokens_per_sec: float,
        efficiency_pct: float,
        target_tps: float = 150.0,
        target_efficiency_pct: float = 95.0,
    ) -> Dict[str, Any]:
        """Validates performance criteria."""
        return validate_criteria(
            tokens_per_sec, efficiency_pct, target_tps, target_efficiency_pct
        )

    def calculate(
        self,
        tokens_per_sec: float,
        model_weight_bytes: Union[int, float],
        target_tps: float = 150.0,
        target_efficiency_pct: float = 95.0,
    ) -> Dict[str, Any]:
        """Performs full bandwidth evaluation and returns structured results dictionary.

        Args:
            tokens_per_sec: Decode speed in tokens per second.
            model_weight_bytes: Total model size in bytes.
            target_tps: Minimum required tokens/sec (default 150.0).
            target_efficiency_pct: Minimum required efficiency percentage (default 95.0%).

        Returns:
            Dictionary containing achieved bandwidth, efficiency, theoretical max, and validation status.
        """
        achieved_gbps = self.calculate_achieved_bandwidth(
            tokens_per_sec, model_weight_bytes
        )
        efficiency_pct = self.calculate_efficiency(achieved_gbps)
        validation = self.validate(
            tokens_per_sec, efficiency_pct, target_tps, target_efficiency_pct
        )

        return {
            "tokens_per_sec": float(tokens_per_sec),
            "model_size_bytes": int(model_weight_bytes),
            "calculated_bandwidth_gbps": round(achieved_gbps, 4),
            "gpu_theoretical_max_gbps": round(self.theoretical_max_gbps, 4),
            "bandwidth_efficiency_pct": round(efficiency_pct, 4),
            "criteria_passed": validation["passed"],
            "validation_details": validation,
        }
