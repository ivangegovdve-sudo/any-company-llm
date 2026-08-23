"""Detect the host's RAM, VRAM and CPU count.

Every GPU probe is fail-silent: a machine with no NVIDIA driver, no CUDA and no
optional dependencies still gets a usable profile with ``gpu_vram_gb == 0.0``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1024**3


@dataclass
class HardwareProfile:
    total_ram_gb: float
    gpu_vram_gb: float  # 0.0 if no GPU
    cpu_count: int
    has_gpu: bool

    def describe(self) -> str:
        gpu = f"{self.gpu_vram_gb:.1f} GB VRAM" if self.has_gpu else "no GPU detected"
        return f"{self.total_ram_gb:.1f} GB RAM, {self.cpu_count} CPUs, {gpu}"


def _bytes_to_gb(value: float) -> float:
    return round(value / _BYTES_PER_GB, 2)


def _probe_pynvml() -> float:
    """Largest single-GPU VRAM in GB via NVML, or 0.0 if unavailable."""
    try:
        import pynvml
    except Exception:  # noqa: BLE001 - ImportError, or a broken install
        logger.debug("pynvml not available", exc_info=True)
        return 0.0

    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - no driver / no device
        logger.debug("nvmlInit failed", exc_info=True)
        return 0.0

    try:
        best = 0.0
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            best = max(best, _bytes_to_gb(total))
        return best
    except Exception:  # noqa: BLE001
        logger.debug("NVML query failed", exc_info=True)
        return 0.0
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _probe_torch() -> float:
    """Largest single-GPU VRAM in GB via torch.cuda, or 0.0 if unavailable."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        logger.debug("torch not available", exc_info=True)
        return 0.0

    try:
        if not torch.cuda.is_available():
            return 0.0
        best = 0.0
        for index in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(index).total_memory
            best = max(best, _bytes_to_gb(total))
        return best
    except Exception:  # noqa: BLE001
        logger.debug("torch.cuda query failed", exc_info=True)
        return 0.0


def detect_vram_gb() -> float:
    """VRAM of the largest GPU in GB. 0.0 when nothing reports a GPU."""
    vram = _probe_pynvml()
    if vram > 0.0:
        return vram
    return _probe_torch()


def scan_hardware() -> HardwareProfile:
    """Build a HardwareProfile for the current host."""
    total_ram_gb = _bytes_to_gb(psutil.virtual_memory().total)
    cpu_count = psutil.cpu_count(logical=True) or 1
    gpu_vram_gb = detect_vram_gb()
    return HardwareProfile(
        total_ram_gb=total_ram_gb,
        gpu_vram_gb=gpu_vram_gb,
        cpu_count=cpu_count,
        has_gpu=gpu_vram_gb > 0.0,
    )
