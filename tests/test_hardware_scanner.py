"""Unit tests for hardware_scanner. No network, no real GPU required."""

from __future__ import annotations

import sys
import types

import pytest

from anycloudllm import hardware_scanner
from anycloudllm.hardware_scanner import HardwareProfile, scan_hardware

GB = 1024**3


def _fake_pynvml(vram_bytes_per_gpu: list[int], *, init_raises: bool = False) -> types.ModuleType:
    """Minimal stand-in for the pynvml module."""
    module = types.ModuleType("pynvml")

    def nvml_init() -> None:
        if init_raises:
            raise RuntimeError("NVML Shared Library Not Found")

    module.nvmlInit = nvml_init
    module.nvmlShutdown = lambda: None
    module.nvmlDeviceGetCount = lambda: len(vram_bytes_per_gpu)
    module.nvmlDeviceGetHandleByIndex = lambda index: index
    module.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(
        total=vram_bytes_per_gpu[handle]
    )
    return module


def _fake_torch(vram_bytes_per_gpu: list[int]) -> types.ModuleType:
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(
        is_available=lambda: bool(vram_bytes_per_gpu),
        device_count=lambda: len(vram_bytes_per_gpu),
        get_device_properties=lambda index: types.SimpleNamespace(
            total_memory=vram_bytes_per_gpu[index]
        ),
    )
    return module


@pytest.fixture
def no_gpu_libs(monkeypatch):
    """Make both `import pynvml` and `import torch` raise ImportError."""
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.fixture
def fake_psutil(monkeypatch):
    """Patch psutil with a controllable RAM / CPU count."""

    def _apply(ram_gb: float, cpu_count: int = 8):
        monkeypatch.setattr(
            hardware_scanner.psutil,
            "virtual_memory",
            lambda: types.SimpleNamespace(total=int(ram_gb * GB)),
        )
        monkeypatch.setattr(
            hardware_scanner.psutil, "cpu_count", lambda logical=True: cpu_count
        )

    return _apply


class TestVramProbes:
    def test_pynvml_missing_returns_zero(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        assert hardware_scanner._probe_pynvml() == 0.0

    def test_pynvml_init_failure_is_silent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([8 * GB], init_raises=True))
        assert hardware_scanner._probe_pynvml() == 0.0

    def test_pynvml_reports_largest_gpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([6 * GB, 24 * GB]))
        assert hardware_scanner._probe_pynvml() == 24.0

    def test_torch_missing_returns_zero(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert hardware_scanner._probe_torch() == 0.0

    def test_torch_without_cuda_returns_zero(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch([]))
        assert hardware_scanner._probe_torch() == 0.0

    def test_torch_reports_largest_gpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch([12 * GB]))
        assert hardware_scanner._probe_torch() == 12.0

    def test_detect_vram_prefers_pynvml(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([16 * GB]))
        monkeypatch.setitem(sys.modules, "torch", _fake_torch([4 * GB]))
        assert hardware_scanner.detect_vram_gb() == 16.0

    def test_detect_vram_falls_back_to_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch([4 * GB]))
        assert hardware_scanner.detect_vram_gb() == 4.0

    def test_detect_vram_zero_when_nothing_available(self, no_gpu_libs):
        assert hardware_scanner.detect_vram_gb() == 0.0


class TestScanHardware:
    def test_returns_zero_vram_gracefully_without_pynvml(self, no_gpu_libs, fake_psutil):
        fake_psutil(ram_gb=32, cpu_count=16)
        profile = scan_hardware()
        assert isinstance(profile, HardwareProfile)
        assert profile.gpu_vram_gb == 0.0
        assert profile.has_gpu is False
        assert profile.total_ram_gb == 32.0
        assert profile.cpu_count == 16

    def test_reports_gpu_when_pynvml_present(self, monkeypatch, fake_psutil):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([8 * GB]))
        fake_psutil(ram_gb=16, cpu_count=8)
        profile = scan_hardware()
        assert profile.gpu_vram_gb == 8.0
        assert profile.has_gpu is True

    def test_cpu_count_never_zero(self, no_gpu_libs, monkeypatch, fake_psutil):
        fake_psutil(ram_gb=8)
        monkeypatch.setattr(hardware_scanner.psutil, "cpu_count", lambda logical=True: None)
        assert scan_hardware().cpu_count == 1

    def test_describe_mentions_ram_and_gpu_state(self, no_gpu_libs, fake_psutil):
        fake_psutil(ram_gb=8, cpu_count=4)
        text = scan_hardware().describe()
        assert "8.0 GB RAM" in text
        assert "4 CPUs" in text
        assert "no GPU detected" in text
