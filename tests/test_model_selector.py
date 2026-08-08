"""Unit tests for model_selector. No network calls — hf_hub_download is mocked."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from anycloudllm.hardware_scanner import HardwareProfile
from anycloudllm.model_selector import (
    CACHE_DIR_ENV,
    LLAMA_1B_REPO,
    LLAMA_3B_REPO,
    LLAMA_8B_REPO,
    DownloadError,
    ModelSelection,
    airllm_options,
    cache_dir,
    download_model,
    expected_path,
    is_cached,
    select_model,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "models"))
    return tmp_path / "models"


def profile(ram_gb: float, vram_gb: float = 0.0, cpu_count: int = 8) -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=ram_gb,
        gpu_vram_gb=vram_gb,
        cpu_count=cpu_count,
        has_gpu=vram_gb > 0.0,
    )


class TestTiers:
    def test_tier1_big_gpu_gets_8b_q8(self):
        s = select_model(profile(ram_gb=64, vram_gb=24))
        assert s.repo_id == LLAMA_8B_REPO
        assert s.filename == "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

    def test_tier1_boundary_exactly_8gb_vram(self):
        assert select_model(profile(ram_gb=16, vram_gb=8.0)).filename == "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

    def test_tier2_midrange_gpu_gets_8b_q4(self):
        s = select_model(profile(ram_gb=8, vram_gb=6))
        assert s.repo_id == LLAMA_8B_REPO
        assert s.filename == "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    def test_tier2_boundary_exactly_4gb_vram(self):
        assert select_model(profile(ram_gb=8, vram_gb=4.0)).filename == "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    def test_tier2_reached_by_ram_alone(self):
        s = select_model(profile(ram_gb=32, vram_gb=0.0))
        assert s.repo_id == LLAMA_8B_REPO
        assert s.filename == "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    def test_tier2_boundary_exactly_16gb_ram(self):
        assert select_model(profile(ram_gb=16.0)).filename == "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    def test_tier2_boundary_exactly_15gb_ram(self):
        assert select_model(profile(ram_gb=15.0)).filename == "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    def test_tier3_modest_ram_gets_3b(self):
        s = select_model(profile(ram_gb=12, vram_gb=2))
        assert s.repo_id == LLAMA_3B_REPO
        assert s.filename == "Llama-3.2-3B-Instruct-Q5_K_M.gguf"

    def test_tier3_boundary_exactly_8gb_ram(self):
        assert select_model(profile(ram_gb=8.0)).filename == "Llama-3.2-3B-Instruct-Q5_K_M.gguf"

    def test_tier4_fallback_gets_1b(self):
        s = select_model(profile(ram_gb=4, vram_gb=0.0))
        assert s.repo_id == LLAMA_1B_REPO
        assert s.filename == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    def test_tier4_tiny_box(self):
        assert select_model(profile(ram_gb=1.5, cpu_count=1)).filename == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    @pytest.mark.parametrize(
        ("ram", "vram", "filename"),
        [
            (64, 24, "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
            (8, 6, "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
            (32, 0, "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
            (12, 0, "Llama-3.2-3B-Instruct-Q5_K_M.gguf"),
            (4, 0, "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        ],
    )
    def test_all_tiers_parametrized(self, ram, vram, filename):
        assert select_model(profile(ram_gb=ram, vram_gb=vram)).filename == filename


class TestReason:
    def test_tier1_reason_cites_vram(self):
        reason = select_model(profile(ram_gb=64, vram_gb=24)).reason
        assert "24.0 GB VRAM" in reason and "Q8_0" in reason

    def test_tier2_reason_cites_both(self):
        reason = select_model(profile(ram_gb=32, vram_gb=0.0)).reason
        assert "0.0 GB VRAM" in reason and "32.0 GB RAM" in reason and "Q4_K_M" in reason

    def test_tier3_reason_cites_ram(self):
        reason = select_model(profile(ram_gb=12, vram_gb=2)).reason
        assert "12.0 GB RAM" in reason and "Q5_K_M" in reason

    def test_tier4_reason_says_falling_back(self):
        reason = select_model(profile(ram_gb=4)).reason
        assert "falling back" in reason and "4.0 GB RAM" in reason

    def test_reason_is_non_empty_for_every_tier(self):
        for ram, vram in [(64, 24), (8, 6), (32, 0), (12, 0), (4, 0)]:
            assert select_model(profile(ram_gb=ram, vram_gb=vram)).reason.strip()


class TestCacheAndDownload:
    def test_cache_dir_honours_env_override(self, isolated_cache):
        assert cache_dir() == isolated_cache

    def test_fresh_selection_has_no_local_path(self):
        assert select_model(profile(ram_gb=64, vram_gb=24)).local_path is None

    def test_selection_picks_up_existing_cached_file(self, isolated_cache):
        isolated_cache.mkdir(parents=True)
        cached = isolated_cache / "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
        cached.write_bytes(b"gguf")
        selection = select_model(profile(ram_gb=64, vram_gb=24))
        assert selection.local_path == cached
        assert is_cached(selection)

    def test_expected_path_is_cache_dir_plus_filename(self, isolated_cache):
        selection = select_model(profile(ram_gb=4))
        assert expected_path(selection) == isolated_cache / selection.filename

    def test_download_skips_network_when_already_cached(self, isolated_cache, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        isolated_cache.mkdir(parents=True)
        selection = ModelSelection(LLAMA_1B_REPO, "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "test")
        target = isolated_cache / selection.filename
        target.write_bytes(b"gguf")
        assert download_model(selection) == target
        assert selection.local_path == target

    def test_download_calls_hf_hub_download_with_expected_args(self, isolated_cache, monkeypatch):
        calls = []
        fake_hub = types.ModuleType("huggingface_hub")

        def hf_hub_download(repo_id, filename, local_dir):
            calls.append({"repo_id": repo_id, "filename": filename, "local_dir": local_dir})
            path = Path(local_dir) / filename
            path.write_bytes(b"gguf")
            return str(path)

        fake_hub.hf_hub_download = hf_hub_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        selection = select_model(profile(ram_gb=4))
        result = download_model(selection)
        assert calls == [{"repo_id": LLAMA_1B_REPO, "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "local_dir": str(isolated_cache)}]
        assert result == isolated_cache / selection.filename
        assert selection.local_path == result

    def test_label_is_repo_slash_filename(self):
        selection = select_model(profile(ram_gb=4))
        assert selection.label == f"{LLAMA_1B_REPO}/{selection.filename}"

    def test_download_raises_download_error_on_network_failure(self, isolated_cache, monkeypatch):
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.hf_hub_download = lambda **kw: (_ for _ in ()).throw(OSError("Connection refused"))
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        selection = ModelSelection(LLAMA_1B_REPO, "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "test")
        with pytest.raises(DownloadError):
            download_model(selection)

    def test_download_raises_download_error_on_404(self, isolated_cache, monkeypatch):
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.hf_hub_download = lambda **kw: (_ for _ in ()).throw(Exception("Repository Not Found"))
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        selection = ModelSelection(LLAMA_1B_REPO, "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "test")
        with pytest.raises(DownloadError, match="not found on HuggingFace"):
            download_model(selection)

    def test_download_raises_download_error_when_hub_missing(self, isolated_cache, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        selection = ModelSelection(LLAMA_1B_REPO, "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "test")
        with pytest.raises(DownloadError, match="huggingface_hub is not installed"):
            download_model(selection)

    def test_disk_space_check_raises_when_full(self, isolated_cache, monkeypatch):
        import shutil
        monkeypatch.setattr(shutil, "disk_usage", lambda p: types.SimpleNamespace(free=0, total=100 * 1024**3, used=100 * 1024**3))
        selection = ModelSelection(LLAMA_1B_REPO, "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "test")
        with pytest.raises(DownloadError, match="Not enough disk space"):
            download_model(selection)


class TestAirLLMOptions:
    def test_airllm_options_returns_list(self):
        assert len(airllm_options()) >= 1

    def test_airllm_summary_contains_slowdown(self):
        for opt in airllm_options():
            assert "slower" in opt.summary()

    def test_airllm_catalog_has_valid_entries(self):
        for opt in airllm_options():
            assert "/" in opt.repo_id
            assert opt.filename.endswith(".gguf")
            assert opt.size_gb > 0
            assert opt.num_layers > 0

    def test_airllm_catalog_largest_first(self):
        sizes = [o.size_gb for o in airllm_options()]
        assert sizes == sorted(sizes, reverse=True)

    def test_tokens_per_minute_scales_with_bandwidth(self):
        opt = airllm_options()[0]
        assert opt.tokens_per_minute(disk_bw_gb_s=3.0) > opt.tokens_per_minute(disk_bw_gb_s=0.1)

    def test_no_entry_mismatches_repo_and_filename(self):
        for opt in airllm_options():
            if "12B" in opt.param_label or "13B" in opt.param_label:
                assert "8B" not in opt.repo_id, f"{opt.param_label} points to 8B repo: {opt.repo_id}"
            if "70B" in opt.param_label:
                assert "8B" not in opt.repo_id


class TestPortValidation:
    def test_valid_port_accepted(self):
        from anycloudllm.server import resolve_port
        assert resolve_port(8080) == 8080
        assert resolve_port(1) == 1
        assert resolve_port(65535) == 65535

    def test_port_zero_raises(self):
        from anycloudllm.server import resolve_port
        with pytest.raises(ValueError, match="out of range"):
            resolve_port(0)

    def test_port_too_high_raises(self):
        from anycloudllm.server import resolve_port
        with pytest.raises(ValueError, match="out of range"):
            resolve_port(65536)

    def test_negative_port_raises(self):
        from anycloudllm.server import resolve_port
        with pytest.raises(ValueError, match="out of range"):
            resolve_port(-1)
