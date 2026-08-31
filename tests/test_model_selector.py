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
    ModelSelection,
    cache_dir,
    download_model,
    expected_path,
    is_cached,
    select_model,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the model cache at a temp dir so nothing looks 'already downloaded'."""
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
        selection = select_model(profile(ram_gb=64, vram_gb=24))
        assert selection.repo_id == LLAMA_8B_REPO
        assert selection.filename == "Llama-3.1-8B-Instruct-Q8_0.gguf"

    def test_tier1_boundary_exactly_8gb_vram(self):
        assert select_model(profile(ram_gb=16, vram_gb=8.0)).filename == (
            "Llama-3.1-8B-Instruct-Q8_0.gguf"
        )

    def test_tier2_midrange_gpu_gets_8b_q4(self):
        selection = select_model(profile(ram_gb=8, vram_gb=6))
        assert selection.repo_id == LLAMA_8B_REPO
        assert selection.filename == "Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    def test_tier2_boundary_exactly_4gb_vram(self):
        assert select_model(profile(ram_gb=8, vram_gb=4.0)).filename == (
            "Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        )

    def test_tier2_reached_by_ram_alone(self):
        selection = select_model(profile(ram_gb=32, vram_gb=0.0))
        assert selection.repo_id == LLAMA_8B_REPO
        assert selection.filename == "Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    def test_tier2_boundary_exactly_16gb_ram(self):
        assert select_model(profile(ram_gb=16.0)).filename == "Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    def test_tier3_modest_ram_gets_3b(self):
        selection = select_model(profile(ram_gb=12, vram_gb=2))
        assert selection.repo_id == LLAMA_3B_REPO
        assert selection.filename == "Llama-3.2-3B-Instruct-Q5_K_M.gguf"

    def test_tier3_boundary_exactly_8gb_ram(self):
        assert select_model(profile(ram_gb=8.0)).filename == "Llama-3.2-3B-Instruct-Q5_K_M.gguf"

    def test_tier4_fallback_gets_1b(self):
        selection = select_model(profile(ram_gb=4, vram_gb=0.0))
        assert selection.repo_id == LLAMA_1B_REPO
        assert selection.filename == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    def test_tier4_tiny_box(self):
        assert select_model(profile(ram_gb=1.5, cpu_count=1)).filename == (
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        )

    @pytest.mark.parametrize(
        ("ram", "vram", "filename"),
        [
            (64, 24, "Llama-3.1-8B-Instruct-Q8_0.gguf"),
            (8, 6, "Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
            (32, 0, "Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
            (12, 0, "Llama-3.2-3B-Instruct-Q5_K_M.gguf"),
            (4, 0, "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        ],
    )
    def test_all_tiers_parametrized(self, ram, vram, filename):
        assert select_model(profile(ram_gb=ram, vram_gb=vram)).filename == filename


class TestReason:
    def test_tier1_reason_cites_vram(self):
        reason = select_model(profile(ram_gb=64, vram_gb=24)).reason
        assert "24.0 GB VRAM" in reason
        assert "Q8_0" in reason

    def test_tier2_reason_cites_both(self):
        reason = select_model(profile(ram_gb=32, vram_gb=0.0)).reason
        assert "0.0 GB VRAM" in reason
        assert "32.0 GB RAM" in reason
        assert "Q4_K_M" in reason

    def test_tier3_reason_cites_ram(self):
        reason = select_model(profile(ram_gb=12, vram_gb=2)).reason
        assert "12.0 GB RAM" in reason
        assert "Q5_K_M" in reason

    def test_tier4_reason_says_falling_back(self):
        reason = select_model(profile(ram_gb=4)).reason
        assert "falling back" in reason
        assert "4.0 GB RAM" in reason

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
        cached = isolated_cache / "Llama-3.1-8B-Instruct-Q8_0.gguf"
        cached.write_bytes(b"gguf")
        selection = select_model(profile(ram_gb=64, vram_gb=24))
        assert selection.local_path == cached
        assert is_cached(selection)

    def test_expected_path_is_cache_dir_plus_filename(self, isolated_cache):
        selection = select_model(profile(ram_gb=4))
        assert expected_path(selection) == isolated_cache / selection.filename

    def test_download_skips_network_when_already_cached(self, isolated_cache, monkeypatch):
        # Any attempt to import huggingface_hub here would be a bug.
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

        assert calls == [
            {
                "repo_id": LLAMA_1B_REPO,
                "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                "local_dir": str(isolated_cache),
            }
        ]
        assert result == isolated_cache / selection.filename
        assert selection.local_path == result

    def test_label_is_repo_slash_filename(self):
        selection = select_model(profile(ram_gb=4))
        assert selection.label == f"{LLAMA_1B_REPO}/{selection.filename}"


class TestAirLLMOptions:
    """Tests for the AirLLM catalog and speed estimates."""

    def test_airllm_options_returns_list(self):
        from anycloudllm.model_selector import airllm_options
        opts = airllm_options()
        assert len(opts) >= 1

    def test_airllm_summary_contains_slowdown(self):
        from anycloudllm.model_selector import airllm_options
        for opt in airllm_options():
            s = opt.summary()
            assert "slower" in s

    def test_tier2_boundary_15gb_ram(self):
        """15 GB RAM (no GPU) should now reach tier-2 (8B Q4_K_M)."""
        from anycloudllm.hardware_scanner import HardwareProfile
        from anycloudllm.model_selector import select_model
        p = HardwareProfile(total_ram_gb=15.0, gpu_vram_gb=0.0, cpu_count=8, has_gpu=False)
        sel = select_model(p)
        assert "Q4_K_M" in sel.filename
        assert "8B" in sel.filename

    def test_tokens_per_minute_scales_with_bandwidth(self):
        from anycloudllm.model_selector import airllm_options
        opt = airllm_options()[0]  # 70B
        tpm_nvme = opt.tokens_per_minute(disk_bw_gb_s=3.0)
        tpm_hdd = opt.tokens_per_minute(disk_bw_gb_s=0.1)
        assert tpm_nvme > tpm_hdd
