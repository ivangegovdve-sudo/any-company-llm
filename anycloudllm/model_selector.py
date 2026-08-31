"""Pick a GGUF model that fits the host, and fetch it on demand.

Selection is a first-match-wins ladder over the HardwareProfile. Downloads go
through huggingface_hub into ``~/.cache/anycloudllm/models/`` so a second
run is offline.

AirLLM option: ``airllm_options()`` returns an ordered list of larger models
that can run on *any* RAM via layer-by-layer disk streaming.  Speed depends on
disk bandwidth -- see ``AirLLMOption.summary()`` for a quick estimate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from anycloudllm.hardware_scanner import HardwareProfile

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "anycloudllm" / "models"
CACHE_DIR_ENV = "ANYCLOUDLLM_CACHE_DIR"

LLAMA_8B_REPO = "bartowski/Llama-3.1-8B-Instruct-GGUF"
LLAMA_3B_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
LLAMA_1B_REPO = "bartowski/Llama-3.2-1B-Instruct-GGUF"


@dataclass
class ModelSelection:
    repo_id: str
    filename: str
    reason: str  # human-readable explanation of why this model was chosen
    local_path: Optional[Path] = None  # populated after download

    @property
    def label(self) -> str:
        return f"{self.repo_id}/{self.filename}"


# (predicate, repo_id, filename, reason-template) — first match wins.
_RULES: list[tuple[Callable[[HardwareProfile], bool], str, str, str]] = [
    (
        lambda p: p.gpu_vram_gb >= 8.0,
        LLAMA_8B_REPO,
        "Llama-3.1-8B-Instruct-Q8_0.gguf",
        "{vram:.1f} GB VRAM (>= 8 GB) fits an 8B model at Q8_0 near-lossless quality",
    ),
    (
        lambda p: p.gpu_vram_gb >= 4.0 or p.total_ram_gb >= 15.0,
        LLAMA_8B_REPO,
        "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "{vram:.1f} GB VRAM / {ram:.1f} GB RAM fits an 8B model at Q4_K_M",
    ),
    (
        lambda p: p.total_ram_gb >= 8.0,
        LLAMA_3B_REPO,
        "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        "{ram:.1f} GB RAM and no usable GPU headroom - 3B at Q5_K_M is the comfortable fit",
    ),
    (
        lambda p: True,
        LLAMA_1B_REPO,
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "only {ram:.1f} GB RAM and {vram:.1f} GB VRAM - falling back to 1B at Q4_K_M",
    ),
]


def cache_dir() -> Path:
    """Where downloaded GGUF files live. Overridable for tests via env var."""
    override = os.environ.get(CACHE_DIR_ENV)
    return Path(override) if override else DEFAULT_CACHE_DIR


def select_model(profile: HardwareProfile) -> ModelSelection:
    """Choose a model for this hardware. First matching rule wins."""
    for predicate, repo_id, filename, template in _RULES:
        if predicate(profile):
            reason = template.format(vram=profile.gpu_vram_gb, ram=profile.total_ram_gb)
            selection = ModelSelection(repo_id=repo_id, filename=filename, reason=reason)
            cached = expected_path(selection)
            if cached.exists():
                selection.local_path = cached
            return selection
    raise RuntimeError("no model rule matched - the fallback rule is missing")


def expected_path(selection: ModelSelection) -> Path:
    """Path the GGUF occupies once downloaded."""
    return cache_dir() / selection.filename


def is_cached(selection: ModelSelection) -> bool:
    return expected_path(selection).exists()


def download_model(selection: ModelSelection) -> Path:
    """Download the GGUF if missing, set ``local_path`` and return it.

    huggingface_hub is imported lazily so hardware scanning and selection work
    on a machine that only installed the base package.
    """
    target = expected_path(selection)
    if target.exists():
        selection.local_path = target
        return target

    from huggingface_hub import hf_hub_download

    target.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=selection.repo_id,
        filename=selection.filename,
        local_dir=str(target.parent),
    )
    selection.local_path = Path(path)
    return selection.local_path


# ── AirLLM (layer-streaming) options ────────────────────────────────────────
#
# AirLLM (github.com/lyogavin/airllm) loads one transformer layer at a time
# from disk, so the effective RAM requirement drops to a few GB regardless of
# model size.  The cost is speed: each forward pass reads the full model from
# disk, so throughput is bounded by disk bandwidth rather than memory bandwidth.
#
# Slowdown formula (rough NVMe estimate, 3 GB/s sustained read):
#   time_per_token ≈ model_size_gb / disk_bw_gb_s
#
# Actual performance varies with OS page-cache warmth, NUMA topology, and
# quantisation; treat the numbers as order-of-magnitude guidance only.


@dataclass
class AirLLMOption:
    """A model that can be run via AirLLM layer-streaming inference."""

    param_label: str       # human-readable size, e.g. "70B"
    repo_id: str
    filename: str
    size_gb: float         # approximate GGUF size on disk
    num_layers: int        # transformer layer count (determines per-token passes)

    @property
    def label(self) -> str:
        return f"{self.repo_id}/{self.filename}"

    def tokens_per_minute(self, disk_bw_gb_s: float = 3.0) -> float:
        """Estimated throughput, assuming disk reads dominate.

        *disk_bw_gb_s* defaults to a typical NVMe sustained read (3 GB/s).
        Pass 0.5 for a SATA SSD or 0.1 for an HDD.
        """
        time_per_token_s = self.size_gb / max(disk_bw_gb_s, 0.001)
        return 60.0 / max(time_per_token_s, 0.001)

    def slowdown_vs_baseline(
        self,
        baseline_tok_per_sec: float = 3.0,
        disk_bw_gb_s: float = 3.0,
    ) -> float:
        """How many times slower than a CPU-resident 8B baseline (~3 tok/s)."""
        baseline_tpm = baseline_tok_per_sec * 60.0
        return baseline_tpm / max(self.tokens_per_minute(disk_bw_gb_s), 0.001)

    def summary(self, disk_bw_gb_s: float = 3.0) -> str:
        tpm = self.tokens_per_minute(disk_bw_gb_s)
        slowdown = self.slowdown_vs_baseline(disk_bw_gb_s=disk_bw_gb_s)
        if tpm >= 1.0:
            speed_str = f"~{tpm:.0f} tok/min"
        else:
            sec_per_tok = 60.0 / tpm
            speed_str = f"~{sec_per_tok:.0f} s/token"
        return (
            f"{self.param_label} ({self.size_gb:.0f} GB on disk)  "
            f"{speed_str} on NVMe  "
            f"[~{slowdown:.0f}x slower than 8B CPU baseline]"
        )


# Catalog ordered largest-to-smallest so the user sees the most capable first.
_AIRLLM_CATALOG: list[AirLLMOption] = [
    AirLLMOption(
        param_label="70B",
        repo_id="bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        filename="Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        size_gb=40.0,
        num_layers=80,
    ),
    AirLLMOption(
        param_label="34B",
        repo_id="bartowski/Meta-Llama-3-34B-Instruct-GGUF",
        filename="Meta-Llama-3-34B-Instruct-Q4_K_M.gguf",
        size_gb=20.0,
        num_layers=60,
    ),
    AirLLMOption(
        param_label="13B",
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Llama-3.2-13B-Instruct-Q4_K_M.gguf",
        size_gb=8.0,
        num_layers=40,
    ),
    AirLLMOption(
        param_label="8B (via AirLLM)",
        repo_id=LLAMA_8B_REPO,
        filename="Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        size_gb=5.0,
        num_layers=32,
    ),
]


def airllm_options() -> list[AirLLMOption]:
    """Return the full AirLLM model catalog, largest first.

    All entries work on any machine regardless of RAM — AirLLM streams layers
    from disk.  Requires ``pip install airllm`` and the model to be downloaded.
    """
    return list(_AIRLLM_CATALOG)