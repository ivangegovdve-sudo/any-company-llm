"""Pick a GGUF model that fits the host, and fetch it on demand.

Selection is a first-match-wins ladder over the HardwareProfile. Downloads go
through huggingface_hub into ``~/.cache/any-company-llm/models/`` so a second
run is offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from any_company_llm.hardware_scanner import HardwareProfile

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "any-company-llm" / "models"
CACHE_DIR_ENV = "ANY_COMPANY_LLM_CACHE_DIR"

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
        lambda p: p.gpu_vram_gb >= 4.0 or p.total_ram_gb >= 16.0,
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
