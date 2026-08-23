"""Optional Hermes integration.

Only works if hermes_agents is installed. Import fail-silently if not — the
standalone app never depends on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from anycloudllm.server import DEFAULT_PORT

try:
    from poc.anycloudllm.adapter import AnyCloudLLMAdapter

    _HERMES_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the host environment
    _HERMES_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    from poc.anycloudllm.adapter import AnyCloudLLMAdapter


def hermes_available() -> bool:
    """True when hermes_agents' adapter could be imported."""
    return _HERMES_AVAILABLE


def get_hermes_adapter(port: int = DEFAULT_PORT) -> Optional["AnyCloudLLMAdapter"]:
    """Return an AnyCloudLLMAdapter pointed at the local server, or None if Hermes not installed."""
    if not _HERMES_AVAILABLE:
        return None
    return AnyCloudLLMAdapter(
        base_url=f"http://127.0.0.1:{port}",
        api_key="",
        model_id="local",
    )
