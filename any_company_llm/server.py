"""Serve a local GGUF over an OpenAI-compatible HTTP API.

Wraps ``llama_cpp.server.app`` — the OpenAI-compat FastAPI app shipped with
llama-cpp-python — so the endpoints are the upstream ones:
``/v1/chat/completions``, ``/v1/completions``, ``/v1/models`` (plus ``/health``,
which this module adds).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
PORT_ENV = "ANY_COMPANY_LLM_PORT"
HOST_ENV = "ANY_COMPANY_LLM_HOST"
MODEL_ALIAS = "local"


@dataclass
class ServerConfig:
    model_path: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    n_ctx: int = 4096
    n_gpu_layers: int = 0  # -1 offloads every layer to the GPU
    model_alias: str = MODEL_ALIAS

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def resolve_port(explicit: int | None = None) -> int:
    """CLI flag wins, then ``ANY_COMPANY_LLM_PORT``, then 8080."""
    if explicit is not None:
        return explicit
    raw = os.environ.get(PORT_ENV)
    if not raw:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", PORT_ENV, raw)
        return DEFAULT_PORT


def resolve_host(explicit: str | None = None) -> str:
    return explicit or os.environ.get(HOST_ENV) or DEFAULT_HOST


def build_app(config: ServerConfig):
    """Build the OpenAI-compatible FastAPI app for this model."""
    try:
        from llama_cpp.server.app import create_app
        from llama_cpp.server.settings import ModelSettings, ServerSettings
    except ImportError as exc:  # pragma: no cover - install-time problem
        raise RuntimeError(
            "llama-cpp-python's server extra is missing. "
            "Install it with: pip install 'llama-cpp-python[server]'"
        ) from exc

    server_settings = ServerSettings(host=config.host, port=config.port)
    model_settings = [
        ModelSettings(
            model=str(config.model_path),
            model_alias=config.model_alias,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
        )
    ]
    app = create_app(server_settings=server_settings, model_settings=model_settings)

    # Upstream has no /health; add one so callers can poll readiness.
    if not any(getattr(route, "path", None) == "/health" for route in app.routes):

        @app.get("/health")
        def health() -> dict:
            return {"status": "ok", "model": config.model_alias}

    return app


def run_server(config: ServerConfig) -> None:
    """Start the server. Blocks until interrupted."""
    import uvicorn

    app = build_app(config)
    logger.info("serving %s at %s", config.model_path.name, config.url)
    uvicorn.run(app, host=config.host, port=config.port)
