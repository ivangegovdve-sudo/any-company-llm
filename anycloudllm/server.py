"""Serve a local GGUF over an OpenAI-compatible HTTP API, plus a chat UI.

Wraps ``llama_cpp.server.app`` — the OpenAI-compat FastAPI app shipped with
llama-cpp-python — so the endpoints are the upstream ones:
``/v1/chat/completions``, ``/v1/completions``, ``/v1/models`` (plus ``/health``
and ``/api/info``, which this module adds).

``/`` serves a self-contained chat page that talks to ``/v1/chat/completions``
on the same origin, so one process is the whole product: no separate frontend,
no CORS, no configuration step.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
PORT_ENV = "ANYCLOUDLLM_PORT"
HOST_ENV = "ANYCLOUDLLM_HOST"
MODEL_ALIAS = "local"

_MIN_PORT = 1
_MAX_PORT = 65535


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
    """CLI flag wins, then ``ANYCLOUDLLM_PORT``, then 8080.

    Raises ``ValueError`` if the resolved port is outside 1–65535.
    """
    if explicit is not None:
        port = explicit
    else:
        raw = os.environ.get(PORT_ENV)
        if not raw:
            return DEFAULT_PORT
        try:
            port = int(raw)
        except ValueError:
            logger.warning("ignoring non-numeric %s=%r", PORT_ENV, raw)
            return DEFAULT_PORT

    if not (_MIN_PORT <= port <= _MAX_PORT):
        raise ValueError(
            f"Port {port} is out of range. Must be between {_MIN_PORT} and {_MAX_PORT}."
        )
    return port


def resolve_host(explicit: str | None = None) -> str:
    return explicit or os.environ.get(HOST_ENV) or DEFAULT_HOST


_LOOPBACK_V4 = "127.0.0.1"
_LOOPBACK_V6 = "::1"


def _can_bind(family: int, host: str, port: int) -> bool:
    import socket

    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError:  # family unavailable on this host
        return True
    with sock:
        # No SO_REUSEADDR: on Windows it would let us bind over a live listener
        # and report a busy port as free.
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _something_answers(family: int, host: str, port: int, timeout: float = 0.25) -> bool:
    import socket

    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError:  # family unavailable on this host
        return False
    with sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
        except OSError:
            return False
    return True


def port_is_free(host: str, port: int) -> bool:
    """True if nothing answers on ``host:port`` and we can bind it.

    A bind test alone is not enough on Windows: a specific-address bind happily
    coexists with an existing wildcard listener rather than failing with
    EADDRINUSE the way Linux does. So a dual-stack server already serving
    ``:::8080`` does not stop us binding ``127.0.0.1:8080`` — we would start,
    look healthy, and the user's browser would still reach the other process.

    Connecting is the test that matches what the browser will experience. We
    probe ``::1`` too when serving loopback, since ``localhost`` resolves to
    either family.
    """
    import socket

    if _something_answers(socket.AF_INET, host, port):
        return False
    if host == _LOOPBACK_V4 and _something_answers(socket.AF_INET6, _LOOPBACK_V6, port):
        logger.info("nothing on 127.0.0.1:%d but [::1]:%d answers; busy", port, port)
        return False
    return _can_bind(socket.AF_INET, host, port)


def ensure_free_port(host: str, port: int, chosen: bool, span: int = 20) -> int:
    """Resolve a bindable port *before* the model spends a minute loading.

    uvicorn would otherwise raise the conflict only after the load, which reads
    as a slow crash. If the user named a port, a conflict is an error — we must
    not quietly serve somewhere they are not looking. If we picked the default
    for them, walk forward to the next free port instead.
    """
    if port_is_free(host, port):
        return port

    if chosen:
        raise OSError(
            f"Port {port} on {host} is already in use. "
            f"Pass --port with a free port, or set {PORT_ENV}."
        )

    for candidate in range(port + 1, port + 1 + span):
        if candidate <= _MAX_PORT and port_is_free(host, candidate):
            logger.info("default port %d busy, using %d", port, candidate)
            print(f"  Port {port} is busy; using {candidate} instead.")
            return candidate

    raise OSError(
        f"Ports {port}-{port + span} on {host} are all in use. "
        f"Pass --port with a free port, or set {PORT_ENV}."
    )


def check_llama_cpp_server() -> None:
    """Verify that llama-cpp-python's server extra is installed.

    Raises ``RuntimeError`` with an actionable message if the import fails.
    Call this *before* downloading the model so the user learns about the
    missing dependency immediately rather than after a multi-GB download.
    """
    try:
        from llama_cpp.server.app import create_app  # noqa: F401
        from llama_cpp.server.settings import ModelSettings, ServerSettings  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python's server extra is missing.\n"
            "Install it with: pip install 'llama-cpp-python[server]'\n\n"
            "For GPU support, install with the appropriate backend first — see:\n"
            "  https://github.com/abetlen/llama-cpp-python#installation-with-specific-hardware"
        ) from exc


@dataclass
class MemoryPlan:
    """How much of llama.cpp's optional memory spending this host can afford."""

    use_mlock: bool
    use_extra_bufts: bool
    model_gb: float
    available_gb: float
    note: str


def plan_memory(model_path: Path, available_bytes: int | None = None) -> MemoryPlan:
    """Decide whether to pin the model and whether to repack its weights.

    Two llama.cpp features cost real, non-mmap RAM on top of the mapped model:

    * ``mlock`` pins every mapped page so it can never be evicted. Note that
      llama-cpp-python's *server* defaults this to ``True`` even though
      llama.cpp's own default is ``False``.
    * Weight repacking (ggml "extra buffer types") allocates a second copy of
      the quantized tensors laid out for faster SIMD kernels — roughly another
      model-sized block.

    On a host with room to spare both are worth having. On a tight one either
    turns a slow-but-working load into a hard failure:
    ``alloc_tensor_range: failed to allocate CPU_REPACK buffer``. So enable each
    only when free memory measurably covers it, and let mmap page the rest in
    from disk otherwise.
    """
    try:
        model_bytes = model_path.stat().st_size
    except OSError:
        model_bytes = 0

    if available_bytes is None:
        try:
            import psutil

            available_bytes = psutil.virtual_memory().available
        except Exception:  # pragma: no cover - psutil is a hard dep, be safe anyway
            available_bytes = 0

    model_gb = model_bytes / 1e9
    available_gb = available_bytes / 1e9

    # Repacking needs about another model's worth of resident RAM. Ask for that
    # plus headroom for the KV cache and compute buffers before allowing it.
    use_extra_bufts = available_bytes >= model_bytes * 1.25
    # Pinning needs the whole model resident and never reclaimable — a stricter bar.
    use_mlock = available_bytes >= model_bytes * 2.0

    if use_mlock and use_extra_bufts:
        note = "ample free memory: pinning weights and repacking for speed"
    elif use_extra_bufts:
        note = "moderate free memory: repacking for speed, not pinning"
    else:
        note = (
            f"low free memory ({available_gb:.1f} GB free vs {model_gb:.1f} GB model): "
            "streaming weights via mmap, no pinning or repacking"
        )
    return MemoryPlan(use_mlock, use_extra_bufts, model_gb, available_gb, note)


@contextlib.contextmanager
def _extra_bufts(enabled: bool):
    """Toggle ggml weight repacking around a model load.

    ``use_extra_bufts`` is a field on ``llama_model_params`` but llama-cpp-python
    does not surface it as a ``Llama(...)`` keyword, so the only way through is
    to adjust the defaults struct that ``Llama.__init__`` starts from.
    """
    if enabled:
        yield
        return

    import llama_cpp.llama_cpp as C

    original = C.llama_model_default_params

    def patched():
        params = original()
        params.use_extra_bufts = False
        return params

    C.llama_model_default_params = patched
    try:
        yield
    finally:
        C.llama_model_default_params = original


def _web_dir() -> Path:
    """Locate the bundled UI, both in-repo and inside the PyInstaller bundle."""
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        candidate = Path(bundled) / "anycloudllm" / "web"
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parent / "web"


def load_chat_page() -> str | None:
    """Return the chat UI HTML, or None if it is not present in this install."""
    index = _web_dir() / "index.html"
    try:
        return index.read_text(encoding="utf-8")
    except OSError:
        logger.warning("chat UI not found at %s; serving API only", index)
        return None


def build_app(config: ServerConfig):
    """Build the OpenAI-compatible FastAPI app for this model, plus the chat UI."""
    try:
        from llama_cpp.server.app import create_app
        from llama_cpp.server.settings import ModelSettings, ServerSettings
    except ImportError as exc:  # pragma: no cover - install-time problem
        raise RuntimeError(
            "llama-cpp-python's server extra is missing. "
            "Install it with: pip install 'llama-cpp-python[server]'"
        ) from exc

    plan = plan_memory(config.model_path)
    logger.info("memory plan: %s", plan.note)
    print(f"  Memory: {plan.note}")

    server_settings = ServerSettings(host=config.host, port=config.port)
    model_settings = [
        ModelSettings(
            model=str(config.model_path),
            model_alias=config.model_alias,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            use_mlock=plan.use_mlock,
            # llama-cpp-python's server defaults this to True where the Llama
            # class itself defaults to False. True sizes the logits scratch
            # array (n_ctx, n_vocab) instead of (n_batch, n_vocab) — at 4096 ctx
            # on a 128k-vocab Llama that is a 2 GB contiguous float32 block, and
            # only /v1/completions with echo+logprobs ever reads it.
            logits_all=False,
        )
    ]
    with _extra_bufts(plan.use_extra_bufts):
        app = create_app(server_settings=server_settings, model_settings=model_settings)

    # Upstream has no /health; add one so callers can poll readiness.
    if not any(getattr(route, "path", None) == "/health" for route in app.routes):

        @app.get("/health")
        def health() -> dict:
            return {"status": "ok", "model": config.model_alias}

    @app.get("/api/info")
    def info() -> dict:
        """What the UI shows in the corner."""
        return {
            "model": config.model_alias,
            "file": config.model_path.name,
            "n_ctx": config.n_ctx,
            "gpu": config.n_gpu_layers != 0,
        }

    page = load_chat_page()
    if page is not None:
        from fastapi.responses import HTMLResponse

        @app.get("/", include_in_schema=False)
        def chat_ui() -> HTMLResponse:
            return HTMLResponse(page)

    return app


def run_server(config: ServerConfig) -> None:
    """Start the server. Blocks until interrupted."""
    import uvicorn

    app = build_app(config)
    logger.info("serving %s at %s", config.model_path.name, config.url)
    uvicorn.run(app, host=config.host, port=config.port)