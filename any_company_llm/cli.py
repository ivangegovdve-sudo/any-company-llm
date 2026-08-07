"""``python -m any_company_llm`` — scan, select, download, serve."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from any_company_llm import __version__
from any_company_llm.hardware_scanner import HardwareProfile, scan_hardware
from any_company_llm.model_selector import download_model, expected_path, select_model
from any_company_llm.server import ServerConfig, resolve_host, resolve_port, run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="any-company-llm",
        description="Scan hardware, download the right model, serve locally.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="port to bind (env: ANY_COMPANY_LLM_PORT, default 8080)",
    )
    parser.add_argument(
        "--host", default=None,
        help="host to bind (env: ANY_COMPANY_LLM_HOST, default 127.0.0.1)",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="fail instead of downloading a model that is not cached yet",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="serve this GGUF instead of auto-selecting one",
    )
    parser.add_argument("--n-ctx", type=int, default=4096, help="context window (default 4096)")
    parser.add_argument("--version", action="version", version=f"any-company-llm {__version__}")
    return parser


def resolve_model_path(args: argparse.Namespace, profile: HardwareProfile) -> Path | None:
    """Return the GGUF to serve, or None if one cannot be obtained."""
    if args.model_path is not None:
        path = args.model_path.expanduser()
        if not path.exists():
            print(f"Error: --model-path {path} does not exist", file=sys.stderr)
            return None
        print(f"Using model: {path}")
        return path

    selection = select_model(profile)
    print(f"Selected model: {selection.label} (reason: {selection.reason})")

    if selection.local_path is not None:
        print(f"Already cached: {selection.local_path}")
        return selection.local_path

    if args.no_download:
        print(
            f"Error: {selection.filename} is not cached at {expected_path(selection)} "
            f"and --no-download was set",
            file=sys.stderr,
        )
        return None

    print(f"Downloading {selection.label}... (this may take a few minutes)")
    return download_model(selection)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("Scanning hardware...")
    profile = scan_hardware()
    print(f"  {profile.describe()}")

    model_path = resolve_model_path(args, profile)
    if model_path is None:
        return 1

    config = ServerConfig(
        model_path=model_path,
        host=resolve_host(args.host),
        port=resolve_port(args.port),
        n_ctx=args.n_ctx,
        n_gpu_layers=-1 if profile.has_gpu else 0,
    )
    print(f"Starting server at {config.url}")
    try:
        run_server(config)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
