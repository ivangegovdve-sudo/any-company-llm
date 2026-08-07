"""``python -m anycloudllm`` — scan, select, download, serve."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anycloudllm import __version__
from anycloudllm.hardware_scanner import HardwareProfile, scan_hardware
from anycloudllm.model_selector import (
    DownloadError,
    airllm_options,
    download_model,
    expected_path,
    select_model,
)
from anycloudllm.server import ServerConfig, check_llama_cpp_server, resolve_host, resolve_port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anycloudllm",
        description="Scan hardware, download the right model, serve locally.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="port to bind (env: ANYCLOUDLLM_PORT, default 8080)",
    )
    parser.add_argument(
        "--host", default=None,
        help="host to bind (env: ANYCLOUDLLM_HOST, default 127.0.0.1)",
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
    parser.add_argument("--version", action="version", version=f"anycloudllm {__version__}")
    parser.add_argument(
        "--airllm", action="store_true",
        help=(
            "show AirLLM layer-streaming options for high-parameter models, "
            "then exit without starting a server"
        ),
    )
    parser.add_argument(
        "--airllm-disk-bw", type=float, default=3.0, metavar="GB_S",
        help=(
            "assumed disk read bandwidth in GB/s for AirLLM speed estimates "
            "(default 3.0 = NVMe; use 0.5 for SATA SSD, 0.1 for HDD)"
        ),
    )
    return parser


def print_airllm_options(disk_bw_gb_s: float) -> None:
    """Print the AirLLM catalog with speed estimates and exit."""
    print()
    print("AirLLM layer-streaming options")
    print("=" * 60)
    print(
        "AirLLM (github.com/lyogavin/airllm) loads one transformer layer at a\n"
        "time from disk, letting you run parameter-heavy models on limited RAM.\n"
        "Install: pip install airllm\n"
    )
    print(f"Speed estimates assume {disk_bw_gb_s:.1f} GB/s sustained disk read.")
    print("Pass --airllm-disk-bw <n> to adjust (0.5 = SATA SSD, 0.1 = HDD).\n")
    for opt in airllm_options():
        print(f"  {opt.summary(disk_bw_gb_s)}")
        print(f"    {opt.label}")
        print()
    print(
        "To use an AirLLM model, pass its downloaded GGUF path via --model-path\n"
        "and make sure airllm is installed and configured for layer streaming."
    )


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

    print(f"Downloading {selection.label}...")
    print("  This may take several minutes. Progress is shown below.")
    try:
        return download_model(selection)
    except DownloadError as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        return None
    except KeyboardInterrupt:
        print("\nDownload cancelled.", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Resolve and validate the port before doing any heavy work.
    try:
        port = resolve_port(args.port)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Scanning hardware...")
    profile = scan_hardware()
    print(f"  {profile.describe()}")

    if args.airllm:
        print_airllm_options(args.airllm_disk_bw)
        return 0

    # Check for llama-cpp-python[server] early — before the model download —
    # so the user learns about missing deps immediately.
    if args.model_path is None or True:  # always check; --model-path doesn't skip the server
        try:
            check_llama_cpp_server()
        except RuntimeError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1

    model_path = resolve_model_path(args, profile)
    if model_path is None:
        return 1

    config = ServerConfig(
        model_path=model_path,
        host=resolve_host(args.host),
        port=port,
        n_ctx=args.n_ctx,
        n_gpu_layers=-1 if profile.has_gpu else 0,
    )
    print(f"Starting server at {config.url}")
    print()
    print("Tip: run with --airllm to see high-parameter options via layer streaming.")
    try:
        run_server(config)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())