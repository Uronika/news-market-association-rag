from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the News Market Association RAG WebUI and API server."
    )
    parser.add_argument(
        "--host",
        default=os.getenv("APP_HOST", "127.0.0.1"),
        help="Bind host. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("APP_PORT", "8000")),
        help="Bind port. Default: 8000",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=_env_bool("APP_RELOAD", False),
        help="Enable auto reload during local development.",
    )
    parser.add_argument(
        "--no-reload",
        action="store_false",
        dest="reload",
        help="Disable auto reload even when APP_RELOAD is set.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("APP_LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level. Default: info",
    )
    return parser.parse_args()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("Missing dependency: uvicorn. Run: py -3.11 -m pip install -r requirements.txt")
        return 1

    args = parse_args()
    print(f"Starting WebUI: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
