import argparse
import ipaddress
import json
from pathlib import Path

from .app import create_app


def _controls_enabled(path):
    try:
        return json.loads(Path(path).read_text()).get("controls", {}).get("enabled") is True
    except (OSError, ValueError, AttributeError):
        return False


def _loopback(host):
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the read-only RSI campaign dashboard")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--config", type=Path, required=True)
    serve.add_argument("--cache-dir", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--refresh-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if _controls_enabled(args.config) and not _loopback(args.host):
        parser.error("controls-enabled dashboard must bind to a loopback host")
    app = create_app(args.config, args.cache_dir / "dashboard.sqlite3",
                     refresh_seconds=args.refresh_seconds)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
