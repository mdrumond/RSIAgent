import argparse
from pathlib import Path

from .app import create_app


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
    app = create_app(args.config, args.cache_dir / "dashboard.sqlite3",
                     refresh_seconds=args.refresh_seconds)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
