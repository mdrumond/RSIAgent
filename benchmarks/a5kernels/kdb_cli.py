"""Small offline CLI for inspecting an A5 knowledge database."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .knowledge import KnowledgeDB


def _backend(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("embedding backend must be MODULE:ATTRIBUTE")
    return getattr(importlib.import_module(module_name), attribute)()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--embedding-backend", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index")
    index.add_argument("--root", type=Path, required=True)
    index.add_argument("--collection", required=True)
    index.add_argument("--language", required=True)
    index.add_argument("--manifest", type=Path)
    index.add_argument("sources", nargs="+", type=Path)
    query = commands.add_parser("query")
    query.add_argument("--collection", required=True)
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("text")
    args = parser.parse_args()
    with KnowledgeDB(args.db, _backend(args.embedding_backend)) as database:
        if args.command == "index":
            manifest = database.index(
                args.root,
                args.sources,
                collection=args.collection,
                language=args.language,
                manifest_path=args.manifest,
            )
            print(manifest.to_json(), end="")
        else:
            print(json.dumps([hit.__dict__ for hit in database.query(
                args.collection, args.text, limit=args.limit
            )], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
