"""Ingest verified reference inventories and query through the Knowledge Agent.

Source paths are consumed only during ingestion. Query clients need the index,
the pinned embedding cache, and a query journal, never repository access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from .embeddings import PinnedBGEEmbeddings
from .knowledge import KnowledgeDB
from .knowledge_agent import KnowledgeAgent, KnowledgeQuery, ProgressiveMemoryJournal


def _languages(source: dict) -> tuple[str, ...]:
    if "kernel_languages" in source:
        return tuple(source["kernel_languages"])
    name = source["name"].lower()
    if "triton" in name:
        return ("triton",)
    if "catlass" in name:
        return ("ascend-c", "catlass-dsl")
    return ("ascend-c",)


def ingest(inventory_path: Path, database_path: Path, embeddings, *, batch_size=32):
    """Resume immutable collections in a staging DB, publish only on success."""
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema_version") != 1:
        raise ValueError("inventory schema_version must be 1")
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    staging = database_path.with_suffix(database_path.suffix + ".staging")
    completed = []
    with KnowledgeDB(staging, embeddings) as database:
        for source in sorted(inventory["sources"],
                             key=lambda item: (len(item.get("files", ())), item["name"])):
            if source.get("status") not in {"ok", "available", "downloaded", "complete"}:
                continue
            root = Path(source["root"])
            if not root.is_absolute():
                raise ValueError("inventory source root must be absolute")
            files = []
            for item in source["files"]:
                relative = PurePosixPath(item["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("source file must be a relative path")
                path = root / relative
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError(f"source hash mismatch: {source['name']}/{relative}")
                if len(data) != item["bytes"]:
                    raise ValueError(f"source size mismatch: {source['name']}/{relative}")
                data.decode("utf-8")
                files.append(path)
            if not files:
                raise ValueError(f"available source has no text files: {source['name']}")
            identity = json.dumps({key: value for key, value in source.items()
                                   if key not in {"root", "error"}}, sort_keys=True)
            collection = source["name"] + "-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
            print(f"indexing {collection}: {len(files)} files", file=sys.stderr, flush=True)
            manifest = database.index(
                root, files, collection=collection,
                language=source.get("document_language", "mixed"),
                repository=source["repository"], revision=source["revision"],
                kernel_languages=_languages(source),
                reference_exposed=source.get("reference_exposed", False),
                batch_size=batch_size,
                progress=lambda count: print(f"{collection}: {count} chunks", file=sys.stderr,
                                             flush=True) if count % 1024 < batch_size else None,
            )
            expected = {item["path"]: (item["sha256"], item["bytes"])
                        for item in source["files"]}
            indexed_sources = {item.path: (item.sha256, item.size) for item in manifest.sources}
            if indexed_sources != expected:
                raise ValueError(
                    f"indexed source changed after inventory verification: {source['name']}; "
                    "use a new DB path after restoring the pinned sources"
                )
            completed.append({"collection": collection, "fingerprint": manifest.fingerprint,
                              "repository": manifest.repository, "revision": manifest.revision,
                              "files": len(manifest.sources)})
        if not completed:
            raise ValueError("inventory has no available sources")
        wanted = {entry["collection"] for entry in completed}
        indexed = {row[0] for row in database.connection.execute("SELECT name FROM collections")}
        if indexed != wanted:
            raise ValueError("staging DB contains collections from another inventory; use a new DB path")
    os.replace(staging, database_path)
    report = {"schema_version": 1, "collections": completed,
              "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest()}
    database_path.with_suffix(database_path.suffix + ".inventory.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def query(database_path, embeddings, query_text, journal, *, collections=(),
          kernel_language=None, limit=5):
    with KnowledgeDB.open_read_only(database_path, embeddings) as database:
        selected = collections or tuple(row[0] for row in database.connection.execute(
            "SELECT name FROM collections ORDER BY name"))
        if kernel_language:
            selected = tuple(name for name in selected if kernel_language in
                             database.manifest(name).kernel_languages)
        if not selected:
            raise ValueError("no collections match the requested scope")
        agent = KnowledgeAgent(enabled=True, journal=ProgressiveMemoryJournal(journal),
                               database=database, collections=selected)
        try:
            results = agent.query(KnowledgeQuery(query_text, limit))
            return {"collections": selected, "results": [asdict(result) for result in results]}
        finally:
            agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("ingest")
    index.add_argument("--inventory", type=Path, required=True)
    index.add_argument("--batch-size", type=int, default=32)
    search = commands.add_parser("query")
    search.add_argument("--query", required=True)
    search.add_argument("--journal", type=Path, required=True)
    search.add_argument("--collection", action="append", default=[])
    search.add_argument("--kernel-language", choices=("ascend-c", "catlass-dsl", "triton"))
    search.add_argument("--limit", type=int, default=5)
    for command in (index, search):
        command.add_argument("--db", type=Path, required=True)
        command.add_argument("--model-cache", type=Path)
    args = parser.parse_args()
    embeddings = PinnedBGEEmbeddings(cache_dir=args.model_cache, local_files_only=True)
    if args.command == "ingest":
        result = ingest(args.inventory, args.db, embeddings, batch_size=args.batch_size)
    else:
        result = query(args.db, embeddings, args.query, args.journal,
                       collections=tuple(args.collection), kernel_language=args.kernel_language,
                       limit=args.limit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
