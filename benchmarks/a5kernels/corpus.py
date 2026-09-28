"""Reproducibly prepare and index pinned reference corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

from .embeddings import PinnedBGEEmbeddings
from .knowledge import KnowledgeDB


@dataclass(frozen=True)
class CorpusFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    collection: str
    language: str
    repository: str
    revision: str
    files: tuple[CorpusFile, ...]

    @classmethod
    def load(cls, path: Path) -> "CorpusSpec":
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected = {"name", "collection", "language", "repository", "revision", "files"}
        if set(raw) != expected:
            raise ValueError(f"corpus spec fields must be exactly {sorted(expected)}")
        files = tuple(CorpusFile(**item) for item in raw["files"])
        spec = cls(files=files, **{key: raw[key] for key in expected - {"files"}})
        spec.validate()
        return spec

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.name):
            raise ValueError("corpus name must be a lowercase artifact-directory component")
        if not self.collection or self.language != "en":
            raise ValueError("corpus requires nonempty names and language 'en'")
        if not self.repository.startswith("https://") or self.repository.endswith("/"):
            raise ValueError("repository must be a canonical HTTPS URL without trailing slash")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("revision must be a lowercase 40-character commit SHA")
        if not self.files:
            raise ValueError("corpus must allowlist at least one file")
        paths = [item.path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("corpus files must be unique and sorted by path")
        for item in self.files:
            pure = PurePosixPath(item.path)
            if pure.is_absolute() or ".." in pure.parts or not pure.name:
                raise ValueError(f"unsafe corpus path: {item.path!r}")
            if len(item.sha256) != 64 or any(c not in "0123456789abcdef" for c in item.sha256):
                raise ValueError(f"invalid sha256 for {item.path!r}")

    def source_root(self, artifacts: Path) -> Path:
        return artifacts.resolve() / self.name / self.revision / "sources"


def _run(argv: Sequence[str], *, cwd: Path) -> str:
    result = subprocess.run(
        list(argv), cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()


def prepare_corpus(spec: CorpusSpec, artifacts: Path) -> Path:
    """Fetch one commit, verify allowlisted blobs, and atomically publish them."""
    destination = spec.source_root(artifacts)
    if destination.exists():
        verify_prepared(spec, artifacts)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{spec.name}-", dir=destination.parent) as value:
        staging = Path(value)
        repository = staging / "repo"
        sources = staging / "sources"
        repository.mkdir()
        sources.mkdir()
        _run(["git", "init", "--quiet"], cwd=repository)
        _run(["git", "remote", "add", "origin", spec.repository], cwd=repository)
        _run(["git", "fetch", "--quiet", "--depth=1", "origin", spec.revision], cwd=repository)
        resolved = _run(["git", "rev-parse", "FETCH_HEAD^{commit}"], cwd=repository)
        if resolved != spec.revision:
            raise ValueError(f"fetched revision mismatch: expected {spec.revision}, got {resolved}")
        for item in spec.files:
            data = subprocess.run(
                ["git", "show", f"{spec.revision}:{item.path}"],
                cwd=repository,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
            actual = hashlib.sha256(data).hexdigest()
            if actual != item.sha256:
                raise ValueError(f"hash mismatch for {item.path}: expected {item.sha256}, got {actual}")
            target = sources / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        os.replace(sources, destination)
    verify_prepared(spec, artifacts)
    return destination


def verify_prepared(spec: CorpusSpec, artifacts: Path) -> tuple[Path, ...]:
    root = spec.source_root(artifacts)
    expected = {item.path: item.sha256 for item in spec.files}
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    } if root.is_dir() else {}
    if set(actual_paths) != set(expected):
        raise ValueError("prepared corpus does not exactly match its allowlist")
    for relative, path in actual_paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected[relative]:
            raise ValueError(f"prepared hash mismatch for {relative}")
    return tuple(actual_paths[path] for path in sorted(actual_paths))


def index_corpus(
    spec: CorpusSpec,
    artifacts: Path,
    database_path: Path,
    manifest_path: Path,
    *,
    model_cache: Path | None = None,
) -> None:
    sources = verify_prepared(spec, artifacts)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = PinnedBGEEmbeddings(cache_dir=model_cache, local_files_only=True)
    with KnowledgeDB(database_path, embeddings) as database:
        database.index(
            spec.source_root(artifacts),
            sources,
            collection=spec.collection,
            language=spec.language,
            manifest_path=manifest_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    index = commands.add_parser("index")
    index.add_argument("--db", type=Path, required=True)
    index.add_argument("--manifest", type=Path, required=True)
    index.add_argument("--model-cache", type=Path)
    args = parser.parse_args()
    spec = CorpusSpec.load(args.spec)
    if args.command == "prepare":
        print(prepare_corpus(spec, args.artifacts))
    else:
        index_corpus(spec, args.artifacts, args.db, args.manifest, model_cache=args.model_cache)
        print(args.manifest.resolve())


if __name__ == "__main__":
    main()
