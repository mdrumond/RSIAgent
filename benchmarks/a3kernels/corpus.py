"""Strict A3-owned Ascend C corpus specifications and verified documents."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Mapping


@dataclass(frozen=True)
class CorpusFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class CorpusDocument:
    collection: str
    target: str
    repository: str
    source_revision: str
    path: str
    content_sha256: str
    text: str

    @property
    def digest(self) -> str:
        payload = {
            "collection": self.collection,
            "content_sha256": self.content_sha256,
            "path": self.path,
            "repository": self.repository,
            "source_revision": self.source_revision,
            "target": self.target,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    collection: str
    target: str
    language: str
    repository: str
    revision: str
    files: tuple[CorpusFile, ...]

    @classmethod
    def load(cls, path: Path) -> "CorpusSpec":
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "name", "collection", "target", "language", "repository", "revision", "files"
        }
        if set(raw) != expected:
            raise ValueError(f"corpus spec fields must be exactly {sorted(expected)}")
        files = tuple(CorpusFile(**item) for item in raw["files"])
        spec = cls(files=files, **{key: raw[key] for key in expected - {"files"}})
        spec.validate()
        return spec

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.name):
            raise ValueError("corpus name must be a lowercase artifact-directory component")
        if self.target != "a3":
            raise ValueError("corpus target must be 'a3'")
        if not self.collection or self.language != "en":
            raise ValueError("corpus requires a collection and language 'en'")
        if not self.repository.startswith("https://") or self.repository.endswith("/"):
            raise ValueError("repository must be a canonical HTTPS URL without trailing slash")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be a lowercase 40-character commit SHA")
        paths = [item.path for item in self.files]
        if not paths:
            raise ValueError("corpus must allowlist at least one file")
        if paths != sorted(set(paths)):
            raise ValueError("corpus files must be unique and sorted by path")
        for item in self.files:
            pure = PurePosixPath(item.path)
            if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != item.path:
                raise ValueError(f"unsafe corpus path: {item.path!r}")
            if not re.fullmatch(r"[0-9a-f]{64}", item.sha256):
                raise ValueError(f"invalid sha256 for {item.path!r}")

    def source_root(self, artifacts: Path) -> Path:
        return artifacts.resolve() / self.name / self.revision / "sources"

    def with_files(self, values: Mapping[str, bytes]) -> "CorpusSpec":
        """Return a pinned derived spec, useful for deterministic offline fixtures."""
        files = tuple(
            CorpusFile(path, hashlib.sha256(data).hexdigest())
            for path, data in sorted(values.items())
        )
        spec = replace(self, files=files)
        spec.validate()
        return spec


def load_documents(spec: CorpusSpec, artifacts: Path) -> tuple[CorpusDocument, ...]:
    """Load an exact prepared allowlist, failing closed on missing or changed bytes."""
    root = spec.source_root(artifacts)
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    } if root.is_dir() else {}
    if set(actual) != {item.path for item in spec.files}:
        raise ValueError("prepared corpus does not exactly match its allowlist")
    documents = []
    for item in spec.files:
        data = actual[item.path].read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item.sha256:
            raise ValueError(f"prepared hash mismatch for {item.path}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"corpus file is not UTF-8: {item.path}") from exc
        documents.append(CorpusDocument(
            collection=spec.collection,
            target=spec.target,
            repository=spec.repository,
            source_revision=spec.revision,
            path=item.path,
            content_sha256=digest,
            text=text,
        ))
    return tuple(documents)
