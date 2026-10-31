"""Dependency-free immutable artifacts and canonical evidence serialization."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import math
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping


def canonical_bytes(value: Any) -> bytes:
    """Serialize JSON data identically or reject ambiguous/non-portable data."""

    normalized = _normalize(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _normalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical evidence cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical evidence mappings require string keys")
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(f"unsupported canonical evidence type: {type(value).__name__}")


_SHA256 = re.compile(r"[0-9a-f]{64}")


class ContentAddressedArtifacts:
    """Store exact bytes once and recover them solely from their SHA-256."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"

    def put_bytes(self, content: bytes) -> str:
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        digest = hashlib.sha256(content).hexdigest()
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError("content-addressed artifact collision")
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        return digest

    def retain_from(self, source_store: ContentAddressedArtifacts, digest: str) -> str:
        """Retain verified immutable CAS bytes, sharing storage when possible.

        Only CAS objects may be linked: callers must never mutate stored bytes.
        Removing the source name does not remove the retained object.
        """
        source = source_store._retention_path(digest)
        target = self._path(digest)
        self._verify_path(source, digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except FileExistsError:
            self._verify_path(target, digest)
        except OSError:
            # Cross-device links and filesystems without link support still retain
            # an independent copy. Exclusive creation never overwrites a CAS inode.
            try:
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                self._verify_path(target, digest)
            else:
                try:
                    with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_stream:
                        shutil.copyfileobj(input_stream, output, length=1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    self._verify_path(target, digest)
                except BaseException:
                    target.unlink()
                    raise
        return digest

    def _retention_path(self, digest: str) -> Path:
        """Resolve immutable backing storage; layered stores may override this."""
        return self._path(digest)

    @staticmethod
    def _verify_path(path: Path, digest: str) -> None:
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != digest:
            raise ValueError("artifact content does not match its SHA-256")

    def put_json(self, value: Any) -> str:
        return self.put_bytes(canonical_bytes(value))

    def read_bytes(self, digest: str) -> bytes:
        content = self._path(digest).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("artifact content does not match its SHA-256")
        return content

    def read_json(self, digest: str) -> Any:
        return json.loads(self.read_bytes(digest))

    def verify(self) -> bool:
        if not self.objects.exists():
            return True
        for path in self.objects.glob("*/*"):
            digest = path.parent.name + path.name
            if not _SHA256.fullmatch(digest):
                raise ValueError(f"invalid artifact path {path}")
            self.read_bytes(digest)
        return True

    def _path(self, digest: str) -> Path:
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("artifact identifier must be a lowercase SHA-256")
        return self.objects / digest[:2] / digest[2:]
