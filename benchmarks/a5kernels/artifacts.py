"""Immutable, content-addressed artifacts for A5 experiment evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from .evidence import canonical_bytes


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
