"""Explicit, immutable publication of learned memory and retained evidence."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import ContentAddressedArtifacts
from .knowledge import KnowledgeDB, CollectionManifest, SourceFingerprint
from .phase1_memory import Phase1ProjectMemory


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class MemorySource:
    """A validated memory, its producer root, and portable target/toolchain facts.

    ``root / memory.artifact_store`` locates the original artifact store. Root is
    never published; provenance should contain identities rather than host paths.
    """

    memory: Phase1ProjectMemory
    root: Path
    provenance: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactPage:
    sha256: str
    offset: int
    total_bytes: int
    data: bytes
    next_offset: int | None


class MemorySnapshot:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "snapshot.json").read_text())
        identity = {key: value for key, value in self.manifest.items() if key != "fingerprint"}
        if self.manifest.get("schema") != "memory-kdb-snapshot-v1" or _hash(_bytes(identity)) != self.manifest.get("fingerprint"):
            raise ValueError("invalid memory snapshot manifest")

    @property
    def fingerprint(self) -> str:
        return self.manifest["fingerprint"]

    def _document_paths(self) -> list[Path]:
        paths = []
        for relative, digest in self.manifest["documents"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or _hash(path.read_bytes()) != digest:
                raise ValueError("snapshot document does not match its SHA-256")
            paths.append(path)
        return paths

    def direct_documents(self) -> str:
        """The complete exported document bundle for a matched direct-memory arm."""
        paths = self._document_paths() + [self.root / "snapshot.json"]
        return "\n".join(path.read_text() for path in sorted(paths))

    def bind(self, manifest: CollectionManifest) -> None:
        """Check that the selected DB collection indexes exactly this snapshot."""
        paths = self._document_paths() + [self.root / "snapshot.json"]
        sources = tuple(sorted((SourceFingerprint(
            path.relative_to(self.root).as_posix(), _hash(path.read_bytes()), path.stat().st_size
        ) for path in paths), key=lambda source: source.path))
        if sources != manifest.sources:
            raise ValueError("selected collection does not match memory snapshot")

    def index(self, database: KnowledgeDB, *, collection: str, language: str = "memory") -> CollectionManifest:
        """Index in a *new* database or collection; existing collections stay frozen."""
        return database.index(self.root, self._document_paths() + [self.root / "snapshot.json"],
                              collection=collection, language=language)

    def read_artifact(self, digest: str, *, offset: int = 0, limit: int = 16384) -> ArtifactPage:
        if digest not in self.manifest["artifacts"]:
            raise ValueError("artifact is not admitted by this snapshot")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 65536:
            raise ValueError("artifact reads require a nonnegative offset and 1..65536 byte limit")
        data = ContentAddressedArtifacts(self.root / "artifacts").read_bytes(digest)
        if offset > len(data):
            raise ValueError("artifact offset exceeds artifact length")
        end = min(offset + limit, len(data))
        return ArtifactPage(digest, offset, len(data), data[offset:end], end if end < len(data) else None)


def export_snapshot(sources: Sequence[MemorySource], destination: Path) -> MemorySnapshot:
    """Publish one self-contained snapshot; exclude whole records with broken evidence.

    Documents and artifacts deduplicate by content hash. Every producer provenance
    remains attached to the shared document. No producer path or publication time
    enters the identity. Re-exporting the same sources is byte reproducible.
    Existing destinations are never overwritten, even when identical.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".memory-snapshot-", dir=destination.parent))
    try:
        documents: dict[str, dict] = {}
        artifacts: dict[str, dict] = {}
        excluded = []
        store = ContentAddressedArtifacts(temporary / "artifacts")
        for source in sources:
            memory = source.memory
            payload = memory.as_dict()
            # Producer storage locations are not consumer dependencies.
            payload.pop("artifact_store", None)
            record = _hash(_bytes(payload))
            provenance = {"memory_sha256": record, "metadata": dict(source.provenance)}
            required = set(memory.admitted_artifact_sha256)
            required.update(digest for lesson in memory.lessons for digest in lesson.evidence_artifact_sha256)
            required.update(item.evidence_sha256 for item in memory.host_facts)
            required.update(digest for item in memory.interpretations for digest in item.supports)
            original = ContentAddressedArtifacts(source.root / (memory.artifact_store or "artifacts"))
            try:
                retained = {}
                for digest in sorted(required):
                    retained[digest] = original.read_bytes(digest)
            except (OSError, ValueError) as exc:
                # Report identity and a stable reason, never machine-specific paths.
                excluded.append({**provenance, "reason": type(exc).__name__,
                                 "artifact_sha256": digest})
                continue
            entry = documents.setdefault(record, {"memory": payload, "provenance": []})
            if provenance not in entry["provenance"]:
                entry["provenance"].append(provenance)
            for digest, content in retained.items():
                store.put_bytes(content)
                artifact = artifacts.setdefault(digest, {"size": len(content), "provenance": []})
                if provenance not in artifact["provenance"]:
                    artifact["provenance"].append(provenance)
        def stable(items):
            return sorted(items, key=lambda item: _bytes(item))
        files = {}
        document_root = temporary / "documents"
        document_root.mkdir()
        def write(name, content):
            path = document_root / name
            path.write_bytes(content)
            files[path.relative_to(temporary).as_posix()] = _hash(content)
        for digest, document in sorted(documents.items()):
            document["provenance"] = stable(document["provenance"])
            write(f"memory-{digest}.json", _bytes(document))
        for digest, artifact in sorted(artifacts.items()):
            artifact["provenance"] = stable(artifact["provenance"])
            content = store.read_bytes(digest)
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                continue  # Binary evidence stays available via bounded artifact reads.
            write(f"artifact-{digest}.txt", _bytes({"artifact_sha256": digest, **artifact}) + text.encode())
        manifest = {"schema": "memory-kdb-snapshot-v1", "documents": files,
                    "artifacts": artifacts, "excluded": stable(excluded),
                    "memory_records": len(documents)}
        manifest["fingerprint"] = _hash(_bytes(manifest))
        (temporary / "snapshot.json").write_bytes(_bytes(manifest))
        # Atomic publication; require a new destination, including concurrent writers.
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return MemorySnapshot(destination)
