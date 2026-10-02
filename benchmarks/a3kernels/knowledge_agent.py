"""Treatment-isolated A3 Knowledge Agent and append-only query journal."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .knowledge import CollectionManifest, KnowledgeDB, SearchHit


@dataclass(frozen=True)
class KnowledgeQuery:
    query: str
    limit: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("limit must be an integer")
        if not 1 <= self.limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        object.__setattr__(self, "query", self.query.strip())


@dataclass(frozen=True)
class Citation:
    collection: str
    path: str
    start_line: int
    end_line: int
    chunk_id: str
    source_revision: str
    content_sha256: str


@dataclass(frozen=True)
class KnowledgeResult:
    citation: Citation
    text: str
    score: float


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class QueryJournal:
    """Durable, append-only, hash-chained record of treatment queries."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        enabled: bool,
        query: KnowledgeQuery,
        collection: str | None,
        manifest: CollectionManifest | None,
        citations: tuple[Citation, ...],
    ) -> dict[str, Any]:
        if enabled != (manifest is not None and collection is not None):
            raise ValueError("enabled query provenance requires a collection manifest")
        if not enabled and citations:
            raise ValueError("disabled query provenance cannot contain citations")
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            entries = self._decode(stream.read())
            entry: dict[str, Any] = {
                "sequence": len(entries) + 1,
                "kind": "a3_knowledge_query",
                "enabled": enabled,
                "query": query.query,
                "limit": query.limit,
                "collection": collection if enabled else None,
                "collection_fingerprint": manifest.fingerprint if manifest else None,
                "embedding": (
                    {
                        "model": manifest.embedding_model,
                        "revision": manifest.embedding_revision,
                        "dimension": manifest.dimension,
                    }
                    if manifest else None
                ),
                "citations": [asdict(citation) for citation in citations] if enabled else [],
                "previous_digest": entries[-1]["entry_digest"] if entries else None,
            }
            entry["entry_digest"] = _digest(_canonical(entry))
            stream.seek(0, os.SEEK_END)
            stream.write(_canonical(entry).decode() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            return entry

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            return self._decode(stream.read())

    def recover_incomplete_tail(self) -> int:
        """Remove only an interrupted, non-newline-terminated final append."""
        if not self.path.exists():
            return 0
        with self.path.open("r+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            value = stream.read()
            if not value or value.endswith(b"\n"):
                self._decode(value.decode("utf-8"))
                return 0
            boundary = value.rfind(b"\n") + 1
            prefix = value[:boundary].decode("utf-8")
            tail = value[boundary:]
            try:
                json.loads(tail)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._decode(prefix)
                stream.seek(boundary)
                stream.truncate()
            else:
                self._decode((value + b"\n").decode("utf-8"))
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            return 1

    @staticmethod
    def _decode(value: str) -> list[dict[str, Any]]:
        if value and not value.endswith("\n"):
            raise ValueError("incomplete final knowledge journal entry")
        entries = []
        previous = None
        for sequence, line in enumerate(value.splitlines(), 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid knowledge journal JSON") from exc
            if (
                entry.get("sequence") != sequence
                or entry.get("kind") != "a3_knowledge_query"
                or entry.get("previous_digest") != previous
            ):
                raise ValueError("invalid knowledge journal continuity")
            claimed = entry.get("entry_digest")
            content = {key: item for key, item in entry.items() if key != "entry_digest"}
            if claimed != _digest(_canonical(content)):
                raise ValueError("knowledge journal entry digest mismatch")
            previous = claimed
            entries.append(entry)
        return entries


class KnowledgeAgent:
    """Fixed-collection librarian gate for the KDB treatment ablation."""

    def __init__(
        self,
        *,
        enabled: bool,
        journal: QueryJournal,
        database: KnowledgeDB | None = None,
        collection: str | None = None,
    ) -> None:
        if enabled and (database is None or not collection):
            raise ValueError("enabled Knowledge Agent requires a database and collection")
        if not enabled and (database is not None or collection is not None):
            raise ValueError("disabled Knowledge Agent must not receive KDB access")
        self.enabled = enabled
        self.journal = journal
        self.database = database
        self.collection = collection
        self.manifest = None
        if database is not None:
            if database.connection.execute("PRAGMA query_only").fetchone()[0] != 1:
                raise ValueError("Knowledge Agent requires a read-only database")
            self.manifest = database.manifest(collection)  # raises for cross-collection access

    def query(self, action: KnowledgeQuery) -> tuple[KnowledgeResult, ...]:
        if not isinstance(action, KnowledgeQuery):
            raise TypeError("Knowledge Agent accepts only KnowledgeQuery actions")
        if not self.enabled:
            self.journal.append(
                enabled=False, query=action, collection=None, manifest=None, citations=()
            )
            return ()
        assert self.database is not None and self.collection is not None
        assert self.manifest is not None
        hits = self.database.query(self.collection, action.query, limit=action.limit)
        results = tuple(self._validate(hit) for hit in hits)
        self.journal.append(
            enabled=True,
            query=action,
            collection=self.collection,
            manifest=self.manifest,
            citations=tuple(result.citation for result in results),
        )
        return results

    def _validate(self, hit: SearchHit) -> KnowledgeResult:
        assert self.database is not None and self.collection is not None
        assert self.manifest is not None
        source = next(
            (item for item in self.manifest.sources if item.path == hit.path), None
        )
        row = self.database.connection.execute(
            """SELECT * FROM chunks WHERE collection = ? AND chunk_id = ?""",
            (self.collection, hit.chunk_id),
        ).fetchone()
        if source is None or row is None:
            raise ValueError("knowledge result is not a validated indexed chunk")
        expected = (
            row["path"], row["start_line"], row["end_line"], row["text"],
            row["source_revision"], row["content_sha256"],
        )
        actual = (
            hit.path, hit.start_line, hit.end_line, hit.text,
            hit.source_revision, hit.content_sha256,
        )
        manifest_provenance = (source.source_revision, source.content_sha256)
        identity = {
            "collection": self.collection,
            "content_sha256": source.content_sha256,
            "path": source.path,
            "repository": source.repository,
            "source_revision": source.source_revision,
            "target": "a3",
        }
        document_digest = _digest(_canonical(identity))
        expected_chunk = _digest(
            f"{document_digest}\0{hit.start_line}\0{hit.end_line}\0{hit.text}".encode()
        )
        if (
            expected != actual
            or expected[-2:] != manifest_provenance
            or actual[-2:] != manifest_provenance
            or expected_chunk != hit.chunk_id
        ):
            raise ValueError("knowledge result is not a validated indexed chunk")
        citation = Citation(
            collection=self.collection, path=hit.path, start_line=hit.start_line,
            end_line=hit.end_line, chunk_id=hit.chunk_id,
            source_revision=hit.source_revision, content_sha256=hit.content_sha256,
        )
        return KnowledgeResult(citation, hit.text, hit.score)
