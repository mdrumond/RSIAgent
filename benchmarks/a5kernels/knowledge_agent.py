"""Read-only Knowledge Agent gate and progressive-memory query provenance."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .knowledge import CollectionManifest, KnowledgeDB, SearchHit

if TYPE_CHECKING:
    from core.loop import DomainActionResult


@dataclass(frozen=True)
class KnowledgeQuery:
    """The only action accepted by the librarian integration."""

    query: str
    limit: int = 5
    dup: int = 1

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
    chunk_hash: str


@dataclass(frozen=True)
class KnowledgeResult:
    citation: Citation
    text: str
    score: float


def parse_knowledge_action(value: str) -> Any:
    """Parse a query action, delegating built-in Actor actions to core."""
    try:
        raw = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        raw = None
    if not isinstance(raw, dict) or set(raw) != {"knowledge_query"}:
        from core.loop import parse_turn

        return parse_turn(value)
    payload = raw["knowledge_query"]
    if not isinstance(payload, dict) or not set(payload) <= {"query", "limit"}:
        return None
    try:
        return KnowledgeQuery(payload.get("query"), payload.get("limit", 5))
    except (TypeError, ValueError):
        return None


class ProgressiveMemoryJournal:
    """Append-only query provenance owned by the mutable experiment memory."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        enabled: bool,
        collection: str | None,
        query: str,
        citations: tuple[Citation, ...],
        manifest: CollectionManifest | None = None,
    ) -> dict[str, Any]:
        if enabled != (manifest is not None):
            raise ValueError("enabled provenance requires exactly one collection manifest")
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            entries = self._decode(stream.read())
            entry = {
                "sequence": len(entries) + 1,
                "kind": "knowledge_query",
                "enabled": enabled,
                "query": query,
                "collection": collection if enabled else None,
                "collection_fingerprint": manifest.fingerprint if manifest else None,
                "embedding": (
                    {"model": manifest.embedding_model, "revision": manifest.embedding_revision}
                    if manifest else None
                ),
                "citations": [asdict(citation) for citation in citations] if enabled else [],
            }
            stream.seek(0, os.SEEK_END)
            stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return entry

    def read(self) -> list[dict[str, Any]]:
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            stream.seek(0)
            return self._decode(stream.read())

    @staticmethod
    def _decode(value: str) -> list[dict[str, Any]]:
        entries = []
        for sequence, line in enumerate(value.splitlines(), 1):
            entry = json.loads(line)
            if entry.get("sequence") != sequence or entry.get("kind") != "knowledge_query":
                raise ValueError("invalid progressive-memory knowledge journal")
            entries.append(entry)
        return entries


class KnowledgeAgent:
    """Librarian authority that mediates all optional KDB access.

    Collection selection is fixed by experiment configuration, never by an Actor
    action.  In an ablation with ``enabled=False`` no database is accepted and the
    response carries neither collection metadata, citations, nor retrieved text.
    """

    DISABLED_OBSERVATION = json.dumps(
        {"knowledge_query": {"available": False, "results": []}},
        sort_keys=True,
        separators=(",", ":"),
    )

    def __init__(
        self,
        *,
        enabled: bool,
        journal: ProgressiveMemoryJournal,
        database: KnowledgeDB | None = None,
        collection: str | None = None,
    ):
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
            self.database = database.read_only_view()
            self.manifest = self.database.manifest(collection)

    def close(self) -> None:
        if self.database is not None:
            self.database.close()

    def query(self, action: KnowledgeQuery) -> tuple[KnowledgeResult, ...]:
        if not self.enabled:
            self.journal.append(
                enabled=False, collection=None, query=action.query, citations=()
            )
            return ()
        assert self.database is not None and self.collection is not None
        hits = self.database.query(self.collection, action.query, limit=action.limit)
        results = tuple(self._validated_result(hit) for hit in hits)
        self.journal.append(
            enabled=True,
            collection=self.collection,
            query=action.query,
            citations=tuple(result.citation for result in results),
            manifest=self.manifest,
        )
        return results

    def execute(self, action: KnowledgeQuery) -> "DomainActionResult":
        """Domain-action executor suitable for ``core.loop.run_attempt``."""
        # Keep the KDB/indexing package dependency-light.  The Actor runtime and its
        # provider SDKs are needed only when this adapter is actually executed.
        from core.loop import DomainActionResult

        if not isinstance(action, KnowledgeQuery):
            raise TypeError("Knowledge Agent accepts only KnowledgeQuery actions")
        results = self.query(action)
        if not self.enabled:
            return DomainActionResult(self.DISABLED_OBSERVATION)
        payload = {
            "knowledge_query": {
                "available": True,
                "results": [
                    {"citation": asdict(result.citation), "text": result.text}
                    for result in results
                ],
            }
        }
        return DomainActionResult(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )

    def _validated_result(self, hit: SearchHit) -> KnowledgeResult:
        assert self.database is not None and self.collection is not None
        citation = Citation(
            collection=self.collection,
            path=hit.path,
            start_line=hit.start_line,
            end_line=hit.end_line,
            chunk_hash=hit.chunk_id,
        )
        self.database.validate_citation(
            citation.collection,
            chunk_id=citation.chunk_hash,
            path=citation.path,
            start_line=citation.start_line,
            end_line=citation.end_line,
        )
        return KnowledgeResult(citation=citation, text=hit.text, score=hit.score)
