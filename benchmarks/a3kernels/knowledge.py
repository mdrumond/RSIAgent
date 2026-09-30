"""Immutable A3 knowledge storage with deterministic hybrid retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .corpus import CorpusDocument

CHUNK_LINES = 80
CHUNK_OVERLAP = 20
RRF_K = 60


class EmbeddingBackend(Protocol):
    model: str
    revision: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class SourceFingerprint:
    path: str
    content_sha256: str
    repository: str
    source_revision: str


@dataclass(frozen=True)
class CollectionManifest:
    collection: str
    target: str
    embedding_model: str
    embedding_revision: str
    dimension: int
    sources: tuple[SourceFingerprint, ...]
    fingerprint: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "CollectionManifest":
        try:
            raw = json.loads(value)
            raw["sources"] = tuple(SourceFingerprint(**item) for item in raw["sources"])
            manifest = cls(**raw)
            expected = _manifest_fingerprint(
                manifest.collection,
                manifest.target,
                manifest.embedding_model,
                manifest.embedding_revision,
                manifest.dimension,
                manifest.sources,
            )
            if manifest.target != "a3" or manifest.fingerprint != expected:
                raise ValueError
            return manifest
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid collection manifest") from exc


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    source_revision: str
    content_sha256: str
    score: float
    lexical_rank: int | None
    vector_rank: int


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest_fingerprint(
    collection: str,
    target: str,
    model: str,
    revision: str,
    dimension: int,
    sources: tuple[SourceFingerprint, ...],
) -> str:
    identity = {
        "collection": collection,
        "dimension": dimension,
        "embedding_model": model,
        "embedding_revision": revision,
        "sources": [asdict(source) for source in sources],
        "target": target,
    }
    return _digest(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())


def _normalize(vector: Sequence[float], dimension: int) -> list[float]:
    if len(vector) != dimension or not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding dimension or values are invalid")
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else [0.0] * dimension


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("stored embedding dimension is invalid")
    return sum(a * b for a, b in zip(left, right))


def _chunks(document: CorpusDocument) -> list[tuple[str, int, int, str]]:
    lines = document.text.splitlines(keepends=True)
    if not lines:
        return []
    values = []
    for offset in range(0, len(lines), CHUNK_LINES - CHUNK_OVERLAP):
        selected = lines[offset : offset + CHUNK_LINES]
        text = "".join(selected)
        start, end = offset + 1, offset + len(selected)
        chunk_id = _digest(f"{document.digest}\0{start}\0{end}\0{text}".encode())
        values.append((chunk_id, start, end, text))
        if end == len(lines):
            break
    return values


class KnowledgeDB:
    """File-backed, immutable collection database for verified A3 documents."""

    def __init__(self, path: Path, embeddings: EmbeddingBackend, *, read_only: bool):
        self.path = path.resolve()
        self.embeddings = embeddings
        if read_only:
            self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
            self.connection.execute("PRAGMA query_only = ON")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def create(cls, path: Path, embeddings: EmbeddingBackend) -> "KnowledgeDB":
        if path == Path(":memory:"):
            raise ValueError("KnowledgeDB requires a file-backed database")
        database = cls(path, embeddings, read_only=False)
        database.connection.executescript("""
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY, manifest_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, collection TEXT NOT NULL,
                path TEXT NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
                text TEXT NOT NULL, source_revision TEXT NOT NULL,
                content_sha256 TEXT NOT NULL, vector_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_collection ON chunks(collection);
        """)
        return database

    @classmethod
    def open_read_only(cls, path: Path, embeddings: EmbeddingBackend) -> "KnowledgeDB":
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        database = cls(resolved, embeddings, read_only=True)
        try:
            database.connection.execute("SELECT name, manifest_json FROM collections LIMIT 0")
            database.connection.execute("SELECT chunk_id, vector_json FROM chunks LIMIT 0")
        except Exception:
            database.connection.close()
            raise
        return database

    def __enter__(self) -> "KnowledgeDB":
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.close()

    def manifest(self, collection: str) -> CollectionManifest:
        row = self.connection.execute(
            "SELECT manifest_json FROM collections WHERE name = ?", (collection,)
        ).fetchone()
        if row is None:
            raise KeyError(collection)
        return CollectionManifest.from_json(row["manifest_json"])

    def index(self, documents: Iterable[CorpusDocument]) -> CollectionManifest:
        provided = list(documents)
        if not provided:
            raise ValueError("a collection must contain documents")
        if any(item.target != "a3" for item in provided):
            raise ValueError("knowledge database accepts only target a3")
        collections = {item.collection for item in provided}
        if len(collections) != 1:
            raise ValueError("documents must share one collection")
        unique: dict[str, CorpusDocument] = {}
        for item in provided:
            if _digest(item.text.encode()) != item.content_sha256:
                raise ValueError(f"document digest mismatch for {item.path}")
            previous = unique.get(item.path)
            if previous is not None and previous != item:
                raise ValueError(f"conflicting duplicate document {item.path}")
            unique[item.path] = item
        ordered = tuple(unique[path] for path in sorted(unique))
        sources = tuple(SourceFingerprint(
            item.path, item.content_sha256, item.repository, item.source_revision
        ) for item in ordered)
        collection = next(iter(collections))
        fingerprint = _manifest_fingerprint(
            collection, "a3", self.embeddings.model, self.embeddings.revision,
            self.embeddings.dimension, sources,
        )
        manifest = CollectionManifest(
            collection, "a3", self.embeddings.model, self.embeddings.revision,
            self.embeddings.dimension, sources, fingerprint,
        )
        existing = self.connection.execute(
            "SELECT manifest_json FROM collections WHERE name = ?", (collection,)
        ).fetchone()
        if existing is not None:
            current = CollectionManifest.from_json(existing["manifest_json"])
            if current != manifest:
                raise ValueError(f"collection {collection!r} is immutable")
            return current
        chunks = [(item, chunk) for item in ordered for chunk in _chunks(item)]
        vectors = self.embeddings.embed_documents([chunk[3] for _, chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedding backend returned the wrong number of vectors")
        normalized = [_normalize(vector, self.embeddings.dimension) for vector in vectors]
        with self.connection:
            self.connection.execute(
                "INSERT INTO collections VALUES (?, ?)", (collection, manifest.to_json())
            )
            for (item, (chunk_id, start, end, text)), vector in zip(chunks, normalized):
                self.connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, collection, item.path, start, end, text,
                     item.source_revision, item.content_sha256,
                     json.dumps(vector, separators=(",", ":"))),
                )
        return manifest

    def query(self, collection: str, query: str, *, limit: int = 10) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        manifest = self.manifest(collection)
        if (self.embeddings.model, self.embeddings.revision, self.embeddings.dimension) != (
            manifest.embedding_model, manifest.embedding_revision, manifest.dimension
        ):
            raise ValueError("embedding backend does not match collection manifest")
        query_vectors = self.embeddings.embed_queries([query])
        if len(query_vectors) != 1:
            raise ValueError("embedding backend must return one query embedding")
        query_vector = _normalize(query_vectors[0], manifest.dimension)
        rows = self.connection.execute(
            "SELECT * FROM chunks WHERE collection = ?", (collection,)
        ).fetchall()
        terms = tuple(dict.fromkeys(term.lower() for term in re.findall(r"\w+", query)))
        lexical = sorted(
            ((sum(row["text"].lower().count(term) for term in terms), row) for row in rows),
            key=lambda item: (-item[0], item[1]["chunk_id"]),
        )
        lexical_rank = {
            row["chunk_id"]: rank for rank, (score, row) in enumerate(lexical, 1) if score
        }
        try:
            stored_vectors = {
                row["chunk_id"]: _normalize(json.loads(row["vector_json"]), manifest.dimension)
                for row in rows
            }
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("stored embedding is corrupt or has an invalid dimension") from exc
        vector = sorted(rows, key=lambda row: (
            -_cosine(query_vector, stored_vectors[row["chunk_id"]]), row["chunk_id"]
        ))
        vector_rank = {row["chunk_id"]: rank for rank, row in enumerate(vector, 1)}
        scored = []
        for row in rows:
            ranks = (lexical_rank.get(row["chunk_id"]), vector_rank[row["chunk_id"]])
            score = sum(1 / (RRF_K + rank) for rank in ranks if rank is not None)
            scored.append((score, row["chunk_id"], row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [SearchHit(
            chunk_id=chunk_id, path=row["path"], start_line=row["start_line"],
            end_line=row["end_line"], text=row["text"],
            source_revision=row["source_revision"], content_sha256=row["content_sha256"],
            score=score, lexical_rank=lexical_rank.get(chunk_id),
            vector_rank=vector_rank[chunk_id],
        ) for score, chunk_id, row in scored[:limit]]
