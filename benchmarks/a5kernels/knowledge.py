"""Deterministic, language-isolated retrieval for kernel reference material."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Protocol, Sequence

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
CHUNK_LINES = 80
CHUNK_OVERLAP = 20
RRF_K = 60


class EmbeddingBackend(Protocol):
    """Minimal embedding interface; production loading stays outside the DB."""

    model: str
    revision: str

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class SourceFingerprint:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class _SourceSnapshot:
    fingerprint: SourceFingerprint
    data: bytes


@dataclass(frozen=True)
class CollectionManifest:
    collection: str
    language: str
    embedding_model: str
    embedding_revision: str
    sources: tuple[SourceFingerprint, ...]
    fingerprint: str
    chunk_lines: int = CHUNK_LINES
    chunk_overlap: int = CHUNK_OVERLAP

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, value: str) -> "CollectionManifest":
        raw = json.loads(value)
        raw["sources"] = tuple(SourceFingerprint(**item) for item in raw["sources"])
        return cls(**raw)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    score: float
    lexical_rank: int | None
    lexical_score: float | None
    vector_rank: int | None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    # POSIX paths make manifests stable across host operating systems.
    return PurePosixPath(path.relative_to(root)).as_posix()


def build_manifest(
    root: Path,
    paths: Iterable[Path],
    *,
    collection: str,
    language: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_revision: str = DEFAULT_EMBEDDING_REVISION,
) -> CollectionManifest:
    manifest, _snapshots = _snapshot_sources(
        root,
        paths,
        collection=collection,
        language=language,
        embedding_model=embedding_model,
        embedding_revision=embedding_revision,
    )
    return manifest


def _snapshot_sources(
    root: Path,
    paths: Iterable[Path],
    *,
    collection: str,
    language: str,
    embedding_model: str,
    embedding_revision: str,
) -> tuple[CollectionManifest, tuple[_SourceSnapshot, ...]]:
    snapshots = []
    for path in sorted(paths, key=lambda item: _relative_path(root, item)):
        relative_path = _relative_path(root, path)
        data = path.read_bytes()
        snapshots.append(
            _SourceSnapshot(
                SourceFingerprint(relative_path, _sha256(data), len(data)),
                data,
            )
        )
    immutable_snapshots = tuple(snapshots)
    sources = tuple(snapshot.fingerprint for snapshot in immutable_snapshots)
    identity = {
        "collection": collection,
        "language": language,
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
        "chunk_lines": CHUNK_LINES,
        "chunk_overlap": CHUNK_OVERLAP,
        "sources": [asdict(source) for source in sources],
    }
    fingerprint = _sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    return (
        CollectionManifest(
            collection=collection,
            language=language,
            embedding_model=embedding_model,
            embedding_revision=embedding_revision,
            sources=sources,
            fingerprint=fingerprint,
        ),
        immutable_snapshots,
    )


def chunk_source(path: str, data: bytes) -> list[Chunk]:
    """Split UTF-8 source into fixed windows, retaining exact final newlines."""
    text = data.decode("utf-8")
    lines = text.splitlines(keepends=True)
    if not lines and text == "":
        return []
    if not lines:  # Defensive: splitlines currently only returns [] for empty text.
        lines = [text]
    chunks = []
    step = CHUNK_LINES - CHUNK_OVERLAP
    for offset in range(0, len(lines), step):
        selected = lines[offset : offset + CHUNK_LINES]
        if not selected:
            break
        chunk_text = "".join(selected)
        start, end = offset + 1, offset + len(selected)
        identity = f"{path}\0{start}\0{end}\0{chunk_text}".encode()
        chunks.append(Chunk(_sha256(identity), path, start, end, chunk_text))
        if end == len(lines):
            break
    return chunks


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _normalize(vector: Sequence[float]) -> list[float]:
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embeddings must contain only finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else [0.0 for value in vector]


def _fts_query(value: str) -> str | None:
    """Turn user prose into a literal OR query instead of FTS syntax."""
    terms = re.findall(r"\w+", value, flags=re.UNICODE)
    return " OR ".join(f'"{term}"' for term in terms) or None


class KnowledgeDB:
    """SQLite FTS5 store with deterministic vector/lexical rank fusion."""

    def __init__(self, path: Path, embeddings: EmbeddingBackend):
        self.path = path
        self.embeddings = embeddings
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "KnowledgeDB":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                fts_table TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS chunks (
                rowid INTEGER PRIMARY KEY,
                collection TEXT NOT NULL REFERENCES collections(name),
                chunk_id TEXT NOT NULL,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                UNIQUE(collection, chunk_id)
            );
            """
        )
        columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(collections)")
        }
        if "fts_table" not in columns:
            self._migrate_shared_fts()

    @staticmethod
    def _fts_table_name(collection: str) -> str:
        # Only fixed ASCII plus a digest ever reaches a SQL identifier position.
        return f"collection_fts_{_sha256(collection.encode())}"

    def _ensure_fts_table(self, table: str) -> None:
        self.connection.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5(
                text, content='chunks', content_rowid='rowid', tokenize='unicode61'
            )"""
        )

    def _migrate_shared_fts(self) -> None:
        """Upgrade the pre-release shared FTS schema without changing content."""
        with self.connection:
            self.connection.execute("ALTER TABLE collections ADD COLUMN fts_table TEXT")
            collections = self.connection.execute("SELECT name FROM collections").fetchall()
            for row in collections:
                table = self._fts_table_name(row["name"])
                self._ensure_fts_table(table)
                self.connection.execute(
                    f"""INSERT INTO {table}(rowid, text)
                        SELECT rowid, text FROM chunks WHERE collection = ?""",
                    (row["name"],),
                )
                self.connection.execute(
                    "UPDATE collections SET fts_table = ? WHERE name = ?",
                    (table, row["name"]),
                )
            self.connection.execute("DROP TABLE IF EXISTS chunks_fts")

    def index(
        self,
        root: Path,
        paths: Iterable[Path],
        *,
        collection: str,
        language: str,
        manifest_path: Path | None = None,
    ) -> CollectionManifest:
        root = root.resolve()
        source_paths = tuple(dict.fromkeys(path.resolve() for path in paths))
        if not source_paths:
            raise ValueError("a collection must contain at least one source")
        if any(not path.is_file() or not path.is_relative_to(root) for path in source_paths):
            raise ValueError("all sources must be files beneath the collection root")
        manifest, snapshots = _snapshot_sources(
            root,
            source_paths,
            collection=collection,
            language=language,
            embedding_model=self.embeddings.model,
            embedding_revision=self.embeddings.revision,
        )
        existing = self.connection.execute(
            "SELECT manifest_json FROM collections WHERE name = ?", (collection,)
        ).fetchone()
        if existing:
            current = CollectionManifest.from_json(existing["manifest_json"])
            if current != manifest:
                raise ValueError(
                    f"collection {collection!r} is immutable; use a new collection name"
                )
            if manifest_path:
                manifest_path.write_text(current.to_json(), encoding="utf-8")
            return current

        chunks = [
            chunk
            for snapshot in snapshots
            for chunk in chunk_source(snapshot.fingerprint.path, snapshot.data)
        ]
        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedding backend returned the wrong number of vectors")
        with self.connection:
            fts_table = self._fts_table_name(collection)
            self.connection.execute(
                """INSERT INTO collections
                   (name, language, fingerprint, manifest_json, fts_table)
                   VALUES (?, ?, ?, ?, ?)""",
                (collection, language, manifest.fingerprint, manifest.to_json(), fts_table),
            )
            self._ensure_fts_table(fts_table)
            for chunk, vector in zip(chunks, vectors):
                cursor = self.connection.execute(
                    """INSERT INTO chunks
                       (collection, chunk_id, path, start_line, end_line, text, vector_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        collection,
                        chunk.chunk_id,
                        chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.text,
                        json.dumps(_normalize(vector), separators=(",", ":")),
                    ),
                )
                self.connection.execute(
                    f"INSERT INTO {fts_table}(rowid, text) VALUES (?, ?)",
                    (cursor.lastrowid, chunk.text),
                )
        if manifest_path:
            manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        return manifest

    def manifest(self, collection: str) -> CollectionManifest:
        row = self.connection.execute(
            "SELECT manifest_json FROM collections WHERE name = ?", (collection,)
        ).fetchone()
        if not row:
            raise KeyError(collection)
        return CollectionManifest.from_json(row["manifest_json"])

    def query(self, collection: str, query: str, *, limit: int = 10) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        collection_row = self.connection.execute(
            "SELECT manifest_json, fts_table FROM collections WHERE name = ?", (collection,)
        ).fetchone()
        if not collection_row:
            raise KeyError(collection)
        manifest = CollectionManifest.from_json(collection_row["manifest_json"])
        actual_embedding = (self.embeddings.model, self.embeddings.revision)
        expected_embedding = (manifest.embedding_model, manifest.embedding_revision)
        if actual_embedding != expected_embedding:
            raise ValueError(
                "embedding backend does not match collection manifest: "
                f"expected {expected_embedding[0]}@{expected_embedding[1]}, "
                f"got {actual_embedding[0]}@{actual_embedding[1]}"
            )
        candidate_limit = max(limit * 4, 20)
        lexical_query = _fts_query(query)
        fts_table = self._fts_table_name(collection)
        if collection_row["fts_table"] != fts_table:
            raise RuntimeError(f"invalid FTS metadata for collection {collection!r}")
        lexical_rows = [] if lexical_query is None else self.connection.execute(
            f"""SELECT c.*, bm25({fts_table}) AS lexical_score
               FROM {fts_table} JOIN chunks c ON c.rowid = {fts_table}.rowid
               WHERE {fts_table} MATCH ? AND c.collection = ?
               ORDER BY lexical_score ASC, c.chunk_id ASC LIMIT ?""",
            (lexical_query, collection, candidate_limit),
        ).fetchall()
        query_vectors = self.embeddings.embed([query])
        if len(query_vectors) != 1:
            raise ValueError("embedding backend must return one query vector")
        all_rows = self.connection.execute(
            "SELECT * FROM chunks WHERE collection = ?", (collection,)
        ).fetchall()
        normalized_query = _normalize(query_vectors[0])
        vector_rows = sorted(
            all_rows,
            key=lambda row: (
                -_cosine(normalized_query, json.loads(row["vector_json"])),
                row["chunk_id"],
            ),
        )[:candidate_limit]
        lexical_rank = {row["chunk_id"]: rank for rank, row in enumerate(lexical_rows, 1)}
        lexical_score = {row["chunk_id"]: row["lexical_score"] for row in lexical_rows}
        vector_rank = {row["chunk_id"]: rank for rank, row in enumerate(vector_rows, 1)}
        rows_by_id = {row["chunk_id"]: row for row in (*lexical_rows, *vector_rows)}
        scored = []
        for chunk_id in set(lexical_rank) | set(vector_rank):
            score = sum(
                1.0 / (RRF_K + rank)
                for rank in (lexical_rank.get(chunk_id), vector_rank.get(chunk_id))
                if rank is not None
            )
            scored.append((score, chunk_id, rows_by_id[chunk_id]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchHit(
                chunk_id=chunk_id,
                path=row["path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                text=row["text"],
                score=score,
                lexical_rank=lexical_rank.get(chunk_id),
                lexical_score=lexical_score.get(chunk_id),
                vector_rank=vector_rank.get(chunk_id),
            )
            for score, chunk_id, row in scored[:limit]
        ]
