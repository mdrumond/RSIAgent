from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from benchmarks.a5kernels.knowledge import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    KnowledgeDB,
    build_manifest,
    chunk_source,
)


class FakeEmbeddings:
    model = DEFAULT_EMBEDDING_MODEL
    revision = DEFAULT_EMBEDDING_REVISION

    def embed(self, texts):
        # Stable and dependency-free; dimensions roughly represent kernel terms.
        terms = ("vector", "matrix", "pipeline")
        return [[float(text.lower().count(term)) for term in terms] for text in texts]


def test_chunk_source_uses_80_lines_with_20_line_overlap():
    data = "".join(f"line {number}\n" for number in range(1, 142)).encode()

    chunks = chunk_source("src/kernel.py", data)

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 80),
        (61, 140),
        (121, 141),
    ]
    assert chunks[0].text.splitlines()[-1] == "line 80"
    assert chunks[1].text.splitlines()[0] == "line 61"
    assert chunk_source("src/kernel.py", data) == chunks


def _write_sources(root: Path) -> tuple[Path, Path]:
    first = root / "vector.py"
    second = root / "matrix.py"
    first.write_text("vector add kernel\n" * 85, encoding="utf-8")
    second.write_text("matrix multiply pipeline\n" * 10, encoding="utf-8")
    return first, second


def test_index_manifest_is_stable_and_collection_is_immutable(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    sources = _write_sources(source_root)
    manifest_path = tmp_path / "manifest.json"

    with KnowledgeDB(tmp_path / "kdb.sqlite", FakeEmbeddings()) as database:
        manifest = database.index(
            source_root,
            reversed(sources),
            collection="catlass-v1",
            language="catlass",
            manifest_path=manifest_path,
        )
        manifest_path.unlink()
        repeated = database.index(
            source_root,
            sources,
            collection="catlass-v1",
            language="catlass",
            manifest_path=manifest_path,
        )
        assert repeated == manifest
        assert manifest.sources[0].path == "matrix.py"
        assert manifest.embedding_revision == DEFAULT_EMBEDDING_REVISION
        assert manifest_path.read_text() == manifest.to_json()

        sources[0].write_text("changed\n", encoding="utf-8")
        with pytest.raises(ValueError, match="immutable"):
            database.index(
                source_root,
                sources,
                collection="catlass-v1",
                language="catlass",
            )


def test_index_uses_one_immutable_read_for_manifest_and_chunks(tmp_path, monkeypatch):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "kernel.py"
    initial = b"vector kernel from snapshot\n"
    later = b"matrix kernel from a later read\n"
    source.write_bytes(initial)
    original_read_bytes = Path.read_bytes
    source_reads = 0

    def changing_read_bytes(path):
        nonlocal source_reads
        if path == source:
            source_reads += 1
            return initial if source_reads == 1 else later
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", changing_read_bytes)

    with KnowledgeDB(tmp_path / "kdb.sqlite", FakeEmbeddings()) as database:
        manifest = database.index(
            source_root,
            [source],
            collection="catlass-snapshot",
            language="catlass",
        )
        indexed_text = database.connection.execute(
            "SELECT text FROM chunks WHERE collection = ?",
            ("catlass-snapshot",),
        ).fetchone()["text"]

    assert source_reads == 1
    assert manifest.sources[0].sha256 == hashlib.sha256(initial).hexdigest()
    assert manifest.sources[0].size == len(initial)
    assert indexed_text == initial.decode()
    assert indexed_text != later.decode()


def test_hybrid_query_is_language_isolated_and_repeatable(tmp_path):
    catlass = tmp_path / "catlass"
    ascendc = tmp_path / "ascendc"
    catlass.mkdir()
    ascendc.mkdir()
    catlass_file = catlass / "guide.txt"
    ascendc_file = ascendc / "guide.txt"
    catlass_file.write_text("vector vector kernel\n", encoding="utf-8")
    ascendc_file.write_text("vector secret ascend c\n", encoding="utf-8")

    with KnowledgeDB(tmp_path / "kdb.sqlite", FakeEmbeddings()) as database:
        database.index(catlass, [catlass_file], collection="catlass", language="catlass")
        database.index(ascendc, [ascendc_file], collection="ascendc", language="ascendc")

        first = database.query("catlass", "how does vector-add work?", limit=5)
        second = database.query("catlass", "vector", limit=5)

    assert first[0].chunk_id == second[0].chunk_id
    assert [hit.path for hit in first] == ["guide.txt"]
    assert "secret" not in first[0].text
    assert first[0].lexical_rank == 1
    assert first[0].vector_rank == 1


def test_vector_ties_use_chunk_hash_as_stable_tiebreaker(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    sources = []
    for name in ("z.txt", "a.txt"):
        path = root / name
        path.write_text("unrelated\n", encoding="utf-8")
        sources.append(path)

    with KnowledgeDB(tmp_path / "kdb.sqlite", FakeEmbeddings()) as database:
        database.index(root, sources, collection="docs", language="catlass")
        hits = database.query("docs", "vector", limit=2)

    assert [hit.chunk_id for hit in hits] == sorted(hit.chunk_id for hit in hits)
    expected = hashlib.sha256(("a.txt\0" + "1\0" + "1\0unrelated\n").encode()).hexdigest()
    assert expected in {hit.chunk_id for hit in hits}


def test_query_rejects_a_different_embedding_space_before_embedding(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "guide.txt"
    source.write_text("vector kernel\n", encoding="utf-8")
    database_path = tmp_path / "kdb.sqlite"
    with KnowledgeDB(database_path, FakeEmbeddings()) as database:
        database.index(root, [source], collection="docs", language="catlass")

    class WrongRevision:
        model = DEFAULT_EMBEDDING_MODEL
        revision = "different-but-dimension-compatible"

        def embed(self, _texts):
            raise AssertionError("mismatched backend must not be invoked")

    with KnowledgeDB(database_path, WrongRevision()) as database:
        with pytest.raises(ValueError, match="does not match collection manifest"):
            database.query("docs", "vector")


def test_bm25_statistics_are_isolated_between_collections(tmp_path):
    first_root = tmp_path / "catlass"
    other_root = tmp_path / "ascendc"
    first_root.mkdir()
    other_root.mkdir()
    first_sources = []
    for name, text in (("vector.txt", "vector vector\n"), ("matrix.txt", "vector matrix\n")):
        path = first_root / name
        path.write_text(text, encoding="utf-8")
        first_sources.append(path)
    other_sources = []
    for number in range(25):
        path = other_root / f"overlap-{number}.txt"
        path.write_text("vector matrix pipeline\n", encoding="utf-8")
        other_sources.append(path)

    with KnowledgeDB(tmp_path / "kdb.sqlite", FakeEmbeddings()) as database:
        database.index(
            first_root, first_sources, collection="catlass", language="catlass"
        )
        before = database.query("catlass", "vector matrix", limit=5)
        database.index(
            other_root, other_sources, collection="ascendc", language="ascendc"
        )
        after = database.query("catlass", "vector matrix", limit=5)

    def lexical_signature(hits):
        return [
            (hit.chunk_id, hit.lexical_rank, hit.lexical_score)
            for hit in hits
            if hit.lexical_rank is not None
        ]

    assert lexical_signature(after) == lexical_signature(before)


def test_shared_fts_schema_is_migrated_to_collection_tables(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "guide.txt"
    source.write_text("vector kernel\n", encoding="utf-8")
    manifest = build_manifest(root, [source], collection="docs", language="catlass")
    chunk = chunk_source("guide.txt", source.read_bytes())[0]
    database_path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE collections (
            name TEXT PRIMARY KEY, language TEXT NOT NULL,
            fingerprint TEXT NOT NULL, manifest_json TEXT NOT NULL
        );
        CREATE TABLE chunks (
            rowid INTEGER PRIMARY KEY, collection TEXT NOT NULL,
            chunk_id TEXT NOT NULL, path TEXT NOT NULL,
            start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
            text TEXT NOT NULL, vector_json TEXT NOT NULL,
            UNIQUE(collection, chunk_id)
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text, content='chunks', content_rowid='rowid', tokenize='unicode61'
        );
        """
    )
    connection.execute(
        "INSERT INTO collections VALUES (?, ?, ?, ?)",
        ("docs", "catlass", manifest.fingerprint, manifest.to_json()),
    )
    cursor = connection.execute(
        "INSERT INTO chunks VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
        ("docs", chunk.chunk_id, chunk.path, 1, 1, chunk.text, json.dumps([1.0, 0.0, 0.0])),
    )
    connection.execute(
        "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
        (cursor.lastrowid, chunk.text),
    )
    connection.commit()
    connection.close()

    with KnowledgeDB(database_path, FakeEmbeddings()) as database:
        hits = database.query("docs", "vector")
        tables = {
            row[0]
            for row in database.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert hits[0].chunk_id == chunk.chunk_id
    assert "chunks_fts" not in tables
    assert any(name.startswith("collection_fts_") for name in tables)
