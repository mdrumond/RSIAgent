from __future__ import annotations

import json
import sqlite3

import pytest

from benchmarks.a3kernels.corpus import CorpusDocument
from benchmarks.a3kernels.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
)
from benchmarks.a3kernels.knowledge import KnowledgeDB


class FakeEmbeddings:
    model = DEFAULT_EMBEDDING_MODEL
    revision = DEFAULT_EMBEDDING_REVISION
    dimension = 3

    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(("documents", list(texts)))
        terms = ("vector", "copy", "pipeline")
        return [[float(text.lower().count(term)) for term in terms] for text in texts]

    def embed_queries(self, texts):
        self.calls.append(("queries", list(texts)))
        return self.embed_documents(texts)


def document(path: str, text: str, *, collection="a3-docs") -> CorpusDocument:
    import hashlib

    digest = hashlib.sha256(text.encode()).hexdigest()
    return CorpusDocument(
        collection=collection,
        target="a3",
        repository="https://example.invalid/ascendc.git",
        source_revision="a" * 40,
        path=path,
        content_sha256=digest,
        text=text,
    )


def test_index_reopen_query_retains_a3_provenance_and_digests(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    encoder = FakeEmbeddings()
    docs = [
        document("z.cpp", "pipeline stages\n"),
        document("a.cpp", "vector copy kernel\n"),
    ]
    with KnowledgeDB.create(path, encoder) as database:
        manifest = database.index(reversed(docs))
        assert [source.path for source in manifest.sources] == ["a.cpp", "z.cpp"]
        assert manifest.target == "a3"
        assert manifest.dimension == 3
        assert len(manifest.fingerprint) == 64

    with KnowledgeDB.open_read_only(path, encoder) as database:
        hits = database.query("a3-docs", "vector copy", limit=2)
        assert hits[0].path == "a.cpp"
        assert hits[0].source_revision == "a" * 40
        assert hits[0].content_sha256 == docs[1].content_sha256
        assert hits == database.query("a3-docs", "vector copy", limit=2)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            database.connection.execute("DELETE FROM chunks")


def test_index_is_immutable_and_duplicate_input_is_deterministic(tmp_path):
    item = document("kernel.cpp", "vector kernel\n")
    with KnowledgeDB.create(tmp_path / "db.sqlite", FakeEmbeddings()) as database:
        first = database.index([item, item])
        second = database.index([item])
        assert first == second
        assert database.connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
        with pytest.raises(ValueError, match="immutable"):
            database.index([document("kernel.cpp", "changed vector kernel\n")])


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda item: CorpusDocument(**{**item.__dict__, "target": "a5"}), "target"),
        (lambda item: CorpusDocument(**{**item.__dict__, "collection": "other"}), "collection"),
        (lambda item: CorpusDocument(**{**item.__dict__, "content_sha256": "0" * 64}), "digest"),
    ],
)
def test_index_rejects_target_collection_and_digest_mismatch(tmp_path, mutation, message):
    item = document("kernel.cpp", "vector kernel\n")
    with KnowledgeDB.create(tmp_path / "db.sqlite", FakeEmbeddings()) as database:
        values = [item, mutation(item)] if message == "collection" else [mutation(item)]
        with pytest.raises(ValueError, match=message):
            database.index(values)


@pytest.mark.parametrize("vectors", [[], [[1.0, 2.0]], [[1.0, 2.0, float("nan")]]])
def test_index_rejects_wrong_count_dimension_or_nonfinite_vectors(tmp_path, vectors):
    class Broken(FakeEmbeddings):
        def embed_documents(self, _texts):
            return vectors

    with KnowledgeDB.create(tmp_path / "db.sqlite", Broken()) as database:
        with pytest.raises(ValueError, match="embedding"):
            database.index([document("kernel.cpp", "vector\n")])
        assert database.connection.execute("SELECT count(*) FROM collections").fetchone()[0] == 0


def test_backend_identity_mismatch_fails_before_query_encoding(tmp_path):
    path = tmp_path / "db.sqlite"
    with KnowledgeDB.create(path, FakeEmbeddings()) as database:
        database.index([document("kernel.cpp", "vector\n")])

    wrong = FakeEmbeddings()
    wrong.revision = "b" * 40
    with KnowledgeDB.open_read_only(path, wrong) as database:
        with pytest.raises(ValueError, match="embedding backend"):
            database.query("a3-docs", "vector")
    assert wrong.calls == []


def test_missing_database_and_ordinary_corruption_fail_cleanly(tmp_path):
    with pytest.raises(FileNotFoundError):
        KnowledgeDB.open_read_only(tmp_path / "missing.sqlite", FakeEmbeddings())
    corrupt = tmp_path / "corrupt.sqlite"
    corrupt.write_text("not sqlite")
    with pytest.raises(sqlite3.DatabaseError):
        KnowledgeDB.open_read_only(corrupt, FakeEmbeddings())


@pytest.mark.parametrize("vector", [json.dumps([1]), "not-json"])
def test_query_detects_corrupt_vector_and_manifest(tmp_path, vector):
    path = tmp_path / "db.sqlite"
    with KnowledgeDB.create(path, FakeEmbeddings()) as database:
        database.index([document("kernel.cpp", "vector\n")])
        database.connection.execute("UPDATE chunks SET vector_json = ?", (vector,))
        database.connection.commit()
        with pytest.raises(ValueError, match="dimension"):
            database.query("a3-docs", "vector")
        database.connection.execute("UPDATE collections SET manifest_json = 'bad'")
        database.connection.commit()
        with pytest.raises(ValueError, match="manifest"):
            database.manifest("a3-docs")


def test_vector_ties_have_stable_digest_order(tmp_path):
    class Ties(FakeEmbeddings):
        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    with KnowledgeDB.create(tmp_path / "db.sqlite", Ties()) as database:
        database.index([document("z.cpp", "one\n"), document("a.cpp", "two\n")])
        hits = database.query("a3-docs", "anything", limit=2)
    assert [hit.chunk_id for hit in hits] == sorted(hit.chunk_id for hit in hits)
