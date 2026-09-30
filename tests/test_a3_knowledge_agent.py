from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from benchmarks.a3kernels.corpus import CorpusDocument
from benchmarks.a3kernels.embeddings import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_REVISION
from benchmarks.a3kernels.knowledge import KnowledgeDB
from benchmarks.a3kernels.knowledge_agent import (
    KnowledgeAgent,
    KnowledgeQuery,
    QueryJournal,
)


class FakeEmbeddings:
    model = DEFAULT_EMBEDDING_MODEL
    revision = DEFAULT_EMBEDDING_REVISION
    dimension = 3

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        terms = ("vector", "copy", "pipeline")
        return [[float(text.lower().count(term)) for term in terms] for text in texts]

    def embed_queries(self, texts):
        self.calls += 1
        terms = ("vector", "copy", "pipeline")
        return [[float(text.lower().count(term)) for term in terms] for text in texts]


def document(path, text, *, collection="a3-docs"):
    return CorpusDocument(
        collection=collection,
        target="a3",
        repository="https://example.invalid/ascendc.git",
        source_revision="a" * 40,
        path=path,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        text=text,
    )


def database(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    encoder = FakeEmbeddings()
    with KnowledgeDB.create(path, encoder) as writable:
        writable.index([
            document("copy.cpp", "vector copy kernel\n"),
            document("pipeline.cpp", "pipeline stages\n"),
        ])
    return path, encoder


@pytest.mark.parametrize(
    "query,limit,error",
    [("", 5, ValueError), ("vector", True, TypeError), ("vector", 0, ValueError),
     ("vector", 21, ValueError)],
)
def test_query_schema_is_exact_and_bounded(query, limit, error):
    with pytest.raises(error):
        KnowledgeQuery(query, limit)
    with pytest.raises(TypeError, match="KnowledgeQuery"):
        KnowledgeAgent(enabled=False, journal=QueryJournal).query({"query": "vector"})


def test_enabled_gate_queries_fixed_collection_and_journals_validated_hits(tmp_path):
    path, encoder = database(tmp_path)
    journal_path = tmp_path / "memory" / "queries.jsonl"
    with KnowledgeDB.open_read_only(path, encoder) as readonly:
        agent = KnowledgeAgent(
            enabled=True,
            journal=QueryJournal(journal_path),
            database=readonly,
            collection="a3-docs",
        )
        results = agent.query(KnowledgeQuery("  vector copy  ", limit=1))

    assert len(results) == 1
    assert results[0].citation.path == "copy.cpp"
    assert results[0].citation.collection == "a3-docs"
    assert results[0].citation.source_revision == "a" * 40
    assert len(results[0].citation.content_sha256) == 64
    entry = QueryJournal(journal_path).read()[0]
    assert entry["query"] == "vector copy"
    assert entry["limit"] == 1
    assert entry["collection"] == "a3-docs"
    assert entry["collection_fingerprint"]
    assert entry["citations"] == [results[0].citation.__dict__]


def test_results_preserve_database_order_and_requested_limit(tmp_path):
    path, encoder = database(tmp_path)
    with KnowledgeDB.open_read_only(path, encoder) as readonly:
        expected = readonly.query("a3-docs", "vector pipeline", limit=2)
        agent = KnowledgeAgent(
            enabled=True, database=readonly, collection="a3-docs",
            journal=QueryJournal(tmp_path / "journal.jsonl"),
        )
        actual = agent.query(KnowledgeQuery("vector pipeline", limit=2))
    assert [item.citation.chunk_id for item in actual] == [item.chunk_id for item in expected]


def test_gate_rejects_other_collection_and_tampered_hit_before_journaling(tmp_path, monkeypatch):
    path, encoder = database(tmp_path)
    journal = QueryJournal(tmp_path / "journal.jsonl")
    with KnowledgeDB.open_read_only(path, encoder) as readonly:
        with pytest.raises(KeyError):
            KnowledgeAgent(
                enabled=True, database=readonly, collection="other", journal=journal
            )
        hit = readonly.query("a3-docs", "vector", limit=1)[0]
        monkeypatch.setattr(readonly, "query", lambda *_args, **_kwargs: [replace(hit, text="bad")])
        agent = KnowledgeAgent(
            enabled=True, database=readonly, collection="a3-docs", journal=journal
        )
        with pytest.raises(ValueError, match="validated indexed chunk"):
            agent.query(KnowledgeQuery("vector"))
    assert journal.read() == []


def test_disabled_treatment_makes_zero_backend_calls_and_leaks_no_content(tmp_path):
    class ForbiddenDatabase:
        def __getattribute__(self, name):
            raise AssertionError(f"disabled treatment touched database: {name}")

    journal = QueryJournal(tmp_path / "journal.jsonl")
    agent = KnowledgeAgent(enabled=False, journal=journal)
    before = FakeEmbeddings().calls
    assert agent.query(KnowledgeQuery("vector secret", 3)) == ()
    assert FakeEmbeddings().calls == before
    assert "secret" in journal.read()[0]["query"]
    with pytest.raises(ValueError, match="must not receive"):
        KnowledgeAgent(enabled=False, journal=journal, database=ForbiddenDatabase())


def test_journal_is_append_only_hash_chained_and_reopens(tmp_path):
    path = tmp_path / "queries.jsonl"
    first = QueryJournal(path)
    agent = KnowledgeAgent(enabled=False, journal=first)
    agent.query(KnowledgeQuery("first"))
    raw_first = path.read_bytes()
    KnowledgeAgent(enabled=False, journal=QueryJournal(path)).query(KnowledgeQuery("second"))
    entries = QueryJournal(path).read()
    assert path.read_bytes().startswith(raw_first)
    assert [entry["sequence"] for entry in entries] == [1, 2]
    assert entries[1]["previous_digest"] == entries[0]["entry_digest"]


def test_journal_rejects_middle_corruption_and_recovers_incomplete_tail(tmp_path):
    path = tmp_path / "queries.jsonl"
    journal = QueryJournal(path)
    KnowledgeAgent(enabled=False, journal=journal).query(KnowledgeQuery("first"))
    valid = path.read_bytes()
    path.write_bytes(valid + b'{"sequence":2')
    with pytest.raises(ValueError, match="incomplete final"):
        journal.read()
    assert journal.recover_incomplete_tail() == 1
    assert journal.read()[0]["query"] == "first"

    entry = json.loads(path.read_text().splitlines()[0])
    entry["query"] = "changed"
    path.write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValueError, match="digest"):
        journal.read()
    with pytest.raises(ValueError, match="only an incomplete"):
        journal.recover_incomplete_tail()
