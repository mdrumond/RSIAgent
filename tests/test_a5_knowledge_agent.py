from __future__ import annotations

import json

import pytest

from benchmarks.a5kernels.knowledge import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    KnowledgeDB,
)
from benchmarks.a5kernels.knowledge_agent import (
    KnowledgeAgent,
    KnowledgeQuery,
    ProgressiveMemoryJournal,
    parse_knowledge_action,
)


class FakeEmbeddings:
    model = DEFAULT_EMBEDDING_MODEL
    revision = DEFAULT_EMBEDDING_REVISION

    def embed(self, texts):
        terms = ("vector", "matrix", "pipeline")
        return [[float(text.lower().count(term)) for term in terms] for text in texts]


def _database(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "guide.py"
    source.write_text("vector add kernel\n" * 5, encoding="utf-8")
    database = KnowledgeDB(tmp_path / "knowledge.sqlite", FakeEmbeddings())
    database.index(root, [source], collection="catlass", language="catlass")
    return database


@pytest.mark.parametrize(
    "value, expected",
    [
        ('{"knowledge_query":{"query":"vector add"}}', KnowledgeQuery("vector add")),
        ('{"knowledge_query":{"query":"matrix", "limit":2}}', KnowledgeQuery("matrix", 2)),
        ('{"program":{"code":"read db"}}', None),
        ('{"knowledge_query":{"query":"x", "collection":"ascendc"}}', None),
        ('{"knowledge_query":{"query":"", "limit":1}}', None),
        ('{"knowledge_query":{"query":"x", "limit":21}}', None),
    ],
)
def test_restricted_action_parser(value, expected):
    assert parse_knowledge_action(value) == expected


def test_gate_returns_validated_citations_and_journals_provenance(tmp_path):
    database = _database(tmp_path)
    journal = ProgressiveMemoryJournal(tmp_path / "memory" / "knowledge.jsonl")
    agent = KnowledgeAgent(
        enabled=True, database=database, collection="catlass", journal=journal
    )

    outcome = agent.execute(KnowledgeQuery("vector add", limit=1))
    payload = json.loads(outcome.observation)["knowledge_query"]

    assert payload["available"] is True
    assert payload["results"][0]["text"].startswith("vector add kernel")
    citation = payload["results"][0]["citation"]
    assert citation["collection"] == "catlass"
    assert citation["path"] == "guide.py"
    assert (citation["start_line"], citation["end_line"]) == (1, 5)
    assert len(citation["chunk_hash"]) == 64
    assert journal.read() == [
        {
            "sequence": 1,
            "kind": "knowledge_query",
            "enabled": True,
            "query": "vector add",
            "collection": "catlass",
            "collection_fingerprint": database.manifest("catlass").fingerprint,
            "embedding": {
                "model": DEFAULT_EMBEDDING_MODEL,
                "revision": DEFAULT_EMBEDDING_REVISION,
            },
            "citations": [citation],
        }
    ]


def test_gate_detects_citation_location_or_content_tampering(tmp_path):
    database = _database(tmp_path)
    hit = database.query("catlass", "vector", limit=1)[0]

    with pytest.raises(ValueError, match="location"):
        database.validate_citation(
            "catlass",
            chunk_id=hit.chunk_id,
            path="wrong.py",
            start_line=hit.start_line,
            end_line=hit.end_line,
        )
    database.connection.execute(
        "UPDATE chunks SET text = 'tampered' WHERE chunk_id = ?", (hit.chunk_id,)
    )
    with pytest.raises(ValueError, match="hash"):
        database.validate_citation(
            "catlass",
            chunk_id=hit.chunk_id,
            path=hit.path,
            start_line=hit.start_line,
            end_line=hit.end_line,
        )


def test_enabled_gate_makes_database_query_only(tmp_path):
    database = _database(tmp_path)
    KnowledgeAgent(
        enabled=True,
        database=database,
        collection="catlass",
        journal=ProgressiveMemoryJournal(tmp_path / "journal.jsonl"),
    )

    with pytest.raises(Exception, match="readonly"):
        database.connection.execute("DELETE FROM chunks")


def test_gate_rejects_embedding_mismatch_without_journaling(tmp_path):
    database = _database(tmp_path)
    database.close()

    class WrongEmbeddings(FakeEmbeddings):
        revision = "wrong-revision"

    journal = ProgressiveMemoryJournal(tmp_path / "memory" / "knowledge.jsonl")
    with KnowledgeDB(tmp_path / "knowledge.sqlite", WrongEmbeddings()) as mismatched:
        agent = KnowledgeAgent(
            enabled=True,
            database=mismatched,
            collection="catlass",
            journal=journal,
        )
        with pytest.raises(ValueError, match="does not match collection manifest"):
            agent.execute(KnowledgeQuery("vector"))

    assert journal.read() == []


def test_treatment_off_has_no_database_or_indexed_content(tmp_path):
    journal = ProgressiveMemoryJournal(tmp_path / "off" / "knowledge.jsonl")
    agent = KnowledgeAgent(enabled=False, journal=journal)

    outcome = agent.execute(KnowledgeQuery("vector secret", limit=3))

    assert outcome.observation == KnowledgeAgent.DISABLED_OBSERVATION
    assert "secret" not in outcome.observation
    assert journal.read() == [
        {
            "sequence": 1,
            "kind": "knowledge_query",
            "enabled": False,
            "query": "vector secret",
            "collection": None,
            "collection_fingerprint": None,
            "embedding": None,
            "citations": [],
        }
    ]
    with pytest.raises(ValueError, match="must not receive KDB"):
        KnowledgeAgent(
            enabled=False,
            journal=journal,
            database=object(),
            collection="catlass",
        )


def test_progressive_memory_journal_appends_stable_sequence(tmp_path):
    journal = ProgressiveMemoryJournal(tmp_path / "memory" / "knowledge.jsonl")
    agent = KnowledgeAgent(enabled=False, journal=journal)

    agent.query(KnowledgeQuery("first"))
    agent.query(KnowledgeQuery("second"))

    assert [entry["sequence"] for entry in journal.read()] == [1, 2]
    assert [entry["query"] for entry in journal.read()] == ["first", "second"]
