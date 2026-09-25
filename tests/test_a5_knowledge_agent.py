from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.loop as loop
from config.settings import load
from core.actor import Done, Program
from core.trace import ArtifactSink
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
        ('{"knowledge_query":{"query":"x", "collection":"ascendc"}}', None),
        ('{"knowledge_query":{"query":"", "limit":1}}', None),
        ('{"knowledge_query":{"query":"x", "limit":21}}', None),
    ],
)
def test_restricted_action_parser(value, expected):
    assert parse_knowledge_action(value) == expected


@pytest.mark.parametrize(
    "query, limit, error",
    [
        ("   ", 5, ValueError),
        ("x", True, TypeError),
        ("x", 0, ValueError),
        ("x", 21, ValueError),
    ],
)
def test_public_query_enforces_parser_invariants(query, limit, error):
    with pytest.raises(error):
        KnowledgeQuery(query, limit=limit)


def test_public_query_normalizes_whitespace():
    assert KnowledgeQuery("  vector add  ").query == "vector add"


def test_parser_delegates_builtin_done():
    assert isinstance(parse_knowledge_action('{"done":null}'), Done)


@pytest.mark.parametrize(
    "value",
    [
        'I should consult the reference first.\n{"knowledge_query":{"query":"vector"}}',
        '```json\n{"knowledge_query":{"query":"vector"}}\n```',
    ],
)
def test_parser_accepts_normal_actor_reply_shapes(value):
    assert parse_knowledge_action(value) == KnowledgeQuery("vector")


def test_parser_uses_last_knowledge_query_and_reports_duplicates():
    action = parse_knowledge_action(
        '{"knowledge_query":{"query":"draft"}}\n'
        'Correction: {"knowledge_query":{"query":"vector", "limit":2}}'
    )

    assert action == KnowledgeQuery("vector", limit=2, dup=2)


def test_parser_preserves_builtin_keep_working_precedence():
    action = parse_knowledge_action(
        '{"program":{"lang":"python","code":"print(1)"}}\n'
        '{"knowledge_query":{"query":"vector"}}'
    )

    assert isinstance(action, Program)
    assert action.code == "print(1)"


def test_knowledge_query_precedes_terminal_done_in_mixed_reply():
    action = parse_knowledge_action(
        '{"knowledge_query":{"query":"vector"}}\n{"done":null}'
    )

    assert action == KnowledgeQuery("vector")


def test_parser_preserves_whole_reply_rejection_for_api_envelopes():
    action = parse_knowledge_action(
        '{"name":"bash","arguments":{"code":"echo not executed"}}\n'
        '{"knowledge_query":{"query":"vector"}}'
    )

    assert action is None


def test_parser_preserves_duplicate_count_across_action_families():
    action = parse_knowledge_action(
        '{"knowledge_query":{"query":"vector"}}\n'
        '{"done":null}\n{"done":null}'
    )

    assert action == KnowledgeQuery("vector", dup=2)


def test_parser_propagates_query_duplicates_to_builtin_action():
    action = parse_knowledge_action(
        '{"program":{"lang":"python","code":"print(1)"}}\n'
        '{"knowledge_query":{"query":"draft"}}\n'
        '{"knowledge_query":{"query":"vector"}}'
    )

    assert isinstance(action, Program)
    assert action.dup == 2


def test_parser_preserves_done_when_query_candidate_is_invalid():
    action = parse_knowledge_action(
        '{"done":null}\n'
        '{"knowledge_query":{"query":"vector"},"metadata":"draft"}'
    )

    assert isinstance(action, Done)


def test_query_precedes_done_within_one_mixed_object():
    action = parse_knowledge_action(
        '{"knowledge_query":{"query":"vector"},"done":null}'
    )

    assert action == KnowledgeQuery("vector")


def test_parser_selects_last_valid_query_candidate():
    action = parse_knowledge_action(
        '{"knowledge_query":{"query":"vector"}}\n'
        '{"knowledge_query":{"query":""}}'
    )

    assert action == KnowledgeQuery("vector")


def test_read_only_view_preserves_relative_database_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    database = KnowledgeDB(Path("knowledge.sqlite"), FakeEmbeddings())
    original_path = database.path
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with database.read_only_view() as view:
        assert view.path == original_path
        assert view.connection.execute("SELECT count(*) FROM collections").fetchone()[0] == 0

    database.close()


def test_journal_preserves_relative_path_after_chdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    journal = ProgressiveMemoryJournal(Path("memory") / "knowledge.jsonl")
    original_path = journal.path
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    journal.append(
        enabled=False,
        collection=None,
        query="vector",
        citations=(),
    )

    assert journal.path == original_path
    assert original_path.is_file()
    assert journal.read()[0]["query"] == "vector"


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
    agent = KnowledgeAgent(
        enabled=True,
        database=database,
        collection="catlass",
        journal=ProgressiveMemoryJournal(tmp_path / "journal.jsonl"),
    )

    with pytest.raises(Exception, match="readonly"):
        agent.database.connection.execute("DELETE FROM chunks")
    database.connection.execute("CREATE TABLE caller_remains_writable (value TEXT)")
    agent.close()


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


def test_two_journal_instances_serialize_sequence_assignment(tmp_path):
    path = tmp_path / "memory" / "knowledge.jsonl"
    first = ProgressiveMemoryJournal(path)
    second = ProgressiveMemoryJournal(path)

    first.append(enabled=False, collection=None, query="first", citations=())
    second.append(enabled=False, collection=None, query="second", citations=())

    assert [entry["sequence"] for entry in first.read()] == [1, 2]


def test_query_then_builtin_done_terminates_actor_loop(monkeypatch, tmp_path):
    class VM:
        def run_command(self, *_args, **_kwargs):
            return ""

    replies = iter([
        '{"knowledge_query":{"query":"vector"}}',
        '{"done":null}',
    ])
    monkeypatch.setattr(loop, "chat", lambda *_args, **_kwargs: next(replies))
    cfg = load(None)
    cfg.max_iters = 3
    cfg.wall_clock_secs = 60
    cfg.history_keep_pairs = 0
    cfg.independent_verify = False
    cfg.practice_mode = True
    agent = KnowledgeAgent(
        enabled=False,
        journal=ProgressiveMemoryJournal(tmp_path / "memory" / "knowledge.jsonl"),
    )

    result, _ = loop.run_attempt(
        "consult references",
        VM(),
        cfg,
        ArtifactSink(str(tmp_path / "trace")),
        allow_noop_done=True,
        turn_parser=parse_knowledge_action,
        action_executor=agent.execute,
    )

    assert result.status == "done"
    assert result.iters == 2
