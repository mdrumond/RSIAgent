"""Curriculum retrieval changes decisions and retains gated evidence across resume."""
import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent, CurriculumSession
from benchmarks.a5kernels.phase1_adaptive_production import curriculum_knowledge, _bind_campaign_identity
from benchmarks.a5kernels.knowledge_agent import KnowledgeQuery, ProgressiveMemoryJournal
from benchmarks.ascend.production import NativeProductionConfig
from test_a5_knowledge_agent import _database, FakeEmbeddings
from test_a5_phase1_curriculum_selector import context, project, wave
from test_a5_phase1_adaptive_production import _cell


def gate(tmp_path):
    db = _database(tmp_path)
    path = db.path
    db.close()
    return curriculum_knowledge(SimpleNamespace(kdb=path, collection="catlass", root=tmp_path), FakeEmbeddings())


def test_repeated_queries_drive_selection_and_survive_session_serialization(tmp_path):
    query, identity = gate(tmp_path)
    calls = []

    def complete(messages):
        calls.append(messages)
        if len(calls) <= 2:
            return json.dumps({"knowledge_query": {"query": "vector", "limit": 1}})
        observation = json.loads(messages[-1]["content"])["knowledge_query"]
        result = observation["results"][0]
        # Selection consumes the retrieved source and its host-validated citation.
        assert result["citation"]["collection"] == "catlass"
        assert result["text"].startswith("vector add kernel")
        return wave([project("retrieved-example")])

    decision, session = CurriculumAgent(complete, query_knowledge=query).select(context())
    assert decision.wave.projects[0].spec.project_id == "retrieved-example"
    assert session.turns == 3
    restored = CurriculumSession(**json.loads(json.dumps({"messages": session.messages, "turns": session.turns})))
    assert list(restored.messages) == list(session.messages)
    observations = [json.loads(row["content"])["knowledge_query"] for row in restored.messages
                    if row["role"] == "user" and row["content"].startswith('{"knowledge_query"')]
    assert len(observations) == 2
    assert observations[0]["results"][0]["citation"] == observations[1]["results"][0]["citation"]
    journal = ProgressiveMemoryJournal(tmp_path / "curriculum-knowledge.jsonl").read()
    assert len(journal) == 2
    assert identity["manifests"]["catlass"]


def test_disabled_retrieval_and_invalid_query_do_not_call_gate():
    replies = iter(['{"knowledge_query":{"query":"vector"}}', 'DECISION: STALLED\n{"reason":"No access"}'])
    decision, session = CurriculumAgent(lambda _: next(replies)).select(context())
    assert decision.status == "STALLED"
    assert session.turns == 2
    calls = []
    replies = iter(['{"knowledge_query":{"query":"vector","collection":"secret"}}',
                    'DECISION: SATURATED\n{"reason":"Enough evidence"}'])
    decision, _ = CurriculumAgent(lambda _: next(replies), query_knowledge=calls.append).select(context())
    assert decision.status == "SATURATED"
    assert calls == []


def test_retrieval_is_optional_when_enabled():
    calls = []
    decision, session = CurriculumAgent(lambda _: 'DECISION: STALLED\n{"reason":"Blocked"}',
                                      query_knowledge=calls.append).select(context())
    assert decision.status == "STALLED"
    assert session.turns == 1
    assert calls == []


def test_manifest_change_rejects_retrieval(tmp_path):
    query, _ = gate(tmp_path)
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    source = tmp_path / "sources" / "guide.py"
    source.write_text("matrix pipeline\n")
    # Simulate a configured database path being replaced by a different snapshot.
    (tmp_path / "knowledge.sqlite").unlink()
    with KnowledgeDB(tmp_path / "knowledge.sqlite", FakeEmbeddings()) as db:
        db.index(source.parent, [source], collection="catlass", language="catlass")
    with pytest.raises(ValueError, match="identity changed"):
        query(KnowledgeQuery("vector"))


def test_identity_preserves_historical_off_and_rejects_access_or_manifest_changes(tmp_path):
    cell = _cell()
    _bind_campaign_identity(tmp_path, cell)
    historical = (tmp_path / "campaign-identity.json").read_bytes()
    _bind_campaign_identity(tmp_path, cell, curriculum_kdb=None)
    assert (tmp_path / "campaign-identity.json").read_bytes() == historical
    enabled = {"enabled": True, "manifests": {"catlass": "a" * 64}}
    with pytest.raises(ValueError, match="different treatment"):
        _bind_campaign_identity(tmp_path, cell, curriculum_kdb=enabled)
    other = tmp_path / "enabled"
    _bind_campaign_identity(other, cell, curriculum_kdb=enabled)
    _bind_campaign_identity(other, cell, curriculum_kdb=enabled)
    for selection in (None, {"enabled": True, "manifests": {"catlass": "b" * 64}}):
        with pytest.raises(ValueError, match="different treatment"):
            _bind_campaign_identity(other, cell, curriculum_kdb=selection)


def test_native_without_kdb_rejects_curriculum_access(tmp_path):
    with pytest.raises(ValueError, match="configured KDB"):
        NativeProductionConfig(tmp_path, "a3", "arch", "local", "a" * 64, "b" * 40,
                               curriculum_kdb=True)
