"""KDB-only delivery exercises real retrieval and ordinary Actor action admission."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.a5kernels.trial import ActorOutcome
from benchmarks.ascend.subject_runtime import build_subject_campaign
from benchmarks.ascend.subjects import Subject
from test_a5_knowledge import FakeEmbeddings
from test_a5_phase1_adaptive_actor import SourceBackend, _request, _spec
from test_subject_knowledge import proposal, SOURCE
from test_subject_runtime import bound


class RetrievalActor:
    def __init__(self):
        self.calls = []
        self.expected_memory_source = None
        self.expected_memory_snapshot = None
        self.expected_memory_document = None
        self.expect_candidate = False
        self.immutable_context = b''

    def __call__(self, instruction, *, profile, workspace, context, turn_parser, action_executor):
        assert json.loads(context.read_bytes()) == {}
        artifacts = action_executor.artifacts
        for digest in [self.expected_memory_source, self.expected_memory_snapshot, self.expected_memory_document]:
            with pytest.raises(FileNotFoundError):
                action_executor.read_artifacts.read_bytes(digest)
        def act(value):
            return action_executor(turn_parser(json.dumps(value)))
        query = act({'query-knowledge': {'query': 'vector increments', 'limit': 5}})
        if self.expect_candidate:
            assert action_executor.read_artifacts.read_bytes(self.expected_memory_source) == SOURCE
            assert action_executor.admitted[self.expected_memory_source] == 'knowledge'
            page = json.loads(act({'read-artifact': {'sha256': self.expected_memory_source}}).observation)['host']
            assert page['text'] == SOURCE.decode()
        else:
            with pytest.raises(FileNotFoundError):
                action_executor.read_artifacts.read_bytes(self.expected_memory_source)
        # The complete retrieval snapshot never becomes a readable artifact.
        with pytest.raises(FileNotFoundError):
            action_executor.read_artifacts.read_bytes(self.expected_memory_snapshot)
        self.calls.append((action_executor.request.actor_memory_delivery,
                           hashlib.sha256(action_executor.request.pre_wave_memory).hexdigest(), query))
        act({'write': {'slot': 'kernel', 'content': 'fixed source'}})
        act({'run': {}})
        final = act({'submit': {}})
        assert final.terminal
        return ActorOutcome(final.status, 4, 1, admitted_artifacts=final.admitted_artifacts)


def campaign(tmp_path):
    db = tmp_path/'knowledge.sqlite'
    reference = tmp_path/'reference.txt'
    reference.write_text('Vector increment reference')
    embeddings = FakeEmbeddings()
    with KnowledgeDB(db, embeddings) as database:
        database.index(tmp_path, [reference], collection='reference', language='ascend-c')
    actor = RetrievalActor()
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path/'backend.jsonl'))
    adaptive = AdaptiveActorProjectExecutor(executor, actor, object())
    def compose(settings, **options):
        result = bound(options['root'], kdb=db, embeddings=embeddings)
        result['actor'] = adaptive
        return result
    runtime = build_subject_campaign({'actor_memory_delivery':'kdb-only'}, target='a3', root=tmp_path/'campaign',
        subjects=[Subject('s','copies')], initial_memory=b'', compose=compose)
    return runtime, actor


def test_kdb_only_candidate_admission_and_arm_isolation(tmp_path, monkeypatch):
    from benchmarks.a5kernels import phase1_adaptive_actor
    presented = []
    original = phase1_adaptive_actor.memory_presentation
    def presentation(raw, artifacts):
        presented.append(raw)
        return original(raw, artifacts)
    monkeypatch.setattr(phase1_adaptive_actor, 'memory_presentation', presentation)
    runtime, actor = campaign(tmp_path)
    draft = dict(memory_id='draft', proposal=proposal(runtime.artifacts), provenance={})
    baseline = runtime.memory.snapshot()
    treatment = json.dumps(dict(schema='subject-memory-snapshot-v1', initial_memory='', memories=[draft])).encode()
    item = dict(project_id=_spec().project_id, spec=_spec().as_dict(), repairs=[])
    actor.expected_memory_source = draft['proposal']['after']['sha256']
    from benchmarks.ascend.subject_memory import digest
    actor.expected_memory_document = digest(draft)
    for name, snapshot, expect in [('control',baseline,False), ('treatment',treatment,True), ('control-after',baseline,False)]:
        actor.expected_memory_snapshot = hashlib.sha256(snapshot).hexdigest()
        actor.expect_candidate = expect
        if expect:
            retained = ContentAddressedArtifacts(tmp_path/name/'artifacts')
            retained.put_bytes(snapshot)
            retained.put_bytes(SOURCE)
            retained.put_json(draft)
        outcome = runtime.execute(item, snapshot, tmp_path/name, remaining_seconds=30)
        assert outcome.host_verified
    assert presented == [b'{}'] * 3
    assert [call[:2] for call in actor.calls] == [
        ('kdb-only', hashlib.sha256(value).hexdigest()) for value in [baseline,treatment,baseline]]
    assert runtime.memory.records() == []
    # Checkpoint identity binds the real retrieval view and the delivery route.
    with pytest.raises(ValueError, match='frozen inputs'):
        runtime.execute(item, baseline, tmp_path/'treatment', remaining_seconds=30)
    assert len(actor.calls) == 3


def test_delivery_mode_change_cannot_reuse_graph_checkpoint(tmp_path):
    from benchmarks.ascend.subject_execution import SubjectExecutor
    from test_subject_execution import graph_actor
    graph, actor = graph_actor(tmp_path, ['{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}','{"submit":{}}'])
    adapter = SubjectExecutor(graph)
    request = _request(tmp_path, graph.executor)
    assert adapter.graph(request).host_verified
    with pytest.raises(ValueError, match='frozen inputs'):
        adapter.graph(replace(request, actor_memory_delivery='kdb-only'))
    assert len(actor.observations) == 3


@pytest.mark.parametrize('mode', ['kdb-only', 'unknown'])
def test_disabled_kdb_and_unknown_modes_fail_before_actor(tmp_path, mode):
    executor = ProjectExecutor(SourceBackend(), EvidenceLedger(tmp_path/'backend.jsonl'))
    graph = AdaptiveActorProjectExecutor(executor, lambda *a, **k:pytest.fail('provider called'), object())
    with pytest.raises(ValueError):
        graph(replace(_request(tmp_path, executor), actor_memory_delivery=mode))


def test_campaign_kdb_only_rejects_disabled_knowledge(tmp_path):
    with pytest.raises(ValueError, match='requires enabled KDB'):
        build_subject_campaign({'actor_memory_delivery':'kdb-only'}, target='a3', root=tmp_path,
            subjects=[Subject('s','copies')], initial_memory=b'',
            compose=lambda settings,**options:bound(options['root']))


def test_kdb_only_rejects_campaign_seed_before_composition(tmp_path):
    with pytest.raises(ValueError, match='nonempty initial memory'):
        build_subject_campaign({'actor_memory_delivery':'kdb-only'}, target='a3', root=tmp_path,
            subjects=[Subject('s','copies')], initial_memory=b'prior lessons',
            compose=lambda *a, **k:pytest.fail('infrastructure composed'))
    assert not (tmp_path/'runtime-config.json').exists()


def test_kdb_only_rejects_external_snapshot_seed_before_execution(tmp_path):
    runtime, actor = campaign(tmp_path)
    item = dict(project_id=_spec().project_id, spec=_spec().as_dict(), repairs=[])
    snapshot = json.dumps(dict(schema='subject-memory-snapshot-v1',
        initial_memory='prior lessons', memories=[])).encode()
    with pytest.raises(ValueError, match='nonempty initial memory'):
        runtime.execute(item, snapshot, tmp_path/'branch', remaining_seconds=30)
    assert actor.calls == []
    assert not (tmp_path/'branch').exists()


def test_direct_delivery_seed_guard_preserves_indexed_default(tmp_path, monkeypatch):
    from test_subject_execution import graph_actor
    graph, actor = graph_actor(tmp_path, ['{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}','{"submit":{}}'])
    graph.query_knowledge = lambda *a, **k:pytest.fail('unexpected retrieval')
    snapshot = json.dumps(dict(schema='subject-memory-snapshot-v1',
        initial_memory='prior lessons', memories=[])).encode()
    request = replace(_request(tmp_path, graph.executor), pre_wave_memory=snapshot)
    original_prepare = graph.executor.prepare
    monkeypatch.setattr(graph.executor, 'prepare', lambda *a, **k:pytest.fail('infrastructure prepared'))
    with pytest.raises(ValueError, match='nonempty initial memory'):
        graph(replace(request, actor_memory_delivery='kdb-only'))
    assert actor.observations == []
    monkeypatch.setattr(graph.executor, 'prepare', original_prepare)
    # Default indexed delivery still admits the complete seed through its normal context.
    contexts = []
    def indexed_actor(instruction, *, context, turn_parser, action_executor, **options):
        contexts.append(context.read_bytes())
        for reply in actor.replies:
            outcome = action_executor(turn_parser(reply))
        return ActorOutcome(outcome.status, 3, 1, admitted_artifacts=outcome.admitted_artifacts)
    graph.actor = indexed_actor
    assert graph(replace(request, publish_learning=False)).evidence.verified
    assert contexts == [snapshot]
