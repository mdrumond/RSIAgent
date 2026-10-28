import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.ascend.memory_consolidation import MemoryConsolidator, discover, submit_consolidation
from benchmarks.ascend.subject_memory import SubjectMemory, StaleMemory


def candidate(name, artifacts, *, applicability='float32 tail', summary='Repair padded copy'):
    source = artifacts.setdefault(hashlib.sha256(name.encode()).hexdigest(), name.encode())
    sha = hashlib.sha256(source).hexdigest()
    return dict(id=name, provenance={'project': name}, proposal=dict(operation='create', base_versions=[],
        summary=summary, applicability=applicability, preconditions='tail is not aligned',
        expectations='matches retained oracle', limitations='tested shapes only',
        before=None, before_unavailable_reason='not retained',
        after=dict(sha256=sha, start_line=1, end_line=1), evidence=[sha], citations=[]))


class Provider:
    def __init__(self, relations=None):
        self.relations = relations or {}
        self.contexts = []

    def __call__(self, messages, *, remaining_seconds):
        context = json.loads(messages[1]['content'])
        read = {json.loads(message['content'])['sha256'] for message in messages[2:]
                if message['role'] == 'user' and 'sha256' in json.loads(message['content'])}
        missing = set(context['required_artifacts']) - read
        if missing:
            return json.dumps(dict(read_artifact=dict(sha256=sorted(missing)[0])))
        self.contexts.append(context)
        pair = frozenset([context['left']['id'], context['right']['id']])
        return json.dumps(dict(relation=self.relations.get(pair, 'equivalent'),
                               reason='same operation and checked conditions', evidence=sorted(read)))


def prepare(tmp_path, rows, artifacts, provider=None, memory=None):
    agent = MemoryConsolidator(provider or Provider())
    result = agent.prepare(rows[0], rows, memory=memory or SubjectMemory(tmp_path/'memory'),
        read_artifact=lambda sha, **kw: artifacts[sha], max_seconds=10)
    return result


def test_preserves_variants_evidence_and_routes_admission(tmp_path):
    artifacts = {}
    rows = [candidate('a', artifacts), candidate('b', artifacts, applicability='float16 tail')]
    original = json.loads(json.dumps(rows))
    result = prepare(tmp_path, rows, artifacts, Provider({frozenset(['a', 'b']): 'variants'}))
    assert result['members'] == ['a', 'b']
    assert result['proposal']['evidence'] == [rows[0]['proposal']['evidence'][0], rows[1]['proposal']['evidence'][0]]
    claims = result['provenance']['consolidation']['claims']
    assert [claim['fields']['applicability'] for claim in claims] == ['float32 tail', 'float16 tail']
    assert result['provenance']['consolidation']['members'] == original
    assert rows == original
    calls = []
    gate = SimpleNamespace(memory=SubjectMemory(tmp_path/'empty'),
                           submit=lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(ValueError, match='admit every member'):
        submit_consolidation(gate, result, {'project': 'a'}, remaining_seconds=30)
    assert calls == []


def test_conflicts_stay_separate_and_pairwise_check_prevents_bridge(tmp_path):
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b', 'c']]
    provider = Provider({frozenset(['b', 'c']): 'conflict'})
    result = prepare(tmp_path, rows, artifacts, provider)
    assert result['members'] == ['a', 'b']
    assert any(row['relation'] == 'conflict' for row in result['comparisons'])


def test_no_merge_has_no_submission(tmp_path):
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b']]
    result = prepare(tmp_path, rows, artifacts, Provider({frozenset(['a', 'b']): 'unrelated'}))
    assert result['proposal'] is None
    assert submit_consolidation(None, result, {}, remaining_seconds=1) is None


def test_accepted_replacement_retains_history_and_detects_stale_base(tmp_path):
    artifacts = {}
    a, b = [candidate(name, artifacts) for name in ['a', 'b']]
    memory = SubjectMemory(tmp_path/'memory')
    before = memory.accept('a', a['proposal'], a['provenance'])
    result = prepare(tmp_path, [b], artifacts, memory=memory)
    assert result['proposal']['base_versions'] == [before]
    admitted = memory.accept('merged', result['proposal'], result['provenance'])
    assert len(memory.records()) == 1
    assert memory.records()[0]['proposal'] == result['proposal']
    assert memory.read(before['version'])['proposal'] == a['proposal']
    assert memory.search('padded')[0]['version'] == admitted['version']
    with pytest.raises(StaleMemory):
        memory.accept('stale', result['proposal'], result['provenance'])


def test_unread_evidence_cannot_pass_and_deadline_is_enforced(tmp_path):
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b']]
    tick = [0]
    def provider(messages, **kw):
        tick[0] += 1
        return json.dumps(dict(relation='equivalent', reason='unsupported', evidence=list(artifacts)))
    agent = MemoryConsolidator(provider, clock=lambda: tick[0])
    result = agent.prepare(rows[0], rows, memory=SubjectMemory(tmp_path/'memory'),
                      read_artifact=lambda sha, **kw: artifacts[sha], max_seconds=3)
    assert result['proposal'] is None
    assert result['discovery']['incomplete'] is True


def test_artifact_hash_mismatch_never_reaches_decision(tmp_path):
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b']]
    tick = [0]
    def provider(messages, **kw):
        tick[0] += 1
        return json.dumps(dict(read_artifact=dict(sha256=rows[0]['proposal']['evidence'][0])))
    agent = MemoryConsolidator(provider, clock=lambda: tick[0])
    result = agent.prepare(rows[0], rows, memory=SubjectMemory(tmp_path/'memory'),
                      read_artifact=lambda sha, **kw: b'changed', max_seconds=3)
    assert result['proposal'] is None
    assert result['discovery']['incomplete'] is True


def test_retrieval_order_is_deterministic_and_excludes_self():
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'c', 'b']]
    assert [row['id'] for row in discover(rows[0], rows)] == ['b', 'c']


def test_bounded_discovery_and_comparison_cache(tmp_path):
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b', 'c', 'd']]
    provider = Provider()
    memory = SubjectMemory(tmp_path/'memory')
    agent = MemoryConsolidator(provider, max_peers=1, cache_root=tmp_path/'cache')
    first = agent.prepare(rows[0], rows, memory=memory,
        read_artifact=lambda sha, **kw: artifacts[sha], max_seconds=10)
    assert first['members'] == ['a', 'b']
    assert first['discovery']['candidates'] == ['b', 'c', 'd']
    assert len(provider.contexts) == 1
    second = agent.prepare(rows[0], rows, memory=memory,
        read_artifact=lambda sha, **kw: artifacts[sha], max_seconds=10)
    assert second == first
    assert len(provider.contexts) == 1


def test_submission_requires_current_admitted_members_and_uses_normal_gate(tmp_path):
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b']]
    memory = SubjectMemory(tmp_path/'memory')
    for row in rows:
        memory.accept(row['id'], row['proposal'], row['provenance'])
    result = prepare(tmp_path, rows, artifacts, memory=memory)
    calls = []
    gate = SimpleNamespace(memory=memory, submit=lambda *args, **kw: calls.append((args, kw)))
    identity = submit_consolidation(gate, result, {'project': 'a'}, remaining_seconds=30)
    assert calls[0][0] == (identity, result['proposal'], result['provenance'], {'project': 'a'})
    assert calls[0][1] == {'remaining_seconds': 30}
    assert len(memory.records()) == 2
    memory.accept('replacement', result['proposal'], result['provenance'])
    with pytest.raises(ValueError, match='admit every member'):
        submit_consolidation(gate, result, {}, remaining_seconds=30)


def test_published_kdb_view_excludes_old_versions_and_exposes_variant_sources(tmp_path):
    from benchmarks.ascend.subject_knowledge import SubjectKnowledge
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b']]
    # Secondary source need not originally appear in evidence.
    extra = b'compiler failure retained evidence'
    extra_sha = hashlib.sha256(extra).hexdigest()
    artifacts[extra_sha] = extra
    rows[1]['proposal']['evidence'] = [extra_sha]
    memory = SubjectMemory(tmp_path/'memory')
    for row in rows:
        memory.accept(row['id'], row['proposal'], row['provenance'])
    from benchmarks.ascend.subject_memory import digest
    original_versions = {record['memory_id']: digest(record) for record in memory.records()}
    result = prepare(tmp_path, rows, artifacts, memory=memory)
    admitted = memory.accept('merged', result['proposal'], result['provenance'])
    store = ContentAddressedArtifacts(tmp_path/'artifacts')
    for value in artifacts.values():
        store.put_bytes(value)
    knowledge = SubjectKnowledge(tmp_path, store, lambda: memory)
    response = knowledge.query({'query': 'padded copy'}, remaining_seconds=10)
    assert len(response['memories']) == 1
    hit = response['memories'][0]
    assert hit['version'] == admitted['version']
    assert hit['version'] not in original_versions.values()
    assert rows[1]['proposal']['after']['sha256'] in hit['artifacts'].values()
    assert extra_sha in hit['artifacts'].values()
    assert knowledge.read_artifact(rows[1]['proposal']['after']['sha256'], remaining_seconds=10) == b'b'


def test_discovery_discounts_shared_boilerplate():
    artifacts = {}
    rows = [candidate(str(i), artifacts, summary=summary) for i, summary in enumerate([
        'Repair padded copy', 'Repair padded copy', 'Reduce axis typing',
        'Matmul accumulation', 'Gather indexing', 'Relu clipping'])]
    assert [row['id'] for row in discover(rows[0], rows)] == ['1']


def test_relation_provider_does_not_receive_unrelated_producer_manifest(tmp_path):
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b']]
    rows[0]['provenance']['request'] = {'artifacts': {
        'unrelated-' + str(i): hashlib.sha256(str(i).encode()).hexdigest() for i in range(1500)}}
    provider = Provider()
    result = prepare(tmp_path, rows, artifacts, provider)
    sent = provider.contexts[0]
    assert set(sent['left']) == {'id', 'proposal'}
    assert sent['required_artifacts'] == sorted(artifacts)
    assert len(json.dumps(sent)) < 5000
    assert result['provenance']['consolidation']['members'][0]['provenance'] == rows[0]['provenance']


@pytest.mark.parametrize('prior_operation', ['append', 'merge'])
def test_reconsolidation_targets_only_current_versions_after_prior_update(tmp_path, prior_operation):
    from copy import deepcopy
    from benchmarks.ascend.subject_memory import digest
    artifacts = {}
    a, b, c = [candidate(name, artifacts) for name in ['a', 'b', 'c']]
    memory = SubjectMemory(tmp_path/'memory')
    old_a = memory.accept('a', a['proposal'], a['provenance'])
    old_b = memory.accept('b', b['proposal'], b['provenance'])
    update = deepcopy(a['proposal'])
    historical = [old_a, old_b] if prior_operation == 'merge' else [old_a]
    update.update(operation=prior_operation, base_versions=historical, limitations='updated tested scope')
    current_a = memory.accept('updated-a', update, a['provenance'])
    if prior_operation == 'merge':
        other = memory.accept('c', c['proposal'], c['provenance'])
    else:
        other = old_b
    result = prepare(tmp_path, [a], artifacts, memory=memory)
    expected = {base['memory_id']: base['version'] for base in [current_a, other]}
    assert {base['memory_id']: base['version'] for base in result['proposal']['base_versions']} == expected
    lineage_a = next(row for row in result['provenance']['consolidation']['members'] if row['id'] == 'a')
    assert lineage_a['proposal']['base_versions'] == historical
    assert lineage_a['base'] == current_a
    calls = []
    gate = SimpleNamespace(memory=memory, submit=lambda *args, **kw: calls.append((args, kw)))
    identity = submit_consolidation(gate, result, {}, remaining_seconds=30)
    assert calls[0][0][0] == identity
    final = memory.accept(identity, result['proposal'], result['provenance'])
    assert [digest(row) for row in memory.records()] == [final['version']]
    assert memory.read(old_a['version'])['proposal'] == a['proposal']
    if prior_operation == 'merge':
        assert memory.read(old_b['version'])['proposal'] == b['proposal']
