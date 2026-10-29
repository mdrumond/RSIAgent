import hashlib
import json

import pytest

from benchmarks.ascend.memory_consolidation import MemoryConsolidator, discover
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
        read.update(item['sha256'] for item in context.get('prefetched_artifacts', []))
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


def test_preserves_variants_and_evidence(tmp_path):
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


def test_conflicts_stay_separate_and_pairwise_check_prevents_bridge(tmp_path):
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b', 'c']]
    provider = Provider({frozenset(['b', 'c']): 'conflict'})
    result = prepare(tmp_path, rows, artifacts, provider)
    assert result['members'] == ['a', 'b']
    assert any(row['relation'] == 'conflict' for row in result['comparisons'])


def test_unrelated_memories_have_no_merge_proposal(tmp_path):
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b']]
    result = prepare(tmp_path, rows, artifacts, Provider({frozenset(['a', 'b']): 'unrelated'}))
    assert result['proposal'] is None


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
    agent = MemoryConsolidator(provider, clock=lambda: tick[0], prefetch_bytes=0)
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
    agent = MemoryConsolidator(provider, clock=lambda: tick[0], prefetch_bytes=0)
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
    memory = SubjectMemory(tmp_path/'memory', ledger=knowledge.ledger)
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
    final = memory.accept('reconsolidated', result['proposal'], result['provenance'])
    assert [digest(row) for row in memory.records()] == [final['version']]
    assert memory.read(old_a['version'])['proposal'] == a['proposal']
    if prior_operation == 'merge':
        assert memory.read(old_b['version'])['proposal'] == b['proposal']


def test_structured_citations_deduplicate_by_json_identity_and_preserve_variants(tmp_path):
    from copy import deepcopy
    artifacts = {}
    rows = [candidate(name, artifacts) for name in ['a', 'b']]
    citation = dict(collection='ops-math-7a9a19c3d2a8', path='math/cdist/op_kernel/arch35/cdist_reduce_hl.h',
        start_line=1, end_line=80, chunk_hash='24e1dc26c5957d4f17e7e75f6b771b7c05db493648b1438408f7d04fd5652cef',
        repository='https://gitcode.com/cann/ops-math.git', revision='64322f6487162c0934ec0364af57ed2f51175a03',
        source_sha256='a23214348e000d23fff4400d67989f1e007098b3c8ee496e1b24438df7783717', reference_exposed=False)
    changed = dict(citation, end_line=96)
    nested = dict(collection={'name': 'subject-memory', 'revision': 'v1'},
                  document={'path': 'memory.json'}, chunk={'lines': [1, 8], 'hash': 'chunk-a'})
    rows[0]['proposal']['citations'] = [citation, nested]
    rows[1]['proposal']['citations'] = [dict(reversed(list(citation.items()))), changed, deepcopy(nested)]
    original = deepcopy(rows)
    result = prepare(tmp_path, rows, artifacts)
    assert result['proposal']['citations'] == [citation, nested, changed]
    assert rows == original
    result['proposal']['citations'][1]['chunk']['lines'].append(9)
    assert rows == original
    assert result['provenance']['consolidation']['members'] == original


def test_reversed_pair_reuses_canonical_prompt_and_cached_judgment(tmp_path):
    from copy import deepcopy
    artifacts = {}
    a, b = [candidate(name, artifacts) for name in ['a', 'b']]
    a['provenance']['request'] = {'artifacts': {'historical-result': 'retained'}}
    original = deepcopy([a, b])
    provider = Provider()
    agent = MemoryConsolidator(provider, cache_root=tmp_path/'cache')
    first = agent.compare(b, a, lambda sha, **kw: artifacts[sha], remaining=lambda: 10)
    assert provider.contexts[0]['left']['id'] == 'a'
    assert provider.contexts[0]['right']['id'] == 'b'
    calls_after_first = len(agent.last_trace)
    second = agent.compare(a, b, lambda sha, **kw: artifacts[sha], remaining=lambda: 10)
    assert second == first
    assert len(provider.contexts) == 1
    assert len(agent.last_trace) == calls_after_first
    assert [a, b] == original
    memory = SubjectMemory(tmp_path/'memory')
    result = agent.prepare(b, [a, b], memory=memory,
        read_artifact=lambda sha, **kw: artifacts[sha], max_seconds=10)
    assert result['provenance']['consolidation']['members'] == [b, a]
    assert len(provider.contexts) == 1


def test_small_raw_bundle_allows_first_request_terminal_judgment(tmp_path):
    artifacts = {}
    a, b = [candidate(name, artifacts) for name in ['a', 'b']]
    requests = []
    def provider(messages, **kw):
        requests.append(messages)
        context = json.loads(messages[1]['content'])
        raw = {item['sha256']: item['text'].encode('utf-8') for item in context['prefetched_artifacts']}
        assert raw == artifacts
        return json.dumps(dict(relation='equivalent', reason='raw examples match', evidence=sorted(raw)))
    result = MemoryConsolidator(provider).compare(a, b, lambda sha, **kw: artifacts[sha], remaining=lambda: 10)
    assert result['relation'] == 'equivalent'
    assert len(requests) == 1


def test_large_artifact_keeps_paging_and_terminal_feedback_identifies_unread_offsets(tmp_path):
    data = b'x' * 9000
    artifacts = {}
    a, b = [candidate(name, artifacts) for name in ['a', 'b']]
    sha = hashlib.sha256(data).hexdigest()
    artifacts[sha] = data
    b['proposal']['evidence'].append(sha)
    requests = []
    def provider(messages, **kw):
        requests.append(messages)
        if len(requests) == 1:
            context = json.loads(messages[1]['content'])
            assert sha not in [item['sha256'] for item in context['prefetched_artifacts']]
            return json.dumps(dict(read_artifact=dict(sha256=sha, offset=0, limit=8192)))
        if len(requests) == 2:
            return json.dumps(dict(relation='variants', reason='premature', evidence=sorted(artifacts)))
        if len(requests) == 3:
            feedback = json.loads(messages[-1]['content'])
            assert feedback['missing_artifacts'] == [dict(sha256=sha, next_offset=8192)]
            return json.dumps(dict(read_artifact=dict(sha256=sha, offset=8192, limit=8192)))
        return json.dumps(dict(relation='variants', reason='all raw evidence read', evidence=sorted(artifacts)))
    result = MemoryConsolidator(provider, prefetch_bytes=100).compare(a, b,
        lambda sha, **kw: artifacts[sha], remaining=lambda: 10)
    assert result['relation'] == 'variants'
    assert len(requests) == 4


def test_prefetched_hash_mismatch_fails_before_provider_call():
    artifacts = {}
    a, b = [candidate(name, artifacts) for name in ['a', 'b']]
    calls = []
    agent = MemoryConsolidator(lambda *args, **kw: calls.append(args))
    with pytest.raises(ValueError, match='artifact bytes do not match'):
        agent.compare(a, b, lambda sha, **kw: b'wrong', remaining=lambda: 10)
    assert calls == []
