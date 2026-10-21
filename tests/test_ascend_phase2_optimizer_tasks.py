import base64
import json

import pytest

from benchmarks.ascend.phase2_optimizer_tasks import select_optimizer_task, reject_optimizer_task
from optimizer.jobs import canonical, sha
from optimizer.memory import freeze_phase1_source, verify_snapshot


def seed(entries=()):
    artifact = b'original kernel code'
    body = dict(schema='optimizer-memory-snapshot-v1', memory_entries=list(entries),
                files={'source.c': dict(sha256=sha(artifact), base64=base64.b64encode(artifact).decode())})
    return {**body, 'snapshot_id': sha(canonical(body))}


def proposal(source='import torch\ndef forward(a,b,r): return torch.relu(a @ b + r)'):
    return dict(task_id='matmul-residual-relu', rationale='new composed kernel', citations=[],
                submission=dict(source=source, entrypoint='forward', signatures=[dict(args=[
                    dict(shape=[4, 8], dtype='float32'), dict(shape=[8, 16], dtype='float32'),
                    dict(shape=[4, 16], dtype='float32')])]))


def select(tmp_path, complete, **kwargs):
    return select_optimizer_task(complete=complete, snapshot=seed(), objective='learn',
                                 root=tmp_path, **kwargs)


@pytest.mark.parametrize('source', [
    'import torch\ndef forward(a,b,r): return torch.relu(a @ b + r)',
    'import torch\ndef forward(a,b,r): return (a @ b).sum(dim=1)',
])
def test_optimizer_compositions_without_graph_grammar(tmp_path, source):
    value = proposal(source)
    assert select(tmp_path, lambda _: json.dumps(value)) == value
    assert select(tmp_path, lambda _: pytest.fail('selected task should be retained')) == value
    assert json.loads((tmp_path / 'selection.json').read_bytes())['proposal_count'] == 1


def test_invalid_submission_feedback_then_correction(tmp_path):
    invalid = proposal()
    invalid['submission']['entrypoint'] = 'not a name'
    calls = []
    def complete(messages):
        calls.append(messages)
        return json.dumps(invalid if len(calls) == 1 else proposal())
    assert select(tmp_path, complete)['task_id'] == proposal()['task_id']
    feedback = json.loads(calls[-1][-1]['content'])['feedback']
    assert 'entrypoint' in feedback


def test_gated_kdb_and_hash_read_are_retained_across_provider_failure(tmp_path):
    responses = iter([json.dumps({'query': 'native matmul residual source'}),
                      json.dumps({'read_artifact': {'sha256': sha(b'original kernel code')}})])
    queries = []
    def query(text):
        queries.append(text)
        return {'results': [{'citation': {'repository': 'ops', 'path': 'kernel.cpp'}}]}
    def complete(messages):
        try:
            return next(responses)
        except StopIteration:
            assert 'original kernel code' in messages[-1]['content']
            raise RuntimeError('provider offline')
    with pytest.raises(RuntimeError, match='offline'):
        select(tmp_path, complete, query_knowledge=query)
    result = select(tmp_path, lambda _: json.dumps(proposal()), query_knowledge=query)
    assert len(queries) == 1
    assert result['citations'] == [{'repository': 'ops', 'path': 'kernel.cpp'}]
    assert json.loads((tmp_path / 'selection.json').read_bytes())['proposal_count'] == 1


def test_requires_kdb_until_elapsed_allocation_expires(tmp_path):
    now = [0.0]
    def complete(_):
        now[0] += 1
        return json.dumps(proposal())
    with pytest.raises(TimeoutError, match='elapsed-time'):
        select(tmp_path, complete, query_knowledge=lambda _: {}, max_seconds=5, clock=lambda: now[0])
    state = json.loads((tmp_path / 'selection.json').read_bytes())
    assert state['proposal_count'] == 4
    assert state['stage'] == 'failed'
    with pytest.raises(TimeoutError):
        select(tmp_path, lambda _: pytest.fail('spent allocation was reset'),
               query_knowledge=lambda _: {}, max_seconds=5, clock=lambda: now[0])


def test_host_rejection_is_actionable_and_counts_same_budget(tmp_path):
    select(tmp_path, lambda _: json.dumps(proposal()))
    reject_optimizer_task(tmp_path, 'representative case-000: matmul dimensions disagree')
    def complete(messages):
        assert 'matmul dimensions disagree' in messages[-1]['content']
        return json.dumps(proposal())
    select(tmp_path, complete)
    assert json.loads((tmp_path / 'selection.json').read_bytes())['proposal_count'] == 2
    with pytest.raises(ValueError, match='identity changed'):
        select_optimizer_task(complete=complete, snapshot=seed(), objective='different', root=tmp_path)


def test_practice_has_same_submission_format_and_weakness_context(tmp_path):
    def complete(messages):
        user = json.loads(messages[-1]['content'])
        assert user['kind'] == 'practice'
        assert user['context']['weakness'] == 'strided residual'
        return json.dumps(proposal())
    assert select(tmp_path, complete, kind='practice', context=dict(weakness='strided residual'))['submission']


def test_selection_resumes_with_json_normalized_context(tmp_path):
    context = {'shape': (4, 16), 'attempts': {1: ('compile', 'verify')}}
    expected = select(tmp_path, lambda _: json.dumps(proposal()), context=context)
    assert select(tmp_path, lambda _: pytest.fail('resume must reuse selection'), context=context) == expected


def test_reference_is_never_executed_in_curriculum(tmp_path):
    result = select(tmp_path, lambda _: json.dumps(proposal('raise RuntimeError("worker only")')))
    assert result['submission']['source'].startswith('raise')


def test_memory_is_bounded_and_all_lessons_remain_hash_readable(tmp_path):
    entries = [dict(project_spec=dict(project_id=f'old-{i}'),
                    agent_lessons=['long lesson ' * 10000]) for i in range(64)]
    def complete(messages):
        assert len(canonical(messages)) < 100_000
        memory = json.loads(messages[-1]['content'])['memory']
        assert len(memory['projects']) == 64
        assert all('sha256' in row for row in memory['projects'])
        return json.dumps(proposal())
    select_optimizer_task(complete=complete, snapshot=seed(entries), objective='learn', root=tmp_path)


def test_reviewed_phase1_seed_has_real_provenance_without_phase2(tmp_path):
    from tests.test_a3_reviewed_phase2 import native_source
    source, review, memory = native_source(tmp_path)
    snapshot = freeze_phase1_source(source, b'', review, snapshot_root=tmp_path / 'seed')
    assert snapshot['phase'] == 'reviewed-phase1'
    assert snapshot['terminal'] is None
    assert snapshot['memory_entries'] == [memory]
    assert not (tmp_path / 'seed' / 'state.json').exists()
    assert 'reviewed-seed/review.json' in snapshot['files']
    assert any('/objects/' in name for name in snapshot['files'])
    verify_snapshot(snapshot)
    select_optimizer_task(complete=lambda _: json.dumps(proposal()), snapshot=snapshot,
                          objective='learn', root=tmp_path / 'selection')


def test_phase1_corrupted_publication_rejected(tmp_path):
    from tests.test_a3_reviewed_phase2 import native_source
    source, review, _ = native_source(tmp_path)
    next(source.glob('waves/*/published.json')).write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        freeze_phase1_source(source, b'', review, snapshot_root=tmp_path / 'seed')


def test_phase1_export_rejects_changed_manifest(tmp_path):
    from benchmarks.ascend.phase2 import admit_native_snapshot
    from optimizer.memory import freeze_reviewed_phase1
    from tests.test_a3_reviewed_phase2 import native_source
    source, review, _ = native_source(tmp_path)
    admitted = admit_native_snapshot(source, b'', review, tmp_path / 'seed')
    manifest = admitted.root / 'snapshot.json'
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='manifest changed'):
        freeze_reviewed_phase1(admitted)


def test_corrections_and_tool_reads_have_no_count_limit(tmp_path):
    responses = iter(['bad'] * 4 + [json.dumps({'query': 'kernel'})] * 7
                     + [json.dumps({'read_artifact': {'sha256': sha(b'original kernel code')}})] * 8
                     + [json.dumps(proposal())])
    now = [0.0]
    def complete(_):
        now[0] += 1
        return next(responses)
    result = select(tmp_path, complete, query_knowledge=lambda _: {'results': []},
                    max_seconds=30, clock=lambda: now[0])
    assert result['task_id'] == proposal()['task_id']
    assert now[0] == 20
