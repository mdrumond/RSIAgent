"""Synthetic publication and recovery behavior; no models or device calls."""
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from benchmarks.a5kernels.phase1_wave_runtime import LiveWaveCoordinator
from benchmarks.ascend.production import NativeProjectPool
from tests.test_a5_phase1_adaptive import SEED, _project, _result, _wave
from tests.test_a5_phase1_wave_runtime import MEMORY, wave, wave_result


def test_partial_wave_publishes_only_finalized_siblings_and_reopens_without_execution(tmp_path):
    seen = []
    def execute(request):
        name = request.assignment.project_id
        seen.append(name)
        if name == 'offline':
            raise OSError('CPL observation unavailable; retained remote:fake:job:owned')
        return wave_result(name.encode(), name != 'incorrect')
    spec = wave('first', 'offline', 'incorrect', 'last')
    runner = LiveWaveCoordinator(tmp_path, spec, MEMORY, execute, 2,
                                 allow_partial=True)
    result = runner.run()
    assert [c.project_id for c in result.commits] == ['first', 'last']
    publication = json.loads((tmp_path/'published.json').read_bytes())
    assert publication['completion'] == 'partial'
    assert publication['skipped_projects'] == ['offline', 'incorrect']
    assert publication['skipped_reasons']['offline']['classification'] == 'infrastructure-failure'
    assert 'remote:fake:job:owned' in publication['skipped_reasons']['offline']['summary']
    assert not (tmp_path/'branches/offline/memory-delta.bin').exists()
    assert runner.resume() == result
    assert set(seen) == {'first', 'offline', 'incorrect', 'last'} and len(seen) == 4
    with pytest.raises(ValueError, match='complete wave'):
        LiveWaveCoordinator(tmp_path, spec, MEMORY, execute, 1).resume()


def test_deadline_excludes_late_completion_and_never_executes_queued_project(tmp_path):
    now = [0]
    seen = []
    def execute(request):
        name = request.assignment.project_id
        seen.append(name)
        if name == 'late':
            now[0] = 10
        return wave_result(name.encode())
    runner = LiveWaveCoordinator(tmp_path, wave('good', 'late', 'queued'), MEMORY,
        execute, 1, allow_partial=True, remaining_seconds=lambda: 10-now[0])
    assert [c.project_id for c in runner.run().commits] == ['good']
    assert seen == ['good', 'late']
    for name in ['late', 'queued']:
        assert not (tmp_path/'branches'/name/'outcome.json').exists()
        record = json.loads((tmp_path/'branches'/name/'interruption.json').read_bytes())
        assert record['classification'] == 'deadline-exhausted'


def test_empty_partial_wave_closes_without_manufacturing_memory(tmp_path):
    def execute(_):
        raise TimeoutError('provider elapsed limit')
    runner = LiveWaveCoordinator(tmp_path, wave('a', 'b'), MEMORY, execute, 1, allow_partial=True)
    assert runner.run().commits == ()
    assert json.loads((tmp_path/'published.json').read_bytes())['completion'] == 'empty'
    assert json.loads(runner.read_published_memory())['deltas'] == []


def test_corrupt_retained_success_is_never_salvaged_as_partial(tmp_path):
    runner = LiveWaveCoordinator(tmp_path, wave('a'), MEMORY, lambda _: wave_result(b'lesson'),
                                 1, allow_partial=True)
    runner.run()
    (tmp_path/'branches/a/memory-delta.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='does not match outcome'):
        runner.finalize_retained()


def test_quarantine_preserves_healthy_worker_and_does_not_reuse_uncertain_worker():
    seen = []
    def broken(request):
        seen.append(('broken', request))
        raise OSError('unsettled remote job')
    def healthy(request):
        seen.append(('healthy', request))
        return request
    pool = NativeProjectPool([broken, healthy])
    with pytest.raises(OSError): pool('first')
    assert pool('second') == 'second'
    assert pool('third') == 'third'
    assert seen == [('broken', 'first'), ('healthy', 'second'), ('healthy', 'third')]
    assert pool.quarantined == [broken]
    with pytest.raises(TimeoutError): pool(SimpleNamespace(remaining_seconds=0))
    assert pool('fourth') == 'fourth'  # No work was dispatched on the expired request.


@pytest.mark.parametrize('close_method', ['finalize_pending_wave', 'resume'])
def test_retained_only_recovery_after_budget_exhaustion_is_atomic_and_idempotent(tmp_path, monkeypatch, close_method):
    names = ['good-a', 'good-b', 'good-c', 'good-d', 'bad', 'offline-a', 'offline-b', 'offline-c']
    def select(messages):
        specs = [_project(n, 'gm-ub-movement') for n in names]
        for index, spec in enumerate(specs):
            spec['shape'] = [33+index]
            spec['graph']['inputs'][0]['shape'] = [33+index]
        return _wave(messages[-1]['content'], specs)
    def execute(request):
        name = request.spec.project_id
        if name.startswith('offline'):
            raise OSError('retained job unavailable')
        return _result(request.spec, name != 'bad', 'host observation')
    strict = AdaptivePhase1Coordinator(tmp_path, 'Recover', SEED, CurriculumAgent(select), execute,
                                       max_parallel=3, project_target=8, max_seconds=10)
    proposal = strict.propose_wave()
    strict.approve_wave(proposal, names)
    with pytest.raises(RuntimeError, match='failed before verification'):
        strict.run_approved_wave()
    state_path = tmp_path/'state.json'
    state = json.loads(state_path.read_bytes());state['elapsed_seconds'] = 10
    state_path.write_text(json.dumps(state))
    def forbidden(*args, **kwargs):
        pytest.fail('recovery must not call curriculum or executor')
    recovery = AdaptivePhase1Coordinator(tmp_path, 'Recover', SEED, CurriculumAgent(forbidden), forbidden,
        max_parallel=3, project_target=8, max_seconds=10, allow_partial=True)
    with (tmp_path/'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already active'): recovery.finalize_pending_wave()
    original = recovery._atomic_json
    def crash(path, value):
        if path == state_path and value['wave_index'] == 2:
            raise OSError('crash before adaptive publication')
        original(path, value)
    monkeypatch.setattr(recovery, '_atomic_json', crash)
    with pytest.raises(OSError, match='crash before'): getattr(recovery, close_method)()
    monkeypatch.undo()
    report = getattr(recovery, close_method)()
    assert report['published_projects'] == names[:4]
    assert len(report['memory']) == 4 and report['failed_projects'] == 4
    assert report['elapsed_seconds'] == 10 and report['remaining_seconds'] == 0
    assert report['wave_results'] == [{'wave_id': 'wave-001', 'status': 'partial',
        'accepted_projects': names[:4], 'skipped_projects': names[4:], 'closure_reason': 'recovery'}]
    assert not (tmp_path/'pending.json').exists()
    assert recovery.finalize_pending_wave() == report
    assert recovery.resume()['published_projects'] == names[:4]


def test_closed_partial_wave_requires_explicit_next_proposal(tmp_path):
    calls, executed = [], []
    def select(messages):
        calls.append(messages)
        specs = ([_project('good', 'gm-ub-movement'), _project('offline', 'simd-vector')]
                 if len(calls) == 1 else [_project('fresh', 'padded-tail')])
        return _wave(messages[-1]['content'], specs)
    def execute(request):
        executed.append(request.spec.project_id)
        if request.spec.project_id == 'offline': raise OSError('unavailable')
        return _result(request.spec, True, 'host pass')
    runner = AdaptivePhase1Coordinator(tmp_path, 'Learn', SEED, CurriculumAgent(select), execute,
        max_parallel=2, project_target=3, max_seconds=100, allow_partial=True)
    report = runner.run()
    assert report['status'] == 'WAVE_CLOSED' and report['successful_projects'] == 1
    assert runner.resume()['published_projects'] == ['good']
    assert len(calls) == 1 and sorted(executed) == ['good', 'offline']
    proposal = runner.propose_wave()
    runner.approve_wave(proposal, ['fresh'])
    report = runner.run_approved_wave()
    assert report['published_projects'] == ['good', 'fresh']
    assert report['wave_results'][-1]['status'] == 'complete'
    assert executed.count('offline') == 1 and len(calls) == 2
