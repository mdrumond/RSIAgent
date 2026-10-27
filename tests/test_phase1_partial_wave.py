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


def test_partial_usefulness_only_assesses_published_memory_not_unfinalized_delta(tmp_path):
    assessed = []
    def select(messages):
        return _wave(messages[-1]['content'], [_project('good', 'gm-ub-movement'),
                                              _project('offline', 'simd-vector')])
    def execute(request):
        if request.spec.project_id == 'offline':
            # A process can disappear after staging a delta but before outcome.
            (request.branch_root / 'memory-delta.bin').write_bytes(b'unfinalized delta')
            raise OSError('lost before terminal outcome')
        return _result(request.spec, True, 'host pass')
    runner = AdaptivePhase1Coordinator(tmp_path, 'Learn', SEED, CurriculumAgent(select), execute,
        max_parallel=1, project_target=2, allow_partial=True)
    runner.usefulness_verifier = SimpleNamespace(enabled=True,
        assess_memory=lambda memory, prior, source: assessed.append(source))
    report = runner.run()
    assert report['status'] == 'WAVE_CLOSED' and report['published_projects'] == ['good']
    assert assessed == ['wave-001/good']
    assert not (tmp_path / 'waves/wave-001/branches/offline/outcome.json').exists()


def test_project_admission_timeout_does_not_quarantine_a_healthy_worker():
    from benchmarks.a5kernels.phase1_wave_runtime import ProjectAdmissionTimeout
    calls = []
    def worker(request):
        calls.append(request)
        if request == 'expired':
            raise ProjectAdmissionTimeout('expired before work admission')
        return 'completed'
    pool = NativeProjectPool([worker])
    with pytest.raises(ProjectAdmissionTimeout):
        pool('expired')
    assert pool('fresh') == 'completed'
    assert pool.quarantined == [] and calls == ['expired', 'fresh']


@pytest.mark.parametrize('next_partial', [False, True])
def test_explicit_continuation_acknowledges_only_current_partial_boundary(tmp_path, next_partial):
    selected, executed, now = [], [], [0.0]
    def select(messages):
        selected.append(messages)
        names = ['good', 'offline'] if len(selected) == 1 else ['fresh', 'offline-next' if next_partial else 'last']
        specs = [_project(name, focus) for name, focus in zip(names,
            ['gm-ub-movement', 'simd-vector'] if len(selected) == 1 else ['padded-tail', 'simd-vector'])]
        return _wave(messages[-1]['content'], specs)
    def execute(request):
        executed.append(request.spec.project_id)
        now[0] += 1
        if request.spec.project_id.startswith('offline'):
            raise OSError('unavailable')
        return _result(request.spec, True, 'host pass')
    def build(target):
        return AdaptivePhase1Coordinator(tmp_path, 'Learn', SEED, CurriculumAgent(select), execute,
            max_parallel=1, project_target=target, max_seconds=100, clock=lambda: now[0], allow_partial=True)
    runner = build(2)
    first = runner.run()
    assert first['status'] == 'WAVE_CLOSED' and first['published_projects'] == ['good']
    wave_root = tmp_path / 'waves/wave-001'
    retained = {path.relative_to(wave_root): path.read_bytes() for path in wave_root.rglob('*') if path.is_file()}
    assert runner.resume()['status'] == 'WAVE_CLOSED'
    assert len(selected) == 1 and executed == ['good', 'offline']
    elapsed = first['elapsed_seconds']
    assert runner.continue_campaign(3)['elapsed_seconds'] == elapsed
    # The acknowledgement survives reconstructing the controller before dispatch.
    resumed = build(3)
    result = resumed.resume()
    assert len(selected) == 2 and executed[:2] == ['good', 'offline']
    assert result['elapsed_seconds'] == elapsed + 2
    assert retained == {path.relative_to(wave_root): path.read_bytes() for path in wave_root.rglob('*') if path.is_file()}
    if next_partial:
        assert result['status'] == 'WAVE_CLOSED'
        assert result['published_projects'] == ['good', 'fresh']
        assert resumed.resume()['status'] == 'WAVE_CLOSED'
        assert len(selected) == 2 and len(executed) == 4
    else:
        assert result['status'] == 'TARGET_REACHED'
        assert result['published_projects'] == ['good', 'fresh', 'last']


def test_explicit_partial_continuation_does_not_renew_exhausted_allocation(tmp_path):
    selected, executed = [], []
    def select(messages):
        selected.append(messages)
        return _wave(messages[-1]['content'], [_project('good', 'gm-ub-movement'),
                                              _project('offline', 'simd-vector')])
    def execute(request):
        executed.append(request.spec.project_id)
        if request.spec.project_id == 'offline':
            raise OSError('unavailable')
        return _result(request.spec, True, 'host pass')
    runner = AdaptivePhase1Coordinator(tmp_path, 'Learn', SEED, CurriculumAgent(select), execute,
        max_parallel=1, project_target=2, max_seconds=100, clock=lambda: 0, allow_partial=True)
    assert runner.run()['status'] == 'WAVE_CLOSED'
    path = tmp_path / 'state.json'
    state = json.loads(path.read_bytes())
    state['elapsed_seconds'] = 100
    path.write_text(json.dumps(state))
    runner.continue_campaign(3)
    result = runner.resume()
    assert result['status'] == 'TIME_BUDGET_EXHAUSTED' and result['remaining_seconds'] == 0
    assert len(selected) == 1 and executed == ['good', 'offline']


def test_first_resume_after_active_allocation_crash_publishes_retained_siblings(tmp_path):
    def select(messages):
        return _wave(messages[-1]['content'], [_project('good', 'gm-ub-movement'),
                                              _project('offline', 'simd-vector')])
    def execute(request):
        if request.spec.project_id == 'offline':
            raise OSError('lost observation')
        return _result(request.spec, True, 'host pass')
    strict = AdaptivePhase1Coordinator(tmp_path, 'Recover', SEED, CurriculumAgent(select), execute,
        max_parallel=1, project_target=2, max_seconds=100, clock=lambda: 0)
    proposal = strict.propose_wave()
    strict.approve_wave(proposal, ['good', 'offline'])
    with pytest.raises(RuntimeError, match='failed before verification'):
        strict.run_approved_wave()
    state_path = tmp_path/'state.json'
    state = json.loads(state_path.read_bytes())
    state.update(allocation_active=True, elapsed_seconds=3)
    state_path.write_text(json.dumps(state))
    retained = {path.relative_to(tmp_path): path.read_bytes()
                for path in (tmp_path/'waves/wave-001/branches/good').rglob('*') if path.is_file()}
    def forbidden(*args):
        pytest.fail('crash recovery must not select or execute new work')
    recovery = AdaptivePhase1Coordinator(tmp_path, 'Recover', SEED, CurriculumAgent(forbidden), forbidden,
        max_parallel=1, project_target=2, max_seconds=100, clock=lambda: 0, allow_partial=True)
    report = recovery.resume()
    assert report['published_projects'] == ['good'] and report['failed_projects'] == 1
    assert report['status'] == 'TIME_BUDGET_EXHAUSTED' and report['remaining_seconds'] == 0
    assert report['elapsed_seconds'] == 100 and not (tmp_path/'pending.json').exists()
    assert report['wave_results'][-1]['closure_reason'] == 'deadline'
    assert all((tmp_path/path).read_bytes() == data for path, data in retained.items())
    assert recovery.resume() == report


def test_automatic_remediation_keeps_infrastructure_partial_boundary(tmp_path):
    from benchmarks.ascend.automatic_remediation import AutomaticRemediation
    selected, executed = [], []
    def select(messages):
        selected.append(messages)
        return _wave(messages[-1]['content'], [_project('bad', 'gm-ub-movement'),
                                              _project('offline', 'simd-vector')])
    def execute(request):
        executed.append(request.spec.project_id)
        if request.spec.project_id == 'offline':
            raise OSError('unsettled transport observation')
        return _result(request.spec, False, 'host mismatch')
    runner = AdaptivePhase1Coordinator(tmp_path, 'Learn', SEED, CurriculumAgent(select), execute,
        max_parallel=1, project_target=2, max_seconds=100, allow_partial=True)
    AutomaticRemediation(runner, target='a3')
    report = runner.run()
    assert report['status'] == 'WAVE_CLOSED'
    assert len(selected) == 1 and executed == ['bad', 'offline']
    assert report['published_projects'] == [] and report['failed_projects'] == 2


def test_settled_semantic_partial_enters_remediation_without_manual_resume(tmp_path):
    from benchmarks.ascend.automatic_remediation import AutomaticRemediation
    selected, executed = [], []
    automatic = None
    def select(messages):
        selected.append(messages)
        if len(selected) == 1:
            return _wave(messages[-1]['content'], [_project('bad', 'gm-ub-movement'),
                                                  _project('good', 'simd-vector')])
        key, = automatic.feedback.ids
        return 'DECISION: SATURATED\n' + json.dumps(dict(reason='defer selected failure',
            failure_investigations=[dict(feedback_id=key, status='deferred', classification='learnable',
                reason='await another investigation')]))
    def execute(request):
        executed.append(request.spec.project_id)
        return _result(request.spec, request.spec.project_id == 'good', 'host observation')
    runner = AdaptivePhase1Coordinator(tmp_path, 'Learn', SEED, CurriculumAgent(select), execute,
        max_parallel=1, project_target=2, max_seconds=100, allow_partial=True)
    automatic = AutomaticRemediation(runner, target='a3')
    report = runner.run()
    assert len(selected) == 2 and executed == ['bad', 'good']
    assert report['status'] == 'SATURATED' and report['published_projects'] == ['good']
    assert len(automatic.store.records()) == 2
    assert next(row for row in automatic.store.records() if row['outcome']['attempt_id'] == 'bad')['lifecycle'] == 'failed'
