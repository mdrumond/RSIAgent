"""Behavioral parent scheduling tests; no device or provider is fabricated."""
import json
from types import SimpleNamespace

import pytest

from rsi import project_wave as module
from rsi.project_wave import ProjectWave
from rsi.python_execution import atomic_json


def wave(tmp_path, projects=('a', 'b'), parallel=2):
    return ProjectWave(tmp_path, projects, ('physical-0', 'physical-1'),
                       b'frozen memory', b'frozen skills', max_parallel=parallel)


def processes(monkeypatch, *, failures=(), stop=None):
    requests, live = [], []
    peak = [0]

    class Process:
        def __init__(self, target, args):
            self.request = args[1]
            self.alive = True

        def start(self):
            requests.append(self.request)
            live.append(self)
            peak[0] = max(peak[0], len(live))
            assert len({p.request.device for p in live}) == len(live)
            if stop:
                stop()

        def is_alive(self):
            # First polling pass settles this synthetic process. No actor loop.
            self.alive = False
            return False

        def join(self):
            if self in live:
                live.remove(self)
                r = self.request
                if r.project_id not in failures:
                    atomic_json(r.root.parent / 'wave-result.json', dict(
                        dispatch_id=r.dispatch_id, actor_seconds=12.5,
                        result={'status': 'accepted'}, settled=True))

    monkeypatch.setattr(module.multiprocessing, 'get_context',
                        lambda name: SimpleNamespace(Process=Process))
    return requests, peak


def test_concurrency_snapshot_barrier_and_retained_restart(tmp_path, monkeypatch):
    pool = wave(tmp_path, ('a', 'b', 'c'))
    requests, peak = processes(monkeypatch)
    with pytest.raises(RuntimeError):
        pool.publish()
    state = pool.run(None)
    assert peak[0] == 2
    assert [r.project_id for r in requests] == ['a', 'b', 'c']
    assert all(r.memory == b'frozen memory' and r.skills == b'frozen skills' for r in requests)
    assert all(r.actor_seconds == 7200 and r.actor_elapsed == 0 for r in requests)
    assert all(r['status'] == 'completed' for r in state['projects'].values())
    published = pool.publish()
    reopened = wave(tmp_path, ('a', 'b', 'c'))
    reopened.run(None)
    assert len(requests) == 3
    assert reopened.publish() == published
    assert [r['project_id'] for r in published['outcomes']] == ['a', 'b', 'c']
    with pytest.raises(ValueError):
        ProjectWave(tmp_path, ('a', 'b', 'c'), ('physical-0', 'physical-1'),
                    b'changed memory', b'frozen skills', max_parallel=2)


def test_crash_holds_lease_and_original_handle_until_observed(tmp_path, monkeypatch):
    pool = wave(tmp_path, ('a', 'b', 'c'))
    requests, _ = processes(monkeypatch, failures=('a',))
    state = pool.run(None)
    assert state['projects']['a']['status'] == 'pending'
    assert state['projects']['a']['device'] == 'physical-0'
    assert [r.device for r in requests] == ['physical-0', 'physical-1', 'physical-1']
    checkpoint = tmp_path / 'projects/a/runner'
    checkpoint.mkdir()
    atomic_json(checkpoint / 'workflow.json', {'pending': {'handle': 'original-handle'}})
    reopened = wave(tmp_path, ('a', 'b', 'c'))
    reopened.run(None)
    assert len(requests) == 3
    assert reopened.status()['projects']['a']['workflow_pending']['handle'] == 'original-handle'
    with pytest.raises(RuntimeError):
        reopened.publish()
    with pytest.raises(ValueError):
        reopened.reconcile('a', lambda req, row: {'closed': False})
    observed = []
    def observe(request, row):
        observed.append(request.dispatch_id)
        return dict(closed=True, observation='original controller and target closed',
                    actor_seconds=40., resume_ready=True, harness_failure=True)
    reopened.reconcile('a', observe)
    assert observed == [requests[0].dispatch_id]
    reopened.grant_harness_recovery('a', observation='transport harness recovery')
    with pytest.raises(ValueError):
        reopened.grant_harness_recovery('a', observation='second recovery')
    more, _ = processes(monkeypatch, failures=('a',))
    reopened.run(None)
    assert len(more) == 1
    assert more[0].reopen and more[0].actor_elapsed == 40.
    assert more[0].actor_seconds == 14400.


def test_stop_drains_only_admitted_and_resume_keeps_results(tmp_path, monkeypatch):
    pool = wave(tmp_path, parallel=1)
    requests, _ = processes(monkeypatch, stop=pool.stop)
    state = pool.run(None)
    # Stop is observed before the next admission, after the current worker.
    assert len(requests) == 1
    assert state['projects']['a']['status'] == 'completed'
    assert state['projects']['b']['status'] == 'ready'
    pool.resume()
    more, _ = processes(monkeypatch)
    pool.run(None)
    assert [r.project_id for r in more] == ['b']
    pool.publish()


def test_parent_writer_exclusion_and_elapsed_no_refund(tmp_path, monkeypatch):
    pool = wave(tmp_path)
    with pool._lock():
        with pytest.raises(RuntimeError):
            wave(tmp_path)
    processes(monkeypatch, failures=('a',))
    pool.run(None)
    pool.reconcile('a', lambda request, row: dict(closed=True,
        observation='closed with reconciled accounting', actor_seconds=100., resume_ready=False))
    with pytest.raises(ValueError):
        pool.reconcile('a', lambda request, row: dict(closed=True,
            observation='incorrect refund', actor_seconds=10., resume_ready=True))
    assert pool.status()['projects']['a']['actor_seconds'] == 100.


def test_worker_delegates_canonical_runner_and_preserves_elapsed(tmp_path, monkeypatch):
    from rsi import project_runner
    from rsi.contracts import BudgetLimits
    calls = []
    request = module.WaveRequest('a', tmp_path / 'runner', 'physical-0',
                                b'memory', b'skills', 7200., 20., True, 'dispatch')

    class Runner:
        root = request.root
        workflow = SimpleNamespace(limits=BudgetLimits(
            actor_seconds=7200, design_iterations=None,
            implementation_iterations=None, execution_tries=None),
            actor_allocation=lambda: 7200.,
            accounting=SimpleNamespace(seconds=lambda: 21.), pending=None)

        def run(self):
            calls.append('canonical-run')
            return {'status': 'accepted'}

    monkeypatch.setattr(project_runner, 'ProjectRunner', Runner)
    module._worker(lambda req: Runner(), request)
    retained = json.loads((tmp_path / 'wave-result.json').read_text())
    assert calls == ['canonical-run']
    assert retained['settled'] is True
    assert retained['actor_seconds'] == 21.
    Runner.workflow.accounting.seconds = lambda: 1.
    with pytest.raises(ValueError):
        module._worker(lambda req: Runner(), request)
    assert calls == ['canonical-run']


def test_orphan_completion_requires_closure_before_publication(tmp_path, monkeypatch):
    pool = wave(tmp_path, ('a',), parallel=1)
    requests, _ = processes(monkeypatch, failures=('a',))
    pool.run(None)
    request = requests[0]
    atomic_json(request.root.parent / 'wave-result.json', dict(
        dispatch_id=request.dispatch_id, actor_seconds=60.,
        result={'status': 'accepted'}, settled=True))
    reopened = wave(tmp_path, ('a',), parallel=1)
    reopened.run(None)
    with pytest.raises(RuntimeError):
        reopened.publish()
    reopened.reconcile('a', lambda req, row: dict(closed=True,
        observation='original worker reaped and target closed', actor_seconds=75.))
    outcome = reopened.publish()['outcomes'][0]
    assert outcome['actor_seconds'] == 75.
    assert outcome['original_receipt']['actor_seconds'] == 60.
    assert reopened.status()['projects']['a']['receipt']['actor_seconds'] == 60.


class ClosedScriptedSession:
    def prepare(self, *args, **kwargs):
        pass

    def next_action(self, *args, **kwargs):
        from coding.pi.rpc import PiProviderError
        raise PiProviderError('scripted local interruption')

    def close(self):
        pass


def spawned_runner(request):
    import sys
    from rsi.contracts import BudgetLimits, ProjectSpec
    from rsi.project_runner import ProjectRunner
    from rsi.skills import builtin_bundle
    source = request.root.parent / 'input'
    source.mkdir()
    (source / 'solution.py').write_text('value = 1\n')
    verifier = source / 'verify.py'
    verifier.write_text('assert True\n')
    command = [sys.executable, '-c', 'pass']
    spec = ProjectSpec(request.project_id, 'local lifecycle test', tuple(command),
        budgets=BudgetLimits(actor_seconds=7200., design_iterations=None,
                             implementation_iterations=None, execution_tries=None))
    return ProjectRunner.create(request.root, spec, builtin_bundle(),
        dict(source=str(source), verifier=str(verifier),
             commands=dict(compile=command, test=[sys.executable, '{verifier}'],
                           validation=[sys.executable, '{verifier}'])),
        query=lambda *args: None, backend_identity='TEST-WAVE',
        session_factory=ClosedScriptedSession, test_backend=True)


def test_real_spawn_retains_canonical_pending_without_provider(tmp_path):
    pool = wave(tmp_path, ('a',), parallel=1)
    state = pool.run(spawned_runner)
    row = state['projects']['a']
    assert row['status'] == 'pending'
    assert row['device'] == 'physical-0'
    assert row['worker']['pid'] > 0
    assert row['workflow_pending'] is not None
    assert (tmp_path / 'projects/a/runner/runner.json').exists()
    assert row['receipt']['result']['status'] == 'suspended'
    assert row['actor_seconds'] >= 0
    # Reopening must not spawn another controller for that retained operation.
    reopened = wave(tmp_path, ('a',), parallel=1)
    assert reopened.run(spawned_runner)['projects']['a']['worker'] == row['worker']


def successful_spawned_runner(request):
    import time
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    from rsi.project_runner import ProjectRunner
    from rsi.strategy import knowledge_db_query
    from test_rsi_project_runner import Session, TestEmbedding
    runner = spawned_runner(request)
    database = KnowledgeDB(request.root.parent / 'knowledge.sqlite3', TestEmbedding())
    source = request.root.parent / 'input'
    database.index(source, [source / 'solution.py'], collection='examples',
                   language='python', revision='test-wave')
    atomic_json(request.root.parent / 'observed-snapshots.json', dict(
        memory=request.memory.hex(), skills=request.skills.hex()))
    deadline = time.monotonic() + 10
    while not all((request.root.parent.parent / p / 'observed-snapshots.json').exists()
                  for p in ('a', 'b')):
        if time.monotonic() >= deadline:
            raise TimeoutError('both real children must overlap before either actor runs')
        time.sleep(.01)

    class IsolatedSession(Session):
        def next_action(self, context, allowed, **kwargs):
            name, args = super().next_action(context, allowed, **kwargs)
            if context['stage'] == 'implementation' and name == 'perform':
                snapshot = context['source']
                args['edit'] = dict(path=snapshot['path'], sha256=snapshot['sha256'],
                    source=f'value = {request.project_id!r}\n')
            return name, args

    runner = ProjectRunner(request.root,
        query=knowledge_db_query(database, ('examples',), negative_query=lambda _: []), backend_identity='TEST-WAVE',
        session_factory=IsolatedSession, test_backend=True)
    canonical_run = runner.run
    def run_and_close():
        try:
            return canonical_run()
        finally:
            database.close()
    runner.run = run_and_close
    return runner


def test_two_spawned_children_keep_snapshots_results_and_sources_isolated(tmp_path):
    pool = wave(tmp_path)
    state = pool.run(successful_spawned_runner)
    rows = state['projects']
    assert all(row['status'] == 'completed' for row in rows.values()), rows
    assert all(row['receipt']['result']['status'] == 'accepted' for row in rows.values())
    assert rows['a']['worker']['pid'] != rows['b']['worker']['pid']
    for project in ('a', 'b'):
        branch = tmp_path / 'projects' / project
        assert json.loads((branch / 'observed-snapshots.json').read_text()) == dict(
            memory=b'frozen memory'.hex(), skills=b'frozen skills'.hex())
        assert (branch / 'runner/project/solution.py').read_text() == f'value = {project!r}\n'
        assert (branch / 'input/solution.py').read_text() == 'value = 1\n'
    published = pool.publish()
    reopened = wave(tmp_path)
    assert reopened.run(successful_spawned_runner)['projects'] == rows
    assert reopened.publish() == published
