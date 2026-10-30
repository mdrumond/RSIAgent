"""Behavioral job ownership checks; no provider or device required."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

import pytest

from rsi import jobs
from rsi.configuration import _bytes, _publish
from rsi.job_worker import run_owned


@pytest.fixture
def instance(tmp_path):
    root = tmp_path / 'instance'
    (root / 'config').mkdir(parents=True)
    (root / 'runs').mkdir()
    (root / 'instance.json').write_text(json.dumps({'schema_version': 1, 'files': {}}))
    (root / 'config/instance.json').write_text(json.dumps({'machinery': 'machinery'}))
    return root


def record(instance, *, process=None, state='running'):
    _, directory = jobs._directory(instance)
    job_id = str(uuid.uuid4())
    value = {'job_id': job_id, 'state': state, 'process': process,
             'request': {'mode': 'phase1'},
             'configuration': {'revision': 'frozen', 'payload': {'prompts': {}}}}
    _publish(jobs._path(directory, job_id), _bytes(value))
    return job_id, value


def test_reused_pid_is_interrupted_and_never_signalled(instance, monkeypatch):
    identity = jobs._identity(os.getpid())
    identity['start_ticks'] = 'wrong'
    job_id, original = record(instance, process=identity)
    signals = []
    monkeypatch.setattr(signal, 'pidfd_send_signal', lambda *args: signals.append(args))
    assert jobs.status(instance, job_id)['state'] == 'interrupted'
    assert jobs.stop(instance, job_id)['state'] == 'interrupted'
    assert signals == []
    assert json.loads(jobs._path(instance / 'runs/jobs', job_id).read_text()) == original


def test_stop_before_worker_identity_is_durable(instance):
    job_id, _ = record(instance, state='launching')
    # A launching worker has not yet published its PID, but stop must not be lost.
    jobs.stop(instance, job_id)
    assert (instance / f'runs/jobs/{job_id}.stop').exists()


def test_worker_persists_events_and_passes_frozen_snapshot(instance):
    job_id, original = record(instance)
    lock = os.open(instance / 'runs/jobs', os.O_RDONLY | os.O_DIRECTORY)
    seen = []
    def service(root, request, snapshot, *, should_stop, emit):
        seen.append((root, request, snapshot, should_stop()))
        emit({'operation_id': 'retained', 'stage': 'design'})
        return {'pending_operation': 'retained', 'state': 'suspended'}
    run_owned(instance, job_id, lock, service)
    result = jobs.evidence(instance, job_id)
    assert seen == [(instance, original['request'], original['configuration'], False)]
    assert result['job']['result']['pending_operation'] == 'retained'
    assert result['events'] == [{'operation_id': 'retained', 'stage': 'design'}]


def test_worker_stop_reaches_service_before_work(instance):
    job_id, _ = record(instance)
    (instance / f'runs/jobs/{job_id}.stop').write_text('{}')
    lock = os.open(instance / 'runs/jobs', os.O_RDONLY | os.O_DIRECTORY)
    def service(root, request, snapshot, *, should_stop, emit):
        assert should_stop()
        return {'actor_seconds': 0, 'state': 'stopped'}
    run_owned(instance, job_id, lock, service)
    assert jobs.status(instance, job_id)['state'] == 'stopped'


def test_worker_failure_is_not_replayed_and_retains_request(instance):
    job_id, original = record(instance)
    lock = os.open(instance / 'runs/jobs', os.O_RDONLY | os.O_DIRECTORY)
    def service(*args, **kwargs):
        raise RuntimeError('unknown pending operation')
    with pytest.raises(RuntimeError):
        run_owned(instance, job_id, lock, service)
    value = jobs.status(instance, job_id)
    assert value['state'] == 'failed'
    assert value['request'] == original['request']
    assert value['configuration'] == original['configuration']


def test_old_release_never_dispatches_global_machinery(instance, monkeypatch):
    spawned = []
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: spawned.append(a))
    with pytest.raises(ValueError, match='lacks CLI job support'):
        jobs.start(instance, {'action': 'start', 'mode': 'phase1'})
    assert spawned == []


def test_live_owned_process_stops_but_other_process_survives(instance):
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        job_id, _ = record(instance, process=jobs._identity(process.pid))
        jobs.stop(instance, job_id)
        process.wait(timeout=5)
        assert other.poll() is None
    finally:
        for child in (process, other):
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


def test_cli_resume_preserves_retained_cycle_and_mode(instance, monkeypatch):
    from rsi.cli import main
    job_id, _ = record(instance, state='failed')
    path = jobs._path(instance / 'runs/jobs', job_id)
    value = json.loads(path.read_text())
    value['request']['cycle_id'] = 'retained-cycle'
    _publish(path, _bytes(value))
    calls = []
    monkeypatch.setattr(jobs, 'start', lambda root, request: calls.append(request) or {})
    assert main(['--instance', str(instance), 'phase1', 'resume', '--job', job_id]) == 0
    assert calls == [{'mode': 'phase1', 'cycle_id': 'retained-cycle',
                      'action': 'resume', 'previous_job': job_id}]
    assert main(['--instance', str(instance), 'phase2', 'resume', '--job', job_id]) == 2
    assert len(calls) == 1


def test_cli_inference_disables_recovery_and_freezes_project(instance, tmp_path, monkeypatch):
    from rsi.cli import main
    project = tmp_path / 'project.json'
    project.write_text(json.dumps({'project_id': 'numeric', 'goal': 'accurate sum',
                                   'command': ['python', 'verify.py']}))
    calls = []
    monkeypatch.setattr(jobs, 'start', lambda root, request: calls.append(request) or {})
    assert main(['--instance', str(instance), 'inference', 'start', '--project',
                 str(project), '--no-recovery']) == 0
    assert calls[0]['recover'] is False
    assert calls[0]['projects'][0]['project_id'] == 'numeric'
    project.unlink()
    assert calls[0]['projects'][0]['command'] == ['python', 'verify.py']


def test_instance_cli_uses_pinned_copy_outside_global_checkout(instance, monkeypatch):
    import hashlib
    from rsi.pinned_cli import dispatch
    machinery = instance / 'machinery/rsi'
    machinery.mkdir(parents=True)
    source = machinery / 'cli.py'
    source.write_text('def main(): return 19\n')
    manifest = {'machinery': {'revision': 'pinned'}, 'files': {
        'machinery/rsi/cli.py': hashlib.sha256(source.read_bytes()).hexdigest()}}
    (instance / 'instance.json').write_text(json.dumps(manifest))
    called = []
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kwargs:
                        called.append((argv, kwargs)) or subprocess.CompletedProcess(argv, 19))
    assert dispatch(instance, ['--instance', str(instance), 'job', 'list']) == 19
    argv, options = called[0]
    assert argv[:3] == [sys.executable, '-I', '-c']
    assert str(instance / 'machinery') in argv
    assert options['cwd'] == Path.cwd()
    source.write_text('def main(): return 0\n')
    with pytest.raises(ValueError, match='integrity'):
        dispatch(instance, ['job', 'list'])
    assert len(called) == 1


def test_partial_event_write_is_not_presented_as_complete_evidence(instance):
    job_id, _ = record(instance, state='failed')
    (instance / f'runs/jobs/{job_id}.events.jsonl').write_bytes(
        b'{"operation":"complete"}\n{"operation":"partial')
    assert jobs.evidence(instance, job_id)['events'] == [{'operation': 'complete'}]


@pytest.fixture
def service_surface(monkeypatch):
    """Exercise CLI adaptation decisions without launching a second agent loop."""
    from types import SimpleNamespace
    from rsi import services
    calls = []
    scheduler = SimpleNamespace(state={'phase': 'original', 'stopped': False, 'pending': None})
    scheduler.extension_events = lambda: []
    scheduler.resume = lambda: calls.append('resume')
    cycle = SimpleNamespace(scheduler=scheduler,
        controller=SimpleNamespace(state={'pending': None}),
        spec=SimpleNamespace(mode=SimpleNamespace(value='inference'), configuration_ref='frozen-ref'))
    cycle.close = lambda: calls.append('close')
    def stop():
        calls.append('canonical-stop')
        scheduler.state['stopped'] = True
    cycle.request_stop = stop
    cycle.reconcile = lambda: calls.append('observe-only')
    def step():
        calls.append('step')
        scheduler.state['phase'] = 'complete'
        return {'outcome': 'accepted'}
    cycle.step = step
    class Services:
        def __init__(self, root):
            calls.append(('instance', root))
        def create_cycle(self, identity, mode, **kwargs):
            calls.append(('create', identity, mode, kwargs))
            return cycle
        def open_cycle(self, identity):
            calls.append(('open', identity))
            return cycle
        def status(self, identity):
            return json.loads(json.dumps({'cycle.json': scheduler.state}))
    monkeypatch.setattr(services, 'InstanceServices', Services)
    return calls, cycle


def test_job_binding_admits_exact_frozen_inputs_and_real_recovery_override(instance, service_surface):
    from rsi.job_service import run_job
    calls, cycle = service_surface
    snapshot = {'revision': 'original', 'payload': {'prompts': {'actor': 'frozen'}}}
    request = {'action': 'revisit', 'mode': 'inference', 'cycle_id': 'new',
               'previous_cycle': 'old', 'recover': False, 'projects': [{'project_id': 'a'}]}
    events = []
    result = run_job(instance, request, snapshot, should_stop=lambda: False, emit=events.append)
    assert result['state'] == 'complete'
    assert calls[1] == ('create', 'new', 'inference', {
        'projects': request['projects'], 'configuration': snapshot,
        'recover': False, 'lineage': 'old'})
    assert calls.count('step') == 1 and calls[-1] == 'close'
    assert events[0]['configuration_ref'] == 'frozen-ref'


def test_resume_unknown_pending_observes_without_dispatch(instance, service_surface):
    from rsi.job_service import run_job
    from rsi.runtime import WorkflowError
    calls, cycle = service_surface
    cycle.scheduler.state['pending'] = {'id': 'retained'}
    def observe():
        calls.append('observe-only')
        raise WorkflowError('existing operation still unknown')
    cycle.reconcile = observe
    result = run_job(instance, {'action': 'resume', 'mode': 'inference', 'cycle_id': 'same'},
        {}, should_stop=lambda: False, emit=lambda _: None)
    assert result['state'] == 'suspended'
    assert 'step' not in calls and 'resume' not in calls
    assert calls[-2:] == ['observe-only', 'close']


def test_stop_unwinds_active_service_before_persisting_canonical_stop(instance, service_surface):
    from rsi.job_service import run_job
    calls, cycle = service_surface
    def interrupted_step():
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            calls.append('child-cleanup-and-elapsed-charge')
    cycle.step = interrupted_step
    job_id, _ = record(instance)
    path = jobs._path(instance / 'runs/jobs', job_id)
    value = json.loads(path.read_text())
    value['request'] = {'action': 'start', 'mode': 'inference', 'cycle_id': 'same'}
    _publish(path, _bytes(value))
    lock = os.open(instance / 'runs/jobs', os.O_RDONLY | os.O_DIRECTORY)
    run_owned(instance, job_id, lock, run_job)
    result = jobs.status(instance, job_id)
    assert result['state'] == 'stopped'
    assert calls[-3:] == ['child-cleanup-and-elapsed-charge', 'canonical-stop', 'close']
    assert result['result']['evidence']['cycle.json']['stopped'] is True


def test_blocked_controller_returns_without_busy_loop(instance, service_surface):
    from rsi.job_service import run_job
    calls, cycle = service_surface
    cycle.step = lambda: calls.append('step')
    result = run_job(instance, {'action': 'start', 'mode': 'inference', 'cycle_id': 'same'},
        {}, should_stop=lambda: False, emit=lambda _: None)
    assert result['state'] == 'blocked'
    assert calls.count('step') == 1


def test_competing_dispatch_is_refused_by_instance_lock(instance, monkeypatch):
    import fcntl
    _, directory = jobs._directory(instance)
    lock = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(jobs, '_machinery', lambda root: root / 'machinery')
    try:
        with pytest.raises(ValueError, match='already has an owned running job'):
            jobs.start(instance, {'action': 'start', 'mode': 'phase1', 'cycle_id': 'a'})
        assert list(directory.glob('*.json')) == []
    finally:
        os.close(lock)
