"""Exercise real local subprocesses without a remote client or device dependency."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import pytest

from benchmarks.ascend.execution import (LocalProcessRunner, apply_worker_config,
    execution_mode, local_environment)
from optimizer.control import ObservationUnavailable, WorkerStopped
from optimizer.worker import load_config


def config(tmp_path):
    return {'execution': 'local', 'workspace': str(tmp_path / 'work'),
            'physical_device': 0, 'arch': 'discovered-architecture', 'max_seconds': 10}


def test_local_config_needs_no_remote_fields_and_remote_default_is_preserved(tmp_path):
    settings = config(tmp_path) | {'guide': str(tmp_path / 'guide')}
    path = tmp_path / 'worker.json'; path.write_text(json.dumps(settings))
    result = load_config(path)
    assert result['execution_profile'] == 'local'
    assert 'cpl_remote' not in result and 'validation_wrapper' not in result
    assert execution_mode({}) == 'remote'
    with pytest.raises(ValueError, match='execution must'):
        apply_worker_config({'execution': 'automatic'})
    with pytest.raises(ValueError, match='workspace'):
        apply_worker_config(settings | {'workspace': 'relative'})


def test_shared_worker_preserves_product_separate_from_transport_host(tmp_path):
    path = tmp_path / 'worker.json'
    path.write_text(json.dumps({'execution': 'remote', 'target': 'bz-a3-1', 'profile': 'bz-a3-1'}))
    settings = apply_worker_config({'target': 'actual-product'}, path)
    assert settings['target'] == 'actual-product'
    assert settings['transport_target'] == 'bz-a3-1'


def test_process_receipt_and_reobservation_do_not_repeat_kernel_command(tmp_path, monkeypatch):
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)
    path = tmp_path / 'calls'
    command = [sys.executable, '-c', f"from pathlib import Path; p=Path({str(path)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x'); print('kernel receipt')"]
    handles = []
    runner = LocalProcessRunner(config(tmp_path), retain=handles.append)
    first = runner.run(command, tmp_path / 'operation')
    second = LocalProcessRunner(config(tmp_path)).run(command, tmp_path / 'operation')
    assert first.stdout == second.stdout == 'kernel receipt\n'
    assert path.read_text() == 'x'
    assert len(handles) == 1 and handles[0].startswith('local:')
    with pytest.raises(ValueError, match='original command'):
        runner.run([sys.executable, '-c', 'print(2)'], tmp_path / 'operation')


def test_observer_reattaches_to_running_local_operation(tmp_path, monkeypatch):
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)
    command = [sys.executable, '-c', "import time; time.sleep(.3); print('done')"]
    directory = tmp_path / 'operation'
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(LocalProcessRunner(config(tmp_path)).run, command, directory)
        until = time.monotonic() + 5
        while not (directory / 'transport.json').exists() or json.loads((directory / 'transport.json').read_text())['state'] == 'dispatching':
            assert time.monotonic() < until
            time.sleep(.01)
        second = LocalProcessRunner(config(tmp_path)).run(command, directory)
        assert second.stdout == first.result().stdout == 'done\n'


@pytest.mark.parametrize('boundary', ['deadline', 'stop'])
def test_local_cancellation_terminates_owned_work_and_keeps_receipt(tmp_path, monkeypatch, boundary):
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)
    started = time.monotonic()
    runner = LocalProcessRunner(config(tmp_path),
        deadline=started + (.4 if boundary == 'deadline' else 10),
        stopped=lambda: boundary == 'stop' and time.monotonic() - started > .4)
    with pytest.raises(WorkerStopped):
        runner.run([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path / 'operation')
    assert time.monotonic() - started < 5
    receipt = json.loads((tmp_path / 'operation/transport.json').read_text())
    assert receipt['state'] == 'cancelled'


def test_uncertain_dispatch_never_submits_a_replacement(tmp_path):
    directory = tmp_path / 'operation'; directory.mkdir()
    # Use a real first receipt to retain its exact command identity.
    command = [sys.executable, '-c', 'print(1)']
    runner = LocalProcessRunner(config(tmp_path))
    runner.run(command, directory)
    path = directory / 'transport.json'
    receipt = json.loads(path.read_text()); receipt.update(state='dispatching'); receipt.pop('pid')
    path.write_text(json.dumps(receipt))
    with pytest.raises(ObservationUnavailable, match='dispatch interrupted'):
        runner.run(command, directory)


def test_device_cannot_escape_inherited_visible_set(tmp_path, monkeypatch):
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '2,3')
    with pytest.raises(ValueError, match='excluded'):
        local_environment(config(tmp_path))
    assert local_environment(config(tmp_path) | {'physical_device': 3})['ASCEND_RT_VISIBLE_DEVICES'] == '3'


def test_outer_attempt_deadline_bounds_local_child(tmp_path):
    from benchmarks.ascend.execution import process_deadline
    runner = LocalProcessRunner(config(tmp_path))
    with process_deadline(time.monotonic() + .3), pytest.raises(WorkerStopped):
        runner.run([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path / 'operation')


def test_confirmed_cancel_can_resume_in_a_fresh_generation(tmp_path):
    command = [sys.executable, '-c', "import time; time.sleep(.4); print('resumed')"]
    directory = tmp_path / 'operation'
    with pytest.raises(WorkerStopped):
        LocalProcessRunner(config(tmp_path), deadline=time.monotonic() + .15).run(command, directory)
    old = json.loads((directory / 'transport.json').read_text())
    result = LocalProcessRunner(config(tmp_path)).run(command, directory)
    assert result.stdout == 'resumed\n'
    assert result.local_handle != old['handle']
    assert (directory / 'retry/transport.json').exists()
    assert json.loads((directory / 'transport.json').read_text())['state'] == 'cancelled'


def test_local_overlay_replaces_a_remote_working_directory(tmp_path):
    path = tmp_path / 'worker.json'; path.write_text(json.dumps(config(tmp_path)))
    settings = apply_worker_config({'remote_workspace': '/previous/remote/account'}, path)
    assert settings['remote_workspace'] == config(tmp_path)['workspace']


def test_a5_collector_uses_worker_config_from_workflow_arguments(tmp_path):
    from tests.test_ascend_campaign_cli import start, write
    from benchmarks.ascend import campaign_cli
    job = start(tmp_path)
    job['target'] = 'a5'
    worker = tmp_path / 'worker.json'; worker.write_text(json.dumps(config(tmp_path)))
    settings = json.loads(Path(job['config']).read_text())
    settings['phase1_arguments'] += ['--worker-config', str(worker)]
    write(Path(job['config']), settings)
    captured = []
    class Collector:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)
        def collect_once(self):
            return {}
    campaign_cli.collector_worker(job, once=True, factory=Collector)
    assert captured == [{'execution': 'local'}]


def test_phase2_optimizer_selects_local_backend_without_wrapper_config(tmp_path, monkeypatch):
    from tests.test_phase2_optimizer_execution import executor, snapshot
    import benchmarks.ascend.phase2_optimizer_execution as module
    example, agent, _backend = executor(tmp_path / 'example', [])
    settings = config(tmp_path)
    calls = []
    monkeypatch.setattr(module, 'preflight_local', lambda c: calls.append('preflight'))
    sentinel = type('Backend', (), {})()
    monkeypatch.setattr(module, 'LocalBackend', lambda *a: sentinel)
    monkeypatch.setattr(module, 'RemoteBackend', lambda *a: pytest.fail('remote backend selected'))
    task = module.OptimizerTaskExecutor(tmp_path / 'local', example.task, snapshot(), settings, agent=agent)
    assert task._loop().backend is sentinel
    assert calls == ['preflight']
    assert task._device_lock().name.endswith('-0.lock')


def test_local_preflight_uses_bounded_cold_runtime_timeout(tmp_path, monkeypatch):
    import benchmarks.ascend.execution as execution
    import subprocess
    settings = dict(execution='local', workspace=str(tmp_path), physical_device=0,
                    arch='test', max_seconds=45)
    def timeout_probe(*args, **kwargs):
        assert kwargs['timeout'] == 45
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(execution.subprocess, 'run', timeout_probe)
    with pytest.raises(RuntimeError, match='exceeded 45 seconds'):
        execution.preflight_local(settings)
