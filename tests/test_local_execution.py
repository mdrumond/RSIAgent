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
