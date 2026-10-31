"""Local CLI behavior tests; no target, provider or hardware required."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

import pytest

from rsi.cpl_execution import CplConfig, CplExecutor, _RUNNER
from rsi.python_execution import CommandRequest, PythonExecution, digest
from rsi.runtime import WorkflowError


FAKE = r'''#!/usr/bin/env python3
import json, os, pathlib, shutil, subprocess, sys, time
args = sys.argv[2:]
action = args.pop(0)
root = pathlib.Path(__file__).parent.parent
with (root / 'calls.jsonl').open('a') as stream:
    stream.write(json.dumps([action, *args]) + '\n')
value = dict(target='fake', state='completed')
if action == 'upload':
    destination = pathlib.Path(args[2])
    if not destination.is_absolute():
        destination = root.parent / 'remote' / destination
    shutil.copyfile(args[1], destination)
    if '--receipt-file' in args:
        value.update(handle='remote:fake:transfer:input', state='queued')
        pathlib.Path(args[args.index('--receipt-file') + 1]).write_text(json.dumps(value))
elif action == 'run':
    receipt = pathlib.Path(args[args.index('--receipt-file') + 1])
    value.update(handle='remote:fake:job:original', state='queued')
    receipt.write_text(json.dumps(value))
    script = args[args.index('--file') + 1]
    worker = ('import os,pathlib,subprocess,sys,time; '
              'time.sleep(float(os.environ.get("CPL_FAKE_QUEUE_SECONDS",0))); '
              'sys.exit(0) if pathlib.Path(sys.argv[2]+".cancel").exists() else None; '
              'r=subprocess.run(["bash",sys.argv[1]]); '
              'pathlib.Path(sys.argv[2]).write_text(str(r.returncode))')
    subprocess.Popen([sys.executable, '-c', worker, script, str(root / 'job-exit')],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
elif action == 'observe':
    if ':job:' in args[0]:
        deadline = time.monotonic() + (int(args[args.index('--timeout') + 1])
                                       if '--wait' in args else 0)
        while not (root / 'job-exit').exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        value['state'] = 'completed' if (root / 'job-exit').exists() else 'running'
    value.update(handle=args[0], exit=0)
elif action == 'cancel':
    (root / 'job-exit.cancel').touch()
    (root / 'job-exit').write_text('-15')
    value.update(handle=args[0], state='cancelled')
elif action == 'fetch':
    shutil.copyfile((root / 'download-source').read_text(), args[1])
    value.update(handle=args[0])
elif action == 'download':
    source = pathlib.Path(args[1])
    if not source.is_absolute():
        source = root.parent / 'remote' / source
    if os.environ.get('CPL_FAKE_LONG_DOWNLOAD'):
        assert '--timeout' in args
        time.sleep(int(args[args.index('--timeout') + 1]))
        (root / 'download-source').write_text(str(source))
        value.update(handle='remote:fake:transfer:output', state='running')
    else:
        shutil.copyfile(source, args[2])
else:
    raise RuntimeError(action)
print(json.dumps(value))
'''


def fixture(tmp_path, code='print("hello")', argv=None):
    installed = tmp_path / 'skill'
    (installed / 'scripts').mkdir(parents=True)
    (installed / 'SKILL.md').write_text('Installed test policy')
    client = installed / 'scripts' / 'cpl-remote'
    client.write_text(FAKE)
    client.chmod(0o700)
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'main.py').write_text(code)
    verifier = tmp_path / 'verify.py'
    verifier.write_text('print("verified")')
    root = tmp_path / 'task'
    backend = PythonExecution.snapshot(root, source, verifier, commands={'run': ['python3', 'main.py']})
    workspace = tmp_path / 'remote'
    workspace.mkdir()
    (workspace / 'commands').mkdir()
    config = CplConfig('fake', 'test-runtime', str(workspace), 2, client, 'frozen',
                       command_cwd=str(workspace / 'commands'),
                       remote_python=sys.executable)
    executor = CplExecutor('frozen', config)
    request = CommandRequest('operation', 'run', tuple(argv or [sys.executable, 'main.py']),
                             backend.project, backend.verifier, backend.project_hash(),
                             digest(backend.verifier.read_bytes()), 'execution', None)
    observations = []
    observe = lambda phase, **data: observations.append((phase, data))
    return executor, request, observe, observations, installed


def test_real_cli_staging_receipt_and_reconciliation(tmp_path):
    executor, request, observe, observations, installed = fixture(
        tmp_path, 'from pathlib import Path\nPath("main.py").write_text("changed")\nprint("hello")')
    result = executor.execute(request, observe=observe, remaining=lambda: 60)
    assert result.returncode == 0 and result.stdout == b'hello\n'
    assert (request.project / 'main.py').read_text() == 'changed'
    calls = [json.loads(line) for line in (installed / 'calls.jsonl').read_text().splitlines()]
    run = next(call for call in calls if call[0] == 'run')
    assert run[run.index('--runtime') + 1] == 'test-runtime'
    assert run[run.index('--npu') + 1] == '2'
    assert any(data.get('handle') == 'remote:fake:job:original' for _, data in observations)
    executor.reconcile(request, observe=observe, remaining=lambda: 60)
    calls = [json.loads(line) for line in (installed / 'calls.jsonl').read_text().splitlines()]
    assert sum(call[0] == 'run' for call in calls) == 1
    assert all(call[1] == 'remote:fake:job:original' for call in calls if call[0] == 'observe')
    with pytest.raises(WorkflowError, match='reserved'):
        executor.execute(request, observe=observe, remaining=lambda: 60)


def test_ambiguous_reservation_never_replays(tmp_path):
    executor, request, observe, _, installed = fixture(tmp_path)
    directory = executor._directory(request)
    directory.mkdir(parents=True)
    with pytest.raises(WorkflowError, match='reserved'):
        executor.execute(request, observe=observe, remaining=lambda: 60)
    assert not (installed / 'calls.jsonl').exists()


def test_budget_exhaustion_does_not_dispatch(tmp_path):
    executor, request, observe, _, installed = fixture(tmp_path)
    with pytest.raises(WorkflowError, match='budget'):
        executor.execute(request, observe=observe, remaining=lambda: 0)
    assert not (installed / 'calls.jsonl').exists()


def test_frozen_factory_identity(tmp_path):
    executor, *_ = fixture(tmp_path)
    with pytest.raises(ValueError, match='profile'):
        CplExecutor('different', executor.config)


def test_driver_environment_isolated_from_registry_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('CATLASS_DSL_CACHE_DIR', '/registry/shared-cache')
    monkeypatch.setenv('RSI_EXPECTED_SOURCE_HASH', 'stale')
    code = ('import json, os\nprint(json.dumps({key:os.environ[key] for key in '
            '("RSI_EXPECTED_SOURCE_HASH", "CATLASS_DSL_CACHE_DIR", '
            '"ASCEND_RT_VISIBLE_DEVICES")}))')
    executor, request, observe, _, _ = fixture(tmp_path, code)
    result = executor.execute(request, observe=observe, remaining=lambda: 60)
    values = json.loads(result.stdout)
    assert values['RSI_EXPECTED_SOURCE_HASH'] == request.source_hash
    assert values['ASCEND_RT_VISIBLE_DEVICES'] == '2'
    cache = Path(values['CATLASS_DSL_CACHE_DIR'])
    state = json.loads((executor._directory(request) / 'transport.json').read_text())
    assert cache == Path(state['remote']) / 'catlass-cache'


@pytest.mark.parametrize('remaining_value,expected', [(None, 45), (120, 45), (3, 3)])
def test_each_client_call_has_finite_timeout(tmp_path, monkeypatch, remaining_value, expected):
    executor, *_ = fixture(tmp_path)
    captured = []
    def run(argv, **kwargs):
        captured.append(kwargs['timeout'])
        return type('Result', (), dict(returncode=0, stdout=b'{}'))()
    monkeypatch.setattr('rsi.cpl_execution.subprocess.run', run)
    executor._cli(['observe', 'remote:fake:job:original'], lambda: remaining_value)
    assert captured == [expected]


def test_missing_executable_is_terminal_failure(tmp_path):
    executor, request, observe, _, _ = fixture(tmp_path, argv=['/missing/trusted-executable'])
    result = executor.execute(request, observe=observe, remaining=lambda: 60)
    assert result.returncode == 127
    assert result.stderr


def test_listener_relative_transfers_and_admitted_cwd(tmp_path):
    executor, request, observe, _, installed = fixture(tmp_path)
    executor = CplExecutor('frozen', replace(executor.config, command_cwd='commands',
                           remote_agent_transfers=True, transfer_relative_paths=True))
    assert executor.execute(request, observe=observe, remaining=lambda: 60).returncode == 0
    calls = [json.loads(line) for line in (installed / 'calls.jsonl').read_text().splitlines()]
    upload = next(call for call in calls if call[0] == 'upload')
    download = next(call for call in calls if call[0] == 'download')
    run = next(call for call in calls if call[0] == 'run')
    assert not Path(upload[3]).is_absolute()
    assert not Path(download[2]).is_absolute()
    assert '--receipt-file' in upload and '--no-wait' in upload
    assert run[run.index('--cwd') + 1] == 'commands'
    assert (executor._directory(request) / 'upload-submission.json').exists()


def test_explicit_recovery_allowance_never_launches_new_work(tmp_path):
    executor, request, observe, _, installed = fixture(tmp_path)
    executor.execute(request, observe=observe, remaining=lambda: 60)
    result = executor.reconcile(request, observe=observe, remaining=lambda: 0,
                                recovery_remaining=lambda: 60)
    assert result.returncode == 0
    calls = [json.loads(line) for line in (installed / 'calls.jsonl').read_text().splitlines()]
    assert sum(call[0] == 'run' for call in calls) == 1


def test_nonzero_cli_preserves_recoverable_handle(tmp_path, monkeypatch):
    executor, *_ = fixture(tmp_path)
    value = dict(target='fake', handle='remote:fake:transfer:retained', state='running')
    def run(argv, **kwargs):
        return type('Result', (), dict(returncode=1, stdout=json.dumps(value).encode()))()
    monkeypatch.setattr('rsi.cpl_execution.subprocess.run', run)
    receipt = tmp_path / 'response.json'
    assert executor._cli(['download'], lambda: 60, receipt) == value
    assert json.loads(receipt.read_text()) == value


def test_frozen_paths_preserve_arguments_and_project_cwd(tmp_path):
    executor, request, observe, _, _ = fixture(tmp_path)
    driver = tmp_path / 'trusted-driver.py'
    driver.write_text('import json, os, sys\nprint(json.dumps([os.getcwd(), sys.argv[1:]]))')
    executor = CplExecutor('frozen', replace(executor.config, trusted_resources=(driver,)))
    request = replace(request, argv=(sys.executable, str(driver), str(request.verifier),
                                      '--source=' + str(request.project / 'main.py'),
                                      'argument with spaces', 'literal;$value'))
    result = executor.execute(request, observe=observe, remaining=lambda: 60)
    cwd, arguments = json.loads(result.stdout)
    remote = json.loads((executor._directory(request) / 'transport.json').read_text())['remote']
    assert cwd == remote + '/project'
    assert arguments == [remote + '/verifier', '--source=' + remote + '/project/main.py',
                         'argument with spaces', 'literal;$value']


def test_owned_workload_timeout_returns_terminal_artifacts(tmp_path):
    executor, request, observe, _, _ = fixture(tmp_path, 'import time\ntime.sleep(30)')
    result = executor.execute(request, observe=observe, remaining=lambda: 2)
    assert result.timed_out and result.returncode != 0
    assert isinstance(result.stdout, bytes) and isinstance(result.stderr, bytes)


def test_changed_receipt_identity_blocks_observation(tmp_path):
    executor, request, observe, _, installed = fixture(tmp_path)
    executor.execute(request, observe=observe, remaining=lambda: 60)
    count = len((installed / 'calls.jsonl').read_text().splitlines())
    changed = replace(request, source_hash='different-source')
    with pytest.raises(WorkflowError, match='identity'):
        executor.reconcile(changed, observe=observe, remaining=lambda: 60)
    assert len((installed / 'calls.jsonl').read_text().splitlines()) == count


def test_unknown_submission_receipt_never_dispatches_during_recovery(tmp_path):
    executor, request, observe, _, installed = fixture(tmp_path)
    directory = executor._directory(request)
    directory.mkdir(parents=True)
    (directory / 'transport.json').write_text(json.dumps(dict(identity=executor._identity(request),
                                                             status='dispatching')))
    with pytest.raises(FileNotFoundError):
        executor.reconcile(request, observe=observe, remaining=lambda: 0,
                           recovery_remaining=lambda: 60)
    assert not (installed / 'calls.jsonl').exists()


def test_slow_download_returns_handle_before_client_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv('CPL_FAKE_LONG_DOWNLOAD', '1')
    executor, request, observe, _, installed = fixture(tmp_path)
    executor = CplExecutor('frozen', replace(executor.config, client_timeout_seconds=4))
    result = executor.execute(request, observe=observe, remaining=lambda: 60)
    assert result.returncode == 0
    calls = [json.loads(line) for line in (installed / 'calls.jsonl').read_text().splitlines()]
    download = next(call for call in calls if call[0] == 'download')
    assert int(download[download.index('--timeout') + 1]) < 4
    assert sum(call[0] == 'download' for call in calls) == 1
    assert any(call[:2] == ['fetch', 'remote:fake:transfer:output'] for call in calls)


def test_queue_expiry_cancels_owned_handle_without_late_work(tmp_path, monkeypatch):
    monkeypatch.setenv('CPL_FAKE_QUEUE_SECONDS', '3')
    executor, request, observe, _, installed = fixture(tmp_path)
    deadline = time.monotonic() + 1.5
    with pytest.raises(WorkflowError):
        executor.execute(request, observe=observe, remaining=lambda: max(0, deadline-time.monotonic()))
    state = json.loads((executor._directory(request) / 'transport.json').read_text())
    assert state['status'] == 'producer-unknown'
    assert state['lease_retained'] is True
    assert state['terminal']['handle'] == 'remote:fake:job:original'
    assert not (Path(state['remote']) / 'project').exists()
    calls = [json.loads(line) for line in (installed / 'calls.jsonl').read_text().splitlines()]
    assert any(call[:2] == ['cancel', 'remote:fake:job:original'] for call in calls)
    assert sum(call[0] == 'run' for call in calls) == 1


@pytest.mark.parametrize('kind,expected_billable', [('transfer', 0), ('job', 7)])
def test_retained_wait_accounts_transfer_copy_but_charges_job_queue(tmp_path, monkeypatch,
                                                                 kind, expected_billable):
    executor, request, _, _, _ = fixture(tmp_path)
    now, phase, charged = 0, 'command', 0
    def observe(next_phase, **metadata):
        nonlocal phase
        phase = next_phase
    def cli(args, remaining, response_file=None):
        nonlocal now, charged
        assert remaining() == 100 - charged
        now += 7
        if phase == 'command':
            charged += 7
        return dict(target='fake', handle=f'remote:fake:{kind}:owned', state='completed')
    monkeypatch.setattr(executor, '_cli', cli)
    executor._wait(f'remote:fake:{kind}:owned', request, observe, lambda: 100-charged)
    assert now == 7 and charged == expected_billable
    assert phase == 'command'


def test_disconnected_deadline_cancellation_retains_owned_lease(tmp_path, monkeypatch):
    executor, request, observe, _, _ = fixture(tmp_path)
    directory = executor._directory(request)
    directory.mkdir(parents=True)
    state = dict(identity=executor._identity(request), status='dispatching')
    budget = [5]
    def submit(*args):
        (directory / 'submission.json').write_text(json.dumps(dict(
            target='fake', handle='remote:fake:job:original', state='queued')))
        budget[0] = 0
        raise WorkflowError('submission observation disconnected')
    seen = []
    def cli(args, remaining, response_file=None):
        seen.append(args)
        raise WorkflowError('cancellation disconnected')
    monkeypatch.setattr(executor, '_submit', submit)
    monkeypatch.setattr(executor, '_cli', cli)
    with pytest.raises(WorkflowError):
        executor._supervise(request, directory, state, directory / 'run.sh', observe, lambda: budget[0])
    saved = json.loads((directory / 'transport.json').read_text())
    assert saved['handle'] == 'remote:fake:job:original'
    assert saved['lease_retained'] and saved['status'] == 'deadline-observation-pending'
    assert all(call == ['cancel', 'remote:fake:job:original'] for call in seen)


def start_runner(root, code, limit=10):
    """Run the shipped producer, independently of the fake CPL client."""
    (root / 'project').mkdir()
    (root / 'project' / 'main.py').write_text(code)
    (root / 'verifier').write_text('verifier')
    (root / 'runner.py').write_text(_RUNNER)
    identity = dict(source_hash='frozen', operation_id='owned')
    (root / 'spec.json').write_text(json.dumps(dict(
        identity=identity, files={}, physical_device=0,
        argv=[sys.executable, '-u', 'main.py'])))
    return subprocess.Popen([sys.executable, str(root / 'runner.py'), str(limit)],
                            start_new_session=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)


def wait_for_file(path, process):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, str(path)
        time.sleep(0.01)


def stop_test_runner(process, root):
    # Only IDs belonging to this test's runner and its recorded child are eligible.
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            receipt = root / 'producer.json'
            if receipt.exists():
                child = json.loads(receipt.read_text()).get('child_pgid')
                if child:
                    try:
                        os.killpg(child, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            process.kill()
            process.wait(timeout=3)
    process.communicate()


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
def test_actual_runner_signal_closes_separate_child_group(tmp_path, signum):
    code = '''import os, pathlib, signal, sys, time
def stop(signum, frame):
    print('forwarded ' + str(signum), flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
print('partial stdout', flush=True)
print('partial stderr', file=sys.stderr, flush=True)
pathlib.Path('child-pgid').write_text(str(os.getpgrp()))
while True:
    pathlib.Path('heartbeat').write_text(str(time.monotonic_ns()))
    time.sleep(0.01)
'''
    process = start_runner(tmp_path, code)
    try:
        heartbeat = tmp_path / 'project' / 'heartbeat'
        wait_for_file(heartbeat, process)
        child_pgid = int((tmp_path / 'project' / 'child-pgid').read_text())
        assert child_pgid != os.getpgid(process.pid)
        process.send_signal(signum)
        process.communicate(timeout=5)
        producer = json.loads((tmp_path / 'producer.json').read_text())
        assert producer['child_pgid'] == child_pgid
        assert producer['signal'] == signum
        assert CplExecutor._producer_closed(producer, producer['identity'])
        with pytest.raises(ProcessLookupError):
            os.killpg(child_pgid, 0)
        final_heartbeat = heartbeat.read_bytes()
        time.sleep(0.05)
        assert heartbeat.read_bytes() == final_heartbeat
        with tarfile.open(tmp_path / 'results.tar') as archive:
            result = json.load(archive.extractfile('result.json'))
            assert result['producer'] == producer
            assert result['returncode'] != 0
            assert archive.extractfile('stdout').read() == (
                'partial stdout\nforwarded ' + str(signum) + '\n').encode()
            assert archive.extractfile('stderr').read() == b'partial stderr\n'
    finally:
        stop_test_runner(process, tmp_path)


@pytest.mark.parametrize('code,limit,returncode,timed_out', [
    ('print("ordinary")', 5, 0, False),
    ('import sys; print("nonzero"); sys.exit(7)', 5, 7, False),
    ('import time; print("partial", flush=True); time.sleep(30)', 0.5, -9, True),
])
def test_actual_runner_terminal_artifacts(tmp_path, code, limit, returncode, timed_out):
    process = start_runner(tmp_path, code, limit)
    try:
        process.communicate(timeout=5)
        result = json.loads((tmp_path / 'result.json').read_text())
        assert result['returncode'] == returncode
        assert result['timed_out'] is timed_out
        assert CplExecutor._producer_closed(result['producer'], result['identity'])
        assert (tmp_path / 'stdout').read_bytes()
        assert (tmp_path / 'results.tar').exists()
    finally:
        stop_test_runner(process, tmp_path)


@pytest.mark.parametrize('legacy_complete', [False, True])
def test_legacy_artifact_requires_previously_known_completion(tmp_path, legacy_complete):
    executor, request, observe, _, installed = fixture(tmp_path)
    executor.execute(request, observe=observe, remaining=lambda: 60)
    directory = executor._directory(request)
    output = directory / 'results.tar'
    unpacked = directory / 'legacy'
    with tarfile.open(output) as archive:
        archive.extractall(unpacked, filter='data')
    result_path = unpacked / 'result.json'
    result = json.loads(result_path.read_text())
    del result['producer']
    result_path.write_text(json.dumps(result))
    with tarfile.open(output, 'w') as archive:
        for path in unpacked.iterdir():
            archive.add(path, arcname=path.name)
    state_path = directory / 'transport.json'
    state = json.loads(state_path.read_text())
    state.pop('producer_version')
    state.pop('producer')
    state.update(status='complete' if legacy_complete else 'terminal')
    state_path.write_text(json.dumps(state))
    if legacy_complete:
        assert executor.reconcile(request, observe=observe, remaining=lambda: 60).returncode == 0
    else:
        with pytest.raises(WorkflowError, match='closure unknown'):
            executor.reconcile(request, observe=observe, remaining=lambda: 60)
        assert json.loads(state_path.read_text())['lease_retained'] is True
    calls = [json.loads(line) for line in (installed / 'calls.jsonl').read_text().splitlines()]
    assert sum(call[0] == 'run' for call in calls) == 1


@pytest.mark.parametrize('proof', ['missing', 'open', 'different-operation'])
def test_new_terminal_outer_job_does_not_substitute_for_closure(tmp_path, proof):
    executor, request, observe, _, _ = fixture(tmp_path)
    executor.execute(request, observe=observe, remaining=lambda: 60)
    directory = executor._directory(request)
    unpacked = directory / 'unproven'
    with tarfile.open(directory / 'results.tar') as archive:
        archive.extractall(unpacked, filter='data')
    result_path = unpacked / 'result.json'
    result = json.loads(result_path.read_text())
    if proof == 'missing':
        result.pop('producer')
    elif proof == 'open':
        result['producer']['closed'] = False
    else:
        result['producer']['identity'] = dict(operation_id='another')
    result_path.write_text(json.dumps(result))
    with tarfile.open(directory / 'results.tar', 'w') as archive:
        for path in unpacked.iterdir():
            archive.add(path, arcname=path.name)
    with pytest.raises(WorkflowError, match='closure unknown'):
        executor.reconcile(request, observe=observe, remaining=lambda: 60)
    state = json.loads((directory / 'transport.json').read_text())
    assert state['lease_retained'] is True
    assert state['status'] == 'producer-unknown'
