import json
from pathlib import Path
import subprocess
import time

import pytest

from benchmarks.ascend.cpl_native import CplNativeCandidateBackend, CplNativeMeasurements
from benchmarks.a3kernels.native_smoke import smoke_plan
from benchmarks.a3kernels.remote_candidate import _PendingObservation
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError


EXPECTED = tuple(float(3*i-1) for i in range(32))


class Remote:
    def __init__(self):
        self.calls, self.jobs = [], {}
        self.unavailable = False
        self.exit_code = 0
    def __call__(self, argv, **options):
        self.calls.append(argv)
        action = argv[2]
        if action == 'capabilities':
            value = dict(state='available', target='fixture', backend='remote-agent', capabilities=dict(upload=True))
        elif action == 'upload':
            value = dict(state='completed', target='fixture', exit=0)
        elif action == 'run':
            handle = 'remote:fixture:job:'+str(len(self.jobs)+1)
            script = Path(argv[argv.index('--file')+1]).read_text()
            if 'A3REMOTE_STAGE=compile' in script:
                output = 'A3CANDIDATE_COMPILED='+'b'*64
            elif 'native_measurement_driver.py' in script:
                output = 'A3KERNEL_OUTPUT='+json.dumps(EXPECTED)+'\nA3TIMING_US=10\nA3TIMING_US=12\n'
                output += 'A3PROFILE_META='+json.dumps(dict(mode='timing', kernel_name='sum_three',
                    target='discovered-a3', language='ascend-c', runtime='native-ascend-c', logical_device=0))
            else:
                output = 'A3KERNEL_OUTPUT='+json.dumps(EXPECTED)
            self.jobs[handle] = output
            value = dict(state='queued', handle=handle)
        elif action in ('observe', 'result'):
            if self.unavailable:
                value = dict(state='observation-unavailable', handle=argv[3])
                self.unavailable = False
                raise subprocess.TimeoutExpired(argv, 1)
            value = dict(state='completed' if self.exit_code == 0 else 'failed', handle=argv[3], exit=self.exit_code)
        else:
            assert action == 'logs'
            value = dict(state='completed', handle=argv[3],
                         content=self.jobs[argv[3]] if argv[argv.index('--stream')+1] == 'stdout' else '')
        return subprocess.CompletedProcess(argv, 0, json.dumps(value), '')


def backend(tmp_path, remote, **kwargs):
    kwargs.setdefault('submission_reserve_seconds', 0)
    kwargs.setdefault('cleanup_reserve_seconds', 0)
    real_clock_probe = kwargs.pop('real_clock_probe', False)
    terminal_receipts = kwargs.pop('terminal_receipts', False)
    candidate = CplNativeCandidateBackend(cpl_remote='/skill/cpl-remote', target='fixture', runtime='named-runtime',
        remote_workspace=kwargs.pop('remote_workspace', '/owned/task'), physical_device=11, state_directory=tmp_path/'transport',
        process_runner=remote, **kwargs)
    if not real_clock_probe:
        candidate._sample_remote_clock = lambda key, expires_at, cwd: dict(
            remote_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            remote_deadline=time.monotonic()+expires_at-candidate.wall_clock())
    if not terminal_receipts:
        candidate._terminal_receipt = lambda *args, **options: None
        candidate._receipt_capable = lambda path: False
    return candidate


def test_cpl_compiles_executes_and_reuses_retained_handle_without_resubmit(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    compilation = candidate.compile(plan, tmp_path/'bundle')
    receipt = candidate.execute_receipt(compilation)
    assert json.loads((candidate.root / plan.execution_id / 'candidate.json').read_bytes()) == dict(project_id=plan.project_id)
    assert receipt.output == EXPECTED
    assert receipt.job_handle.startswith('remote:fixture:job:')
    assert candidate.compile(plan, tmp_path/'bundle') == compilation
    assert candidate.execute_receipt(compilation) == receipt
    submits = [call for call in remote.calls if call[2] == 'run']
    assert len(submits) == 2
    assert all(call[call.index('--runtime')+1] == 'named-runtime' for call in submits)
    assert all(call[0] == '/skill/cpl-remote' for call in remote.calls)


def test_resume_observes_same_compile_after_disconnect(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    remote.unavailable = True
    with pytest.raises(_PendingObservation):
        candidate.compile(plan, tmp_path/'bundle')
    compilation = backend(tmp_path, remote).compile(plan, tmp_path/'bundle')
    assert compilation.library_sha256 == 'b'*64
    assert len([call for call in remote.calls if call[2] == 'run']) == 1
    assert {call[3] for call in remote.calls if call[2] == 'observe'} == {'remote:fixture:job:1'}


def test_ambiguous_submission_is_never_repeated(tmp_path):
    calls = []
    def disconnected(argv, **options):
        calls.append(argv)
        if argv[2] == 'capabilities':
            return Remote()(argv, **options)
        if argv[2] == 'upload':
            return subprocess.CompletedProcess(argv, 0, '{"state":"completed"}', '')
        raise subprocess.TimeoutExpired(argv, 1)
    candidate = backend(tmp_path, disconnected)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    for _ in range(2):
        with pytest.raises(_PendingObservation):
            candidate.compile(plan, tmp_path/'bundle')
    assert len([call for call in calls if call[2] == 'run']) == 1


def test_foreign_observation_cannot_become_candidate_evidence(tmp_path):
    remote = Remote()
    def foreign(argv, **options):
        result = remote(argv, **options)
        if argv[2] == 'observe':
            result.stdout = '{"state":"completed","exit":0,"handle":"remote:other:job:1"}'
        return result
    with pytest.raises(ProjectInfrastructureError, match='foreign'):
        backend(tmp_path, foreign).compile(smoke_plan(target='discovered-a3', arch='discovered-arch'), tmp_path/'bundle')


def test_timing_requires_host_output_and_retains_route_evidence(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    compilation = candidate.compile(plan, tmp_path/'bundle')
    measurement = CplNativeMeasurements(compilation=compilation, candidate_backend=candidate,
        expected_output=EXPECTED, evidence_directory=tmp_path/'measurements',
        environment_identity=dict(cann_version='frozen-cann', profiler_version='frozen-msprof'))
    result = measurement.timing(warm_up=1, launch_count=2)
    assert result['summary']['median_us'] == 11
    assert result['summary']['instrumented'] is False
    assert result['identity']['execution']['runtime'] == 'named-runtime'
    assert measurement.timing(warm_up=1, launch_count=2) == result
    assert len([call for call in remote.calls if call[2] == 'run']) == 3
    with pytest.raises(ValueError, match='host verification'):
        CplNativeMeasurements(compilation=compilation, candidate_backend=candidate,
            expected_output=(0.,)*32, evidence_directory=tmp_path/'wrong',
            environment_identity=dict(cann_version='cann', profiler_version='msprof')).timing(launch_count=2)


def test_parent_elapsed_exhaustion_does_not_submit(tmp_path):
    remote = Remote()
    with pytest.raises(_PendingObservation):
        backend(tmp_path, remote, remaining_seconds=lambda: 0).compile(
            smoke_plan(target='discovered-a3', arch='discovered-arch'), tmp_path/'bundle')
    assert remote.calls == []


def test_remote_agent_uses_relative_transfer_and_cwd_with_absolute_payloads(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote, registry_workspace='/owned')
    compilation = candidate.compile(smoke_plan(target='discovered-a3', arch='discovered-arch'), tmp_path/'bundle')
    assert compilation.library_sha256 == 'b'*64
    upload = next(call for call in remote.calls if call[2] == 'upload')
    assert upload[5].startswith('task/')
    submit = next(call for call in remote.calls if call[2] == 'run')
    assert submit[submit.index('--cwd')+1] == 'task'
    assert '/owned/task/candidates/' in Path(submit[submit.index('--file')+1]).read_text()


def test_queue_expiry_cancels_owned_handle_without_resubmitting(tmp_path):
    now = [1000.0]
    remote = Remote()
    def delayed(argv, **options):
        if argv[2] == 'observe':
            remote.calls.append(argv)
            if '--wait' in argv:
                now[0] += 11
            return subprocess.CompletedProcess(argv, 0, json.dumps(dict(state='queued', handle=argv[3])), '')
        if argv[2] == 'cancel':
            remote.calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, json.dumps(dict(state='cancelled', handle=argv[3])), '')
        return remote(argv, **options)
    candidate = backend(tmp_path, delayed, timeout=60, cleanup_reserve_seconds=50, clock=lambda: now[0], wall_clock=lambda: now[0])
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    for _ in range(2):
        with pytest.raises(_PendingObservation, match='elapsed allocation expired'):
            candidate.compile(plan, tmp_path/'bundle')
    assert len([call for call in remote.calls if call[2] == 'run']) == 1
    assert [call[3] for call in remote.calls if call[2] == 'cancel'] == ['remote:fixture:job:1']
    record = json.loads((candidate.root/plan.execution_id/'compile'/'operation.json').read_text())
    assert record['expired'] is True
    assert record['cancel_requested'] is True


def test_payload_dispatched_after_frozen_expiry_does_not_execute(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote, timeout=10, remote_workspace=str(tmp_path/'remote'))
    candidate._sample_remote_clock = lambda *a, **k: dict(remote_deadline=time.monotonic()-100,
        remote_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    marker = tmp_path/'must-not-be-created'
    candidate.run_remote(('python', '-c', 'import pathlib,sys;pathlib.Path(sys.argv[1]).touch()', str(marker)), key='late')
    script = candidate.root/'late'/'command.sh'
    result = subprocess.run(('bash', str(script)), capture_output=True, text=True, check=False)
    assert result.returncode == 124
    assert not marker.exists()


def test_distinct_replays_run_fresh_but_same_replay_uses_cache(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    compilation = candidate.compile(plan, tmp_path/'bundle')
    measurements = CplNativeMeasurements(compilation=compilation, candidate_backend=candidate,
        expected_output=EXPECTED, evidence_directory=tmp_path/'measurements',
        environment_identity=dict(cann_version='cann', profiler_version='msprof'))
    first = measurements.timing(warm_up=1, launch_count=2, replay_id='first')
    assert measurements.timing(warm_up=1, launch_count=2, replay_id='first') == first
    measurements.timing(warm_up=1, launch_count=2, replay_id='second')
    scripts = [Path(call[call.index('--file')+1]).read_text() for call in remote.calls if call[2] == 'run']
    assert sum('native_measurement_driver.py' in script for script in scripts) == 2


def test_compile_from_two_directories_then_measurement_reuses_canonical_bundle(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    first = candidate.compile(plan, tmp_path/'actor-compile')
    second = candidate.compile(plan, tmp_path/'measurement-compile')
    assert second == first
    result = CplNativeMeasurements(compilation=second, candidate_backend=candidate,
        expected_output=EXPECTED, evidence_directory=tmp_path/'measurements',
        environment_identity=dict(cann_version='cann', profiler_version='msprof')).timing(launch_count=2)
    assert result['summary']['median_us'] == 11
    assert len([call for call in remote.calls if call[2] == 'upload']) == 1
    assert len([call for call in remote.calls if call[2] == 'run']) == 3


def test_compile_error_returned_when_terminal_poll_crosses_deadline(tmp_path):
    now = [1000.]
    remote = Remote()
    remote.exit_code = 1
    def compiler_error(argv, **options):
        result = remote(argv, **options)
        if argv[2] == 'observe':
            now[0] += 11
        if argv[2] == 'logs':
            stream = argv[argv.index('--stream')+1]
            result.stdout = json.dumps(dict(handle=argv[3], content=(
                'A3CANDIDATE_COMPILE_FAILED=1' if stream == 'stdout' else
                "candidate.cpp: error: no matching function for call to 'Mul'")))
        return result
    candidate = backend(tmp_path, compiler_error, timeout=60, cleanup_reserve_seconds=50,
                        clock=lambda: now[0], wall_clock=lambda: now[0])
    result = candidate.compile(smoke_plan(target='discovered-a3', arch='discovered-arch'), tmp_path/'bundle')
    assert result.error_type == 'NativeCompileError'
    assert "no matching function for call to 'Mul'" in result.detail
    assert not any(call[2] == 'cancel' for call in remote.calls)
    assert len([call for call in remote.calls if call[2] == 'run']) == 1


def test_expired_retained_job_is_observed_before_cancellation(tmp_path):
    now = [1000.]
    remote = Remote()
    remote.unavailable = True
    candidate = backend(tmp_path, remote, timeout=10, clock=lambda: now[0], wall_clock=lambda: now[0])
    with pytest.raises(_PendingObservation):
        candidate.run_remote(('false',), key='compile')
    now[0] += 20
    remote.exit_code = 1
    candidate.recover_owned_operations()
    result = candidate.run_remote(('false',), key='compile')
    assert result.returncode == 1
    assert len([call for call in remote.calls if call[2] == 'run']) == 1
    assert not any(call[2] == 'cancel' for call in remote.calls)
    assert result.job_handle == 'remote:fixture:job:1'


def test_terminal_streams_resume_after_elapsed_parent_and_log_disconnect(tmp_path):
    remaining = [100.]
    remote = Remote()
    failed = [False]
    def disconnected_stderr(argv, **options):
        if argv[2] == 'logs' and argv[argv.index('--stream')+1] == 'stderr' and not failed[0]:
            failed[0] = True
            remaining[0] = 0
            raise subprocess.TimeoutExpired(argv, 1)
        return remote(argv, **options)
    candidate = backend(tmp_path, disconnected_stderr, remaining_seconds=lambda: remaining[0])
    with pytest.raises(_PendingObservation):
        candidate.run_remote(('true',), key='compile')
    record = json.loads((candidate.root/'compile/operation.json').read_text())
    assert record['terminal_result']['returncode'] == 0
    assert 'stdout' in record['terminal_result']
    candidate.recover_owned_operations()
    result = candidate.run_remote(('true',), key='compile')
    assert result.returncode == 0
    assert len([c for c in remote.calls if c[2] == 'run']) == 1
    assert len([c for c in remote.calls if c[2] == 'logs' and c[c.index('--stream')+1] == 'stdout']) == 1


def test_expiry_observation_rejects_foreign_handle_without_cancelling(tmp_path):
    now = [1000.]
    remote = Remote()
    remote.unavailable = True
    candidate = backend(tmp_path, remote, timeout=10, clock=lambda: now[0], wall_clock=lambda: now[0])
    with pytest.raises(_PendingObservation):
        candidate.run_remote(('true',), key='compile')
    now[0] += 5
    def foreign(argv, **options):
        return subprocess.CompletedProcess(argv, 0, json.dumps(dict(handle='remote:other:job:1', state='failed', exit=1)), '')
    candidate._run = foreign
    with pytest.raises(ProjectInfrastructureError, match='foreign'):
        candidate.run_remote(('true',), key='compile')
    assert not any(c[2] == 'cancel' for c in remote.calls)


@pytest.mark.parametrize('log_seconds', [11, 46])
def test_terminal_collection_has_own_bounded_read_only_allocation(tmp_path, log_seconds):
    now = [1000.]
    remote = Remote()
    def slow_logs(argv, **options):
        result = remote(argv, **options)
        if argv[2] == 'logs':
            now[0] += log_seconds
        return result
    candidate = backend(tmp_path, slow_logs, timeout=10, clock=lambda: now[0], wall_clock=lambda: now[0])
    if log_seconds > 45:
        with pytest.raises(_PendingObservation, match='terminal result collection pending'):
            candidate.run_remote(('true',), key='compile')
    else:
        assert candidate.run_remote(('true',), key='compile').returncode == 0
    assert len([c for c in remote.calls if c[2] == 'logs']) == (1 if log_seconds > 45 else 2)
    assert not any(c[2] == 'cancel' for c in remote.calls)


@pytest.mark.parametrize('stage', ['compile', 'execute'])
def test_deadline_exit_is_infrastructure_interruption_not_candidate_failure(tmp_path, stage):
    remote = Remote()
    candidate = backend(tmp_path, remote)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    compilation = candidate.compile(plan, tmp_path/'bundle') if stage == 'execute' else None
    remote.exit_code = 124
    for _ in range(2):
        with pytest.raises(_PendingObservation, match='execution allocation expired'):
            if stage == 'compile':
                candidate.compile(plan, tmp_path/'bundle')
            else:
                candidate.execute_receipt(compilation)
    submits = [call for call in remote.calls if call[2] == 'run']
    assert len(submits) == (2 if stage == 'execute' else 1)
    operation = candidate.root/plan.execution_id/stage/'operation.json'
    assert json.loads(operation.read_text())['result']['returncode'] == 124


def test_cpl_records_stage_and_nested_cleanup_timing_without_resubmission(tmp_path):
    remote = Remote()
    now = [100.]
    def runner(argv, **kwargs):
        now[0] += 1
        return remote(argv, **kwargs)
    candidate = backend(tmp_path, runner, clock=lambda:now[0], wall_clock=lambda:now[0])
    candidate.compile(smoke_plan(target='discovered-a3', arch='discovered-arch'), tmp_path/'bundle')
    spans = [json.loads(line) for line in (candidate.root/'timing.jsonl').read_text().splitlines()]
    assert {row['stage'] for row in spans} == {'upload', 'compile', 'cleanup'}
    assert all(row['ended_epoch']-row['started_epoch'] == row['elapsed_seconds'] for row in spans)
    compile_span = next(row for row in spans if row['stage']=='compile')
    assert compile_span['elapsed_seconds'] == 4
    assert len([c for c in remote.calls if c[2]=='run']) == 1


def test_cpl_stop_metadata_distinguishes_operator_stop_from_deadline(tmp_path):
    def remaining():return 0
    remaining.stop_requested=lambda:True
    candidate = backend(tmp_path, Remote(), remaining_seconds=remaining)
    with pytest.raises(_PendingObservation) as caught:
        candidate.run_remote(('true',), key='compile')
    assert caught.value.interruption_reason == 'operator-stop'
    remaining.stop_requested=lambda:False
    with pytest.raises(_PendingObservation) as caught:
        candidate.run_remote(('true',), key='compile')
    assert caught.value.interruption_reason == 'deadline-exhausted'


def test_non_json_cli_failure_retains_bounded_diagnostics_without_replay(tmp_path):
    calls = []
    def rejected(argv, **options):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 2, '', 'x'*9000+'HTTP400 cwd validation failed')
    candidate = backend(tmp_path, rejected)
    with pytest.raises(_PendingObservation) as caught:
        candidate.run_remote(('true',), key='health')
    diagnostic = caught.value.transport_diagnostic
    assert diagnostic['returncode']==2 and len(diagnostic['stderr'])==8192
    assert diagnostic['stderr'].endswith('HTTP400 cwd validation failed')
    record = json.loads((candidate.root/'health/operation.json').read_text())
    assert record['observation_failure']==diagnostic
    assert record['state']=='submitting' and not record.get('handle')
    assert caught.value.interruption_reason=='transport-unresolved'
    with pytest.raises(_PendingObservation, match='ambiguous'):
        backend(tmp_path, rejected).run_remote(('true',), key='health')
    assert len(calls)==1


def test_readonly_recovery_settles_disconnected_job_without_submission(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote)
    remote.unavailable = True
    with pytest.raises(_PendingObservation):
        candidate.run_remote(['python', '-c', 'pass'], key='health/interrupted')
    before = len([call for call in remote.calls if call[2] == 'run'])
    candidate.recover_owned_operations()
    record = json.loads((candidate.root/'health/interrupted/operation.json').read_text())
    assert record['state'] == 'completed'
    assert len([call for call in remote.calls if call[2] == 'run']) == before == 1
    assert candidate.run_remote(['python', '-c', 'pass'], key='health/interrupted').returncode == 0


def test_readonly_recovery_preserves_ambiguous_and_foreign_allocation_records(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote)
    identity = dict(route=candidate.runtime_identity, workspace='/owned/task', arguments=[])
    for name, row in {'ambiguous':dict(identity=identity, state='submitting'),
            'foreign':dict(identity=dict(identity, route=dict(candidate.runtime_identity, physical_device=2)),
                           state='running', handle='remote:fixture:job:other')}.items():
        path = candidate.root/name/'operation.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row))
    candidate.recover_owned_operations()
    assert not remote.calls
    assert json.loads((candidate.root/'ambiguous/operation.json').read_text())['state']=='submitting'


@pytest.mark.parametrize('deadline', [False, True])
def test_real_launcher_distinguishes_deadline_escalation_from_normal_137(tmp_path, deadline):
    remote = Remote()
    def launch(argv, **options):
        if argv[2] != 'run':
            return remote(argv, **options)
        remote.calls.append(argv)
        script = Path(argv[argv.index('--file')+1])
        result = subprocess.run(('bash', str(script)), capture_output=True, text=True, timeout=15)
        handle = 'remote:fixture:job:real-launcher'
        remote.jobs[handle] = result.stdout
        remote.exit_code = result.returncode
        return subprocess.CompletedProcess(argv, 0, json.dumps(dict(state='completed', handle=handle, exit=result.returncode)), '')
    candidate = backend(tmp_path, launch, timeout=46 if deadline else 10,
                        cleanup_reserve_seconds=45 if deadline else 0, remote_workspace=str(tmp_path/'remote'))
    marker = tmp_path/'started'
    command = ('python', '-c',
        'import signal,time,pathlib,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
        'pathlib.Path(sys.argv[1]).touch(); time.sleep(60)', str(marker)) if deadline else (
        'python', '-c', 'import sys;sys.exit(137)')
    if deadline:
        with pytest.raises(_PendingObservation, match='execution allocation expired'):
            candidate.run_remote(command, key='real')
        assert marker.exists()
        assert remote.exit_code == 124
    else:
        assert candidate.run_remote(command, key='real').returncode == 137
    assert len([call for call in remote.calls if call[2]=='run']) == 1


@pytest.mark.parametrize('legacy', [False, True])
def test_expired_legacy_137_is_held_without_rewriting_or_resubmitting(tmp_path, legacy):
    now = [1000.0]
    remote = Remote()
    remote.exit_code = 137
    candidate = backend(tmp_path, remote, wall_clock=lambda:now[0])
    argv = ['python', '-c', 'pass']
    directory = candidate.root/'retained'
    if legacy:
        directory.mkdir(parents=True)
        (directory/'command.json').write_text(json.dumps(dict(argv=argv, expires_at=1010.0)))
        script = directory/'command.sh'
        script.write_text("exec timeout --signal=TERM --kill-after=10s 10s python -c 'pass'\n")
        handle = 'remote:fixture:job:legacy'
        remote.jobs[handle] = ''
        arguments = ['run','fixture','--runtime','named-runtime','--cwd','/owned/task','--file',str(script.resolve())]
        (directory/'operation.json').write_text(json.dumps(dict(state='running',handle=handle,expires_at=1010.0,
            identity=dict(route=candidate.runtime_identity,workspace='/owned/task',arguments=arguments))))
    assert candidate.run_remote(argv, key='retained').returncode == 137
    script = (directory/'command.sh').read_bytes()
    payload = (directory/'command.json').read_bytes()
    now[0] = 2000.0
    if legacy:
        with pytest.raises(_PendingObservation, match='execution allocation expired'):
            candidate.run_remote(argv, key='retained')
    else:
        assert candidate.run_remote(argv, key='retained').returncode == 137
    assert (directory/'command.sh').read_bytes() == script
    assert (directory/'command.json').read_bytes() == payload
    assert len([call for call in remote.calls if call[2]=='run']) == (0 if legacy else 1)


def test_real_launcher_kills_surviving_descendant_after_leader_exits(tmp_path):
    import os
    import signal
    import sys
    remote = Remote()
    candidate = backend(tmp_path, remote, timeout=2, remote_workspace=str(tmp_path/'remote'))
    ready = tmp_path/'descendant.pid'
    parent = tmp_path/'parent.py'
    descendant = "import os,pathlib,signal,time,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    parent.write_text("import pathlib,subprocess,sys,time\n"
        + "subprocess.Popen([sys.executable,'-c'," + repr(descendant) + ",sys.argv[1]])\n"
        + "while not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.01)\n"
        + "time.sleep(60)\n")
    # Generate the actual remote launcher; the fixture transport does not execute it.
    candidate.run_remote([sys.executable, str(parent), str(ready)], key='tree')
    script = candidate.root/'tree/command.sh'
    pid = None
    try:
        result = subprocess.run(('bash', str(script)), capture_output=True, text=True, timeout=17)
        assert ready.exists(), result.stderr
        pid = int(ready.read_text())
        assert result.returncode == 124
        status = Path(f'/proc/{pid}/stat')
        # An orphan may briefly remain as a zombie until the system reaper runs.
        settled = time.monotonic() + 1
        while status.exists():
            try: state = status.read_text().split(') ',1)[1].split()[0]
            except FileNotFoundError: break
            if state in ('X','Z'): break
            if time.monotonic() >= settled: pytest.fail('deadline descendant remains alive')
            time.sleep(0.01)
    finally:
        if pid is None and ready.exists(): pid = int(ready.read_text())
        if pid is not None:
            try: os.kill(pid, signal.SIGKILL)
            except ProcessLookupError: pass


@pytest.mark.parametrize('cancel_signal', ['SIGTERM', 'SIGINT'])
def test_real_launcher_cancellation_kills_isolated_payload_group(tmp_path, cancel_signal):
    import os
    import signal
    import sys
    ready = tmp_path/'ready.json'
    descendant = (
        "import json,os,pathlib,signal,sys,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "signal.signal(signal.SIGINT,signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getppid(),os.getpid()])); "
        "time.sleep(60)"
    )
    parent = tmp_path/'parent.py'
    parent.write_text("import signal,subprocess,sys,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "signal.signal(signal.SIGINT,signal.SIG_IGN)\n"
        + "subprocess.Popen([sys.executable,'-c'," + repr(descendant) + ",sys.argv[1]])\n"
        + "time.sleep(60)\n")
    candidate = backend(tmp_path, Remote(), remote_workspace=str(tmp_path/'remote'))
    candidate.run_remote([sys.executable, str(parent), str(ready)], key='cancelled-tree')
    process = subprocess.Popen(('bash', str(candidate.root/'cancelled-tree/command.sh')),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    pids = []
    try:
        deadline = time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:
            time.sleep(0.01)
        assert ready.exists(), 'payload group did not become ready'
        pids = json.loads(ready.read_text())
        os.killpg(process.pid, getattr(signal,cancel_signal))
        assert process.wait(timeout=2) == 124
        deadline = time.monotonic()+1
        for pid in pids:
            status = Path(f'/proc/{pid}/stat')
            while status.exists():
                try: state = status.read_text().split(') ',1)[1].split()[0]
                except FileNotFoundError: break
                if state in ('X','Z'): break
                assert time.monotonic()<deadline, 'cancelled payload remained alive'
                time.sleep(0.01)
    finally:
        if not pids and ready.exists(): pids = json.loads(ready.read_text())
        for pid in pids:
            try: os.kill(pid,signal.SIGKILL)
            except ProcessLookupError: pass
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL)
        process.communicate(timeout=3)


@pytest.mark.parametrize('legacy', [False, True])
def test_pending_measurement_binds_replay_before_submission_and_recovers_same_handle(tmp_path, legacy):
    remote = Remote()
    candidate = backend(tmp_path, remote)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    compilation = candidate.compile(plan, tmp_path/'bundle')
    measurements = CplNativeMeasurements(compilation=compilation, candidate_backend=candidate,
        expected_output=EXPECTED, evidence_directory=tmp_path/'measurements',
        environment_identity=dict(cann_version='cann', profiler_version='msprof'))
    def disconnect(argv, **options):
        result = remote(argv, **options)
        if argv[2] == 'run' and 'native_measurement_driver.py' in Path(argv[argv.index('--file')+1]).read_text():
            remote.unavailable = True
        return result
    candidate._run = disconnect
    with pytest.raises(_PendingObservation):
        measurements.timing(warm_up=1, launch_count=2, replay_id='pending')
    receipts = list(measurements.root.glob('*/measurement.json'))
    assert len(receipts) == 1
    if legacy:
        receipts[0].unlink()  # Actual pre-upgrade layout: only transport journals.
    pending = json.loads(next((candidate.root/'measurements').glob('*/operation.json')).read_text())
    before = len([call for call in remote.calls if call[2]=='run'])
    observation_start = len(remote.calls)
    with pytest.raises(ProjectInfrastructureError):
        measurements.timing(warm_up=2, launch_count=2, replay_id='pending')
    assert len([call for call in remote.calls if call[2]=='run']) == before
    candidate._run = remote
    result = measurements.timing(warm_up=1, launch_count=2, replay_id='pending')
    assert result['summary']['median_us'] == 11
    assert [call[3] for call in remote.calls[observation_start:] if call[2]=='observe'] == [pending['handle']]
    assert len([call for call in remote.calls if call[2]=='run']) == before
    measurement_runs = [call for call in remote.calls if call[2]=='run'
        and 'native_measurement_driver.py' in Path(call[call.index('--file')+1]).read_text()]
    assert len(measurement_runs) == 1


def test_short_remaining_allocation_does_not_start_upload(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote, remaining_seconds=lambda: .37,
                        submission_reserve_seconds=30, cleanup_reserve_seconds=45)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    with pytest.raises(_PendingObservation) as error:
        candidate.compile(plan, tmp_path/'bundle')
    assert error.value.interruption_reason == 'deadline-exhausted'
    assert remote.calls == []
    path = candidate.root/plan.execution_id/'upload/operation.json'
    assert not path.exists() or json.loads(path.read_text())['state'] == 'new'


@pytest.mark.parametrize('operation', ['run', 'upload'])
def test_submission_receipt_survives_cli_loss_and_never_resubmits(tmp_path, operation):
    remote = Remote()
    lost = [False]
    handle = 'remote:fixture:'+('job' if operation == 'run' else 'transfer')+':receipt'
    def submitted(argv, **options):
        if argv[2] == operation and not lost[0]:
            lost[0] = True
            remote.calls.append(argv)
            receipt = dict(target='fixture', operation=operation, state='queued', handle=handle)
            if operation == 'run':
                receipt['runtime'] = 'named-runtime'
                remote.jobs[handle] = 'diagnostic'
            Path(argv[argv.index('--receipt-file')+1]).write_text(json.dumps(receipt))
            if operation == 'upload':
                assert '--no-wait' in argv
            raise subprocess.TimeoutExpired(argv, 1)
        return remote(argv, **options)
    candidate = backend(tmp_path, submitted)
    def execute():
        return candidate.run_remote(('true',), key='compile') if operation == 'run' else candidate._operation(
            'upload', ('upload', 'fixture', '/local/archive', '/owned/task/archive'), kind='transfer')
    with pytest.raises(_PendingObservation):
        execute()
    result = execute()
    assert result.get('state') == 'completed' if isinstance(result, dict) else result.returncode == 0
    assert len([call for call in remote.calls if call[2] == operation]) == 1
    assert {call[3] for call in remote.calls if call[2] == 'observe'} == {handle}


@pytest.mark.parametrize('change', ['foreign-target', 'foreign-runtime', 'malformed'])
def test_invalid_submission_receipt_cannot_authenticate_a_job(tmp_path, change):
    remote = Remote()
    def disconnected(argv, **options):
        path = Path(argv[argv.index('--receipt-file')+1])
        receipt = dict(target='fixture', operation='run', runtime='named-runtime',
                       state='queued', handle='remote:fixture:job:owned')
        if change == 'foreign-target': receipt['target'] = 'other'
        if change == 'foreign-runtime': receipt['runtime'] = 'other'
        path.write_text('{' if change == 'malformed' else json.dumps(receipt))
        remote.calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 1)
    candidate = backend(tmp_path, disconnected)
    for _ in range(2):
        with pytest.raises(ProjectInfrastructureError, match='receipt'):
            candidate.run_remote(('true',), key='compile')
    assert len(remote.calls) == 1


def test_missing_exit_preserves_diagnostics_without_candidate_failure(tmp_path):
    remote = Remote()
    def no_exit(argv, **options):
        result = remote(argv, **options)
        if argv[2] in ('observe', 'result'):
            result.returncode = 1  # A valid terminal JSON response survives CLI nonzero.
            result.stdout = json.dumps(dict(handle=argv[3], state='failed', exit=None))
        return result
    candidate = backend(tmp_path, no_exit)
    with pytest.raises(_PendingObservation, match='authenticated exit'):
        candidate.run_remote(('true',), key='compile')
    record = json.loads((candidate.root/'compile/operation.json').read_text())
    assert record['terminal_result']['stdout']
    assert record['terminal_result']['stderr'] == ''
    assert 'returncode' not in record['terminal_result'] and 'result' not in record
    candidate.recover_owned_operations()
    assert len([call for call in remote.calls if call[2] == 'run']) == 1


def test_ambiguous_immutable_upload_is_quarantined_but_execution_stays_blocking(tmp_path):
    from benchmarks.ascend.runtime_allocation import unsettled_operations
    remote = Remote()
    def lost_upload(argv, **options):
        if argv[2] == 'capabilities':
            return remote(argv, **options)
        remote.calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 1)
    candidate = backend(tmp_path, lost_upload)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    for _ in range(2):
        with pytest.raises(_PendingObservation):
            candidate.compile(plan, tmp_path/'bundle')
    assert unsettled_operations(candidate.root) == []
    record = json.loads((candidate.root/plan.execution_id/'upload/operation.json').read_text())
    assert record['state'] == 'quarantined'
    assert len([call for call in remote.calls if call[2] == 'upload']) == 1
    with pytest.raises(_PendingObservation):
        candidate.run_remote(('true',), key='compile')
    assert len(unsettled_operations(candidate.root)) == 1


def test_explicit_readonly_recovery_quarantines_legacy_unknown_upload(tmp_path):
    from benchmarks.ascend.runtime_allocation import unsettled_operations
    candidate = backend(tmp_path, Remote())
    key = 'a'*64
    path = candidate.root/key/'upload/operation.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(state='submitting', expires_at=0, identity=dict(
        route=candidate.runtime_identity, workspace='/owned/task',
        arguments=['upload', 'fixture', '/local/archive', '/owned/task/'+key+'.tar']))))
    assert len(unsettled_operations(candidate.root)) == 1
    candidate.recover_owned_operations()
    assert unsettled_operations(candidate.root) == []
    assert candidate._run.calls == []


def test_terminal_cleanup_deadline_is_frozen_across_resume(tmp_path):
    now = [1000.]
    remote = Remote()
    def slow_logs(argv, **options):
        result = remote(argv, **options)
        if argv[2] == 'logs':
            assert options['timeout'] <= 45
            now[0] += 46
        return result
    candidate = backend(tmp_path, slow_logs, clock=lambda:now[0], wall_clock=lambda:now[0])
    for _ in range(2):
        with pytest.raises(_PendingObservation, match='terminal result collection pending'):
            candidate.run_remote(('true',), key='compile')
    record = json.loads((candidate.root/'compile/operation.json').read_text())
    assert record['cleanup_expires_at'] == 1045
    assert len([call for call in remote.calls if call[2] == 'logs']) == 1
    assert len([call for call in remote.calls if call[2] == 'run']) == 1


def test_terminal_looking_submission_without_job_handle_remains_blocking(tmp_path):
    from benchmarks.ascend.runtime_allocation import unsettled_operations
    calls = []
    def missing(argv, **options):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, json.dumps(dict(state='failed', exit=None)), '')
    candidate = backend(tmp_path, missing)
    with pytest.raises(_PendingObservation, match='retained handle'):
        candidate.run_remote(('true',), key='compile')
    assert len(unsettled_operations(candidate.root)) == 1
    with pytest.raises(_PendingObservation, match='ambiguous'):
        candidate.run_remote(('true',), key='compile')
    assert len(calls) == 1


def test_failed_upload_recovery_does_not_block_next_environment_admission(tmp_path, monkeypatch):
    from test_environment_admission import setup
    from benchmarks.ascend.runtime_allocation import unsettled_operations
    remote = Remote()
    handle = 'remote:fixture:transfer:failed-copy'
    def transfer(argv, **options):
        if argv[2] == 'upload':
            remote.calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, json.dumps(dict(state='queued', handle=handle)), '')
        return remote(argv, **options)
    candidate = backend(tmp_path, transfer)
    remote.unavailable = True
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    with pytest.raises(_PendingObservation):
        candidate.compile(plan, tmp_path/'bundle')
    remote.exit_code = 1
    admission, health, probes, _ = setup(tmp_path/'environment', monkeypatch)
    admission.transports = (candidate,)
    admission.pending_root = tmp_path
    admission(lambda:100)
    record = json.loads((candidate.root/plan.execution_id/'upload/operation.json').read_text())
    assert record['state'] == 'failed' and record['terminal_failure']['handle'] == handle
    assert 'result' not in record and unsettled_operations(tmp_path) == []
    observations = len([call for call in remote.calls if call[2] == 'observe'])
    admission(lambda:100)
    assert len([call for call in remote.calls if call[2] == 'observe']) == observations
    assert health == [11,11] and probes == ['probe']
    remote.exit_code = 0
    assert candidate.run_remote(('true',), key='independent').returncode == 0


def test_ssh_upload_uses_bounded_sync_copy_and_caches_public_capabilities(tmp_path):
    from dataclasses import replace
    remote = Remote()
    def ssh(argv, **options):
        result = remote(argv, **options)
        if argv[2] == 'capabilities':
            value = json.loads(result.stdout)
            value['backend'] = 'ssh'
            result.stdout = json.dumps(value)
        elif argv[2] == 'upload':
            assert '--receipt-file' not in argv and '--no-wait' not in argv
            assert 0 < options['timeout'] <= 600
        elif argv[2] == 'run':
            assert '--receipt-file' in argv
        return result
    candidate = backend(tmp_path, ssh)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    assert candidate.compile(plan, tmp_path/'bundle').library_sha256 == 'b'*64
    assert candidate.compile(replace(plan, source=plan.source+'\n'), tmp_path/'next').library_sha256 == 'b'*64
    assert len([call for call in remote.calls if call[2] == 'capabilities']) == 1
    assert len([call for call in remote.calls if call[2] == 'upload']) == 2


def test_interrupted_ssh_copy_quarantines_only_its_immutable_destination(tmp_path):
    from benchmarks.ascend.runtime_allocation import unsettled_operations
    calls = []
    def ssh(argv, **options):
        calls.append(argv)
        if argv[2] == 'capabilities':
            return subprocess.CompletedProcess(argv, 0, json.dumps(dict(target='fixture', state='available',
                backend='ssh', capabilities=dict(upload=True))), '')
        assert argv[2] == 'upload' and '--receipt-file' not in argv and '--no-wait' not in argv
        raise subprocess.TimeoutExpired(argv, options['timeout'])
    candidate = backend(tmp_path, ssh)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    for _ in range(2):
        with pytest.raises(_PendingObservation):
            candidate.compile(plan, tmp_path/'bundle')
    assert unsettled_operations(candidate.root) == []
    assert len([call for call in calls if call[2] == 'upload']) == 1
