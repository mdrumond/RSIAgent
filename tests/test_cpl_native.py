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
        if action == 'upload':
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
    return CplNativeCandidateBackend(cpl_remote='/skill/cpl-remote', target='fixture', runtime='named-runtime',
        remote_workspace='/owned/task', physical_device=11, state_directory=tmp_path/'transport',
        process_runner=remote, **kwargs)


def test_cpl_compiles_executes_and_reuses_retained_handle_without_resubmit(tmp_path):
    remote = Remote()
    candidate = backend(tmp_path, remote)
    plan = smoke_plan(target='discovered-a3', arch='discovered-arch')
    compilation = candidate.compile(plan, tmp_path/'bundle')
    receipt = candidate.execute_receipt(compilation)
    assert receipt.output == EXPECTED
    assert receipt.job_handle.startswith('remote:fixture:job:')
    assert candidate.compile(plan, tmp_path/'bundle') == compilation
    assert candidate.execute_receipt(compilation) == receipt
    submits = [call for call in remote.calls if call[2] == 'run']
    assert len(submits) == 2
    assert all(call[call.index('--runtime')+1] == 'named-runtime' for call in submits)
    script = Path(submits[0][submits[0].index('--file')+1]).read_text()
    assert 'ASCEND_RT_VISIBLE_DEVICES=11' in script  # Actual executed shell contract.
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
    assert upload[-1].startswith('task/')
    submit = next(call for call in remote.calls if call[2] == 'run')
    assert submit[submit.index('--cwd')+1] == 'task'
    assert '/owned/task/candidates/' in Path(submit[submit.index('--file')+1]).read_text()


def test_queue_expiry_cancels_owned_handle_without_resubmitting(tmp_path):
    now = [1000.0]
    remote = Remote()
    def delayed(argv, **options):
        if argv[2] == 'observe':
            remote.calls.append(argv)
            now[0] += 11
            return subprocess.CompletedProcess(argv, 0, json.dumps(dict(state='queued', handle=argv[3])), '')
        if argv[2] == 'cancel':
            remote.calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, json.dumps(dict(state='cancelled', handle=argv[3])), '')
        return remote(argv, **options)
    candidate = backend(tmp_path, delayed, timeout=10, clock=lambda: now[0], wall_clock=lambda: now[0])
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
    candidate = backend(tmp_path, remote, timeout=10, wall_clock=lambda: time.time()-100)
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
