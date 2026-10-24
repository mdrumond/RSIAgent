"""Native A3 execution through the user-wide CPL registry and retained handles."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
from statistics import median
import subprocess
import time

from benchmarks.a3kernels.candidate import CandidateCompilation, _parse_output
from benchmarks.a3kernels.native_candidate import NativeCandidatePlan
from benchmarks.a3kernels.native_measurements import NativeMeasurementBackend
from benchmarks.a3kernels.phase1_evidence import canonical_digest
from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt, FailedEvidence, attest
from benchmarks.a3kernels.profiling import parse_timing_output
from benchmarks.a3kernels.remote_candidate import GZA3RemoteCandidateBackend, _COMPILE_SCRIPT, _EXECUTE_SCRIPT, _PendingObservation
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2))
    temporary.replace(path)


class CplNativeCandidateBackend:
    """A named registry runtime owns activation; no SSH or validation adapters."""
    build_bundle = GZA3RemoteCandidateBackend.build_bundle

    def __init__(self, *, cpl_remote, target, runtime, remote_workspace, physical_device,
                 state_directory, profile='bz-a3-1', timeout=600, remaining_seconds=None,
                 process_runner=subprocess.run, clock=time.monotonic):
        workspace = PurePosixPath(remote_workspace)
        if not workspace.is_absolute() or '..' in workspace.parts:
            raise ValueError('remote_workspace must be an absolute registry task workspace')
        if type(physical_device) is not int or physical_device < 0 or timeout <= 0:
            raise ValueError('device and elapsed allocation must be valid')
        if not target or not runtime:
            raise ValueError('explicit registry target and runtime are required')
        self._cpl, self.target, self.runtime = str(cpl_remote), target, runtime
        self._workspace, self._device, self.profile = workspace, physical_device, profile
        self.root, self.timeout = Path(state_directory), timeout
        self.remaining_seconds, self._run, self.clock = remaining_seconds, process_runner, clock
        self.runtime_identity = dict(target=target, runtime=runtime, profile=profile,
                                     physical_device=physical_device, logical_device=0)

    def _allowance(self):
        remaining = self.timeout if self.remaining_seconds is None else min(self.timeout, self.remaining_seconds())
        if not math.isfinite(remaining) or remaining <= 0:
            raise _PendingObservation('CPL native elapsed allocation exhausted')
        return remaining

    def _call(self, arguments, *, seconds):
        try:
            completed = self._run((self._cpl, '--json', *arguments), text=True,
                                 capture_output=True, check=False, timeout=max(.001, seconds))
            value = json.loads(completed.stdout)
        except (subprocess.TimeoutExpired, OSError, ValueError) as error:
            raise _PendingObservation('CPL command outcome unavailable; inspect retained state before retry') from error
        if not isinstance(value, dict):
            raise ProjectInfrastructureError('CPL response must be an object')
        return value

    def _operation(self, key, arguments, *, kind):
        """Never submit again after an uncertain submission or observable handle."""
        path = self.root / key / 'operation.json'
        identity = dict(route=self.runtime_identity, workspace=str(self._workspace), arguments=list(arguments))
        record = json.loads(path.read_text()) if path.exists() else dict(identity=identity, state='new')
        if record['identity'] != identity:
            raise ProjectInfrastructureError('retained CPL operation identity changed')
        if 'result' in record:
            return record['result']
        deadline = self.clock() + self._allowance()
        if not record.get('handle'):
            if record['state'] != 'new':
                raise _PendingObservation('CPL submission outcome is ambiguous; do not resubmit')
            record['state'] = 'submitting'
            _save(path, record)
            value = self._call(arguments, seconds=max(.001, deadline-self.clock()))
            handle = value.get('handle')
            if handle is not None:
                if not isinstance(handle, str) or not handle.startswith(f'remote:{self.target}:{kind}:'):
                    raise ProjectInfrastructureError('CPL returned a foreign retained handle')
                record['handle'] = handle
            record['state'] = value.get('state')
            _save(path, record)
        else:
            value = dict(state=record['state'], handle=record['handle'])
        while value.get('state') not in ('completed', 'failed', 'cancelled'):
            if not record.get('handle') or self.clock() >= deadline:
                raise _PendingObservation('CPL operation pending; retain and observe '+str(record.get('handle')))
            value = self._call(('observe', record['handle'], '--wait', '--timeout',
                                str(max(1, min(30, int(deadline-self.clock()))))),
                               seconds=max(.001, deadline-self.clock()))
            if value.get('handle') != record['handle']:
                raise ProjectInfrastructureError('CPL observation returned a foreign handle')
            record['state'] = value.get('state')
            _save(path, record)
        if kind == 'transfer':
            if value.get('state') != 'completed':
                raise ProjectInfrastructureError('CPL upload did not complete')
            result = value
        else:
            if not record.get('handle') or type(value.get('exit')) is not int:
                # A resumed terminal handle needs its final exit status fetched.
                value = self._call(('result', record['handle']), seconds=self._allowance())
            if value.get('handle') != record.get('handle') or type(value.get('exit')) is not int:
                raise ProjectInfrastructureError('CPL terminal result lacks its authenticated exit status')
            result = dict(handle=record['handle'], returncode=value['exit'])
            for stream in ('stdout', 'stderr'):
                logs = self._call(('logs', record['handle'], '--stream', stream, '--tail', '1000000'),
                                  seconds=self._allowance())
                if logs.get('handle') != record['handle'] or not isinstance(logs.get('content'), str):
                    raise ProjectInfrastructureError('CPL logs returned a foreign or invalid result')
                result[stream] = logs['content']
        record['result'] = result
        _save(path, record)
        return result

    def remote_candidate_directory(self, plan):
        return self._workspace / 'candidates' / plan.execution_id

    def run_remote(self, argv, *, key, cwd=None):
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        script = directory / 'command.sh'
        body = 'set -eu\nexport ASCEND_RT_VISIBLE_DEVICES='+str(self._device)+'\n'
        body += 'export ASCEND_VISIBLE_DEVICES='+str(self._device)+'\n'
        body += 'timeout --signal=TERM --kill-after=10s '+str(self._allowance())+'s '
        body += shlex.join(tuple(str(a) for a in argv))+'\n'
        # Freeze original server-side elapsed allowance on resume.
        if not script.exists():
            script.write_text(body)
        arguments = ('run', self.target, '--runtime', self.runtime,
                     '--cwd', str(cwd or self._workspace), '--file', str(script.resolve()))
        value = self._operation(key, arguments, kind='job')
        result = subprocess.CompletedProcess(argv, value['returncode'], value['stdout'], value['stderr'])
        result.job_handle = value['handle']
        return result

    def _require_plan(self, plan):
        if not isinstance(plan, NativeCandidatePlan) or plan.execution_profile != self.profile:
            raise ValueError('CPL native transport requires its selected native execution profile')

    def compile(self, plan, local_directory):
        self._require_plan(plan)
        key = plan.execution_id
        bundle = self.build_bundle(plan, Path(local_directory) / (key+'.tar'))
        archive = self._workspace / (key+'.tar')
        self._operation(key+'/upload', ('upload', self.target, str(bundle.path.resolve()), str(archive)), kind='transfer')
        result = self.run_remote(('bash', '-c', _COMPILE_SCRIPT, 'rsi-native', str(archive),
            str(self.remote_candidate_directory(plan)), bundle.sha256, key, plan.source_fingerprint), key=key+'/compile')
        if result.returncode:
            kind = 'NativeCompileError' if 'A3CANDIDATE_COMPILE_FAILED=1' in result.stdout.splitlines() else 'NativeHostError'
            return FailedEvidence.create(plan, stage='compile', error_type=kind, detail=result.stderr or result.stdout)
        markers = [line.split('=', 1)[1] for line in result.stdout.splitlines() if line.startswith('A3CANDIDATE_COMPILED=')]
        if len(markers) != 1 or len(markers[0]) != 64 or any(c not in '0123456789abcdef' for c in markers[0]):
            raise ProjectInfrastructureError('native compilation returned an invalid library identity')
        body = dict(plan=plan, library_sha256=markers[0], stdout=result.stdout, stderr=result.stderr)
        return CandidateCompilation(**body, attestation_sha256=attest(body))

    def execute_receipt(self, compilation):
        plan = compilation.plan
        self._require_plan(plan)
        result = self.run_remote(('bash', '-c', _EXECUTE_SCRIPT, 'rsi-native',
            str(self.remote_candidate_directory(plan)), plan.execution_id, plan.source_fingerprint,
            compilation.library_sha256), key=plan.execution_id+'/execute')
        if result.returncode:
            return FailedEvidence.create(plan, stage='execute', error_type='RemoteExecutionError', detail=result.stderr or result.stdout)
        return ExecutionReceipt(exit_code=0, output=_parse_output(result.stdout, plan.padded_length),
            stdout=result.stdout, stderr=result.stderr, job_handle=result.job_handle,
            metadata=(('library_sha256', compilation.library_sha256),
                      ('remote_candidate_directory', str(self.remote_candidate_directory(plan)))))


class CplNativeMeasurements(NativeMeasurementBackend):
    """Host-verified native timing/profiling; raw vendor metrics retain semantics."""
    def __init__(self, *, compilation, candidate_backend, expected_output, environment_identity,
                 evidence_directory, atol=1e-5, rtol=1e-5,
                 supported_metrics=('Basic', 'PipeUtilization'), **_unused):
        if not isinstance(compilation.plan, NativeCandidatePlan):
            raise ValueError('measurements require a native candidate')
        if any(not environment_identity.get(k) for k in ('cann_version', 'profiler_version')):
            raise ValueError('measurements require discovered CANN/profiler identities')
        self._compilation, self._candidate_backend = compilation, candidate_backend
        self._expected, self._environment = tuple(expected_output), dict(environment_identity)
        if len(self._expected) != compilation.plan.output.numel or any(not math.isfinite(x) for x in self._expected):
            raise ValueError('host reference must match finite native output')
        if any(not math.isfinite(v) or v < 0 for v in (atol, rtol)):
            raise ValueError('native tolerances must be finite nonnegative')
        self._atol, self._rtol = atol, rtol
        self._supported_metrics = frozenset(supported_metrics)
        self.root = Path(evidence_directory)

    def _measure(self, mode, warm_up, launch_count, replay_id, metric):
        if type(warm_up) is not int or type(launch_count) is not int or warm_up < 0 or launch_count < 1:
            raise ValueError('invalid native measurement workload')
        plan = self._compilation.plan
        backend = self._candidate_backend
        identity = dict(execution_id=plan.execution_id, source_fingerprint=plan.source_fingerprint,
            library_sha256=self._compilation.library_sha256, mode=mode, warm_up=warm_up,
            launch_count=launch_count, metric=metric, environment=self._environment,
            execution=backend.runtime_identity, host_reference_sha256=canonical_digest(self._expected),
            atol=self._atol, rtol=self._rtol, kernel_name=plan.kernel_name, target=plan.target, arch=plan.arch,
            inputs=[dict(name=t.name, shape=list(t.shape), dtype=t.dtype, logical_shape=list(t.logical_shape)) for t in plan.inputs],
            output=dict(shape=list(plan.output.shape), logical_shape=list(plan.output.logical_shape), dtype=plan.output.dtype))
        request_id = canonical_digest(identity)
        replay = replay_id or request_id
        record = self.root / hashlib.sha256(replay.encode()).hexdigest() / 'measurement.json'
        if record.exists():
            retained = json.loads(record.read_text())
            if retained['request_id'] != request_id:
                raise ProjectInfrastructureError('native measurement replay identity changed')
            return retained['result']
        verification = self._verify()
        directory = backend.remote_candidate_directory(plan)
        argv = ['python', str(directory/'native_measurement_driver.py'), mode, '--candidate-dir', str(directory),
                '--kernel', plan.kernel_name, '--execution-id', plan.execution_id,
                '--source-fingerprint', plan.source_fingerprint, '--library-sha256', self._compilation.library_sha256,
                '--warm-up', str(warm_up), '--launch-count', str(launch_count)]
        if metric:
            argv += ['--metric', 'ArithmeticUtilization' if metric == 'Basic' else metric]
        result = backend.run_remote(argv, key='measurements/'+request_id, cwd=directory)
        if result.returncode:
            raise ProjectInfrastructureError(result.stderr or result.stdout or 'native measurement failed')
        self._verify_values(_parse_output(result.stdout, len(self._expected)))
        meta = self._one_json_marker(result.stdout, 'A3PROFILE_META=', 'native metadata')
        if any(meta.get(k) != v for k, v in dict(mode=mode, kernel_name=plan.kernel_name,
                target=plan.target, language='ascend-c', runtime='native-ascend-c', logical_device=0).items()):
            raise ProjectInfrastructureError('native measurement returned foreign metadata')
        if mode == 'timing':
            samples = parse_timing_output(result.stdout)
            if len(samples) != launch_count:
                raise ProjectInfrastructureError('native timing sample count differs')
            summary = dict(measurement='launch-plus-synchronize', samples_us=list(samples),
                           median_us=median(samples), instrumented=False)
        else:
            summary = self._one_json_marker(result.stdout, 'A3PROFILE_COMPACT=', 'native profile')
            if (summary.get('exported_kernels') != [plan.kernel_name] or not summary.get('selected_columns')
                    or not summary.get('report_sha256') or not (summary.get('metric_values') or summary.get('timeline'))):
                raise ProjectInfrastructureError('native profile lacks exact-kernel report evidence')
            summary.update(instrumented=True, metric=metric,
                           duration_semantics='instrumented diagnostic; not canonical latency')
        raw = dict(stdout=result.stdout, stderr=result.stderr, verification=verification, metadata=meta)
        raw_hash = canonical_digest(raw)
        measured = dict(identity=identity, summary=summary, evidence_sha256=raw_hash, artifact_payloads={raw_hash: raw})
        _save(record, dict(request_id=request_id, result=measured, evidence=dict(profile=backend.profile,
            handle=result.job_handle, status='completed', physical_device=backend._device, logical_device=0,
            remote_report=meta.get('remote_report'))))
        return measured
