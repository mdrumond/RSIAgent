"""Native A3 execution through the user-wide CPL registry and retained handles."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import median
import subprocess
import time
import uuid

from benchmarks.a3kernels.candidate import CandidateCompilation, _parse_output
from benchmarks.a3kernels.native_candidate import NativeCandidatePlan
from benchmarks.a3kernels.native_measurements import NativeMeasurementBackend
from benchmarks.a3kernels.phase1_evidence import canonical_digest
from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt, FailedEvidence, attest
from benchmarks.a3kernels.profiling import parse_timing_output
from .actor_time import preparation, current_budget
from .remote_command import (LAUNCHER, clock_source, launcher_source, payload_digest,
                             receipt_reader_source, validate_clock, validate_receipt)
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
                 state_directory, profile='bz-a3-1', timeout=600, remaining_seconds=None, registry_workspace=None,
                 process_runner=subprocess.run, clock=time.monotonic, wall_clock=time.time,
                 submission_reserve_seconds=30, cleanup_reserve_seconds=45):
        workspace = PurePosixPath(remote_workspace)
        if not workspace.is_absolute() or '..' in workspace.parts:
            raise ValueError('remote_workspace must be an absolute registry task workspace')
        if type(physical_device) is not int or physical_device < 0 or timeout <= 0:
            raise ValueError('device and elapsed allocation must be valid')
        if not target or not runtime:
            raise ValueError('explicit registry target and runtime are required')
        self._cpl, self.target, self.runtime = str(cpl_remote), target, runtime
        self._workspace, self._device, self.profile = workspace, physical_device, profile
        self.registry_workspace = PurePosixPath(registry_workspace) if registry_workspace is not None else None
        if self.registry_workspace is not None:
            if not self.registry_workspace.is_absolute() or '..' in self.registry_workspace.parts:
                raise ValueError('registry_workspace must be an absolute configured base')
            workspace.relative_to(self.registry_workspace)
        self.root, self.timeout = Path(state_directory), timeout
        self.remaining_seconds, self._run, self.clock = remaining_seconds, process_runner, clock
        self.wall_clock = wall_clock
        self._capabilities = None
        self.submission_reserve = submission_reserve_seconds
        self.cleanup_reserve = cleanup_reserve_seconds
        if min(self.submission_reserve, self.cleanup_reserve) < 0:
            raise ValueError('elapsed reserves must be nonnegative')
        self.runtime_identity = dict(target=target, runtime=runtime, profile=profile,
                                     physical_device=physical_device, logical_device=0)

    def _pending(self, message, reason='transport-unresolved'):
        error = _PendingObservation(message)
        stopped = getattr(self.remaining_seconds, 'stop_requested', lambda: False)()
        error.interruption_reason = 'operator-stop' if stopped else reason
        return error

    def _timed(self, stage, callback, *, key=None):
        started, epoch = self.clock(), self.wall_clock()
        try:
            return callback()
        finally:
            elapsed = max(0, self.clock()-started)
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / 'timing.jsonl').open('a') as out:
                out.write(json.dumps(dict(stage=stage, key=key, started_epoch=epoch,
                    ended_epoch=epoch+elapsed, elapsed_seconds=elapsed))+'\n')

    def _allowance(self):
        remaining = self.timeout if self.remaining_seconds is None else min(self.timeout, self.remaining_seconds())
        if not math.isfinite(remaining) or remaining <= 0:
            raise self._pending('CPL native elapsed allocation exhausted', 'deadline-exhausted')
        return remaining

    def _call(self, arguments, *, seconds):
        completed = None
        try:
            completed = self._run((self._cpl, '--json', *arguments), text=True,
                                 capture_output=True, check=False, timeout=max(.001, seconds))
            value = json.loads(completed.stdout)
        except (subprocess.TimeoutExpired, OSError, ValueError) as error:
            pending = self._pending('CPL command outcome unavailable; inspect retained state before retry')
            if completed is not None:
                pending.transport_diagnostic = dict(returncode=completed.returncode,
                    stdout=completed.stdout[-8192:], stderr=completed.stderr[-8192:])
            raise pending from error
        if not isinstance(value, dict):
            raise ProjectInfrastructureError('CPL response must be an object')
        return value

    def _operation(self, key, arguments, *, kind, expires_at=None, controller_deadline=None):
        stage = key.rsplit('/', 1)[-1]
        if stage not in {'upload', 'stage', 'compile', 'execute'}:
            stage = 'transport'
        try:
            return self._timed(stage, lambda: self._operation_inner(key, arguments,
                kind=kind, expires_at=expires_at, controller_deadline=controller_deadline), key=key)
        except _PendingObservation as error:
            path = self.root / key / 'operation.json'
            diagnostic = getattr(error, 'transport_diagnostic', None)
            if diagnostic is not None and path.exists():
                record = json.loads(path.read_text())
                record['observation_failure'] = diagnostic
                _save(path, record)
            # CLI failure is diagnostic evidence, not proof of remote termination.
            raise

    def _operation_inner(self, key, arguments, *, kind, expires_at=None, controller_deadline=None):
        """Never submit again after an uncertain submission or observable handle."""
        path = self.root / key / 'operation.json'
        identity = dict(route=self.runtime_identity, workspace=str(self._workspace), arguments=list(arguments))
        record = json.loads(path.read_text()) if path.exists() else dict(identity=identity, state='new')
        if record['identity'] != identity:
            raise ProjectInfrastructureError('retained CPL operation identity changed')
        if 'result' in record:
            return record['result']
        if 'expires_at' not in record:
            if record['state'] != 'new':
                raise ProjectInfrastructureError('retained operation lacks frozen elapsed expiry; inspect its handle')
            record['expires_at'] = expires_at if expires_at is not None else self.wall_clock()+self._allowance()
            record['controller_deadline'] = controller_deadline if controller_deadline is not None else (
                self.clock()+record['expires_at']-self.wall_clock())
            record['controller_boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            if self._receipt_capable(path): record['requires_terminal_result'] = True
        terminal = ('completed', 'failed', 'cancelled')
        receipt = path.with_name('submission-receipt.json')
        self._restore_receipt(path, record, receipt, kind)
        if record.get('state') != 'new' and not record.get('handle'):
            self._quarantine_upload(path, record)
        if record.get('state') == 'quarantined':
            raise self._pending('CPL upload namespace quarantined; do not resubmit')
        if ('controller_deadline' in record and record.get('controller_boot_id') !=
                Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
            raise ProjectInfrastructureError('controller boot changed; retained work requires read-only reconciliation')
        work_expiry = record['expires_at'] - self.cleanup_reserve
        work_remaining = (record['controller_deadline']-self.cleanup_reserve-self.clock()
            if 'controller_deadline' in record else work_expiry-self.wall_clock())
        available = self.timeout if self.remaining_seconds is None else min(self.timeout, max(0, self.remaining_seconds()))
        deadline = self.clock() + max(0, min(available, work_remaining))

        def expired():
            return (self.clock() >= deadline or
                    self.remaining_seconds is not None and self.remaining_seconds() <= 0)

        def recover_expiry():
            return self._timed('cleanup', recover_expiry_inner)

        def recover_expiry_inner():
            # Execution stops at expiry; allow only bounded observation/cleanup.
            # A failed compiler may already have exited before the final poll.
            record.setdefault('cleanup_expires_at', self.wall_clock()+45)
            _save(path, record)
            cleanup_deadline = self.clock() + max(0, record['cleanup_expires_at']-self.wall_clock())
            def cleanup_call(arguments):
                seconds = cleanup_deadline-self.clock()
                if seconds <= 0:
                    raise self._pending('CPL expiry observation unavailable; retained handle '+str(record.get('handle')), 'deadline-exhausted')
                return self._call(arguments, seconds=seconds)
            if kind == 'job' and record.get('handle') and record['state'] not in terminal:
                observed = cleanup_call(('observe', record['handle']))
                if observed.get('handle') != record['handle']:
                    raise ProjectInfrastructureError('CPL observation returned a foreign handle')
                record['state'] = observed.get('state')
                _save(path, record)
                if record['state'] in ('completed', 'failed'):
                    return self._finish_operation(path, record, observed, kind)
                if record['state'] != 'cancelled' and not record.get('cancel_requested'):
                    cancelled = cleanup_call(('cancel', record['handle']))
                    if cancelled.get('handle') != record['handle']:
                        raise ProjectInfrastructureError('CPL cancellation returned a foreign handle')
                    record.update(state=cancelled.get('state'), cancel_requested=True)
                    _save(path, record)
                    if record['state'] in ('completed', 'failed'):
                        return self._finish_operation(path, record, cancelled, kind)
            elif record['state'] in ('completed', 'failed'):
                return self._finish_operation(path, record, dict(state=record['state'], handle=record.get('handle')), kind)
            if kind == 'job' and record.get('handle'):
                recovered = self._terminal_receipt(path, record, seconds=cleanup_deadline-self.clock())
                if recovered is not None: return recovered
            if record.get('requires_terminal_result'):
                record['remote_state'] = record['state']
                record['state'] = 'observation-unavailable'
            record['expired'] = True
            _save(path, record)
            raise self._pending('CPL elapsed allocation expired; retained handle '+str(record.get('handle')), 'deadline-exhausted')

        if expired():
            return recover_expiry()
        if not record.get('handle'):
            if record['state'] != 'new':
                raise self._pending('CPL submission outcome is ambiguous; do not resubmit')
            if deadline-self.clock() < self.submission_reserve:
                raise self._pending('CPL submission not started: insufficient elapsed allocation', 'deadline-exhausted')
            submitted_arguments = (*arguments, '--receipt-file', str(receipt.resolve()))
            if kind == 'transfer':
                capabilities = self._upload_capabilities(seconds=deadline-self.clock())
                submitted_arguments = (submitted_arguments + ('--no-wait',)
                    if capabilities['backend'] == 'remote-agent' else arguments)
            if deadline-self.clock() < self.submission_reserve:
                raise self._pending('CPL submission not started: insufficient elapsed allocation', 'deadline-exhausted')
            record['state'] = 'submitting'
            _save(path, record)
            try:
                value = self._call(submitted_arguments, seconds=max(.001, deadline-self.clock()))
            except _PendingObservation:
                self._restore_receipt(path, record, receipt, kind)
                if not record.get('handle'):
                    self._quarantine_upload(path, record)
                raise
            handle = value.get('handle')
            if kind == 'job' and handle is None:
                self._restore_receipt(path, record, receipt, kind)
                if not record.get('handle'):
                    raise self._pending('CPL job submission lacks a retained handle; do not resubmit')
                handle = record['handle']
                value = dict(handle=handle, state=record['state'])
            if handle is not None:
                if not isinstance(handle, str) or not handle.startswith(f'remote:{self.target}:{kind}:'):
                    raise ProjectInfrastructureError('CPL returned a foreign retained handle')
                record['handle'] = handle
            record['state'] = value.get('state')
            _save(path, record)
        else:
            value = dict(state=record['state'], handle=record['handle'])
        while value.get('state') not in terminal:
            if expired():
                return recover_expiry()
            if not record.get('handle') or self.clock() >= deadline:
                raise self._pending('CPL operation pending; retain and observe '+str(record.get('handle')))
            try:
                value = self._call(('observe', record['handle'], '--wait', '--timeout',
                                    str(max(1, min(30, int(deadline-self.clock()))))),
                                   seconds=max(.001, deadline-self.clock()))
            except _PendingObservation:
                if expired():
                    return recover_expiry()
                raise
            if value.get('handle') != record['handle']:
                raise ProjectInfrastructureError('CPL observation returned a foreign handle')
            record['state'] = value.get('state')
            _save(path, record)
        return self._finish_operation(path, record, value, kind)

    def _finish_operation(self, path, record, value, kind, *, collection_expires_at=None):
        if 'result' in record: return record['result']
        with preparation('collect terminal logs and authenticated result'):
            return self._timed('cleanup', lambda: self._collect_operation(
                path, record, value, kind, collection_expires_at=collection_expires_at))

    def _collect_operation(self, path, record, value, kind, *, collection_expires_at=None):
        # Result collection cannot dispatch work or extend its frozen expiry.
        # Checkpoint each stream so a disconnect never loses compiler feedback.
        if collection_expires_at is None:
            record.setdefault('cleanup_expires_at', self.wall_clock()+45)
            _save(path, record)
        deadline = self.clock() + max(0, (collection_expires_at if collection_expires_at is not None
            else record['cleanup_expires_at'])-self.wall_clock())
        def collect(arguments):
            seconds = deadline-self.clock()
            if seconds <= 0:
                raise self._pending('CPL terminal result collection pending; retained handle '+str(record.get('handle')))
            return self._call(arguments, seconds=seconds)
        if kind == 'transfer':
            if value.get('state') != 'completed':
                record['terminal_failure'] = value
                _save(path, record)
                raise ProjectInfrastructureError('CPL upload did not complete')
            result = value
        else:
            needs_receipt = self._receipt_capable(path) and value.get('exit') != 0
            if type(value.get('exit')) is not int or needs_receipt or value.get('state') == 'cancelled':
                recovered = self._terminal_receipt(path, record, seconds=max(0, deadline-self.clock()))
                if recovered is not None: return recovered
                if needs_receipt:
                    record.update(state='observation-unavailable', remote_observation=value)
                    _save(path, record)
                    raise self._pending('RSI command terminal receipt is pending')
            if value.get('state') == 'cancelled':
                if record.get('requires_terminal_result'):
                    record.update(state='observation-unavailable', remote_observation=value)
                    _save(path, record)
                raise self._pending('CPL elapsed allocation expired; retained cancelled handle '+str(record.get('handle')), 'deadline-exhausted')
            result = record.setdefault('terminal_result', dict(handle=record.get('handle')))
            if type(value.get('exit')) is int:
                result['returncode'] = value['exit']
                _save(path, record)
            for stream in ('stdout', 'stderr'):
                if stream in result:
                    continue
                logs = collect(('logs', record['handle'], '--stream', stream, '--tail', '1000000'))
                if logs.get('handle') != record['handle'] or not isinstance(logs.get('content'), str):
                    raise ProjectInfrastructureError('CPL logs returned a foreign or invalid result')
                result[stream] = logs['content']
                _save(path, record)
            if 'returncode' not in result:
                value = collect(('result', record['handle']))
                if value.get('handle') != record.get('handle'):
                    raise ProjectInfrastructureError('CPL result returned a foreign handle')
                if type(value.get('exit')) is not int:
                    record.update(state='observation-unavailable', remote_observation=value)
                    _save(path, record)
                    raise self._pending('CPL terminal result lacks its authenticated exit status')
                result['returncode'] = value['exit']
                _save(path, record)
        record['result'] = result
        record.setdefault('result_received_epoch', self.wall_clock())
        record.setdefault('result_received_monotonic', self.clock())
        record.setdefault('result_controller_boot_id', Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        _save(path, record)
        return result

    def recover_owned_operations(self, *, seconds=30, remaining_seconds=None):
        """Observe this binding's retained handles without submitting device work."""
        deadline = self.clock() + min(seconds, self.timeout)
        for path in Path(getattr(self, 'recovery_root', self.root)).rglob('operation.json'):
            record = json.loads(path.read_text())
            if ('result' in record or record.get('state') in ('new', 'quarantined')
                    or record.get('state') == 'cancelled' and not self._receipt_capable(path)
                    or record.get('state') == 'failed' and record.get('terminal_failure')):
                continue
            identity = record.get('identity', {})
            if identity.get('route') != self.runtime_identity or identity.get('workspace') != str(self._workspace):
                continue
            kind = 'transfer' if identity.get('arguments', [None])[0:1] == ['upload'] else 'job'
            self._restore_receipt(path, record, path.with_name('submission-receipt.json'), kind)
            handle = record.get('handle')
            if not handle:
                self._quarantine_upload(path, record)
                continue  # Unknown execution must remain blocked, never resubmitted.
            if not any(handle.startswith(f'remote:{self.target}:{kind}:') for kind in ('job', 'transfer')):
                raise ProjectInfrastructureError('retained CPL operation has a foreign handle')
            remaining = min(deadline-self.clock(), remaining_seconds() if remaining_seconds else float("inf"))
            if remaining <= 0:
                break
            observed = self._call(('observe', handle), seconds=remaining)
            if observed.get('handle') != handle:
                raise ProjectInfrastructureError('CPL observation returned a foreign handle')
            record['state'] = observed.get('state')
            _save(path, record)
            if kind == 'transfer' and record['state'] == 'failed':
                record['terminal_failure'] = observed
                _save(path, record)
                continue
            if record['state'] not in ('completed', 'failed') and kind == 'job':
                left = min(deadline-self.clock(), remaining_seconds() if remaining_seconds else float('inf'))
                try:
                    if self._terminal_receipt(path, record, seconds=max(0, left)) is not None: continue
                except _PendingObservation: pass
                if record.get('requires_terminal_result'):
                    record['remote_state'] = record['state']
                    record['state'] = 'observation-unavailable'
                    _save(path, record)
            if record['state'] in ('completed', 'failed'):
                # Explicit read-only recovery has its own bounded allocation; it
                # never extends the frozen execution deadline or submits work.
                left = min(deadline-self.clock(), remaining_seconds() if remaining_seconds else float('inf'))
                try:
                    self._finish_operation(path, record, observed, kind,
                                           collection_expires_at=self.wall_clock()+max(0, left))
                except _PendingObservation:
                    continue  # Terminal work cannot occupy a device; incomplete logs remain retained.

    def _upload_capabilities(self, *, seconds):
        if self._capabilities is None:
            value = self._call(('capabilities', self.target), seconds=seconds)
            if (value.get('target') != self.target or value.get('backend') not in ('ssh', 'remote-agent')
                    or value.get('state') != 'available' or not value.get('capabilities', {}).get('upload')):
                raise ProjectInfrastructureError('CPL target does not advertise a supported upload backend')
            self._capabilities = value
        return self._capabilities

    def _restore_receipt(self, path, record, receipt, kind):
        if record.get('handle') or not receipt.exists():
            return
        try:
            value = json.loads(receipt.read_text())
        except (ValueError, OSError) as error:
            raise ProjectInfrastructureError('invalid CPL submission receipt') from error
        expected_operation = 'upload' if kind == 'transfer' else 'run'
        handle = value.get('handle')
        if (not isinstance(handle, str) or not handle.startswith(f'remote:{self.target}:{kind}:')
                or value.get('target') != self.target or value.get('operation') != expected_operation
                or kind == 'job' and value.get('runtime') != self.runtime):
            raise ProjectInfrastructureError('CPL submission receipt belongs to a foreign operation')
        record.update(handle=handle, state=value.get('state'))
        _save(path, record)

    def _quarantine_upload(self, path, record):
        # Only immutable, candidate-specific archive uploads have no execution side effects.
        arguments = record['identity'].get('arguments', [])
        candidate = path.parent.parent.name
        if (record.get('handle') or len(candidate) != 64
                or any(c not in '0123456789abcdef' for c in candidate)
                or arguments[:2] != ['upload', self.target]
                or len(arguments) != 4
                or arguments[-1] != self._wire_path(self._workspace / (candidate+'.tar'))):
            return
        record.update(state='quarantined', quarantine=dict(kind='isolated-upload',
            destination=arguments[-1], workspace=str(self._workspace)))
        _save(path, record)

    def _wire_path(self, path):
        selected = PurePosixPath(path)
        return str(selected.relative_to(self.registry_workspace)) if self.registry_workspace is not None else str(selected)

    def remote_candidate_directory(self, plan):
        return self._workspace / 'candidates' / plan.execution_id

    def _readonly_operation(self, key, body, *, seconds, cwd=None, expires_at=None, cleanup=False):
        """Own read helpers separately; exhausted Actor work cannot prevent cleanup."""
        if seconds <= 0:
            raise self._pending('RSI read-only recovery allocation exhausted')
        deadline = self.clock()+seconds
        stopped = getattr(self.remaining_seconds, 'stop_requested', lambda: False)
        def remaining():
            return 0 if stopped() and not cleanup else max(0, deadline-self.clock())
        remaining.stop_requested = stopped
        helper = CplNativeCandidateBackend(cpl_remote=self._cpl, target=self.target,
            runtime=self.runtime, remote_workspace=str(self._workspace), physical_device=self._device,
            state_directory=self.root, profile=self.profile, timeout=max(.001, seconds),
            remaining_seconds=remaining, registry_workspace=self.registry_workspace,
            process_runner=self._run, clock=self.clock, wall_clock=self.wall_clock,
            submission_reserve_seconds=0, cleanup_reserve_seconds=0)
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        script = directory / 'read-command.sh'
        if script.exists() and script.read_text() != body:
            raise ProjectInfrastructureError('retained read-only command changed')
        if not script.exists(): script.write_text(body)
        arguments = ('run', self.target, '--runtime', self.runtime,
            '--cwd', self._wire_path(cwd or self._workspace), '--file', str(script.resolve()))
        with preparation('remote clock admission and terminal receipt recovery'):
            return helper._operation(key, arguments, kind='job', expires_at=expires_at)

    def _sample_remote_clock(self, key, *, expires_at, cwd):
        payload = json.loads((self.root/key/'command.json').read_text())
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if payload['controller_boot_id'] != boot:
            raise ProjectInfrastructureError('controller boot changed before remote command admission')
        local_deadline = payload['controller_deadline']
        seconds = max(0, local_deadline-self.clock())
        value = self._readonly_operation(key+'/clock', clock_source(), seconds=seconds,
            cwd=cwd)
        if value['returncode']:
            raise ProjectInfrastructureError('remote clock probe failed')
        try:
            sample = validate_clock(json.loads(value['stdout']))
            recorded = json.loads((self.root/key/'clock/operation.json').read_text())
            # Remaining time is measured when the clock sample was received, not
            # when it was requested. Probe queue, execution and collection count.
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            if recorded['result_controller_boot_id'] != boot:
                raise ValueError('controller boot changed before command admission')
            remaining = max(0, local_deadline-recorded['result_received_monotonic'])
            return dict(remote_boot_id=sample['boot_id'],
                        remote_deadline=sample['monotonic']+remaining)
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectInfrastructureError('remote clock admission is invalid') from error

    @staticmethod
    def _receipt_capable(path):
        command = path.with_name('command.json')
        return command.exists() and json.loads(command.read_text()).get('launcher') == LAUNCHER

    def _terminal_receipt(self, path, record, *, seconds):
        command = path.with_name('command.json')
        if not command.exists(): return None
        payload = json.loads(command.read_text())
        if payload.get('launcher') != LAUNCHER: return None
        index = record.get('receipt_reads', 0)
        key = str(path.parent.relative_to(self.root))+'/receipt-read-'+str(index)
        try:
            value = self._readonly_operation(key, receipt_reader_source(payload['result_directory']),
                seconds=seconds, cleanup=True)
        except _PendingObservation:
            helper_path = self.root/key/'operation.json'
            helper = json.loads(helper_path.read_text()) if helper_path.exists() else {}
            if helper.get('state') == 'cancelled':
                record['receipt_reads'] = index+1
                _save(path, record)
            raise
        record['receipt_reads'] = index+1
        _save(path, record)
        try:
            if value['returncode']:
                raise ValueError('terminal receipt reader failed')
            envelope = json.loads(value['stdout'])
            result = validate_receipt(envelope, payload)
        except (KeyError, TypeError, ValueError) as error:
            record.update(state='observation-unavailable', receipt_error=str(error))
            _save(path, record)
            raise ProjectInfrastructureError('RSI terminal receipt is invalid') from error
        if result is not None:
            result['handle'] = record['handle']
            record.update(result=result, state='completed' if result['returncode'] == 0 else 'failed',
                command_receipt=dict(path=payload['result_directory']+'/terminal.json',
                    payload_sha256=payload['payload_sha256'], binding=payload['binding'],
                    termination=envelope['receipt']['termination']), result_received_epoch=self.wall_clock())
            _save(path, record)
        return result

    def run_remote(self, argv, *, key, cwd=None):
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        script, command = directory/'command.sh', directory/'command.json'
        if command.exists():
            payload = json.loads(command.read_text())
        else:
            allowance = self._allowance()
            allocation = self.wall_clock()+allowance
            local_allocation = self.clock()+allowance
            nonce = uuid.uuid4().hex
            payload = dict(argv=[str(arg) for arg in argv],
                expires_at=allocation-self.cleanup_reserve, allocation_expires_at=allocation,
                controller_deadline=local_allocation-self.cleanup_reserve,
                allocation_controller_deadline=local_allocation,
                controller_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                launcher=LAUNCHER, binding=dict(route=self.runtime_identity,
                    workspace=str(self._workspace), argv=[str(arg) for arg in argv], operation_nonce=nonce),
                result_directory=str(self._workspace/'.rsi-results'/nonce))
            # Freeze the allocation before a potentially queued clock probe.
            _save(command, payload)
        if payload['argv'] != [str(arg) for arg in argv]:
            raise ProjectInfrastructureError('retained CPL command changed')
        if payload.get('launcher') != LAUNCHER:
            operation = directory/'operation.json'
            retained = json.loads(operation.read_text()) if operation.exists() else {}
            if not retained.get('handle') and 'result' not in retained and not (directory/'submission-receipt.json').exists():
                raise ProjectInfrastructureError('unmapped legacy command has no authenticated handle; start a fresh operation')
        if payload.get('launcher') == LAUNCHER and 'remote_deadline' not in payload:
            if (directory/'operation.json').exists():
                raise ProjectInfrastructureError('submitted command has no frozen remote deadline')
            payload.update(self._sample_remote_clock(key, expires_at=payload['expires_at'], cwd=cwd))
            payload['payload_sha256'] = payload_digest(payload)
            _save(command, payload)
        if payload.get('launcher') == LAUNCHER:
            body = {name: value for name, value in payload.items() if name != 'payload_sha256'}
            if (payload_digest(body) != payload.get('payload_sha256')
                    or payload['binding']['route'] != self.runtime_identity
                    or payload['binding']['workspace'] != str(self._workspace)
                    or payload['binding']['argv'] != payload['argv']):
                raise ProjectInfrastructureError('retained command payload identity changed')
        if not script.exists():
            if payload.get('launcher') != LAUNCHER:
                raise ProjectInfrastructureError('retained legacy command script is unavailable')
            script.write_text(launcher_source(payload))
        arguments = ('run', self.target, '--runtime', self.runtime,
                     '--cwd', self._wire_path(cwd or self._workspace), '--file', str(script.resolve()))
        value = self._operation(key, arguments, kind='job',
            expires_at=payload.get('allocation_expires_at', payload['expires_at']),
            controller_deadline=payload.get('allocation_controller_deadline'))
        if (value['returncode'] == 124 or
                value['returncode'] == 137 and payload.get('launcher') not in ('process-group-v1', LAUNCHER)
                and self.wall_clock() >= payload['expires_at']):
            raise self._pending('CPL command execution allocation expired; retained handle '+value['handle'], 'deadline-exhausted')
        result = subprocess.CompletedProcess(argv, value['returncode'], value['stdout'], value['stderr'])
        result.job_handle = value['handle']
        return result

    def _require_plan(self, plan):
        if not isinstance(plan, NativeCandidatePlan) or plan.execution_profile != self.profile:
            raise ValueError('CPL native transport requires its selected native execution profile')

    def compile(self, plan, local_directory):
        self._require_plan(plan)
        key = plan.execution_id
        _save(self.root / key / 'candidate.json', dict(project_id=plan.project_id))
        with preparation('candidate bundle, upload and staging'):
            bundle = self.build_bundle(plan, self.root / 'bundles' / (key+'.tar'))
            archive = self._workspace / (key+'.tar')
            self._operation(key+'/upload', ('upload', self.target, str(bundle.path.resolve()), self._wire_path(archive)), kind='transfer')
            if current_budget() is not None:
                staging = _COMPILE_SCRIPT.split("printf 'A3REMOTE_STAGE=compile", 1)[0]
                staged = self.run_remote(('bash', '-c', staging, 'rsi-native', str(archive),
                    str(self.remote_candidate_directory(plan)), bundle.sha256, key, plan.source_fingerprint), key=key+'/stage')
                if staged.returncode:
                    raise ProjectInfrastructureError(staged.stderr or staged.stdout or 'candidate staging failed')
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
            if 'result' in retained:
                return retained['result']
        directory = backend.remote_candidate_directory(plan)
        argv = ['python', str(directory/'native_measurement_driver.py'), mode, '--candidate-dir', str(directory),
                '--kernel', plan.kernel_name, '--execution-id', plan.execution_id,
                '--source-fingerprint', plan.source_fingerprint, '--library-sha256', self._compilation.library_sha256,
                '--warm-up', str(warm_up), '--launch-count', str(launch_count)]
        if metric:
            argv += ['--metric', 'ArithmeticUtilization' if metric == 'Basic' else metric]
        replay_key = canonical_digest(dict(request_id=request_id, replay_id=replay))
        if not record.exists():
            # Older versions wrote replay receipts only after success. Reconstruct
            # only the exact retained key/command; an unbound different pending
            # operation cannot safely be assigned to another replay.
            bound_keys = {json.loads(path.read_text()).get('operation_key')
                          for path in self.root.glob('*/measurement.json')}
            for operation in (backend.root / 'measurements').glob('*/operation.json'):
                old = json.loads(operation.read_text())
                key = operation.parent.name
                if key in bound_keys or 'result' in old:
                    continue
                command = operation.parent / 'command.json'
                if (key != replay_key or not command.exists()
                        or json.loads(command.read_text()).get('argv') != argv):
                    raise ProjectInfrastructureError('unbound retained native measurement requires recovery before a new replay')
            _save(record, dict(request_id=request_id, operation_key=replay_key))
        verification = self._verify()
        result = backend.run_remote(argv, key='measurements/'+replay_key, cwd=directory)
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
        _save(record, dict(request_id=request_id, operation_key=replay_key, result=measured, evidence=dict(profile=backend.profile,
            handle=result.job_handle, status='completed', physical_device=backend._device, logical_device=0,
            remote_report=meta.get('remote_report'))))
        return measured
