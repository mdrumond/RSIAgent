"""Retained CPL transport for trusted, frozen command inputs.

Configuration is host injection, never an actor wire command. Construction and
factory binding perform no dispatch. Hardware admission belongs to the caller.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import threading
import time

from rsi.python_execution import CommandResult, PythonExecution, atomic_json, digest
from rsi.runtime import WorkflowError


@dataclass(frozen=True)
class CplConfig:
    target: str
    runtime: str
    workspace: str
    physical_device: int
    installed_client: Path
    profile_identity: str
    command_cwd: str
    trusted_resources: tuple[Path, ...] = ()
    remote_python: str = 'python3'
    client_timeout_seconds: float = 45.0
    remote_agent_transfers: bool = False
    transfer_relative_paths: bool = False

    def __post_init__(self):
        if (not all(isinstance(x, str) and x.strip() for x in
                    (self.target, self.runtime, self.workspace, self.profile_identity,
                     self.remote_python, self.command_cwd))
                or not PurePosixPath(self.workspace).is_absolute()
                or self.command_cwd in ('.', '/', self.workspace)
                or '..' in PurePosixPath(self.command_cwd).parts
                or type(self.physical_device) is not int or self.physical_device < 0):
            raise ValueError('invalid frozen CPL profile')
        if (not math.isfinite(self.client_timeout_seconds)
                or not 0 < self.client_timeout_seconds <= 60):
            raise ValueError('CPL client timeout must be positive and at most 60 seconds')
        object.__setattr__(self, 'installed_client', Path(self.installed_client).resolve())
        object.__setattr__(self, 'trusted_resources', tuple(Path(p).resolve() for p in self.trusted_resources))


# Runs inside the selected registry runtime. Only this live Popen owns the group.
_RUNNER = r'''
import hashlib, json, os, pathlib, signal, subprocess, tarfile, time
root = pathlib.Path(__file__).parent
started = time.monotonic()
spec = json.loads((root / 'spec.json').read_text())
for name, expected in spec['files'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('staged input identity mismatch')
limit = float(__import__('sys').argv[1]) if len(__import__('sys').argv) > 1 else None
if limit is not None and len(__import__('sys').argv) > 2:
    # Shell-local extraction interval, rounded up conservatively. No host epochs.
    limit -= int(__import__('sys').argv[2]) + 1
env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(spec['physical_device']),
           PYTHONPATH=str(root / 'project'), PYTHONPYCACHEPREFIX=str(root / 'cache'),
           RSI_EXPECTED_SOURCE_HASH=spec['identity']['source_hash'],
           CATLASS_DSL_CACHE_DIR=str(root / 'catlass-cache'))
def persist(name, value):
    temporary = root / (name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(root / name)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
producer = dict(version=2, identity=spec['identity'], runner_pid=os.getpid(),
                child_pid=None, child_pgid=None, reaped=False, closed=False)
persist('producer.json', producer)
interrupted = None
def terminate(signum, frame):
    global interrupted
    interrupted = signum
signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
with (root / 'stdout').open('wb') as out, (root / 'stderr').open('wb') as err:
    process = None
    timed_out = False
    returncode = 127
    try:
        if limit is not None:
            limit = max(0, limit - (time.monotonic() - started))
            if limit <= 0:
                raise subprocess.TimeoutExpired(spec['argv'], 0)
        deadline = None if limit is None else time.monotonic() + limit
        if interrupted is None:
            process = subprocess.Popen(spec['argv'], cwd=root / 'project', env=env,
                                       stdout=out, stderr=err, start_new_session=True)
            producer.update(child_pid=process.pid, child_pgid=process.pid)
            def child_exited():
                # Keep the leader waitable until the last group signal, so its
                # PID/PGID cannot be recycled into an unrelated process group.
                return os.waitid(os.P_PID, process.pid,
                                 os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
            persist('producer.json', producer)
            while not child_exited() and interrupted is None:
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(spec['argv'], limit)
                time.sleep(0.02)
    except subprocess.TimeoutExpired:
        timed_out = True
    except OSError as exc:
        err.write(str(exc).encode())
    finally:
        # Reap descendants even when the group leader exited normally.
        if process is not None:
            if interrupted is not None:
                try:
                    os.killpg(process.pid, interrupted)
                except ProcessLookupError:
                    pass
                grace_end = time.monotonic() + 0.2
                while not child_exited() and time.monotonic() < grace_end:
                    time.sleep(0.01)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = process.wait()
            if interrupted is not None and returncode == 0:
                returncode = 128 + interrupted
            producer['reaped'] = True
            # A terminal wrapper or reaped leader alone cannot close descendants.
            end = time.monotonic() + 1
            while True:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    producer['closed'] = True
                    break
                if time.monotonic() >= end:
                    break
                time.sleep(0.02)
        else:
            producer.update(reaped=True, closed=True)
            if interrupted is not None:
                returncode = 128 + interrupted
        producer.update(returncode=returncode, signal=interrupted,
                        elapsed=time.monotonic()-started)
        persist('producer.json', producer)
result = dict(identity=spec['identity'], returncode=returncode,
              timed_out=timed_out, elapsed=time.monotonic()-started,
              producer=producer)
persist('result.json', result)
with tarfile.open(root / 'results.tar.tmp', 'w') as archive:
    for name in ('result.json', 'producer.json', 'stdout', 'stderr', 'project', 'verifier'):
        archive.add(root / name, arcname=name)
    if (root / 'profile').exists():
        archive.add(root / 'profile', arcname='profile')
(root / 'results.tar.tmp').replace(root / 'results.tar')
'''


class CplExecutor:
    """One operation directory and durable receipt per immutable command identity.

    profile is the frozen identity string used by executor_factory. Drivers are
    explicit trusted_resources; command arguments are otherwise left unchanged.
    reconcile() never uploads or launches a workload, including after ambiguity.
    """

    def __init__(self, profile: str, config: CplConfig):
        if profile != config.profile_identity:
            raise ValueError('frozen executor profile does not match factory')
        self.profile, self.config = profile, config

    @property
    def metadata(self):
        c = self.config
        return dict(executor_identity=self.profile, target=c.target, runtime=c.runtime,
                    physical_device=c.physical_device, logical_device=0)

    def _limit(self, remaining):
        value = remaining()
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise WorkflowError('CPL elapsed budget exhausted; reconcile retained work')
        return value

    def _cli(self, args, remaining, response_file=None, *, result_path=None):
        # Read the installed policy afresh before *each* operation, not a copy.
        skill = self.config.installed_client.parent.parent / 'SKILL.md'
        skill.read_text()
        limit = self._limit(remaining)
        ceiling = self.config.client_timeout_seconds
        limit = ceiling if limit is None else min(ceiling, limit)
        try:
            result = subprocess.run([str(self.config.installed_client), '--json', *map(str, args)],
                                    capture_output=True, timeout=limit)
            output, stderr, returncode = result.stdout, getattr(result, 'stderr', b''), result.returncode
        except subprocess.TimeoutExpired as exc:
            output, stderr, returncode = exc.stdout or b'', exc.stderr or b'', 124
        destination = result_path if result_path is not None else response_file
        if destination is not None:
            destination = Path(destination)
            # Preserve native CLI diagnostics before parsing or raising. A failed
            # response is evidence, never a receipt proving backend submission.
            Path(str(destination) + '.stdout').write_bytes(output)
            Path(str(destination) + '.stderr').write_bytes(stderr)
            atomic_json(Path(str(destination) + '.client.json'),
                        dict(returncode=returncode, result_path=destination.name))
        try:
            value = json.loads(output)
        except (ValueError, UnicodeError) as exc:
            raise WorkflowError('invalid CPL response; reconcile retained work') from exc
        if not isinstance(value, dict):
            raise WorkflowError('CPL response must be an object')
        if destination is not None:
            atomic_json(destination, value)
        if returncode and not value.get('handle'):
            raise WorkflowError('CPL client failed; retain receipt and reconcile')
        return value

    def _transfer_path(self, path):
        if self.config.transfer_relative_paths:
            return PurePosixPath(path).relative_to(self.config.workspace).as_posix()
        return path

    def _identity(self, request):
        return dict(operation_id=request.operation_id, command=request.command,
                    source_hash=request.source_hash, verifier_hash=request.verifier_hash,
                    argv=list(request.argv), profile=self.profile, **self.metadata)

    def _directory(self, request):
        return request.project.parent / 'cpl-receipts' / digest(request.operation_id.encode())

    def _phase(self, observe, phase, request, **extra):
        observe(phase, **self.metadata, operation_id=request.operation_id,
                source_hash=request.source_hash, command=request.command, **extra)

    def _handle(self, value, kind='job'):
        handle = value.get('handle')
        if (value.get('target') != self.config.target or not isinstance(handle, str)
                or not handle.startswith(f'remote:{self.config.target}:{kind}:')):
            raise WorkflowError('CPL receipt target/handle mismatch')
        return handle

    def _observation_seconds(self, remaining):
        limit = self._limit(remaining)
        return max(0, min(20, int(self.config.client_timeout_seconds / 2),
                          int(limit / 2) if limit is not None else 20))

    def _wait(self, handle, request, observe, remaining):
        while True:
            phase = 'transfer' if handle.split(':')[2] == 'transfer' else 'command'
            self._phase(observe, phase, request, handle=handle)
            seconds = self._observation_seconds(remaining)
            args = ['observe', handle]
            if seconds >= 1:
                args += ['--wait', '--timeout', seconds]
            try:
                value = self._cli(args, remaining,
                                  self._directory(request) / ('observe-' + digest(handle.encode()) + '.json'))
            finally:
                self._phase(observe, 'command', request, handle=handle)
            if self._handle(value, handle.split(':')[2]) != handle:
                raise WorkflowError('CPL observation changed handle')
            if value.get('state') in ('completed', 'failed', 'cancelled'):
                return value
            if value.get('state') not in ('queued', 'dispatching', 'running', 'reconnecting',
                                           'observation-unavailable'):
                raise WorkflowError('CPL handle cannot be resolved; never redispatch')

    def execute(self, request, *, observe, remaining):
        self._phase(observe, 'command', request)
        self._limit(remaining)
        directory = self._directory(request)
        # Even a directory without a receipt can represent an ambiguous submission.
        directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError as exc:
            raise WorkflowError('CPL operation already reserved; use reconcile') from exc
        identity = self._identity(request)
        remote = str(PurePosixPath(self.config.workspace) / ('rsi-' + digest(str(directory.resolve()).encode())))
        state = dict(identity=identity, remote=remote, status='packaging',
                     producer_version=2, lease_retained=True)
        atomic_json(directory / 'transport.json', state)
        bundle = directory / 'bundle'
        bundle.mkdir()
        backend = PythonExecution(request.project.parent, executor_factory=lambda identity: self)
        if backend.project_hash() != request.source_hash:
            raise WorkflowError('frozen source identity changed before packaging')
        self._copy(request.project, bundle / 'project', remaining)
        self._copy(request.verifier, bundle / 'verifier', remaining)
        if digest((bundle / 'verifier').read_bytes()) != request.verifier_hash:
            raise WorkflowError('frozen verifier identity changed')
        mappings = [(str(request.project), remote + '/project'),
                    (str(request.verifier), remote + '/verifier')]
        for index, resource in enumerate(self.config.trusted_resources):
            destination = bundle / 'trusted' / str(index) / resource.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._copy(resource, destination, remaining)
            mappings.append((str(resource), remote + '/' + destination.relative_to(bundle).as_posix()))
        if request.profile_path is not None:
            mappings.append((str(request.profile_path), remote + '/profile'))
        def mapped(arg):
            # Support --option=/frozen/path as well as positional paths.
            prefix, separator, value = arg.partition('=')
            candidate = value if separator else arg
            for original, replacement in sorted(mappings, key=lambda row: -len(row[0])):
                if candidate == original or candidate.startswith(original + '/'):
                    return (prefix + '=' if separator else '') + replacement + candidate[len(original):]
            return arg
        if backend.project_hash() != request.source_hash:
            raise WorkflowError('source changed while packaging')
        files = {}
        for path in bundle.rglob('*'):
            self._limit(remaining)
            if path.is_file():
                files[path.relative_to(bundle).as_posix()] = digest(path.read_bytes())
        spec = dict(identity=identity, argv=[mapped(arg) for arg in request.argv], files=files,
                    remaining=self._limit(remaining),
                    physical_device=self.config.physical_device)
        (bundle / 'runner.py').write_text(_RUNNER)
        atomic_json(bundle / 'spec.json', spec)
        with tarfile.open(directory / 'inputs.tar', 'w') as archive:
            for path in bundle.iterdir():
                self._limit(remaining)
                archive.add(path, arcname=path.name)
        self._phase(observe, 'transfer', request)
        upload_args = ['upload', self.config.target, directory / 'inputs.tar',
                       self._transfer_path(remote + '.tar')]
        if self.config.remote_agent_transfers:
            upload_args += ['--receipt-file', directory / 'upload-submission.json', '--no-wait']
        try:
            upload = self._cli(upload_args, remaining, directory / 'upload.json')
        finally:
            self._phase(observe, 'command', request)
            upload_receipt = directory / 'upload-submission.json'
            if upload_receipt.exists():
                upload_handle = self._handle(json.loads(upload_receipt.read_text()), 'transfer')
                self._phase(observe, 'command', request, handle=upload_handle)
        if upload.get('handle'):
            upload_handle = self._handle(upload, 'transfer')
            terminal = self._wait(upload_handle, request, observe, remaining)
            if terminal.get('state') != 'completed':
                raise WorkflowError('CPL input transfer did not complete')
        # Exclusive mkdir prevents accidental reuse of another operation's files.
        script = directory / 'run.sh'
        limit = self._limit(remaining)
        invocation = [self.config.remote_python, remote + '/runner.py']
        if limit is not None:
            invocation.append(str(limit))
        script.write_text('#!/bin/bash\nset -eu\nSECONDS=0\n' +
                          f'mkdir -- {shlex.quote(remote)}\n' +
                          f'tar -xf {shlex.quote(remote + ".tar")} -C {shlex.quote(remote)}\n' +
                          shlex.join(invocation) + (' \"$SECONDS\"' if limit is not None else '') + '\n')
        return self._supervise(request, directory, state, script, observe, remaining)

    def _supervise(self, request, directory, state, script, observe, remaining):
        # Supervise controller elapsed allocation through submission, queueing and
        # producer setup. Producer durations never settle the controller ledger.
        self._limit(remaining)
        finished, expired = threading.Event(), threading.Event()
        def cancel_at_expiry():
            # remaining() is the authoritative controller allocation: transfer
            # exclusions can extend the wall envelope, stop closes it immediately.
            while not finished.wait(0.02):
                allowance = remaining()
                if allowance is not None and allowance <= 0:
                    break
            else:
                return
            expired.set()
            receipt = directory / 'submission.json'
            # The installed CLI writes this before waiting. No receipt is unknown,
            # never permission to repeat submission or invent a cancellation ID.
            if not receipt.exists():
                return
            cleanup_end = time.monotonic() + 15
            cleanup = lambda: max(0, cleanup_end - time.monotonic())
            try:
                handle = self._handle(json.loads(receipt.read_text()))
                atomic_json(directory / 'deadline-intent.json',
                            dict(handle=handle, cleanup_only=True, status='cancellation-pending'))
                self._cli(['cancel', handle], cleanup, directory / 'deadline-cancel.json')
            except Exception as exc:
                atomic_json(directory / 'deadline-unknown.json', dict(error=type(exc).__name__))
        watcher = threading.Thread(target=cancel_at_expiry, daemon=True)
        watcher.start()
        try:
            return self._submit(request, directory, state, script, observe, remaining)
        finally:
            allowance = remaining()
            if allowance is not None and allowance <= 0:
                expired.set()
            finished.set()
            watcher.join(timeout=16)
            if expired.is_set() and (directory / 'submission.json').exists():
                cleanup_end = time.monotonic() + 15
                cleanup = lambda: max(0, cleanup_end - time.monotonic())
                handle = self._handle(json.loads((directory / 'submission.json').read_text()))
                state.update(handle=handle, status='deadline-observation-pending',
                             cleanup_only=True, lease_retained=True)
                atomic_json(directory / 'transport.json', state)
                if not (directory / 'deadline-cancel.json').exists():
                    self._cli(['cancel', handle], cleanup, directory / 'deadline-cancel.json')
                self._phase(observe, 'command', request, handle=handle, deadline_expired=True)
                terminal = self._wait(handle, request, observe, cleanup)
                state.update(handle=handle, terminal=terminal, status='deadline-producer-unknown',
                             deadline_clock='controller_elapsed_remaining', cleanup_only=True,
                             lease_retained=True)
                atomic_json(directory / 'transport.json', state)
                # Cleanup allowance retrieves proof only; it grants no work time.
                self.reconcile(request, observe=observe, remaining=cleanup)
                state = json.loads((directory / 'transport.json').read_text())
                state.update(status='deadline-terminal', cleanup_only=True)
                atomic_json(directory / 'transport.json', state)

    def _submit(self, request, directory, state, script, observe, remaining):
        state['status'] = 'dispatching'
        atomic_json(directory / 'transport.json', state)
        try:
            self._cli(['run', self.config.target, '--runtime', self.config.runtime,
                       '--cwd', self.config.command_cwd, '--npu', self.config.physical_device,
                       '--file', script, '--receipt-file', directory / 'submission.json'], remaining, directory / 'submission-response.json')
        finally:
            receipt = directory / 'submission.json'
            response = directory / 'submission-response.json'
            if not receipt.exists() and response.exists():
                value = json.loads(response.read_text())
                if value.get('handle'):
                    self._handle(value)
                    atomic_json(receipt, value)
            if receipt.exists():
                handle = self._handle(json.loads(receipt.read_text()))
                state.update(handle=handle, status='submitted')
                atomic_json(directory / 'transport.json', state)
                self._phase(observe, 'command', request, handle=handle)
        return self.reconcile(request, observe=observe, remaining=remaining)

    def _copy(self, source, destination, remaining):
        self._limit(remaining)
        if source.is_symlink():
            raise WorkflowError('symlinks are not frozen execution inputs')
        if source.is_dir():
            destination.mkdir()
            for child in source.iterdir():
                self._copy(child, destination / child.name, remaining)
        else:
            shutil.copy2(source, destination)
        self._limit(remaining)

    def reconcile(self, request, *, observe, remaining, recovery_remaining=None):
        """Observe the original receipt and retrieve results; never submit work."""
        # Recovery allowance only funds observation/retrieval, never dispatch.
        if recovery_remaining is not None:
            remaining = recovery_remaining
        self._phase(observe, 'command', request)
        directory = self._directory(request)
        state = json.loads((directory / 'transport.json').read_text())
        if state['identity'] != self._identity(request):
            raise WorkflowError('CPL operation identity mismatch')
        # Legacy outputs are readable only when completion was already established.
        # Observing a newly terminal legacy wrapper cannot establish child closure.
        legacy_complete = (state.get('producer_version') is None
                           and (state.get('status') == 'complete'
                                or state.get('legacy_completion_known') is True))
        if legacy_complete:
            state['legacy_completion_known'] = True
        receipt = json.loads((directory / 'submission.json').read_text())
        handle = self._handle(receipt)
        self._phase(observe, 'command', request, handle=handle)
        terminal = self._wait(handle, request, observe, remaining)
        state.update(handle=handle, terminal=terminal, status='producer-unknown',
                     lease_retained=True)
        atomic_json(directory / 'transport.json', state)
        output = directory / 'results.tar'
        transfer_path = directory / 'download.json'
        transfer = json.loads(transfer_path.read_text()) if transfer_path.exists() else {}
        if transfer.get('handle'):
            transfer_handle = self._handle(transfer, 'transfer')
            self._wait(transfer_handle, request, observe, remaining)
            self._phase(observe, 'transfer', request, handle=transfer_handle)
            try:
                self._cli(['fetch', transfer_handle, output], remaining, directory / 'fetch.json')
            finally:
                self._phase(observe, 'command', request, handle=handle)
        elif not output.exists():
            if (directory / 'download-started.json').exists():
                raise WorkflowError('download submission unknown; reconcile transfer receipt without replay')
            atomic_json(directory / 'download-started.json', dict(handle=handle))
            self._phase(observe, 'transfer', request, handle=handle)
            try:
                value = self._cli(['download', self.config.target,
                                   self._transfer_path(state['remote'] + '/results.tar'), output,
                                   '--timeout', self._observation_seconds(remaining)],
                                  remaining, transfer_path)
                if value.get('handle'):
                    atomic_json(transfer_path, value)
            finally:
                self._phase(observe, 'command', request, handle=handle)
            if value.get('handle') and value.get('state') != 'completed':
                return self.reconcile(request, observe=observe, remaining=remaining)
        extracted = directory / 'retrieved'
        if extracted.exists():
            shutil.rmtree(extracted)
        extracted.mkdir()
        with tarfile.open(output) as archive:
            for member in archive.getmembers():
                self._limit(remaining)
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                    raise WorkflowError('invalid CPL output archive member')
            archive.extractall(extracted, filter='data')
        result = json.loads((extracted / 'result.json').read_text())
        if (result.get('identity') != state['identity']
                or digest((extracted / 'verifier').read_bytes()) != request.verifier_hash
                or type(result.get('returncode')) is not int
                or type(result.get('timed_out')) is not bool):
            raise WorkflowError('CPL terminal artifact identity mismatch')
        producer = result.get('producer')
        if not self._producer_closed(producer, state['identity']):
            if not (legacy_complete and producer is None):
                raise WorkflowError('CPL producer closure unknown; retain lease and reconcile')
        else:
            state['producer'] = producer
        state.update(lease_retained=False, status='producer-closed')
        atomic_json(directory / 'transport.json', state)
        self._limit(remaining)
        # Source mutations are returned for PythonExecution's normal classification.
        self._copy_back(extracted / 'project', request.project, remaining)
        if request.profile_path is not None and (extracted / 'profile').is_file():
            self._copy(extracted / 'profile', request.profile_path, remaining)
        state['status'] = 'complete'
        atomic_json(directory / 'transport.json', state)
        return CommandResult(result['returncode'], (extracted / 'stdout').read_bytes(),
                             (extracted / 'stderr').read_bytes(), result['timed_out'])

    @staticmethod
    def _producer_closed(producer, identity):
        if not isinstance(producer, dict):
            return False
        child = producer.get('child_pid')
        return (producer.get('version') == 2 and producer.get('identity') == identity
                and type(producer.get('runner_pid')) is int and producer['runner_pid'] > 0
                and producer.get('closed') is True and producer.get('reaped') is True
                and ((child is None and producer.get('child_pgid') is None)
                     or (type(child) is int and child > 0
                         and producer.get('child_pgid') == child)))

    def _copy_back(self, source, destination, remaining):
        for path in destination.iterdir():
            self._limit(remaining)
            other = source / path.name
            if not other.exists() or path.is_dir() != other.is_dir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        for path in source.iterdir():
            self._limit(remaining)
            other = destination / path.name
            if path.is_dir():
                other.mkdir(exist_ok=True)
                self._copy_back(path, other, remaining)
            else:
                shutil.copy2(path, other)
