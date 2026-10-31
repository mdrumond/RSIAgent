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
spec = json.loads((root / 'spec.json').read_text())
for name, expected in spec['files'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('staged input identity mismatch')
started = time.monotonic()
limit = float(__import__('sys').argv[1]) if len(__import__('sys').argv) > 1 else None
env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(spec['physical_device']),
           PYTHONPATH=str(root / 'project'), PYTHONPYCACHEPREFIX=str(root / 'cache'),
           RSI_EXPECTED_SOURCE_HASH=spec['identity']['source_hash'],
           CATLASS_DSL_CACHE_DIR=str(root / 'catlass-cache'))
with (root / 'stdout').open('wb') as out, (root / 'stderr').open('wb') as err:
    process = None
    timed_out = False
    returncode = 127
    try:
        process = subprocess.Popen(spec['argv'], cwd=root / 'project', env=env,
                                   stdout=out, stderr=err, start_new_session=True)
        process.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out = True
    except OSError as exc:
        err.write(str(exc).encode())
    finally:
        # Reap descendants even when the group leader exited normally.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = process.wait()
result = dict(identity=spec['identity'], returncode=returncode,
              timed_out=timed_out, elapsed=time.monotonic()-started)
(root / 'result.json').write_text(json.dumps(result))
with tarfile.open(root / 'results.tar', 'w') as archive:
    for name in ('result.json', 'stdout', 'stderr', 'project', 'verifier'):
        archive.add(root / name, arcname=name)
    if (root / 'profile').exists():
        archive.add(root / 'profile', arcname='profile')
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

    def _cli(self, args, remaining, response_file=None):
        # Read the installed policy afresh before *each* operation, not a copy.
        skill = self.config.installed_client.parent.parent / 'SKILL.md'
        skill.read_text()
        limit = self._limit(remaining)
        ceiling = self.config.client_timeout_seconds
        limit = ceiling if limit is None else min(ceiling, limit)
        try:
            result = subprocess.run([str(self.config.installed_client), '--json', *map(str, args)],
                                    capture_output=True, timeout=limit)
            output, returncode = result.stdout, result.returncode
        except subprocess.TimeoutExpired as exc:
            output, returncode = exc.stdout or b'', 124
        try:
            value = json.loads(output)
        except (ValueError, UnicodeError) as exc:
            raise WorkflowError('invalid CPL response; reconcile retained work') from exc
        if not isinstance(value, dict):
            raise WorkflowError('CPL response must be an object')
        if response_file is not None:
            atomic_json(response_file, value)
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

    def _wait(self, handle, request, observe, remaining):
        while True:
            self._phase(observe, 'command', request, handle=handle)
            limit = self._limit(remaining)
            seconds = min(30, int(self.config.client_timeout_seconds / 2),
                          int(limit / 2) if limit is not None else 30)
            args = ['observe', handle]
            if seconds >= 1:
                args += ['--wait', '--timeout', seconds]
            value = self._cli(args, remaining,
                              self._directory(request) / ('observe-' + digest(handle.encode()) + '.json'))
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
        state = dict(identity=identity, remote=remote, status='packaging')
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
        script.write_text('#!/bin/bash\nset -eu\n' +
                          f'mkdir -- {shlex.quote(remote)}\n' +
                          f'tar -xf {shlex.quote(remote + ".tar")} -C {shlex.quote(remote)}\n' +
                          shlex.join(invocation) + '\n')
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
        receipt = json.loads((directory / 'submission.json').read_text())
        handle = self._handle(receipt)
        self._phase(observe, 'command', request, handle=handle)
        terminal = self._wait(handle, request, observe, remaining)
        state.update(handle=handle, terminal=terminal, status='terminal')
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
                                   self._transfer_path(state['remote'] + '/results.tar'), output],
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
        self._limit(remaining)
        # Source mutations are returned for PythonExecution's normal classification.
        self._copy_back(extracted / 'project', request.project, remaining)
        if request.profile_path is not None and (extracted / 'profile').is_file():
            self._copy(extracted / 'profile', request.profile_path, remaining)
        state['status'] = 'complete'
        atomic_json(directory / 'transport.json', state)
        return CommandResult(result['returncode'], (extracted / 'stdout').read_bytes(),
                             (extracted / 'stderr').read_bytes(), result['timed_out'])

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
