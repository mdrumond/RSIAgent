"""Portable evidence backend. Host calls only; this is not a process sandbox.

One writer owns a task. The actor edits only through read/edit; arbitrary argv
are frozen trusted task inputs, never actor-provided shell snippets. Frozen
verification remains outside the editable project and is hash-checked each run.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
from typing import Protocol

from core.artifacts import ContentAddressedArtifacts
from rsi.contracts import Stage
from rsi.runtime import WorkflowError


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ingest_elapsed(clock, operation_id, receipt, *, observed_start, observed_end, category,
                   observed_execution_start=None, observed_execution_end=None):
    """Ingest producer duration, never remote epochs (trusted host API).

    Explicit execution bounds must come from authoritative host observation or
    a transport's validated clock mapping. Duration-only placement is synthetic:
    accept it only when overlap-union accounting is independent of placement.
    Uncertain envelopes persist and block budgeted work until exact reconciliation.
    """
    key = 'receipt:' + operation_id
    elapsed = receipt.get('elapsed')
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
            or elapsed < 0 or not all(math.isfinite(x) for x in (observed_start, observed_end))
            or observed_end > clock.clock() or observed_end < observed_start
            or elapsed > observed_end - observed_start or category not in clock.BILLABLE):
        clock.uncertainties[key] = {'error': 'invalid duration or envelope'}
        clock.save()
        raise WorkflowError('missing or invalid producer duration/observation envelope')
    metadata = dict(elapsed=elapsed, envelope=[observed_start, observed_end], category=category)
    exact = observed_execution_start is not None or observed_execution_end is not None
    previous = clock.receipts.get(key)
    if previous and any(previous[field] != metadata[field] for field in metadata):
        clock.uncertainties[key] = metadata
        clock.save()
        raise WorkflowError('conflicting duration or observation envelope')
    if previous and not exact and key not in clock.uncertainties:
        return
    for name, prior in clock.receipts.items():
        if (name != key and prior['placement'] == 'synthetic_union_equivalent'
                and prior['envelope'][0] < observed_end and prior['envelope'][1] > observed_start):
            clock.uncertainties[key] = metadata
            clock.save()
            raise WorkflowError('overlapping synthetic receipt requires authoritative reconciliation')
    if exact:
        start, end = observed_execution_start, observed_execution_end
        if (start is None or end is None or not all(math.isfinite(x) for x in (start, end))
                or start < observed_start or end > observed_end or end < start
                or not math.isclose(end - start, elapsed, rel_tol=0, abs_tol=1e-6)):
            clock.uncertainties[key] = metadata
            clock.save()
            raise WorkflowError('invalid authoritative execution interval')
        metadata['placement'] = 'authoritative_host_interval'
    else:
        intervals = sorted((max(observed_start, start), min(observed_end, clock.clock() if end is None else end))
                           for kind, start, end in clock.records.values()
                           if kind in clock.BILLABLE and start < observed_end
                           and (clock.clock() if end is None else end) > observed_start)
        coverage, boundary = 0., observed_start
        for start, end in intervals:
            coverage += max(0, end - max(start, boundary))
            boundary = max(boundary, end)
        if clock.unresolved or (0 < elapsed < observed_end - observed_start
                                and 0 < coverage < observed_end - observed_start):
            clock.uncertainties[key] = metadata
            clock.save()
            raise WorkflowError('uncertain partial overlap requires authoritative execution interval')
        start, end = observed_end - elapsed, observed_end
        metadata['placement'] = 'synthetic_union_equivalent'
    interval = [category, start, end]
    if (key in clock.records and clock.records[key] != interval
            and (not exact or previous['placement'] != 'synthetic_union_equivalent')):
        clock.uncertainties[key] = metadata
        clock.save()
        raise WorkflowError('conflicting elapsed receipt')
    # Retain producer duration and uncertainty envelope, not fictitious exact times.
    clock.receipts[key] = metadata
    clock.records[key] = interval
    clock.uncertainties.pop(key, None)
    clock.save()


@dataclass(frozen=True)
class CommandRequest:
    operation_id: str
    command: str
    argv: tuple[str, ...]
    project: Path
    verifier: Path
    source_hash: str
    verifier_hash: str
    category: str
    profile_path: Path | None


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes = b''
    stderr: bytes = b''
    timed_out: bool = False


class TrustedExecutor(Protocol):
    """Host injection; owns target path mapping, frozen verifier and transfers.

    observe(phase, **metadata) marks entry into that phase in controller time;
    it is not a retrospective duration report. Call observe('command', ...) before
    running/observing command work, including after setup or output transfers.
    execute must observe actual CPL handles before waiting, retain target/runtime/
    physical-device/candidate-case identity in observation metadata, and honor
    remaining() before each nested command. Only observed setup/transfer phases
    are excluded. Return only after terminal observation and output/source sync.
    An exception means unknown work: callers retain pending and never redispatch.
    No credentials or transport implementation belong in this protocol.
    """

    def execute(self, request: CommandRequest, *, observe, remaining) -> CommandResult: ...


class _UnboundExecutor:
    def execute(self, request, *, observe, remaining):
        raise WorkflowError('reopen snapshot with its trusted executor factory before execution')


class PythonExecution:
    """Task snapshot, CAS command receipts, exact-source edits and verification.

    run() does not complete workflow stages: the actor composition layer supplies
    skill/strategy evidence to Workflow.complete(). A pending receipt must be
    observed/reconciled by the owner and must never be dispatched again.
    """

    def __init__(self, root, workflow=None, *, executor_factory=None):
        self.root = Path(root).resolve()
        self.project = self.root / 'project'
        self.verifier = self.root / 'frozen' / 'verify.py'
        self.store = ContentAddressedArtifacts(self.root / 'artifacts')
        self.workflow = workflow
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.contract = self.store.read_json(self.manifest['contract'])
        identity = self.contract.get('executor_identity')
        if identity is not None and executor_factory is None:
            raise ValueError('frozen executor binding requires a trusted executor factory')
        self.executor = executor_factory(identity) if identity is not None else None
        if identity is not None and not callable(getattr(self.executor, 'execute', None)):
            raise ValueError('trusted executor must implement execute')

    @classmethod
    def snapshot(cls, root, source, verifier, *, commands, performance_seconds=None, editable_files=None,
                 generated_paths=(), executor_identity=None, retained_paths=()):
        if executor_identity is not None and (
                not isinstance(executor_identity, str) or not executor_identity.strip()):
            raise ValueError('executor_identity must be a nonblank trusted binding')
        root, source, verifier = Path(root).absolute(), Path(source).resolve(), Path(verifier).resolve()
        if root.exists() or root.resolve().is_relative_to(source):
            raise ValueError('task must be new and outside the input folder')
        if not source.is_dir() or not verifier.is_file():
            raise ValueError('missing source or correctness verifier')
        if (source / '.git').exists():
            status = subprocess.run(['git', '-C', str(source), 'status', '--porcelain',
                                     '--untracked-files=all'], capture_output=True, check=True)
            if status.stdout:
                raise ValueError('repository input must be clean')
        if performance_seconds is not None and (not math.isfinite(performance_seconds)
                                                or performance_seconds <= 0):
            raise ValueError('invalid performance goal')
        for argv in commands.values():
            if not isinstance(argv, (list, tuple)) or not argv or any(
                    not isinstance(arg, str) or not arg or '\0' in arg for arg in argv):
                raise ValueError('commands must be nonempty argv arrays')
        if any(path.is_symlink() for path in source.rglob('*')):
            raise ValueError('snapshot inputs must not contain symlinks')
        verifier_name = verifier.relative_to(source).as_posix() if verifier.is_relative_to(source) else None
        generated_paths = sorted(set(generated_paths))
        for name in generated_paths:
            if not name or Path(name).is_absolute() or '..' in Path(name).parts or name == '.':
                raise ValueError('generated paths must be explicit project-relative paths')
        if not isinstance(retained_paths, (list, tuple)) or any(
                not isinstance(name, str) for name in retained_paths):
            raise ValueError('retained_paths must be an array of project-relative output paths')
        retained_paths = sorted(set(retained_paths))
        for name in retained_paths:
            path = Path(name)
            if (not name or '\0' in name or path.is_absolute() or '..' in path.parts
                    or path == Path('.') or any(part in {'.git', '__pycache__', '.pytest_cache'}
                                               for part in path.parts)):
                raise ValueError('retained paths must select explicit outputs, not source or caches')
            # Retention declares future outputs, never reclassifies input files.
            selected = source / path
            if selected.is_file() or any(p.is_file() for p in selected.rglob('*')):
                raise ValueError('retained paths must not contain initial source files')
            if verifier_name and (verifier_name == name or Path(verifier_name).is_relative_to(path)):
                raise ValueError('retained paths must not include frozen verifier')
        retained_paths = sorted({Path(name).as_posix() for name in retained_paths})
        def generated(name):
            return (any(part in {'__pycache__', '.pytest_cache'} for part in Path(name).parts)
                    or any(name == item or name.startswith(item.rstrip('/') + '/')
                           for item in (*generated_paths, *retained_paths)))
        execution_inputs = sorted(p.relative_to(source).as_posix() for p in source.rglob('*')
                                  if p.is_file() and '.git' not in p.relative_to(source).parts
                                  and p.relative_to(source).as_posix() != verifier_name
                                  and not generated(p.relative_to(source).as_posix()))
        editable_files = execution_inputs if editable_files is None else sorted(set(editable_files))
        for name in editable_files:
            if (not name or Path(name).is_absolute() or '..' in Path(name).parts
                    or name == verifier_name or generated(name)):
                raise ValueError('invalid editable file manifest')
        root.mkdir(parents=True)
        try:
            shutil.copytree(source, root / 'project', ignore=shutil.ignore_patterns('.git'))
            if verifier_name:
                (root / 'project' / verifier_name).unlink()
            (root / 'frozen').mkdir()
            shutil.copyfile(verifier, root / 'frozen/verify.py')
            (root / 'receipts').mkdir()
            store = ContentAddressedArtifacts(root / 'artifacts')
            contract = dict(commands={key: list(value) for key, value in commands.items()},
                            editable_files=editable_files, execution_inputs=execution_inputs,
                            generated_paths=generated_paths, verifier_name=verifier_name,
                            verifier=digest(verifier.read_bytes()),
                            performance_seconds=performance_seconds)
            if executor_identity is not None:
                contract['executor_identity'] = executor_identity
            if retained_paths:
                contract['retained_paths'] = retained_paths
            atomic_json(root / 'manifest.json', {'contract': store.put_json(contract)})
            # Snapshotting must not construct a target adapter or dispatch work.
            return cls(root, executor_factory=lambda identity: _UnboundExecutor())
        except BaseException:
            shutil.rmtree(root)
            raise

    def project_hash(self):
        entries = []
        names = set(self.contract.get('execution_inputs', self.contract['editable_files']))
        names.update(self.contract['editable_files'])
        # New code/data counts too. Only explicitly generated outputs and narrow
        # interpreter/test caches may be ignored; edit rights are independent.
        for path in self.project.rglob('*'):
            name = path.relative_to(self.project).as_posix()
            if (any(part in {'__pycache__', '.pytest_cache'} for part in Path(name).parts)
                    or any(name == item or name.startswith(item.rstrip('/') + '/')
                           for item in (*self.contract.get('generated_paths', ()),
                                        *self.contract.get('retained_paths', ())))):
                continue
            if path.is_file() or path.is_symlink():
                names.add(name)
        for name in sorted(names):
            path = self._file(name)
            entries.append([name, digest(path.read_bytes()) if path.is_file() else None])
        return digest(json.dumps(entries, separators=(',', ':')).encode())

    def sync_source(self):
        current = self.project_hash()
        if self.workflow and current != self.workflow.source_hash:
            if not (self.workflow.pending and self.workflow.stage in
                    (Stage.IMPLEMENTATION, Stage.DEBUG, Stage.OPTIMISATION)):
                self.workflow.source_changed(current)
        return current

    def _file(self, name):
        if name == self.contract.get('verifier_name'):
            raise ValueError('frozen verifier is outside editable project')
        path = self.project / name
        if Path(name).is_absolute() or '..' in Path(name).parts or not name:
            raise ValueError('file must be inside editable project')
        if not path.resolve().is_relative_to(self.project) or path.is_symlink():
            raise ValueError('file escapes editable project')
        return path

    def read(self, name):
        data = self._file(name).read_bytes()
        return dict(content=data.decode(), file_hash=digest(data), project_hash=self.project_hash())

    def edit(self, name, content, *, expected_file_hash, expected_project_hash):
        if name not in self.contract['editable_files']:
            raise ValueError('file is not in the frozen editable manifest')
        path = self._file(name)
        actual = digest(path.read_bytes()) if path.exists() else None
        if actual != expected_file_hash or self.project_hash() != expected_project_hash:
            raise WorkflowError('source changed since read')
        if self.workflow:
            if self.workflow.outcome or (self.workflow.pending and self.workflow.stage not in
                    (Stage.IMPLEMENTATION, Stage.DEBUG, Stage.OPTIMISATION)):
                raise WorkflowError('workflow stage cannot edit source')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        # Pending mutation stages commit this through complete(new_source_hash=...).
        if self.workflow and self.workflow.pending:
            return self.project_hash()
        return self.sync_source()

    def _receipt_path(self, operation):
        return self.root / 'receipts' / (digest(operation.encode()) + '.json')

    def _retain_outputs(self, result, contract):
        """Collect only frozen output selections; terminal failures count too.

        Executors must synchronize these outputs before returning a terminal
        result. Missing outputs are explicit observations, never fabricated data.
        Collection errors leave the existing receipt pending, preventing replay.
        The reference keys deliberately reuse the canonical export closure.
        """
        selections = contract.get('retained_paths', ())
        if not selections:
            return
        def checked_name(path):
            name = path.relative_to(self.project).as_posix()
            # Check every component before traversing: even an internal symlink
            # can expose source/verifier material outside the selected subtree.
            parent = self.project
            for part in Path(name).parts:
                parent = parent / part
                if parent.is_symlink():
                    raise ValueError('retained output must not contain symlinks: ' + name)
            if not path.resolve().is_relative_to(self.project):
                raise ValueError('retained output escapes project: ' + name)
            return name
        files, missing = {}, []
        for selection in selections:
            selected = self.project / selection
            checked_name(selected)
            paths = [selected]
            if selected.is_dir():
                paths.extend(sorted(selected.rglob('*')))
            if not selected.exists():
                missing.append(selection)
            for path in paths:
                name = path.relative_to(self.project).as_posix()
                if any(part in {'.git', '__pycache__', '.pytest_cache'} for part in Path(name).parts):
                    continue
                checked_name(path)
                if path.is_dir() or not path.exists():
                    continue
                if not path.is_file():
                    raise ValueError('retained output must be a regular file: ' + name)
                data = path.read_bytes()
                files[name] = dict(path=name, bytes=len(data), reference=self.store.put_bytes(data))
        manifest = dict(kind='rsi-retained-output-v1', operation_id=result['operation_id'],
            command=result['command'], source_hash=result['source_hash'],
            classification=result['classification'], returncode=result['returncode'],
            retained_paths=list(selections), missing_paths=missing,
            files=[files[name] for name in sorted(files)])
        result.setdefault('artifact_refs', []).append(self.store.put_json(manifest))

    def _save(self, operation, value):
        atomic_json(self._receipt_path(operation), value)

    def receipt(self, operation):
        value = json.loads(self._receipt_path(operation).read_text())
        if value.get('status') != 'complete':
            raise WorkflowError('pending execution must be observed, never replayed')
        return self.store.read_json(value['artifact'])

    def numeric(self, operation):
        """Run the frozen independent correctness verifier with numeric classification."""
        return self.run(operation, 'test', numeric=True)

    def _execute_target(self, request, record, timeout):
        """Persist observations before further target effects; failures stay pending."""
        clock = self.workflow.accounting
        token = record['clock_token']
        started = time.monotonic()
        charged = 0.0

        def close_interval():
            nonlocal charged
            clock.end(token)
            kind, start, end = clock.records[token]
            if kind in clock.BILLABLE:
                charged += end - start

        def observe(phase, **metadata):
            nonlocal token
            if phase not in ('setup', 'transfer', 'command'):
                raise ValueError('executor phase must be setup, transfer or command')
            # Save even handle-only observations; never synthesize a CPL handle.
            record.setdefault('observations', []).append(dict(phase=phase, **metadata))
            self._save(request.operation_id, record)
            close_interval()
            token = clock.start(request.category if phase == 'command' else phase)
            record['clock_token'] = token
            self._save(request.operation_id, record)

        def remaining():
            if self.workflow.state['stopped'] or self.workflow.outcome:
                return 0.
            limits = [value for value in self.workflow.remaining().values() if value is not None]
            if timeout is not None:
                active = (clock.now() - clock.records[token][1]
                          if clock.records[token][0] in clock.BILLABLE else 0.)
                limits.append(max(0, timeout - charged - active))
            return min(limits) if limits else None

        try:
            result = self.executor.execute(request, observe=observe, remaining=remaining)
            if not isinstance(result, CommandResult):
                raise TypeError('executor must return a terminal CommandResult')
            if not isinstance(result.stdout, bytes) or not isinstance(result.stderr, bytes):
                raise TypeError('command output must be bytes')
            if (type(result.returncode) is not int or type(result.timed_out) is not bool):
                raise TypeError('terminal command requires integer returncode and boolean timed_out')
            return result
        except BaseException:
            clock.uncertainties['executor:' + request.operation_id] = dict(
                status='unknown', receipt=str(self._receipt_path(request.operation_id)))
            clock.save()
            raise
        finally:
            close_interval()
            record['elapsed'] = charged
            record['observed_elapsed'] = time.monotonic() - started
            self._save(request.operation_id, record)

    def run(self, operation, command, *, numeric=False, profile=False, timeout=None):
        if self._receipt_path(operation).exists():
            previous = self.receipt(operation)
            if (previous['command'] != command or previous['source_hash'] != self.project_hash()
                    or previous['numeric'] != numeric or previous['profiling'] != profile):
                raise WorkflowError('operation identity reused for different work or source')
            return previous
        if not self.workflow:
            raise WorkflowError('execution requires a workflow budget')
        source_hash = self.sync_source()
        if digest(self.verifier.read_bytes()) != self.contract['verifier']:
            raise WorkflowError('frozen verifier changed')
        # Reload the CAS contract, detecting corruption instead of trusting edits.
        contract = self.store.read_json(self.manifest['contract'])
        argv = [arg.replace('{verifier}', str(self.verifier)) for arg in contract['commands'][command]]
        if command in ('test', 'validation') and str(self.verifier) not in argv:
            raise WorkflowError('verification must execute frozen verifier')
        remaining = [x for x in self.workflow.remaining().values() if x is not None]
        if timeout is not None:
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('timeout must be positive')
            remaining.append(timeout)
        limit = min(remaining) if remaining else None
        if limit is not None and limit <= 0:
            raise WorkflowError('time budget exhausted')
        category = 'profiling' if profile else 'compilation' if command == 'compile' else 'execution'
        profile_path = self.root / 'receipts' / (digest(operation.encode()) + '.profile')
        if profile:
            if Path(argv[0]).resolve() != Path(sys.executable).resolve() or len(argv) < 2 or argv[1].startswith('-'):
                raise ValueError('profiling requires a declared Python script argv')
            argv = [argv[0], '-m', 'cProfile', '-o', str(profile_path), *argv[1:]]
        env = dict(os.environ, PYTHONPATH=str(self.project),
                   PYTHONPYCACHEPREFIX=str(self.root / 'cache'))
        record = dict(operation_id=operation, source_hash=source_hash, command=command,
                      argv=argv, status='pending', category=category, numeric=numeric, profiling=profile,
                      stage_source_hash=self.workflow.source_hash,
                      stage_operation=self.workflow.pending['id'] if self.workflow.pending else None)
        if self.executor is not None:
            record['executor_identity'] = contract['executor_identity']
        if not self.workflow.reserve_execution(operation):
            raise WorkflowError('reserved command has no receipt; reconcile before dispatch')
        self._save(operation, record)
        token = self.workflow.accounting.start(category)
        record['clock_token'] = token
        self._save(operation, record)
        started = time.monotonic()
        process, timed_out, stdout, stderr, returncode = None, False, b'', b'', None
        interrupted = None
        if self.executor is not None:
            request = CommandRequest(operation, command, tuple(argv), self.project, self.verifier,
                source_hash, contract['verifier'], category, profile_path if profile else None)
            target_result = self._execute_target(request, record, timeout)
            stdout, stderr = target_result.stdout, target_result.stderr
            returncode, timed_out = target_result.returncode, target_result.timed_out
            elapsed = record['elapsed']
        else:
            try:
                process = subprocess.Popen(argv, cwd=self.project, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, start_new_session=True)
                record['pid'] = process.pid
                self._save(operation, record)
                try:
                    if process is not None:
                        stdout, stderr = process.communicate(
                            timeout=None if limit is None else max(0, limit - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    stdout, stderr = process.communicate()
                if process is not None:
                    returncode = process.returncode
            except OSError as exc:
                stderr = str(exc).encode()
            except BaseException as exc:
                interrupted = exc
                # Only this live Popen instance establishes process-group ownership.
                # Never act on a persisted PID after restart.
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    stdout, stderr = process.communicate()
                    returncode = process.returncode
            finally:
                elapsed = time.monotonic() - started
                # Producer elapsed is authoritative, never a subtraction of wall epochs.
                ended = self.workflow.accounting.records[token][1] + elapsed
                record['elapsed'] = elapsed
                self._save(operation, record)
                self.workflow.accounting.reconcile(token, ended)
        return self._finalize(record, stdout=stdout, stderr=stderr, returncode=returncode,
                              timed_out=timed_out, elapsed=elapsed, interrupted=interrupted)

    def _finalize(self, record, *, stdout, stderr, returncode, timed_out, elapsed, interrupted=None):
        operation, command = record['operation_id'], record['command']
        source_hash, numeric = record['source_hash'], record['numeric']
        contract = self.store.read_json(self.manifest['contract'])
        profile_path = self.root / 'receipts' / (digest(operation.encode()) + '.profile')
        current = self.project_hash()
        if digest(self.verifier.read_bytes()) != contract['verifier']:
            raise WorkflowError('frozen verifier changed during command; reconcile pending execution')
        if current != source_hash:
            # Composition must reconcile pending Workflow source edits itself.
            if not self.workflow.pending:
                self.workflow.source_changed(current)
        classification = ('interrupted' if interrupted else 'timeout' if timed_out else 'source_changed' if current != source_hash
                          else ('compile_failure' if command == 'compile' else
                                'numeric_failure' if numeric else 'execution_failure')
                          if returncode != 0 else 'passed')
        goal = contract['performance_seconds']
        if classification == 'passed' and command in ('timing', 'profile') and goal is not None and elapsed > goal:
            classification = 'performance_shortfall'
        result = dict(record, status='complete', returncode=returncode, timed_out=timed_out,
                      elapsed=elapsed,
                      clock_domain='controller_intervals' if self.executor is not None else 'producer_monotonic',
                      current_hash=current,
                      classification=classification, stdout=self.store.put_bytes(stdout),
                      stderr=self.store.put_bytes(stderr))
        if profile_path.exists():
            result['profile'] = self.store.put_bytes(profile_path.read_bytes())
        self._retain_outputs(result, contract)
        artifact = self.store.put_json(result)
        self._save(operation, dict(status='complete', artifact=artifact))
        if interrupted is not None:
            raise interrupted
        return result

    def reconcile_target(self, operation, timing, *, observation_seconds=45.):
        """Observe the original target command and finalize only verified timing.

        timing is a trusted host receipt in the original controller clock domain,
        never raw remote epoch arithmetic. Missing timing/closure stays pending.
        This method calls executor.reconcile only; execute is never a fallback.
        """
        record = json.loads(self._receipt_path(operation).read_text())
        if record['status'] == 'complete':
            return self.receipt(operation)
        if (self.executor is None or not callable(getattr(self.executor, 'reconcile', None))
                or timing['operation_id'] != operation or timing['source_hash'] != record['source_hash']
                or type(observation_seconds) not in (int, float) or not math.isfinite(observation_seconds)
                or observation_seconds <= 0):
            raise WorkflowError('scoped authoritative timing and observation-capable executor required')
        contract = self.store.read_json(self.manifest['contract'])
        if digest(self.verifier.read_bytes()) != contract['verifier']:
            raise WorkflowError('frozen verifier changed')
        argv = [arg.replace('{verifier}', str(self.verifier)) for arg in contract['commands'][record['command']]]
        profile_path = self.root / 'receipts' / (digest(operation.encode()) + '.profile')
        if record['profiling']:
            argv = [argv[0], '-m', 'cProfile', '-o', str(profile_path), *argv[1:]]
        if argv != record['argv'] or record['stage_operation'] != self.workflow.pending['id']:
            raise WorkflowError('pending target dispatch differs from frozen stage/command')
        request = CommandRequest(operation, record['command'], tuple(argv), self.project, self.verifier,
            record['source_hash'], contract['verifier'], record['category'],
            profile_path if record['profiling'] else None)
        clock = self.workflow.accounting
        deadline = time.monotonic() + observation_seconds
        intervals, phase, start = [], record['category'], clock.now()
        def observe(next_phase, **metadata):
            nonlocal phase, start
            if next_phase not in ('command', 'setup', 'transfer'):
                raise ValueError('unknown executor observation phase')
            ended = clock.now()
            intervals.append([phase, start, ended])
            phase, start = record['category'] if next_phase == 'command' else next_phase, ended
            record.setdefault('observations', []).append(dict(phase=next_phase, reconciliation=True, **metadata))
            self._save(operation, record)
        try:
            result = self.executor.reconcile(request, observe=observe,
                remaining=lambda: max(0., deadline - time.monotonic()))
        finally:
            intervals.append([phase, start, clock.now()])
            for index, interval in enumerate(intervals):
                clock.records['observation:' + uuid.uuid4().hex + ':' + str(index)] = interval
            clock.save()
        if (not isinstance(result, CommandResult) or type(result.returncode) is not int
                or not isinstance(result.stdout, bytes) or not isinstance(result.stderr, bytes)
                or type(result.timed_out) is not bool):
            raise WorkflowError('terminal target command observation required')
        for token, ended_at in timing.get('closed_intervals', {}).items():
            if token != record['clock_token']:
                raise WorkflowError('target observation cannot close unrelated actor intervals')
            if token in clock.unresolved:
                clock.reconcile(token, ended_at)
            elif clock.records[token][2] != ended_at:
                raise WorkflowError('retained target interval differs from authoritative closure')
        ingest_elapsed(clock, operation, timing['producer'],
            observed_start=timing['observed_start'], observed_end=timing['observed_end'],
            observed_execution_start=timing['execution_start'], observed_execution_end=timing['execution_end'],
            category=record['category'])
        clock.uncertainties.pop('executor:' + operation, None)
        clock.save()
        clock.seconds()  # Unrelated unknown work cannot be silently discarded.
        return self._finalize(record, stdout=result.stdout, stderr=result.stderr,
            returncode=result.returncode, timed_out=result.timed_out, elapsed=timing['producer']['elapsed'])
