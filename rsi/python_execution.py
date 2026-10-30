"""Portable evidence backend. Host calls only; this is not a process sandbox.

One writer owns a task. The actor edits only through read/edit; arbitrary argv
are frozen trusted task inputs, never actor-provided shell snippets. Frozen
verification remains outside the editable project and is hash-checked each run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
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


class PythonExecution:
    """Task snapshot, CAS command receipts, exact-source edits and verification.

    run() does not complete workflow stages: the actor composition layer supplies
    skill/strategy evidence to Workflow.complete(). A pending receipt must be
    observed/reconciled by the owner and must never be dispatched again.
    """

    def __init__(self, root, workflow=None):
        self.root = Path(root).resolve()
        self.project = self.root / 'project'
        self.verifier = self.root / 'frozen' / 'verify.py'
        self.store = ContentAddressedArtifacts(self.root / 'artifacts')
        self.workflow = workflow
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.contract = self.store.read_json(self.manifest['contract'])

    @classmethod
    def snapshot(cls, root, source, verifier, *, commands, performance_seconds=None, editable_files=None):
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
        if editable_files is None:
            editable_files = [p.relative_to(source).as_posix() for p in source.rglob('*')
                              if p.is_file() and not any(part in {'.git', '__pycache__', '.pytest_cache'}
                                                        for part in p.relative_to(source).parts)
                              and p.relative_to(source).as_posix() != verifier_name]
        editable_files = sorted(set(editable_files))
        for name in editable_files:
            if (not name or Path(name).is_absolute() or '..' in Path(name).parts
                    or name == verifier_name):
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
                            editable_files=editable_files, verifier=digest(verifier.read_bytes()),
                            performance_seconds=performance_seconds)
            atomic_json(root / 'manifest.json', {'contract': store.put_json(contract)})
            return cls(root)
        except BaseException:
            shutil.rmtree(root)
            raise

    def project_hash(self):
        entries = []
        for name in self.contract['editable_files']:
            path = self._file(name)
            entries.append([name, digest(path.read_bytes()) if path.is_file() else None])
        return digest(json.dumps(entries, separators=(',', ':')).encode())

    def sync_source(self):
        current = self.project_hash()
        if self.workflow and current != self.workflow.source_hash:
            self.workflow.source_changed(current)
        return current

    def _file(self, name):
        if name not in self.contract['editable_files']:
            raise ValueError('file is not in the frozen editable manifest')
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
                      argv=argv, status='pending', category=category, numeric=numeric, profiling=profile)
        if not self.workflow.reserve_execution(operation):
            raise WorkflowError('reserved command has no receipt; reconcile before dispatch')
        self._save(operation, record)
        token = self.workflow.accounting.start(category)
        record['clock_token'] = token
        self._save(operation, record)
        started = time.monotonic()
        process, timed_out, stdout, stderr, returncode = None, False, b'', b'', None
        try:
            process = subprocess.Popen(argv, cwd=self.project, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
            record['pid'] = process.pid
            self._save(operation, record)
            try:
                stdout, stderr = process.communicate(
                    timeout=None if limit is None else max(0, limit - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
            returncode = process.returncode
        except OSError as exc:
            stderr = str(exc).encode()
        finally:
            elapsed = time.monotonic() - started
            # Producer elapsed is authoritative, never a subtraction of wall epochs.
            ended = self.workflow.accounting.records[token][1] + elapsed
            record['elapsed'] = elapsed
            self._save(operation, record)
            self.workflow.accounting.reconcile(token, ended)
        current = self.project_hash()
        if current != source_hash:
            # Composition must reconcile pending Workflow source edits itself.
            if not self.workflow.pending:
                self.workflow.source_changed(current)
        classification = ('timeout' if timed_out else 'source_changed' if current != source_hash
                          else ('compile_failure' if command == 'compile' else
                                'numeric_failure' if numeric else 'execution_failure')
                          if returncode != 0 else 'passed')
        goal = contract['performance_seconds']
        if classification == 'passed' and command in ('timing', 'profile') and goal is not None and elapsed > goal:
            classification = 'performance_shortfall'
        result = dict(record, status='complete', returncode=returncode, timed_out=timed_out,
                      elapsed=elapsed, clock_domain='producer_monotonic', current_hash=current,
                      classification=classification, stdout=self.store.put_bytes(stdout),
                      stderr=self.store.put_bytes(stderr))
        if profile_path.exists():
            result['profile'] = self.store.put_bytes(profile_path.read_bytes())
        artifact = self.store.put_json(result)
        self._save(operation, dict(status='complete', artifact=artifact))
        return result
