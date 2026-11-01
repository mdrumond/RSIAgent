"""Durable, single-writer local jobs for a pinned RSI instance.

A dead worker is observable interruption, never permission to replay its request.
Stopping is cooperative: the service must drain its canonical actor before exit.
"""
from __future__ import annotations

import fcntl
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from .configuration import _bytes, _locations, _publish, show_configuration


_TERMINAL = {'completed', 'failed', 'stopped', 'interrupted'}


def _directory(instance):
    root, _ = _locations(instance)
    directory = root / 'runs' / 'jobs'
    directory.mkdir(parents=True, exist_ok=True)
    if directory.resolve() != directory:
        raise ValueError('job directory escapes instance')
    return root, directory


def _path(directory, job_id):
    try:
        if str(uuid.UUID(job_id)) != job_id:
            raise ValueError
    except (ValueError, AttributeError):
        raise ValueError('invalid job ID') from None
    return directory / f'{job_id}.json'


def _identity(pid):
    """Linux process identity, independent of PID reuse or machine restart."""
    raw = Path(f'/proc/{pid}/stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'start_ticks': fields[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'state': fields[0]}


def _pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # Some supported Python builds omit the wrapper despite Linux libc support.
    libc = ctypes.CDLL(None, use_errno=True)
    operation = getattr(libc, 'pidfd_open', None)
    if operation is None:
        raise ValueError('safe process stopping requires Linux pidfd_open support')
    operation.argtypes, operation.restype = (ctypes.c_int, ctypes.c_uint), ctypes.c_int
    descriptor = operation(pid, 0)
    if descriptor < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return descriptor


def _alive(identity):
    if not identity:
        return False
    try:
        current = _identity(identity['pid'])
        return current['state'] != 'Z' and all(
            current[key] == identity[key] for key in ('pid', 'start_ticks', 'boot_id'))
    except (OSError, KeyError):
        return False


def _machinery(root):
    # Bootstrap pins machinery independently of editable runtime inputs.
    manifest = json.loads((root / 'instance.json').read_text())
    machinery = root / 'machinery'
    if machinery.resolve() != machinery:
        raise ValueError('machinery directory must be the instance snapshot')
    for relative in ('rsi/jobs.py', 'rsi/job_worker.py', 'rsi/services.py', 'rsi/service.py'):
        if f'machinery/{relative}' not in manifest['files']:
            raise ValueError('instance release lacks CLI job support; bootstrap an updated instance')
    return machinery


def _launch_owner_present(directory, job_id):
    # A retained replacement proves this launch relinquished ownership before
    # admission. Its successor's instance lock cannot resurrect the old job.
    for path in directory.glob('*.json'):
        if path.stem != job_id:
            successor = json.loads(path.read_text())
            if successor['request'].get('previous_job') == job_id:
                return False
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(descriptor)


def _observed_status(directory, job_id, *, owner_present=None):
    path = _path(directory, job_id)
    record = json.loads(path.read_text())
    if record['state'] == 'launching' and record.get('process') is None:
        if owner_present is None:
            owner_present = _launch_owner_present(directory, job_id)
        # Startup may have published its identity/final result during the probe.
        record = json.loads(path.read_text())
        if record['state'] == 'launching' and record.get('process') is None and owner_present:
            return record  # Ownership is retained; no PID or terminal failure is invented.
    if record['state'] not in _TERMINAL and not _alive(record.get('process')):
        record['state'] = 'interrupted'
    return record


def status(instance, job_id):
    _, directory = _directory(instance)
    return _observed_status(directory, job_id)


def list_jobs(instance):
    _, directory = _directory(instance)
    return [status(instance, path.stem) for path in sorted(directory.glob('*.json'))]


def start(instance, request):
    """Launch once, preserving a failed launch or unknown worker for inspection."""
    root, directory = _directory(instance)
    machinery = _machinery(root)
    snapshot = show_configuration(root)
    if request.get('action') in ('resume', 'reconcile', 'recover'):
        previous = status(root, request['previous_job'])
        if previous['request']['cycle_id'] != request['cycle_id']:
            raise ValueError('resume cannot change retained cycle identity')
        snapshot = previous['configuration']
    lock = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('instance already has an owned running job') from None
        from .instance_dispatch import launch_context
        _, verified_source, interpreter, environment = launch_context(root, required=True)
        if verified_source != machinery:
            raise ValueError('job source changed during runtime selection')
        # This dispatcher's newly acquired lock is not ownership of an older launch.
        previous_jobs = [_observed_status(directory, path.stem, owner_present=False)
                         for path in sorted(directory.glob('*.json'))]
        superseded = {item['request'].get('previous_job') for item in previous_jobs}
        for previous in previous_jobs:
            if previous['state'] == 'interrupted' and previous['job_id'] not in superseded:
                if request.get('action') not in ('resume', 'revisit', 'reconcile', 'recover') or request.get('previous_job') != previous['job_id']:
                    raise ValueError('interrupted job requires explicit resume/revisit with its retained ID')
        job_id = str(uuid.uuid4())
        request = dict(request)
        if request.get('action') != 'resume':
            request['request_id'] = job_id
        path = _path(directory, job_id)
        record = {'job_id': job_id, 'state': 'launching', 'request': request,
                  'configuration': snapshot, 'process': None}
        _publish(path, _bytes(record), immutable=True)
        # -I ignores caller PYTHONPATH/user site; only the verified snapshot is inserted.
        bootstrap = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
                     'from rsi.job_worker import main; main()')
        with (directory / f'{job_id}.log').open('ab', buffering=0) as output:
            try:
                process = subprocess.Popen(
                    [interpreter, '-I', '-c', bootstrap, str(machinery),
                     str(root), job_id, str(lock)], cwd=machinery,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                    start_new_session=True, pass_fds=(lock,), env=environment)
                # Worker alone publishes state, avoiding parent/worker write races.
            except OSError as exc:
                record.update(state='failed', error=str(exc))
                _publish(path, _bytes(record))
                raise
        return {'job_id': job_id, 'pid': process.pid, 'state': 'launching'}
    finally:
        # Child inherits the open description and retains the flock until it exits.
        os.close(lock)


def stop(instance, job_id):
    root, directory = _directory(instance)
    record = json.loads(_path(directory, job_id).read_text())
    if record['state'] in _TERMINAL:
        return record
    identity = record.get('process')
    if not identity:
        # Durable flag covers the launch/identity publication race.
        _publish(directory / f'{job_id}.stop', b'{}\n')
        return record
    if not _alive(identity):
        return status(root, job_id)
    try:
        descriptor = _pidfd_open(identity['pid'])
    except ProcessLookupError:
        return status(root, job_id)
    try:
        if not _alive(identity):
            return status(root, job_id)
        _publish(directory / f'{job_id}.stop', b'{}\n')
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        except ProcessLookupError:
            pass  # Worker completed between identity observation and the signal.
    finally:
        os.close(descriptor)
    return status(root, job_id)


def evidence(instance, job_id):
    root, directory = _directory(instance)
    record = status(root, job_id)
    path = directory / f'{job_id}.events.jsonl'
    lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
    from .job_inspection import inspect_cycle
    return {'job': record, 'inspection': inspect_cycle(root, record['request']), 'events': [json.loads(line) for line in lines if line.endswith(b'\n')],
            'log': str(directory / f'{job_id}.log')}
