"""Checked retained operations. An uncertain dispatch is never replayed."""
from contextlib import contextmanager
from benchmarks.ascend.actor_time import preparing
import fcntl
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import tempfile
import time

from .jobs import atomic, canonical
from .control import ObservationUnavailable, WorkerStopped

CLIENT_TIMEOUT = 120
CANCEL_TIMEOUT = 30


@contextmanager
def exclusive(path, *, wait=False, stopped=lambda: False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    raise
                if stopped():
                    raise WorkerStopped('stopped while waiting for device')
                time.sleep(1)
        yield


class CheckedTransport:
    def __init__(self, config, *, run=None, retain=lambda handle: None,
                 stopped=lambda: False, deadline=float('inf')):
        self.config, self.run, self.retain, self.stopped = config, run, retain, stopped
        self.deadline = deadline
        self.client_timeout = float(config.get('transport_timeout_seconds', CLIENT_TIMEOUT))
        if not math.isfinite(self.client_timeout) or self.client_timeout <= 0:
            raise ValueError('transport_timeout_seconds must be finite and positive')

    def _invoke(self, argv, *, cleanup=False, **kwargs):
        if self.run is not None:
            return self.run(argv, capture_output=True, text=True, **kwargs)
        # Files avoid pipe-buffer deadlocks while polling Stop. Kill only this
        # local client's private process group, never assume its remote job died.
        limit = time.monotonic() + (CANCEL_TIMEOUT if cleanup else self.client_timeout)
        if not cleanup:
            limit = min(limit, self.deadline)
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(argv, stdout=stdout, stderr=stderr,
                                       start_new_session=True, **kwargs)
            interrupted = False
            try:
                while process.poll() is None:
                    if time.monotonic() >= limit or (not cleanup and self.stopped()):
                        interrupted = True
                        break
                    time.sleep(.1)
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        pass
                    try:
                        # The leader may exit before a stalled SSH child does.
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            stdout.seek(0)
            stderr.seek(0)
            result = subprocess.CompletedProcess(argv, process.returncode,
                        stdout.read().decode(errors='replace'), stderr.read().decode(errors='replace'))
            result.interrupted = interrupted
            return result

    def cancel_pending(self, directory):
        """Recover Stop against the exact retained operation, never a new dispatch."""
        directory = Path(directory)
        path = directory / 'transport.json'
        state = json.loads(path.read_text())
        if state.get('state') in {'completed', 'failed', 'cancelled'}:
            return
        if not state.get('native_handle'):
            raise ObservationUnavailable('cannot cancel uncertain dispatch without its native handle')
        result = self._invoke([self.config['cpl_remote'], '--json', 'cancel', state['native_handle']], cleanup=True)
        atomic(directory / 'cancel.log', (result.stdout + result.stderr).encode())
        if result.returncode:
            raise ObservationUnavailable('cancellation unavailable; retained handle remains active')
        state['state'] = 'cancelled'
        atomic(path, canonical(state))

    @preparing('optimizer file transfer')
    def transfer(self, action, source, destination):
        # Transfer sentinels also prevent duplicate uncertain submission.
        local = Path(source if action == 'upload' else destination)
        marker = local.with_name(local.name + '.' + action + '.json')
        state = json.loads(marker.read_text()) if marker.exists() else {}
        if state.get('state') == 'completed' and (action == 'upload' or local.exists()):
            return
        if state and not state.get('handle'):
            raise ObservationUnavailable('uncertain transfer; inspect ' + str(marker))
        if not state:
            atomic(marker, canonical({'state': 'dispatching'}))
            result = self._invoke([self.config['cpl_remote'], '--json', action,
                                   self.config['target'], str(source), str(destination)])
            atomic(local.with_name(local.name + '.' + action + '.log'),
                   (f'returncode={result.returncode}\n' + result.stdout + '\n' + result.stderr).encode())
            try:
                state = json.loads(result.stdout)
            except ValueError:
                raise ObservationUnavailable('transfer returned no JSON; inspect ' + str(marker))
            if state.get('handle'):
                self.retain(state['handle'])
            atomic(marker, canonical(state))
            if getattr(result, 'interrupted', False):
                raise ObservationUnavailable('transfer client interrupted; inspect ' + str(marker))
        while state.get('state') not in {'completed', 'failed', 'cancelled'}:
            if not state.get('handle'):
                raise ObservationUnavailable('transfer returned no handle')
            if self.stopped():
                raise WorkerStopped('stopped while observing transfer')
            result = self._invoke([self.config['cpl_remote'], '--json', 'observe', state['handle'],
                                   '--wait', '--timeout', '5'])
            atomic(local.with_name(local.name + '.' + action + '.observe.log'), (result.stdout + result.stderr).encode())
            if getattr(result, 'interrupted', False):
                raise ObservationUnavailable('transfer observation interrupted; retained handle remains active')
            try:
                state = json.loads(result.stdout)
            except ValueError:
                raise ObservationUnavailable('transfer observation unavailable')
            atomic(marker, canonical(state))
        if state['state'] != 'completed':
            raise RuntimeError('transfer failed: ' + str(marker))
        if action == 'download' and not local.exists():
            staging = local.with_name(local.name + '.fetching')
            result = self._invoke([self.config['cpl_remote'], '--json', 'fetch', state['handle'], str(staging)])
            atomic(local.with_name(local.name + '.fetch.log'), (result.stdout + result.stderr).encode())
            if result.returncode:
                raise ObservationUnavailable('completed transfer fetch unavailable')
            os.replace(staging, local)

    def execute(self, directory, argv):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'transport.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get('state') == 'completed':
            return
        if state and not state.get('handle'):
            raise ObservationUnavailable('dispatch outcome unknown; retained transport.json requires recovery')
        while True:
            if self.stopped():
                if state.get('native_handle'):
                    self.cancel_pending(directory)
                raise WorkerStopped('user requested stop')
            command = [self.config['validation_wrapper'], '--profile', self.config['profile']]
            if state.get('handle'):
                observer = str(Path(self.config['validation_wrapper']).parent / 'bz-a3' / 'run.sh')
                command = [observer, '--profile', self.config['profile'], '--timeout', '5',
                           '--resume', state['handle'].split(':', 1)[1]]
            else:
                # Commit before dispatch; a process crash cannot cause duplicate execution.
                state = {'state': 'dispatching'}
                atomic(path, canonical(state))
                command += ['--operation', 'optimizer-' + directory.name[:40], 'run', '--native',
                            '--runtime', self.config['runtime'], '--device', str(self.config['physical_device']),
                            '--timeout', '5', '--', *argv]
            result = self._invoke(command, env={**os.environ, 'CPL_REMOTE': self.config['cpl_remote']})
            output = result.stdout + '\n' + result.stderr
            with (directory / 'transport.log').open('ab') as stream:
                stream.write(output.encode())
            for line in output.splitlines():
                if line.startswith('BZ_A3_JOB_STATE='):
                    observed = line.split('=', 1)[1].split()[0]
                    if 'exit=124' in line.split():
                        state['deadline_expired'] = True
                    if observed in {'failed', 'cancelled', 'missing'}:
                        state['state'] = 'failed'
                for prefix, key in [('BZ_A3_REMOTE_HANDLE=', 'native_handle'),
                                    ('CATLASS_VALIDATION_HANDLE=', 'handle'),
                                    ('CATLASS_VALIDATION_STATE=', 'state')]:
                    if line.startswith(prefix):
                        value = line[len(prefix):]
                        if key.endswith('handle') and value != state.get(key):
                            self.retain(value)
                        state[key] = value
            if not result.returncode:
                state['state'] = 'completed'
            atomic(path, canonical(state))
            if getattr(result, 'interrupted', False):
                if state.get('state') == 'completed':
                    return  # Terminal remote evidence survives client teardown.
                # A local observation timeout is still uncertain. Only a real
                # Stop/budget boundary plus confirmed exact-handle cancellation
                # can become WorkerStopped (and release optional diagnostics).
                if self.stopped() and state.get('native_handle'):
                    self.cancel_pending(directory)
                    raise WorkerStopped('stopped after cancelling retained operation')
                raise ObservationUnavailable('local command client interrupted; observe retained handle or recover dispatch')
            if state.get('deadline_expired'):
                state['state'] = 'cancelled'
                state['reason'] = 'wall-time budget exhausted'
                atomic(path, canonical(state))
                raise WorkerStopped('remote wall-time budget exhausted')
            if state['state'] == 'completed':
                return
            if state['state'] == 'failed' and state.get('handle'):
                raise RuntimeError('remote operation failed; inspect transport.log')
            if not state.get('handle'):
                raise ObservationUnavailable('dispatch returned no handle; do not resubmit')
