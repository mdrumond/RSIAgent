"""Loopback dashboard ownership for one pinned instance, independent of actor jobs."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .configuration import _bytes, _publish
from .instance_setup import machinery


def _paths(instance):
    root, source, digest = machinery(instance)
    directory = root / 'dashboard'
    if not directory.is_dir() or directory.resolve() != directory:
        raise ValueError('dashboard directory must belong to the instance')
    return root, source, digest, directory


def status(instance):
    from .jobs import _alive
    _, _, _, directory = _paths(instance)
    path = directory / 'process.json'
    if not path.exists():
        return {'state': 'not_started'}
    record = json.loads(path.read_text())
    if record['state'] in ('running', 'launching') and not _alive(record.get('process')):
        record['state'] = 'unknown' if record.get('process') is None else 'interrupted'
    return record


def start(instance, *, port=0, wait_seconds=15):
    if type(port) is not int or not 0 <= port <= 65535 or wait_seconds <= 0:
        raise ValueError('port must be 0..65535 and startup wait positive')
    root, source, digest, directory = _paths(instance)
    setup = json.loads((root / 'runtime/setup.json').read_text())
    if setup.get('state') != 'ready' or setup['machinery_hash'] != digest:
        raise ValueError('run instance setup for this pinned machinery first')
    interpreter = root / 'runtime/python/bin/python'
    if not interpreter.is_file():
        raise ValueError('instance Python environment is missing; run instance setup')
    for name in ('rsi/dashboard_runtime.py', 'rsi/jobs.py'):
        manifest = json.loads((root / 'instance.json').read_text())
        if 'machinery/' + name not in manifest['files']:
            raise ValueError('snapshot lacks dashboard lifecycle support; bootstrap a current release')
    lock = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('dashboard already has an owned process') from None
        previous = status(root)
        if previous['state'] == 'unknown':
            raise ValueError('unknown dashboard launch: inspect retained evidence and explicitly stop it first')
        launch_id = str(uuid.uuid4())
        stop_path = directory / 'stop.json'
        stop_path.unlink(missing_ok=True)
        record = {'state': 'launching', 'launch_id': launch_id, 'process': None,
                  'machinery_hash': digest, 'host': '127.0.0.1', 'requested_port': port}
        _publish(directory / 'process.json', _bytes(record))
        bootstrap = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
                     'from rsi.dashboard_runtime import serve; serve(*sys.argv[1:])')
        try:
            with (directory / 'server.log').open('ab', buffering=0) as output:
                subprocess.Popen([str(interpreter), '-I', '-c', bootstrap, str(source),
                                  str(root), launch_id, str(lock)], cwd=source,
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                 start_new_session=True, pass_fds=(lock,),
                                 env=dict(os.environ, PATH=str(Path(setup['node']).parent) +
                                          os.pathsep + os.environ.get('PATH', os.defpath),
                                          PYTHONNOUSERSITE='1'))
        except OSError:
            record['state'] = 'failed'
            _publish(directory / 'process.json', _bytes(record))
            raise
    finally:
        os.close(lock)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        observed = status(root)
        if observed['state'] not in ('launching', 'unknown'):
            return observed
        time.sleep(.05)
    # Timeout retains launch identity. A subsequent start cannot replay it.
    return status(root)


def stop(instance):
    from .jobs import _alive, _pidfd_open
    _, _, _, directory = _paths(instance)
    record = status(instance)
    if record['state'] in ('not_started', 'stopped', 'failed', 'interrupted'):
        return record
    _publish(directory / 'stop.json', _bytes({'launch_id': record['launch_id']}))
    identity = record.get('process')
    if not identity:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return record
            record['state'] = 'stopped'
            _publish(directory / 'process.json', _bytes(record))
        finally:
            os.close(descriptor)
        return record
    if identity:
        try:
            descriptor = _pidfd_open(identity['pid'])
        except ProcessLookupError:
            return status(instance)
        try:
            if _alive(identity):
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        except ProcessLookupError:
            pass
        finally:
            os.close(descriptor)
    return status(instance)


def serve(instance, launch_id, lock_descriptor):
    """Owned child entrypoint. Bind once; publish the actual kernel-selected port."""
    from .jobs import _identity
    from dashboard.app import create_app
    from werkzeug.serving import make_server
    root, _, _, directory = _paths(instance)
    path = directory / 'process.json'
    record = json.loads(path.read_text())
    if record['launch_id'] != launch_id:
        raise ValueError('dashboard launch identity changed')
    record['process'] = _identity(os.getpid())
    _publish(path, _bytes(record))
    server = None
    def interrupted(*_):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        config = directory / 'config.json'
        _publish(config, _bytes({'campaigns': [], 'rsi_instance': str(root),
                                 'controls': {'enabled': True}}))
        app = create_app(config, directory / 'cache.sqlite')
        server = make_server('127.0.0.1', record['requested_port'], app, threaded=True)
        record.update(state='running', port=server.server_port,
                      url=f'http://127.0.0.1:{server.server_port}')
        _publish(path, _bytes(record))
        if not (directory / 'stop.json').exists():
            server.serve_forever(poll_interval=.1)
        record['state'] = 'stopped'
    except KeyboardInterrupt:
        record['state'] = 'stopped'
    except BaseException as exc:
        record.update(state='failed', error_type=type(exc).__name__)
        raise
    finally:
        if server:
            server.server_close()
        _publish(path, _bytes(record))
        signal.signal(signal.SIGTERM, previous_handler)
        os.close(int(lock_descriptor))
