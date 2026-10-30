"""Owned worker lifecycle; the production service binding is deliberately separate."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import threading

from .configuration import _bytes, _publish
from .jobs import _identity, _path


def run_owned(instance, job_id, lock, run_service):
    """Retain the inherited instance lock through service cleanup and final receipt."""
    root = Path(instance)
    directory = root / 'runs/jobs'
    path = _path(directory, job_id)
    record = json.loads(path.read_text())
    stopped = threading.Event()
    stop_file = directory / f'{job_id}.stop'
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    events_lock = threading.Lock()

    def should_stop():
        return stopped.is_set() or stop_file.exists()

    def emit(event):
        data = _bytes(event)
        with events_lock:
            with (directory / f'{job_id}.events.jsonl').open('ab', buffering=0) as stream:
                stream.write(data)
                os.fsync(stream.fileno())

    try:
        record.update(state='running', process=_identity(os.getpid()))
        _publish(path, _bytes(record))
        result = run_service(root, record['request'], record['configuration'],
                             should_stop=should_stop, emit=emit)
        record.update(state='stopped' if should_stop() else 'completed', result=result)
    except BaseException as exc:
        # Service owns cleanup of canonical actor and children before raising.
        record.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            _publish(path, _bytes(record))
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            os.close(lock)
