"""Retained local monotonic setup measurements; never actor stopping budgets."""
from contextlib import contextmanager
import json
import os
import time


class SetupTelemetry:
    def __init__(self, store):
        self.store = store

    def record(self, category, start, *, success, scope, wall_started=None, error_type=None, actor_billable=False):
        row = dict(kind='setup', category=category, scope=scope,
            elapsed_seconds=time.monotonic() - start, success=success,
            started_monotonic=start, wall_started=wall_started, error_type=error_type,
            actor_billable=actor_billable)
        reference = self.store.put_json(row)
        path = self.store.root / 'setup.jsonl'
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, (json.dumps(dict(reference=reference, **row)) + '\n').encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return reference

    @contextmanager
    def measure(self, category, scope):
        start, wall = time.monotonic(), time.time()
        try:
            yield
        except BaseException as exc:
            self.record(category, start, success=False, scope=scope, wall_started=wall,
                        error_type=type(exc).__name__)
            raise
        else:
            self.record(category, start, success=True, scope=scope, wall_started=wall)
