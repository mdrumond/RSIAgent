"""Restart-safe active-work allowances; elapsed downtime is never charged.

One budget has one execution owner. Checkpoint before dispatch and after each
returned operation; a crash loses only time since the last checkpoint. Retained
remote operations must be observed before dispatching further work.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import os
from pathlib import Path
import time
import threading
import uuid


_current = ContextVar("active_time_budget", default=None)

def current_budget():
    return _current.get()


class ActiveTimeBudget:
    def __init__(self, path, seconds=None, *, clock=time.monotonic, parent=None):
        self.path, self.clock = Path(path), clock
        self._lock = threading.RLock()
        self._started = None
        self.parent = parent
        self._excluded_intervals = []
        self._depth = 0
        if self.path.exists():
            state = json.loads(self.path.read_text())
            self.seconds, self._elapsed = state['seconds'], state['elapsed_seconds']
            if seconds is not None and seconds != self.seconds:
                raise ValueError('retained active-time allowance changed')
        else:
            self.seconds, self._elapsed = seconds, 0.0
        if (type(self.seconds) not in (int, float) or not math.isfinite(self.seconds)
                or self.seconds <= 0):
            raise ValueError('an explicit positive finite time allowance is required')
        if (type(self._elapsed) not in (int, float) or not math.isfinite(self._elapsed)
                or self._elapsed < 0):
            raise ValueError('invalid retained active time')
        self.checkpoint()

    def checkpoint(self):
        with self._lock:
            return self._checkpoint()

    def _checkpoint(self):
        if self._started is not None:
            now = self.clock()
            self._elapsed += max(0.0, now - self._started)
            self._started = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('w') as stream:
                json.dump(dict(schema='active-time-v1', seconds=self.seconds,
                               elapsed_seconds=self._elapsed), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return self._elapsed

    def exclude_interval(self, start, end):
        """Exclude a failed infrastructure call, merging overlapping worker pauses."""
        if not all(math.isfinite(value) for value in (start, end)) or end < start:
            raise ValueError('invalid excluded interval')
        with self._lock:
            self._checkpoint()
            previous = sum(b - a for a, b in self._excluded_intervals)
            intervals = sorted([*self._excluded_intervals, (start, end)])
            merged = []
            for a, b in intervals:
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
                else:
                    merged.append((a, b))
            self._excluded_intervals = merged
            self._elapsed = max(0.0, self._elapsed - (sum(b - a for a, b in merged) - previous))
            self._checkpoint()
        if self.parent is not None:
            self.parent.exclude_interval(start, end)

    def exclude(self, seconds):
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('excluded time must be finite and nonnegative')
        now = self.clock()
        self.exclude_interval(now - seconds, now)

    @property
    def elapsed_seconds(self):
        return self.checkpoint()

    @property
    def remaining_seconds(self):
        remaining = max(0.0, self.seconds - self.checkpoint())
        return min(remaining, self.parent.remaining_seconds) if self.parent else remaining

    @property
    def expired(self):
        return self.remaining_seconds <= 0

    @contextmanager
    def active(self):
        token = _current.set(self)
        with self._lock:
            self._depth += 1
            if self._started is None:
                self._started = self.clock()
        try:
            yield self
        finally:
            with self._lock:
                self._checkpoint()
                self._depth -= 1
                if not self._depth:
                    self._started = None
            _current.reset(token)

    @contextmanager
    def pause(self):
        # Keep one shared clock running and subtract interval unions. Nested
        # pauses and concurrent worker outages must never subtract twice.
        started = self.clock()
        try:
            yield
        finally:
            self.exclude_interval(started, self.clock())
