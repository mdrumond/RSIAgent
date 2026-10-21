"""Durable elapsed-time admission for resumable Phase 2 campaigns."""
from contextlib import contextmanager
import json
import math
import os
import time
from pathlib import Path

from optimizer.jobs import atomic, canonical


def seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('time budgets must be finite positive seconds')
    return value


class Phase2TimeBudget:
    def __init__(self, root, maximum, practice=None, *, clock=time.monotonic):
        self.path = Path(root) / 'time-budget.json'
        self.maximum = seconds(maximum)
        self.practice = seconds(practice) if practice is not None else None
        self.clock, self.started = clock, None
        self.active_practice = False
        self.practice_started = None
        self.identity = dict(max_seconds=maximum, max_practice_seconds=practice)
        self.state = dict(identity=self.identity, elapsed_seconds=0, practice_seconds=0)
        if self.path.exists():
            self.state = json.loads(self.path.read_bytes())
            if self.state['identity'] != self.identity:
                raise ValueError('retained Phase 2 time budgets changed')
            reservation = self.state.get('inflight_seconds', 0)
            owner = self.state.get('owner_pid')
            alive = False
            if owner is not None:
                try:
                    os.kill(owner, 0)
                    alive = True
                except ProcessLookupError:
                    pass
            if 'inflight_seconds' in self.state and not alive:
                # Unknown interrupted work keeps its full admitted reservation.
                self.state['elapsed_seconds'] += reservation
                if self.state.pop('inflight_practice', False):
                    self.state['practice_seconds'] += reservation
                self.state.pop('inflight_seconds', None)
                self.state.pop('owner_pid', None)
                self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic(self.path, canonical(self.state))

    @property
    def remaining(self):
        running = max(0, self.clock() - self.started) if self.started is not None else 0
        return max(0, self.maximum - self.state['elapsed_seconds'] - running)

    @property
    def practice_remaining(self):
        anchor = self.practice_started if self.practice_started is not None else self.started
        running = max(0, self.clock() - anchor) if anchor is not None and self.active_practice else 0
        return max(0, self.practice - self.state['practice_seconds'] - running)

    def reset_practice(self):
        self.state['practice_seconds'] = 0
        self.save()

    @contextmanager
    def practice_work(self):
        """Charge practice selection and qualification inside a total-time scope."""
        if self.started is None or self.active_practice:
            raise RuntimeError('practice selection requires an active campaign operation')
        self.practice_started = self.clock()
        self.active_practice = True
        self.state['inflight_seconds'] = min(self.remaining, self.practice_remaining)
        self.state['inflight_practice'] = True
        self.save()
        try:
            yield
        finally:
            self.state['practice_seconds'] += max(0, self.clock() - self.practice_started)
            self.practice_started = None
            self.active_practice = False
            self.state['inflight_practice'] = False
            self.state['inflight_seconds'] = self.remaining
            self.save()

    @contextmanager
    def active(self, *, practice=False):
        if self.started is not None or 'inflight_seconds' in self.state:
            raise RuntimeError('time budget operation is already active')
        self.started = self.clock()
        self.active_practice = practice
        self.state['inflight_seconds'] = min(self.remaining, self.practice_remaining) if practice else self.remaining
        self.state['inflight_practice'] = practice
        self.state['owner_pid'] = os.getpid()
        self.save()
        try:
            yield
        finally:
            elapsed = max(0, self.clock() - self.started)
            self.started = None
            self.active_practice = False
            self.state['elapsed_seconds'] += elapsed
            if practice:
                self.state['practice_seconds'] += elapsed
            self.state.pop('inflight_seconds', None)
            self.state.pop('owner_pid', None)
            self.state.pop('inflight_practice', None)
            self.save()
