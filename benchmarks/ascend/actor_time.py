"""Controller-owned actor work clocks; preparation never spends work allocation."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from threading import RLock
import json
import math
import time

from optimizer.jobs import atomic, canonical
from .time_budget import deadline_after

_current = ContextVar('actor_work_budget', default=None)
_lease = ContextVar('actor_work_lease', default=None)
_preparing = ContextVar('actor_preparation', default=False)
_linked = ContextVar('linked_actor_work_budgets', default=())


class WorkDeadline:
    """Live monotonic deadline, including pauses after it was handed off."""
    def __init__(self, remaining, clock=time.monotonic):
        self.remaining, self.clock = remaining, clock

    def endpoint(self):
        remaining = self.remaining()
        return self.clock() + remaining if remaining > 0 else float('-inf')
    def __sub__(self, other): return self.endpoint() - other
    def __rsub__(self, other): return other - self.endpoint()
    def __lt__(self, other): return self.endpoint() < other
    def __le__(self, other): return self.endpoint() <= other
    def __gt__(self, other): return self.endpoint() > other
    def __ge__(self, other): return self.endpoint() >= other


class ActorWorkBudget:
    """Persist union work intervals and separate per-actor telemetry.

    The campaign owner opens this ledger; its worker threads share the instance.
    Historical wall-budget usage stays charged. Crashed preparation is free;
    an unclosed working interval is conservatively charged until recovery.
    """
    def __init__(self, path, seconds, *, clock=time.monotonic, wall_clock=time.time):
        deadline_after(seconds, clock=clock)
        self.path, self.clock, self.wall_clock = Path(path), clock, wall_clock
        state = json.loads(self.path.read_bytes()) if self.path.exists() else {}
        self.grants = state.get('grants', {})
        if not any(math.isclose(state.get('max_seconds', seconds), x, rel_tol=1e-12)
                   for x in (seconds, seconds + sum(self.grants.values()))):
            raise ValueError('attempt time allocation changed')
        self.seconds = state.get('max_seconds', seconds)
        self.elapsed = state.get('elapsed_seconds', 0)
        if state.get('active_since') is not None:
            self.elapsed += max(0, wall_clock() - state['active_since'])
        self.actors = state.get('actors', {})
        self.preparation = state.get('preparation_seconds', 0)
        self.wall = state.get('wall_seconds', 0)
        self.workers, self.lock = {}, RLock()
        self.external_active, self.linked_budgets = False, set()
        self.tick = clock()
        self.deadline = WorkDeadline(self.remaining, clock)
        self._save()

    def _update(self):
        now = self.clock(); delta = max(0, now-self.tick); self.tick = now
        self.wall += delta
        if self.external_active or any(w.working for w in self.workers.values()): self.elapsed += delta
        if any(not w.working for w in self.workers.values()): self.preparation += delta
        for w in self.workers.values():
            row = self.actors.setdefault(w.name, {'work_seconds': 0, 'preparation_seconds': 0})
            row['work_seconds' if w.working else 'preparation_seconds'] += delta
            row.update(state='working' if w.working else 'preparing', reason=w.reason)

    def _save(self):
        for budget in self.linked_budgets:
            with budget.lock:
                budget._update()
                budget.external_active = any(w.working and budget in w.tags for w in self.workers.values())
                budget._save()
        for w in self.workers.values():
            self.actors.setdefault(w.name, {'work_seconds': 0, 'preparation_seconds': 0}).update(
                state='working' if w.working else ('ready' if w.reason is None else 'preparing'), reason=w.reason, allocation_seconds=w.limit)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic(self.path, canonical(dict(max_seconds=self.seconds, elapsed_seconds=self.elapsed,
            active_since=self.wall_clock() if self.external_active or any(w.working for w in self.workers.values()) else None,
            policy='actor-work-v1', observed_epoch=self.wall_clock(), preparation_seconds=self.preparation, wall_seconds=self.wall,
            actors=self.actors, **({'grants': self.grants} if self.grants else {}))))

    def remaining(self):
        with self.lock:
            self._update()
            return max(0, self.seconds-self.elapsed)

    def report(self):
        with self.lock:
            self._update(); self._save()
            return dict(policy='actor-work-v1', elapsed_seconds=self.elapsed,
                remaining_seconds=max(0, self.seconds-self.elapsed), preparation_seconds=self.preparation,
                wall_seconds=self.wall, actors=self.actors)

    @contextmanager
    def bind(self):
        token = _current.set(self)
        try: yield self
        finally: _current.reset(token)

    @contextmanager
    def actor(self, name, seconds):
        worker = WorkLease(self, name, seconds)
        with self.lock:
            self._update(); self.workers[id(worker)] = worker
            self.linked_budgets.update(worker.tags); self._save()
        token = _lease.set(worker)
        try:
            yield worker
        except Exception as error:
            with self.lock:
                self._update(); self.actors.setdefault(name, {}).update(state='blocked', reason=str(error) or type(error).__name__)
            raise
        else:
            with self.lock:
                self._update(); self.actors.setdefault(name, {}).update(state='finished', reason=None)
        finally:
            _lease.reset(token)
            with self.lock:
                self.workers.pop(id(worker)); self._save()

    def close(self):
        with self.lock:
            self._update(); self._save()


class WorkLease:
    def __init__(self, budget, name, seconds):
        self.budget, self.name = budget, name
        self.limit = budget.actors.get(name, {}).get('allocation_seconds', seconds)
        self.baseline = 0
        self.depth, self.pauses, self.reason = 0, 0, 'actor initialization'
        self.tags = _linked.get()
        self.deadline = WorkDeadline(self.remaining, budget.clock)

    @property
    def working(self): return self.depth > 0 and self.pauses == 0

    def remaining(self):
        remaining = self.budget.remaining()
        used = self.budget.actors.get(self.name, {}).get('work_seconds', 0)-self.baseline
        return max(0, min(remaining, self.limit-used))

    @contextmanager
    def scope(self, *, work=False, reason=None):
        b = self.budget
        with b.lock:
            b._update(); old = self.reason
            if work: self.depth += 1
            else: self.pauses += 1
            self.reason = reason; b._save()
        try: yield
        finally:
            with b.lock:
                b._update()
                if work: self.depth -= 1
                else: self.pauses -= 1
                self.reason = old; b._save()


def current_budget(): return _current.get()


def live_deadline(default):
    worker = _lease.get()
    return worker.deadline if worker is not None else default


def preparation_active(): return _preparing.get()


@contextmanager
def preparation(reason):
    token = _preparing.set(True)
    worker = _lease.get()
    try:
        if worker is None:
            budget = current_budget()
            if budget is None:
                yield
            else:
                from threading import get_ident
                with budget.actor('controller-preparation:' + str(get_ident()), budget.seconds) as controller:
                    with controller.scope(reason=reason): yield
        else:
            with worker.scope(reason=reason): yield
    finally:
        _preparing.reset(token)


def actor_attempt(callback):
    @wraps(callback)
    def call(self, request):
        budget = current_budget()
        if budget is None: return callback(self, request)
        seconds = min(self.max_seconds, request.remaining_seconds) if request.remaining_seconds is not None else self.max_seconds
        name = str(request.branch_root)
        attempt = Path(request.branch_root) / 'manual-execution.json'
        if attempt.exists():
            name += ':' + json.loads(attempt.read_bytes())['attempt_id']
        with budget.actor(name, seconds):
            return callback(self, request)
    return call


def actor_work(callback):
    @wraps(callback)
    def call(*args, **kwargs):
        worker = _lease.get()
        if worker is None: return callback(*args, **kwargs)
        with worker.scope(work=True, reason='actor work'):
            return callback(*args, **kwargs)
    return call


def dispatch_actor(callback, name, seconds, *args, **kwargs):
    """Plain executor callbacks are work; production adapters own finer scopes."""
    budget = current_budget()
    if budget is None or getattr(callback, 'manages_actor_time', False):
        return callback(*args, **kwargs)
    with budget.actor(name, seconds) as worker:
        with worker.scope(work=True, reason='actor executor'):
            return callback(*args, **kwargs)


def prepare_until_ready(callback, *, stopped=lambda: False, sleep=time.sleep):
    """Retry only identified transient admission failures; never relax checks."""
    from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
    transient = {'device is not idle', 'device health query unavailable',
                 'bounded device health query failed',
                 }
    while True:
        if stopped(): raise TimeoutError('operator stopped preparation')
        try:
            return callback()
        except ProjectInfrastructureError as error:
            if str(error) not in transient:
                raise
            with preparation(str(error)):
                sleep(1)


@contextmanager
def linked_work_budget(budget):
    """An attributable repair budget follows the same Actor intervals."""
    token = _linked.set(tuple(dict.fromkeys((*_linked.get(), budget))))
    try: yield
    finally:
        _linked.reset(token)
        owner = current_budget()
        if owner is not None:
            with owner.lock:
                owner._update()
                if not any(budget in worker.tags for worker in owner.workers.values()):
                    owner._save()
                    owner.linked_budgets.discard(budget)
