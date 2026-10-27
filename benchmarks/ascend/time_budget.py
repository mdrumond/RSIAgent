"""Elapsed-time limits shared by production Ascend host loops."""
import math
import time


def deadline_after(seconds, *, clock=None):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("max_seconds must be a finite positive number")
    return (clock or time.monotonic)() + seconds


class AttemptTimeBudget:
    """Persist elapsed work; recover an interrupted interval using wall time.

    Monotonic timestamps stay local. The wall timestamp is only a recovery
    anchor for a process that disappeared before its final checkpoint.
    """
    def __init__(self, path, seconds, *, clock=None, wall_clock=None):
        import json
        from pathlib import Path
        from optimizer.jobs import atomic, canonical
        self.path = Path(path)
        self.clock = clock or time.monotonic
        self.wall_clock = wall_clock or time.time
        deadline_after(seconds, clock=self.clock)
        state = json.loads(self.path.read_text()) if self.path.exists() else {
            "max_seconds": seconds, "elapsed_seconds": 0,
        }
        granted = sum(state.get('grants', {}).values())
        # Sequential grants and summing their receipt values can round differently.
        if not any(math.isclose(state['max_seconds'], allocation, rel_tol=1e-12, abs_tol=1e-12)
                   for allocation in (seconds, seconds + granted)):
            raise ValueError("attempt time allocation changed")
        seconds = state['max_seconds']
        elapsed = state['elapsed_seconds']
        if state.get('active_since') is not None:
            elapsed += max(0, self.wall_clock()-state['active_since'])
        self.elapsed = elapsed
        self.grants = state.get('grants', {})
        self.started = self.clock()
        self.deadline = self.started + max(0, seconds-elapsed)
        self.seconds = seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic, self._canonical = atomic, canonical
        self._save(active_since=self.wall_clock())

    def _save(self, *, active_since=None):
        self._atomic(self.path, self._canonical({
            'max_seconds': self.seconds, 'elapsed_seconds': self.elapsed,
            'active_since': active_since,
            **({'grants': self.grants} if self.grants else {}),
        }))

    def close(self):
        self.elapsed += max(0, self.clock()-self.started)
        self.started = self.clock()
        self._save()
