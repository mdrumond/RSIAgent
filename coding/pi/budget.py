"""Durable, conservative spend reservations shared by both coding backends."""

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import uuid


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path: Path, limit: float = 3.0, *, window=None, window_limit=None):
        if not math.isfinite(limit) or not 0 < limit <= 10:
            raise ValueError("experiment budget must be in (0, 10] USD")
        self.path, self.limit = Path(path), limit
        if window is not None and (not isinstance(window, str) or not window or
                type(window_limit) not in (int, float) or not math.isfinite(window_limit) or
                not 0 < window_limit <= limit):
            raise ValueError("invalid incremental budget window")
        self.window, self.window_limit = window, window_limit
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(self):
        with self._state() as state:
            return json.loads(json.dumps(state))

    @contextmanager
    def _state(self):
        # Keep an interrupted request's entire reservation. Never retry it free.
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {
                "limit_usd": self.limit, "requests": [],
            }
            if state["limit_usd"] != self.limit:
                raise ValueError("existing budget has a different limit")
            yield state
            temporary = self.path.with_suffix(".tmp")
            with temporary.open("w") as stream:
                json.dump(state, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)

    def reserve(self, request: dict, *, output_tokens: int, rates: tuple[float, float], label: str):
        # UTF-8 bytes + framing allowance deliberately overestimate text tokens.
        input_bound = len(json.dumps(request, ensure_ascii=False).encode()) + 4096
        amount = (input_bound * rates[0] + output_tokens * rates[1]) / 1_000_000
        with self._state() as state:
            if sum(r["charged_bound_usd"] for r in state["requests"]) + amount > self.limit:
                raise BudgetExceeded("request would exceed the shared smoke budget")
            if self.window is not None:
                windows = state.setdefault("windows", {})
                if windows.setdefault(self.window, self.window_limit) != self.window_limit:
                    raise ValueError("existing incremental window has a different limit")
                charged = sum(r["charged_bound_usd"] for r in state["requests"]
                              if r.get("window") == self.window)
                if charged + amount > self.window_limit:
                    raise BudgetExceeded("request would exceed the incremental smoke budget")
            rid = uuid.uuid4().hex
            state["requests"].append({
                "id": rid, "label": label, "reserved_usd": amount,
                "charged_bound_usd": amount, "usage": None,
                **({"window": self.window} if self.window is not None else {}),
            })
        return rid

    def settle(self, rid: str, usage: dict, rates: tuple[float, float]):
        input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if any(type(n) is not int or n < 0 for n in (input_tokens, output_tokens)):
            return  # Unknown billing keeps the reservation.
        amount = (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
        with self._state() as state:
            row = next(r for r in state["requests"] if r["id"] == rid)
            row.update(usage={"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
                       charged_bound_usd=amount)
            if amount > row["reserved_usd"]:
                # Fail closed on subsequent calls if the assumed bound was wrong.
                row["charged_bound_usd"] = max(amount, self.limit)
                row["bound_violation"] = True


@contextmanager
def allocation_lock(directory):
    """Only one paired comparison may use an allocation at a time."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".allocation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BudgetExceeded("another comparison is using this allocation") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
