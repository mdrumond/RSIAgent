"""Historical profile byte-limit metadata and wire-size telemetry."""

from dataclasses import dataclass
import json



class ContextBudgetError(ValueError):
    """A local context limit, not an infrastructure failure or kernel result."""


@dataclass(frozen=True)
class ContextBudget:
    high: int
    target: int
    hard: int

    def __post_init__(self):
        if not all(type(n) is int for n in (self.target, self.high, self.hard)) or not (
            1024 <= self.target < self.high <= self.hard
        ):
            raise ValueError("context budgets require 1024 <= target < high <= hard")


DEFAULT_CONTEXT_BUDGETS = {
    "openai/gpt-5.6-sol": ContextBudget(128 * 1024, 80 * 1024, 192 * 1024),
    "deepseek-flash": ContextBudget(128 * 1024, 80 * 1024, 192 * 1024),
}


def request_bytes(request):
    return len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

