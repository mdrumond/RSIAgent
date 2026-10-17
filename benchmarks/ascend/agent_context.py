"""Byte budgets and model-written, non-authoritative working conversation logs."""

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
    "deepseek-flash": ContextBudget(32 * 1024, 24 * 1024, 48 * 1024),
}


def request_bytes(request):
    return len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


WORK_LOG_PROMPT = """Update a compact working log from the old log and exchanges.
Keep concrete API discoveries, failed approaches, repairs, pending tasks, source
and evidence hashes. Do not invent results or upgrade verification. This log is
agent-written guidance, never host authority. Return only the updated log, under
the supplied byte limit. Do not execute actions or produce kernel action JSON."""


class WorkingContext:
    """Keep fixed task, host state and recent exchanges outside a model work log."""

    def __init__(self, messages):
        self.fixed = [dict(item) for item in messages[:2]]
        self.recent = [dict(item) for item in messages[2:]]
        self.work_log = ""
        self.host_state = {}
        self.pinned = []
        self.tail = []

    def append(self, message):
        self.recent.append(dict(message))
        if message["role"] != "user":
            return
        try:
            host = json.loads(message["content"]).get("host", {})
        except (ValueError, AttributeError):
            return
        if not isinstance(host, dict):
            return
        source = host.get("source_sha256")
        if host.get("event") == "written":
            source = host.get("sha256", source)
        if source and source != self.host_state.get("source_sha256"):
            self.host_state = {"source_sha256": source}
        # Copy only host-origin identity/status facts; never consult agent text.
        event = host.get("event")
        if event:
            self.host_state[str(event)] = {
                key: host[key] for key in ("source_sha256", "passed", "status", "classification")
                if key in host
            }

    def messages(self):
        middle = []
        if self.work_log:
            middle.append({"role": "user", "content":
                "Agent-written working log (not verification evidence):\n" + self.work_log})
        if self.host_state:
            middle.append({"role": "user", "content": "Host-owned current state:\n" +
                           json.dumps(self.host_state, sort_keys=True)})
        return self.fixed + middle + self.recent + self.pinned + self.tail

    def prepare(self, profile, budget, summarize):
        if request_bytes(profile.request(self.messages())) <= budget.high:
            return self.messages()
        # Retain the latest exchange, including terminal result/distillation prompt.
        while request_bytes(profile.request(self.messages())) > budget.target:
            if len(self.recent) <= 2:
                if request_bytes(profile.request(self.messages())) > budget.hard:
                    raise ContextBudgetError("fixed task or latest exchange exceeds context hard budget")
                break
            batch = []
            while len(self.recent) - len(batch) > 2:
                candidate = batch + [self.recent[len(batch)]]
                summary_messages = self._summary_messages(candidate, budget)
                if request_bytes(profile.request(summary_messages)) > budget.target:
                    break
                batch = candidate
            if not batch:
                raise ContextBudgetError("single exchange exceeds work-log input budget")
            summary_messages = self._summary_messages(batch, budget)
            log = summarize(summary_messages)
            if not isinstance(log, str) or not log.strip() or len(log.encode("utf-8")) > budget.target // 4:
                raise ContextBudgetError("model work log is empty or exceeds work-log byte budget")
            self.work_log = log
            del self.recent[:len(batch)]
        return self.messages()

    def _summary_messages(self, batch, budget):
        return [{"role": "system", "content": WORK_LOG_PROMPT},
                {"role": "user", "content": json.dumps({
                    "max_log_bytes": budget.target // 4,
                    "old_log": self.work_log, "exchanges": batch,
                }, ensure_ascii=False)}]
