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
    "deepseek-flash": ContextBudget(128 * 1024, 80 * 1024, 192 * 1024),
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
                # Reserve a bounded correction message on every attempt, including
                # the first, so retries can reuse the exact original batch.
                reserved = summary_messages + [{"role": "user", "content": "x" * 512}]
                if request_bytes(profile.request(reserved)) > budget.target:
                    break
                batch = candidate
            if not batch:
                raise ContextBudgetError("single exchange exceeds work-log input budget")
            summary_messages = self._summary_messages(batch, budget)
            limit = budget.target // 4
            correction = []
            for attempt in range(3):
                log = summarize(summary_messages + correction)
                size = len(log.encode("utf-8")) if isinstance(log, str) else None
                if isinstance(log, str) and log.strip() and size <= limit:
                    break
                measured = f"{size} UTF-8 bytes" if size is not None else "non-text (UTF-8 size unavailable)"
                reason = "empty" if isinstance(log, str) and not log.strip() else "invalid"
                feedback = (f"Previous work log was {reason}: {measured}; limit {limit} UTF-8 bytes. "
                            "Regenerate a non-empty text log from the original input above. "
                            "Return only the updated log within the limit.")
                if attempt == 2:
                    raise ContextBudgetError(f"model work log invalid after 3 attempts: {measured}; "
                                             f"limit {limit} UTF-8 bytes ({reason})")
                correction = [{"role": "user", "content": feedback}]
            self.work_log = log
            del self.recent[:len(batch)]
        return self.messages()

    def _summary_messages(self, batch, budget):
        return [{"role": "system", "content": WORK_LOG_PROMPT},
                {"role": "user", "content": json.dumps({
                    "max_log_bytes": budget.target // 4,
                    "old_log": self.work_log, "exchanges": batch,
                }, ensure_ascii=False)}]
