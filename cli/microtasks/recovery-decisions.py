import copy
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[1])
lease = root / "lease.json"
frozen = b'{"owner":"harness","token":"frozen-fixture"}\n'
assert lease.read_bytes() == frozen, "actor changed harness ownership"
module = runpy.run_path(str(root / "kernel.py"))
assert lease.read_bytes() == frozen, "import changed harness ownership"

events = [
    {"phase": "start", "operation": "retrieve", "at": 40.0, "actor_elapsed": 1168.0},
    {"phase": "start", "operation": "other", "at": 40.2, "actor_elapsed": 1168.2},
    {"phase": "end", "operation": "retrieve", "at": 41.75, "actor_elapsed": 1169.75},
]
for values, expected in [
    (events, 1.75),
    (list(reversed(events)), 1.75),
    ([dict(events[0], at=200), dict(events[-1], at=202.5)], 2.5),
    ([], None),
    (events[:1], None),
    (events + [dict(events[-1])], None),
    ([dict(events[0], at=50), events[-1]], None),
]:
    before = copy.deepcopy(values)
    result = module["tool_seconds"](values, "retrieve")
    assert result == expected, (result, expected)
    assert values == before, "duration calculation mutated evidence"
assert module["tool_seconds"](events, "other") is None


class Actions:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))
        return record


runtime = {"product": "fixture-a5", "torch": "same", "arch": "fixture"}
base = {"expected": runtime, "observed": dict(runtime), "lease_error": None,
        "current_verified": False, "locally_correct": True,
        "remaining_seconds": 120, "verification_seconds": 30, "events": events}
scenarios = [
    ({"observed": dict(runtime, torch="different"), "current_verified": True},
     "blocked", [("report", "runtime-identity")]),
    ({"observed": dict(runtime, product="other"), "lease_error": "missing-receipt"},
     "blocked", [("report", "runtime-identity")]),
    ({"lease_error": "missing-receipt"}, "blocked", [("report", "harness-lease")]),
    ({"lease_error": "host-fence-unresolved", "current_verified": True},
     "blocked", [("report", "harness-lease")]),
    ({"current_verified": True, "remaining_seconds": 0}, "accept", [("accept",)]),
    ({"remaining_seconds": 29}, "checkpoint", [("checkpoint",)]),
    ({"remaining_seconds": 30}, "validate", [("verify",)]),
    ({}, "validate", [("verify",)]),
    ({"locally_correct": False}, "repair", [("repair",)]),
]
for changes, decision, calls in scenarios:
    case = copy.deepcopy({**base, **changes})
    before = copy.deepcopy(case)
    actions = Actions()
    assert module["next_action"](case, actions) == {"action": decision, "tool_seconds": 1.75}
    assert actions.calls == calls, (decision, actions.calls)
    assert case == before, "decision mutated evidence"
    assert lease.read_bytes() == frozen, "decision changed harness ownership"
print("runtime, lease ownership, measured duration and verification decisions passed")
