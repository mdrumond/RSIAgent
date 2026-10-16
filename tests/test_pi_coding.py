import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import requests

from benchmarks.pi_coding.budget import Budget, BudgetExceeded
from benchmarks.pi_coding.experiment import pi_attempt, starter_source
from benchmarks.pi_coding.gateway import Gateway
from benchmarks.pi_coding.remote import LENGTHS, RemoteAgent, RemoteUnavailable, Validator, inputs, validation_script, verify_outputs
from benchmarks.pi_coding.rpc import PiError, PiRPC
from benchmarks.pi_coding.tools import CandidateTools


def test_budget_reservations_survive_unknown_outcome_and_settle(tmp_path):
    path = tmp_path / "budget.json"
    budget = Budget(path, 0.01)
    rid = budget.reserve({}, output_tokens=100, rates=(1, 1), label="first")
    with pytest.raises(BudgetExceeded):
        Budget(path, 0.01).reserve({}, output_tokens=10000, rates=(1, 1), label="retry")
    budget.settle(rid, {"prompt_tokens": 10, "completion_tokens": 20}, (1, 1))
    assert json.loads(path.read_text())["requests"][0]["charged_bound_usd"] == 0.00003
    budget.reserve({}, output_tokens=100, rates=(1, 1), label="next")
    with pytest.raises(ValueError):
        Budget(path, 3).reserve({}, output_tokens=1, rates=(1, 1), label="reset")


def test_budget_unknown_usage_and_bound_violation_fail_closed(tmp_path):
    budget = Budget(tmp_path / "budget.json")
    rid = budget.reserve({}, output_tokens=1, rates=(1, 1), label="x")
    budget.settle(rid, {}, (1, 1))
    assert json.loads(budget.path.read_text())["requests"][0]["usage"] is None
    budget.settle(rid, {"prompt_tokens": 4_000_000, "completion_tokens": 1}, (1, 1))
    with pytest.raises(BudgetExceeded):
        budget.reserve({}, output_tokens=1, rates=(1, 1), label="y")


def test_gateway_rejects_before_provider_and_pins_generation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not contact provider")
    original = requests.post
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", forbidden)
    with Gateway("openai", "not-a-real-key", Budget(tmp_path / "budget.json", 0.001), "test") as gate:
        data = {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 999999, "service_tier": "priority"}
        assert gate.prepare(data)["service_tier"] == "default"
        assert gate.prepare(data)["max_completion_tokens"] == 8192
        r = original(gate.url + "/chat/completions", json=data,
                     headers={"Authorization": "Bearer " + gate.token}, timeout=5)
        assert r.status_code == 402
        assert gate.calls == 0
        data["model"] = "different-model"
        assert original(gate.url + "/chat/completions", json=data,
                        headers={"Authorization": "Bearer " + gate.token}, timeout=5).status_code == 400


def test_responses_gateway_pins_reasoning_budget_and_privacy(tmp_path):
    gateway = Gateway("openai", "fake", Budget(tmp_path / "budget.json"), "test")
    request = gateway.prepare({"model": "gpt-5.6-sol", "input": [{"role": "user", "content": "hi"}],
                               "max_output_tokens": 1000000, "store": True,
                               "reasoning": {"effort": "none"}, "stream": True}, responses=True)
    assert request["reasoning"] == {"effort": "high"}
    assert request["max_output_tokens"] == 8192
    assert request["store"] is False
    assert "stream_options" not in request
    assert "messages" not in request


def test_responses_usage_is_settled(tmp_path, monkeypatch):
    original = requests.post
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_lines(self):
            yield b'event: response.completed'
            yield b'data: {"type":"response.completed","response":{"usage":{"input_tokens":12,"output_tokens":9}}}'
            yield b''
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", lambda *a, **kw: Response())
    budget = Budget(tmp_path / "budget.json")
    with Gateway("openai", "fake", budget, "test") as gate:
        response = original(gate.url + "/responses", json={"model": "gpt-5.6-sol", "input": [
            {"role": "user", "content": "hi"}], "stream": True},
            headers={"Authorization": "Bearer " + gate.token}, timeout=5)
        assert response.status_code == 200
    state = json.loads(budget.path.read_text())
    assert state["requests"][0]["usage"] == {"prompt_tokens": 12, "completion_tokens": 9}
    assert state["requests"][0]["charged_bound_usd"] < state["requests"][0]["reserved_usd"]


def test_container_validation_uses_only_existing_container_and_unique_directory():
    from pathlib import PurePosixPath
    script = validation_script({"container": "existing", "cann_env": "/toolkit/set_env.sh",
                                "python": "/venv/bin/python", "device": 2, "arch": "dav-2201"},
                               PurePosixPath("pi-investigation/unique"))
    assert "docker cp pi-investigation/unique/. existing:/tmp/rsi-pi-unique" in script
    assert "ASCEND_RT_VISIBLE_DEVICES=2" in script
    assert "docker run" not in script
    assert "docker restart" not in script


def test_candidate_tools_scope_and_validation_limit(tmp_path):
    (tmp_path / "kernel.cpp").write_text("old")
    tools = CandidateTools(tmp_path, Validator(tmp_path, None), deadline=time.monotonic() + 20)
    with pytest.raises(ValueError):
        tools.execute("write_candidate", {"source": "new", "path": "../escape"})
    for _ in range(3):
        tools.execute("write_candidate", {"source": "new"})
        result = tools.execute("validate_candidate", {})
        assert result["status"] == "infrastructure_unavailable"
        assert result["passed"] is False
    with pytest.raises(ValueError):
        tools.execute("write_candidate", {"source": "fourth"})
    with pytest.raises(ValueError):
        tools.execute("validate_candidate", {})
    (tmp_path / "kernel.cpp").unlink()
    (tmp_path / "kernel.cpp").symlink_to(tmp_path / "validation.json")
    with pytest.raises(ValueError):
        tools.execute("read_candidate", {})


def test_host_verification_rejects_nan_wrong_shapes_and_wrong_values():
    rows = [[a + b for a, b in zip(*inputs(n))] for n in LENGTHS]
    assert verify_outputs(rows)
    assert not verify_outputs(rows[:-1])
    rows[0][0] = float("nan")
    assert not verify_outputs(rows)
    rows[0][0] = 0
    assert not verify_outputs(rows)


def test_rpc_waits_for_settlement_and_preserves_unicode(tmp_path):
    script = '''import json,sys
c=json.loads(sys.stdin.buffer.readline())
print(json.dumps({"type":"response","id":c["id"],"success":True,"data":{"disposition":"started"}}),flush=True)
print(json.dumps({"type":"agent_end"}),flush=True)
print(json.dumps({"type":"message_end","message":{"role":"assistant","text":"a\\u2028b"}},ensure_ascii=False),flush=True)
print(json.dumps({"type":"agent_settled"}),flush=True)
sys.stdin.buffer.read()
'''
    with PiRPC([sys.executable, "-c", script], cwd=tmp_path, env=dict(os.environ), timeout=2) as rpc:
        events = rpc.prompt("hello")
    assert events[-1]["type"] == "agent_settled"
    assert events[-2]["message"]["text"] == "a\u2028b"
    assert rpc.process.poll() is not None


@pytest.mark.parametrize("script,error", [
    ('import sys; sys.stdin.readline(); print("invalid", flush=True)', PiError),
    ('import sys; sys.stdin.readline()', PiError),
    ('import sys,time; sys.stdin.readline(); time.sleep(20)', TimeoutError),
])
def test_rpc_bad_records_exit_and_deadline_cleanup(tmp_path, script, error):
    with PiRPC([sys.executable, "-c", script], cwd=tmp_path, env=dict(os.environ), timeout=0.1) as rpc:
        with pytest.raises(error):
            rpc.prompt("hello")
    assert rpc.process.poll() is not None


def test_remote_missing_transport_and_ambiguous_submission(tmp_path):
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[-1] == "health":
            return subprocess.CompletedProcess(argv, 0, json.dumps({"status": "ok"}), "")
        raise subprocess.TimeoutExpired(argv, 45)
    agent = RemoteAgent({"access_helper": "/unused/access.py", "remote": "a2"}, tmp_path, runner=runner)
    with pytest.raises(RemoteUnavailable, match="registration"):
        agent.ready()
    with pytest.raises(RemoteUnavailable, match="inspect existing jobs"):
        agent.submit("#!/bin/sh\ntrue\n")
    assert len(calls) == 2  # No retry of ambiguous mutation.
    assert json.loads((tmp_path / "jobs.jsonl").read_text())["status"] == "submitting"


def test_remote_wait_requires_terminal_success_and_zero_exit(tmp_path):
    states = iter([{"status": "running"}, {"status": "succeeded", "result": {"exit_code": 1}}])
    agent = RemoteAgent({"access_helper": "/unused/access.py", "remote": "a2"}, tmp_path,
                        runner=lambda argv, **kw: subprocess.CompletedProcess(argv, 0, json.dumps(next(states)), ""))
    with pytest.raises(RemoteUnavailable, match="exit code"):
        agent.wait("test-id")


def test_remote_deadline_prevents_new_requests(tmp_path):
    calls = []
    agent = RemoteAgent({"access_helper": "/unused/access.py", "remote": "a2"}, tmp_path,
                        deadline=time.monotonic() - 1, runner=lambda *a, **kw: calls.append(a))
    with pytest.raises(RemoteUnavailable, match="deadline"):
        agent.call("health")
    assert calls == []


def test_real_pi_rpc_and_extension_with_fake_provider(tmp_path, monkeypatch):
    """Offline end-to-end Pi tool test; never sends a paid request."""
    executable = Path(__file__).resolve().parents[1] / "integrations/pi/node_modules/.bin/pi"
    if not executable.exists():
        pytest.skip("install the pinned optional Pi package to exercise its real RPC runtime")
    responses = iter([
        {"name": "write_candidate", "arguments": json.dumps({"source": "fixed candidate"})},
        {"name": "validate_candidate", "arguments": "{}"},
        None,
    ])
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_lines(self):
            tool = next(responses)
            delta = {"role": "assistant", "content": "Done"} if tool is None else {
                "role": "assistant", "tool_calls": [{"index": 0, "id": "call_" + tool["name"],
                                                        "type": "function", "function": tool}]}
            for chunk in [
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if tool is None else "tool_calls"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            ]:
                chunk.update(id="test", object="chat.completion.chunk", created=0, model="deepseek-flash")
                yield b"data: " + json.dumps(chunk).encode()
                yield b""
            yield b"data: [DONE]"
            yield b""
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", lambda *a, **kw: Response())
    (tmp_path / "kernel.cpp").write_text(starter_source())
    validator = Validator(tmp_path, None)
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 25)
    trace = []
    with Gateway("deepseek", "fake-only", Budget(tmp_path / "budget.json"), "offline") as gate:
        pi_attempt("deepseek", gate, tools, "Fix the kernel and validate it.", tmp_path, trace, workflow="native")
    assert (tmp_path / "kernel.cpp").read_text() == "fixed candidate"
    assert len(validator.results) == 1
    assert validator.results[0]["status"] == "infrastructure_unavailable"
    assert gate.calls == 3
    assert trace[-1]["type"] == "agent_settled"
