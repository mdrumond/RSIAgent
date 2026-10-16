"""Real Pi runtime tests with deterministic, free provider/validator substitutes."""

import hashlib
import json
from pathlib import Path
import time

import pytest

from benchmarks.pi_coding.budget import Budget
from benchmarks.pi_coding.experiment import compact_observation, parse_args, pi_attempt
from benchmarks.pi_coding.gateway import Gateway
from benchmarks.pi_coding.remote import Validator
from benchmarks.pi_coding.tools import CandidateTools


class ScriptedValidator(Validator):
    def __init__(self, root, outcomes):
        super().__init__(root, None)
        self.outcomes = outcomes

    def validate(self):
        status = self.outcomes[len(self.results)]
        row = {"source_sha256": hashlib.sha256((self.root / "kernel.cpp").read_bytes()).hexdigest(),
               "status": status, "passed": status == "passed",
               "diagnostic": "irrelevant warning\n" * 100 + "kernel.cpp:3: error: fix the kernel\nbad();\n   ^\n"}
        self.results.append(row)
        return row


def fake_stream(source, finish_reason="tool_calls"):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def iter_lines(self):
            tool = {"name": "submit_candidate", "arguments": json.dumps({"source": source})}
            for chunk in [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
                    {"index": 0, "id": "call_submit", "type": "function", "function": tool}]}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            ]:
                chunk.update(id="offline", object="chat.completion.chunk", created=0, model="deepseek-flash")
                yield b"data: " + json.dumps(chunk).encode()
                yield b""
            yield b"data: [DONE]"
            yield b""
    return Response()


@pytest.mark.parametrize("outcomes", [
    ["compile_failed", "runtime_failed", "passed"],
    ["incorrect", "incorrect", "incorrect"],
    ["passed"],
    ["infrastructure_unavailable"],
])
def test_efficient_pi_has_one_request_and_one_validation_per_revision(tmp_path, monkeypatch, outcomes):
    executable = Path(__file__).resolve().parents[1] / "integrations/pi/node_modules/.bin/pi"
    if not executable.exists():
        pytest.skip("install pinned Pi to test its real lifecycle")
    captured = []
    sources = [f"IMPLEMENTATION_{i}();" for i in range(len(outcomes))]
    def provider(*_args, **kwargs):
        index = len(captured)
        assert index < len(outcomes), "Pi attempted an unnecessary model call"
        captured.append(kwargs["json"])
        return fake_stream(sources[index])
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", provider)
    (tmp_path / "kernel.cpp").write_text("STARTER_ONLY")
    validator = ScriptedValidator(tmp_path, outcomes)
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 25)
    trace = []
    prompt = "TASK_AND_USER_REQUEST\nStarter source:\nSTARTER_ONLY"
    with Gateway("deepseek", "offline-key", Budget(tmp_path / "budget.json"), "offline", max_calls=3) as gateway:
        pi_attempt("deepseek", gateway, tools, prompt, tmp_path, trace)
    assert gateway.calls == len(outcomes)
    assert tools.writes == len(outcomes) == len(validator.results)
    assert (tmp_path / "kernel.cpp").read_text() == sources[-1]
    for index, payload in enumerate(captured):
        assert [tool["function"]["name"] for tool in payload["tools"]] == ["submit_candidate"]
        text = json.dumps(payload["messages"])
        assert "TASK_AND_USER_REQUEST" in text
        if index:
            assert sources[index - 1] in text
            assert "STARTER_ONLY" not in text
            for old_source in sources[:index - 1]:
                assert old_source not in text
    assert sum(e.get("type") == "agent_settled" for e in trace) == len(outcomes)


def test_atomic_submission_obeys_both_budgets(tmp_path):
    (tmp_path / "kernel.cpp").write_text("initial")
    validator = ScriptedValidator(tmp_path, ["compile_failed"] * 3)
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 10)
    for i in range(3):
        result = tools.execute("submit_candidate", {"source": f"revision {i}"})
        assert result["status"] == "compile_failed"
    with pytest.raises(ValueError):
        tools.execute("submit_candidate", {"source": "fourth"})
    assert (tmp_path / "kernel.cpp").read_text() == "revision 2"
    assert len(validator.results) == tools.writes == 3


def test_diagnostic_compaction_keeps_actual_errors_and_raw_evidence():
    raw = {"status": "compile_failed", "passed": False,
           "diagnostic": "warning\n" * 500 + "kernel.cpp:19: error: unknown symbol\n    bad();\n    ^\n"}
    compact = compact_observation(raw)
    assert "unknown symbol" in compact["diagnostic"]
    assert "bad();" in compact["diagnostic"]
    assert "warning" not in compact["diagnostic"]
    assert len(raw["diagnostic"]) > len(compact["diagnostic"])
    runtime = compact_observation({"status": "runtime_failed", "diagnostic": "start" + "x" * 8000 + "ROOT_CAUSE"})
    assert runtime["diagnostic"].startswith("start")
    assert runtime["diagnostic"].endswith("ROOT_CAUSE")
    assert len(runtime["diagnostic"]) < 2100


def test_efficient_default_and_native_opt_in():
    base = ["--backend", "pi", "--provider", "deepseek", "--provider-smoke-only"]
    assert parse_args(base).pi_workflow == "efficient"
    assert parse_args(base + ["--pi-workflow", "native"]).pi_workflow == "native"


def test_truncation_uses_a_slot_and_retries_without_partial_history(tmp_path, monkeypatch):
    executable = Path(__file__).resolve().parents[1] / "integrations/pi/node_modules/.bin/pi"
    if not executable.exists():
        pytest.skip("install pinned Pi to test its real lifecycle")
    captured = []
    def provider(*_args, **kwargs):
        index = len(captured)
        assert index < 3, "no hidden recovery request may exceed the three-call budget"
        captured.append(kwargs["json"])
        return fake_stream("TRUNCATED_PARTIAL" if index == 0 else f"valid_{index}();",
                           "length" if index == 0 else "tool_calls")
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", provider)
    (tmp_path / "kernel.cpp").write_text("STARTER")
    validator = ScriptedValidator(tmp_path, ["compile_failed", "passed"])
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 25)
    trace = []
    with Gateway("deepseek", "fake", Budget(tmp_path / "budget.json"), "offline", max_calls=3) as gateway:
        pi_attempt("deepseek", gateway, tools, "Task\nStarter source:\nSTARTER", tmp_path, trace)
    assert gateway.calls == 3
    assert tools.writes == len(validator.results) == 2
    assert (tmp_path / "kernel.cpp").read_text() == "valid_2();"
    assert "TRUNCATED_PARTIAL" not in json.dumps(captured[1]["messages"])
    assert "TRUNCATED_PARTIAL" not in json.dumps(captured[2]["messages"])
    assert sum(e.get("type") == "pi_revision_truncated" for e in trace) == 1
