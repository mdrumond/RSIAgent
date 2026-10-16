"""Persistent Pi lifecycle, revisions, accounting, and paired-run stopping rules."""
import hashlib
import json
from pathlib import Path
import time

import pytest
import requests

from benchmarks.pi_coding.budget import Budget, BudgetExceeded
from benchmarks.pi_coding.experiment import pi_attempt, rsi_attempt, parse_args
from benchmarks.pi_coding.gateway import Gateway, normalize_usage
from benchmarks.pi_coding.remote import Validator
from benchmarks.pi_coding.report import usage_totals
from benchmarks.pi_coding.tools import CandidateTools


class ScriptedValidator(Validator):
    def __init__(self, root, outcomes):
        super().__init__(root, None, max_validations=12)
        self.outcomes = outcomes

    def validate(self):
        status = self.outcomes[len(self.results)]
        result = {"status": status, "passed": status == "passed",
                  "source_sha256": hashlib.sha256((self.root / "kernel.cpp").read_bytes()).hexdigest(),
                  "diagnostic": "kernel.cpp: error: repair the current source"}
        self.results.append(result)
        return result


def stream(tool, args, index, finish="tool_calls"):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": f"call_{index}",
                     "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]}
            for chunk in [{"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                           "usage": {"prompt_tokens": 50, "completion_tokens": 10}}]:
                chunk.update(id=f"test_{index}", object="chat.completion.chunk", created=0, model="deepseek-flash")
                yield b"data: " + json.dumps(chunk).encode()
                yield b""
            yield b"data: [DONE]"
            yield b""
    return Response()


@pytest.mark.parametrize("terminal", ["passed", "infrastructure_unavailable"])
@pytest.mark.usefixtures("pi_runtime")
def test_persistent_real_pi_edits_and_stops_without_closing_request(tmp_path, monkeypatch, terminal):
    captured = []
    actions = [("submit_candidate", {"source": "OLD_SOURCE_0"}),
               ("edit_and_validate", {"old_text": "OLD_SOURCE_0", "new_text": "LATEST_SOURCE_1"}),
               ("edit_and_validate", {"old_text": "LATEST_SOURCE_1", "new_text": "FINAL_SOURCE_2"})]
    def provider(*_, **kwargs):
        i = len(captured)
        assert i < 3, "unexpected paid closing or hidden retry"
        captured.append(kwargs["json"])
        return stream(*actions[i], i)
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", provider)
    (tmp_path / "kernel.cpp").write_text("STARTER")
    validator = ScriptedValidator(tmp_path, ["compile_failed", "incorrect", terminal])
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 30, max_revisions=12)
    trace = []
    with Gateway("deepseek", "fake", Budget(tmp_path / "budget.json", 5), "test", max_calls=30) as gate:
        pi_attempt("deepseek", gate, tools, "TASK\nStarter source:\nSTARTER", tmp_path, trace, workflow="persistent")
    assert gate.calls == tools.writes == len(validator.results) == 3
    assert (tmp_path / "kernel.cpp").read_text() == "FINAL_SOURCE_2"
    assert len(list((tmp_path / "revisions").glob("*.cpp"))) == 3
    assert len(list((tmp_path / "sessions").rglob("*.jsonl"))) == 1
    # Latest complete exchange survives; superseded full-source submission is gone.
    third = captured[2]["messages"]
    assert not any(m.get("role") == "assistant" and m.get("tool_calls", [{}])[0].get("id") == "call_0" for m in third)
    calls = {c["id"] for m in third for c in m.get("tool_calls", [])}
    results = {m["tool_call_id"] for m in third if m.get("role") == "tool"}
    assert calls == results == {"call_1"}
    assert "LATEST_SOURCE_1" in json.dumps(third)
    assert "STARTER" not in json.dumps(third)


def test_targeted_edit_is_atomic_and_obeys_limit(tmp_path):
    (tmp_path / "kernel.cpp").write_text("same same")
    validator = ScriptedValidator(tmp_path, ["passed"])
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 10, max_revisions=1)
    with pytest.raises(ValueError, match="exactly once"):
        tools.execute("edit_and_validate", {"old_text": "same", "new_text": "x"})
    assert tools.writes == 0 and (tmp_path / "kernel.cpp").read_text() == "same same"
    tools.execute("edit_and_validate", {"old_text": "same same", "new_text": "fixed"})
    with pytest.raises(ValueError, match="exhausted"):
        tools.execute("edit_and_validate", {"old_text": "fixed", "new_text": "extra"})
    assert (tmp_path / "kernel.cpp").read_text() == "fixed"


@pytest.mark.usefixtures("pi_runtime")
def test_persistent_truncation_resumes_same_session_without_partial_arguments(tmp_path, monkeypatch):
    captured = []
    def provider(*_, **kwargs):
        i = len(captured)
        captured.append(kwargs["json"])
        assert i < 2
        return stream("submit_candidate", {"source": "PARTIAL" if i == 0 else "fixed"}, i,
                      finish="length" if i == 0 else "tool_calls")
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", provider)
    (tmp_path / "kernel.cpp").write_text("starter")
    validator = ScriptedValidator(tmp_path, ["passed"])
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 30, max_revisions=12)
    trace = []
    with Gateway("deepseek", "fake", Budget(tmp_path / "budget.json", 5), "test", max_calls=30) as gate:
        pi_attempt("deepseek", gate, tools, "TASK\nStarter source:\nstarter", tmp_path, trace, workflow="persistent")
    assert gate.calls == 2 and tools.writes == 1
    assert "PARTIAL" not in json.dumps(captured[1])
    assert len(list((tmp_path / "sessions").rglob("*.jsonl"))) == 1
    assert sum(e.get("type") == "pi_revision_truncated" for e in trace) == 1


def test_usage_subsets_and_unknowns():
    usage = normalize_usage({"input_tokens": 100, "output_tokens": 40,
                             "input_tokens_details": {"cached_tokens": 80},
                             "output_tokens_details": {"reasoning_tokens": 30}})
    result = usage_totals([{"usage": usage}, {"usage": None}])
    assert result == {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140,
                      "cached_input_tokens": 80, "reasoning_tokens": 30,
                      "unknown_requests": 1, "unknown_breakdown_requests": 0}


def test_budget_is_durable_and_separate_from_old_smoke(tmp_path):
    old = Budget(tmp_path / "old.json", 3)
    old.reserve({}, output_tokens=1, rates=(1, 1), label="old")
    before = old.path.read_bytes()
    new = Budget(tmp_path / "new.json", 5)
    new.reserve({}, output_tokens=100000, rates=(1, 40), label="generation")
    with pytest.raises(BudgetExceeded):
        new.reserve({}, output_tokens=100000, rates=(1, 40), label="retry")
    assert old.path.read_bytes() == before
    assert len(new.snapshot()["requests"]) == 1


def responses_stream(index):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            reasoning = {"type": "reasoning", "id": f"rs_{index}", "summary": [], "encrypted_content": f"opaque_{index}"}
            call = {"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}", "name": "submit_candidate",
                    "arguments": json.dumps({"source": f"source_{index}"}), "status": "completed"}
            response = {"id": f"resp_{index}", "status": "completed", "output": [reasoning, call],
                        "usage": {"input_tokens": 100, "output_tokens": 40,
                                  "input_tokens_details": {"cached_tokens": 20},
                                  "output_tokens_details": {"reasoning_tokens": 30}}}
            events = [{"type": "response.created", "response": {"id": f"resp_{index}"}}]
            for i, item in enumerate([reasoning, call]):
                events += [{"type": "response.output_item.added", "output_index": i, "item": item},
                           {"type": "response.output_item.done", "output_index": i, "item": item}]
            events.append({"type": "response.completed", "response": response})
            for event in events:
                yield b"data: " + json.dumps(event).encode()
                yield b""
    return Response()


@pytest.mark.usefixtures("pi_runtime")
def test_persistent_real_pi_responses_preserves_reasoning_and_usage(tmp_path, monkeypatch):
    captured = []
    def provider(*_, **kwargs):
        i = len(captured)
        assert i < 2
        captured.append(kwargs["json"])
        return responses_stream(i)
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", provider)
    (tmp_path / "kernel.cpp").write_text("starter")
    validator = ScriptedValidator(tmp_path, ["compile_failed", "passed"])
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 30, max_revisions=12)
    with Gateway("openai", "fake", Budget(tmp_path / "budget.json", 5), "test", max_calls=30, max_output_tokens=16384) as gate:
        pi_attempt("openai", gate, tools, "TASK\nStarter source:\nstarter", tmp_path, [], workflow="persistent")
    assert tools.writes == gate.calls == 2
    assert captured[0]["max_output_tokens"] == 16384
    assert captured[0]["parallel_tool_calls"] is False
    assert any(item.get("type") == "reasoning" and item.get("encrypted_content") == "opaque_0" for item in captured[1]["input"])
    totals = usage_totals(gate.records)
    assert totals["total_tokens"] == 280
    assert totals["reasoning_tokens"] == 60
    assert totals["cached_input_tokens"] == 40


def test_rsi_responses_adapter_handles_truncation_then_pass(tmp_path, monkeypatch):
    original_post = requests.post
    captured = []
    class Response:
        status_code = 200
        def __init__(self, index): self.index = index
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def json(self):
            return {"status": "incomplete" if self.index == 0 else "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text":
                        json.dumps({"action": "write_source", "source": "fixed"})}]}],
                    "usage": {"input_tokens": 10, "output_tokens": 20}}
    def provider(url, **kwargs):
        if url.startswith("http://127.0.0.1:"):
            return original_post(url, **kwargs)
        assert url.endswith("/responses")
        captured.append(kwargs["json"])
        return Response(len(captured) - 1)
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", provider)
    (tmp_path / "kernel.cpp").write_text("starter")
    validator = ScriptedValidator(tmp_path, ["passed"])
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 15, max_revisions=12)
    trace = []
    with Gateway("openai", "fake", Budget(tmp_path / "budget.json", 5), "test", max_calls=30) as gate:
        rsi_attempt("openai", gate, tools, "TASK\nStarter source:\nstarter", trace, api="responses")
    assert gate.calls == 2 and tools.writes == 1
    assert validator.results[-1]["passed"]
    assert sum(e.get("type") == "rsi_revision_truncated" for e in trace) == 1


def test_paired_run_stops_on_failed_credit_preflight(tmp_path, monkeypatch):
    import benchmarks.pi_coding.compare as compare
    monkeypatch.setattr(compare, "runtime_preflight", lambda: None)
    monkeypatch.setattr(compare, "LOCAL", tmp_path)
    monkeypatch.setattr(compare.Validator, "validate", lambda _: {"passed": True, "status": "passed"})
    monkeypatch.setattr(compare, "provider_preflight", lambda *args: {"passed": False, "error": {"message": "No credits"}})
    monkeypatch.setattr(compare, "run_experiment", lambda _: pytest.fail("must not generate after account failure"))
    connection = tmp_path / "connection.json"
    connection.write_text("{}")
    assert compare.main(["--budget-usd", "10", "--connection", str(connection)]) == 2
    report = json.loads(next((tmp_path / "comparisons").glob("*/comparison.json")).read_text())
    assert report["status"] == "blocked" and report["arms"] == []
    assert report["budget_bounds_usd"] == {"rsi": 0, "pi": 0}


@pytest.mark.usefixtures("pi_runtime")
def test_persistent_revision_limit_extends_past_three_and_stops(tmp_path, monkeypatch):
    count = 0
    def provider(*_, **kwargs):
        nonlocal count
        assert count < 4
        count += 1
        return stream("submit_candidate", {"source": f"source_{count}"}, count)
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", provider)
    (tmp_path / "kernel.cpp").write_text("starter")
    validator = ScriptedValidator(tmp_path, ["compile_failed"] * 4)
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 20, max_revisions=4)
    with Gateway("deepseek", "fake", Budget(tmp_path / "budget.json", 5), "test", max_calls=30) as gate:
        pi_attempt("deepseek", gate, tools, "TASK\nStarter source:\nstarter", tmp_path, [], workflow="persistent")
    assert count == tools.writes == len(validator.results) == 4


@pytest.mark.usefixtures("pi_runtime")
def test_persistent_request_cap_rejects_extra_call_before_provider(tmp_path, monkeypatch):
    from benchmarks.pi_coding.rpc import PiError
    calls = []
    def provider(*_, **kwargs):
        calls.append(kwargs["json"])
        assert len(calls) == 1
        return stream("submit_candidate", {"source": "broken"}, 1)
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", provider)
    (tmp_path / "kernel.cpp").write_text("starter")
    validator = ScriptedValidator(tmp_path, ["compile_failed"])
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 20, max_revisions=12)
    with Gateway("deepseek", "fake", Budget(tmp_path / "budget.json", 5), "test", max_calls=1) as gate:
        with pytest.raises(PiError):
            pi_attempt("deepseek", gate, tools, "TASK\nStarter source:\nstarter", tmp_path, [], workflow="persistent")
    assert gate.stop_reason == "request_limit" and len(calls) == 1
    assert tools.writes == 1


def test_expired_deadline_prevents_edit(tmp_path):
    (tmp_path / "kernel.cpp").write_text("starter")
    tools = CandidateTools(tmp_path, Validator(tmp_path, None), deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        tools.execute("edit_and_validate", {"old_text": "starter", "new_text": "changed"})
    assert (tmp_path / "kernel.cpp").read_text() == "starter"


def test_resume_provider_switch_preserves_existing_budget_and_prompt(tmp_path, monkeypatch):
    import benchmarks.pi_coding.compare as compare
    monkeypatch.setattr(compare, "runtime_preflight", lambda: None)
    monkeypatch.setattr(compare, "LOCAL", tmp_path)
    monkeypatch.setattr(compare.Validator, "validate", lambda _: {"passed": True, "status": "passed"})
    seen = []
    def preflight(env, budget, directory, provider):
        seen.append((provider, budget.snapshot()))
        return {"passed": False, "error": {"message": "offline test stop"}}
    monkeypatch.setattr(compare, "provider_preflight", preflight)
    directory = tmp_path / "comparisons" / "existing"
    directory.mkdir(parents=True)
    budget = Budget(directory / "rsi-budget.json", 5)
    rid = budget.reserve({}, output_tokens=64, rates=(8, 40), label="failed-openai-preflight")
    report = {"status": "blocked", "arms": [], "total_limit_usd": 10, "prompt": "original task",
              "comparison": "existing", "board_preflight": {"passed": True},
              "provider_preflight": {"passed": False, "error": {"message": "no credits"}}}
    (directory / "comparison.json").write_text(json.dumps(report))
    connection = tmp_path / "connection.json"
    connection.write_text("{}")
    assert compare.main(["--resume", "existing", "--provider", "deepseek", "--connection", str(connection)]) == 2
    assert seen[0][0] == "deepseek"
    assert seen[0][1]["requests"][0]["id"] == rid
    saved = json.loads((directory / "comparison.json").read_text())
    assert saved["prompt"] == "original task"
    assert len(saved["previous_provider_preflights"]) == 1
    assert saved["budget_bounds_usd"]["rsi"] > 0
    assert saved["model"] == "deepseek-flash"


def test_deepseek_cache_accounting_uses_provider_hit_counter():
    from benchmarks.pi_coding.gateway import normalize_chat_usage
    usage = normalize_chat_usage({"prompt_tokens": 100, "completion_tokens": 20,
                                  "prompt_cache_hit_tokens": 64, "prompt_cache_miss_tokens": 36}, "deepseek")
    totals = usage_totals([{"usage": usage}])
    assert totals["total_tokens"] == 120 and totals["cached_input_tokens"] == 64
    assert totals["reasoning_tokens"] == 0 and totals["unknown_breakdown_requests"] == 0


def test_reference_rerun_keeps_both_ledgers_and_identical_prompt(tmp_path, monkeypatch):
    import benchmarks.pi_coding.compare as compare
    monkeypatch.setattr(compare, 'runtime_preflight', lambda: None)
    monkeypatch.setattr(compare, 'LOCAL', tmp_path)
    monkeypatch.setattr(compare.Validator, 'validate', lambda _: {'passed': True, 'status': 'passed'})
    monkeypatch.setattr(compare, 'provider_preflight', lambda *args: {'passed': True})
    old = tmp_path / 'comparisons' / 'previous'
    old.mkdir(parents=True)
    (old / 'comparison.json').write_text(json.dumps({'status': 'completed_with_failures', 'total_limit_usd': 10}))
    ids = {}
    for backend in ('rsi', 'pi'):
        ids[backend] = Budget(old / (backend + '-budget.json'), 5).reserve({}, output_tokens=100, rates=(1, 1), label='prior')
    seen = []
    def run(args):
        seen.append(args)
        assert Budget(args.budget_file, 5).snapshot()['requests'][0]['id'] == ids[args.backend]
        return {'backend': args.backend, 'run': args.backend, 'status': 'compile_failed', 'stop_reason': 'revision_limit', 'requests': [], 'validations': [], 'writes': 0, 'seconds': 0}
    monkeypatch.setattr(compare, 'run_experiment', run)
    connection = tmp_path / 'connection.json'
    connection.write_text('{}')
    prompt = tmp_path / 'references.txt'
    prompt.write_text('Same task with identical reference examples.')
    assert compare.main(['--provider', 'deepseek', '--reuse-budget-from', 'previous', '--connection', str(connection), '--prompt-file', str(prompt), '--no-knowledge']) == 2
    assert len(seen) == 2 and seen[0].prompt == seen[1].prompt == prompt.read_text()
    assert not list(tmp_path.glob('comparisons/deepseek-*/rsi-budget.json'))
    for backend in ids:
        assert Budget(old / (backend + '-budget.json'), 5).snapshot()['requests'][0]['id'] == ids[backend]


def test_reference_report_distinguishes_phase_tokens_from_allocation_cost():
    from benchmarks.pi_coding.report import markdown_report
    report = markdown_report({'status': 'completed_with_failures', 'budget_reused_from': 'previous',
                              'phase_cost_bounds_usd': {'rsi': 0.02, 'pi': 0.03},
                              'interruption_note': 'Prior Pi run was intentionally interrupted.'})
    assert 'token counts cover this phase only' in report
    assert 'rsi=0.020000' in report and 'pi=0.030000' in report
    assert 'intentionally interrupted' in report


@pytest.fixture
def pi_runtime():
    executable = Path(__file__).resolve().parents[1] / "integrations/pi/node_modules/.bin/pi"
    if not executable.exists():
        pytest.skip("install the pinned Pi package to test the real RPC runtime")
