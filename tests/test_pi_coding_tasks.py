import copy
import json
import time

import pytest

from benchmarks.pi_coding.experiment import final_status, parse_args, run, starter_source
from benchmarks.pi_coding.remote import Validator, driver_source
from benchmarks.pi_coding.budget import Budget
from benchmarks.pi_coding.gateway import Gateway
from benchmarks.pi_coding.tools import CandidateTools
from benchmarks.pi_coding.tasks import (
    MATMUL_SHAPES, contract_digest, matmul_feedback, matmul_inputs,
    matmul_references, task_contract, verify_matmul_outputs,
)


BASE_ARGS = ["--backend", "pi", "--provider", "deepseek", "--provider-smoke-only"]


def test_cli_supports_trailing_flag_and_file_prompts(tmp_path):
    assert parse_args(BASE_ARGS + ["--task", "matmul-add", "Tile A @ B + C"]).prompt == "Tile A @ B + C"
    assert parse_args(BASE_ARGS + ["--prompt", "Handle tails"]).prompt == "Handle tails"
    path = tmp_path / "prompt.txt"
    path.write_text("Tile the kernel.\nHandle the K tail.\n")
    assert parse_args(BASE_ARGS + ["--prompt-file", str(path)]).prompt == path.read_text()
    default = parse_args(BASE_ARGS)
    assert default.task == "vector-add" and default.prompt is None


@pytest.mark.parametrize("arguments", [[""], [" "], ["x" * 16385], ["first", "--prompt", "second"]])
def test_cli_rejects_invalid_or_conflicting_prompts(arguments):
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGS + arguments)


def test_prompt_file_is_not_shell_evaluated(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("Keep literal $(touch UNWANTED) and `commands` in the request.")
    assert "$(touch UNWANTED)" in parse_args(BASE_ARGS + ["--prompt-file", str(path)]).prompt
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGS + ["request", "--prompt-file", str(path)])


def test_matmul_requires_multiple_output_tiles_and_k_chunks():
    contract = task_contract("matmul-add")
    assert contract["tile_mnk"] == [64, 64, 32]
    assert [((m + 63) // 64) * ((n + 63) // 64) for m, n, k in MATMUL_SHAPES] == [1, 12, 30]
    assert [(k + 31) // 32 for m, n, k in MATMUL_SHAPES] == [2, 9, 7]
    assert all(d % 64 for d in MATMUL_SHAPES[-1][:2])
    assert MATMUL_SHAPES[-1][2] % 32
    assert contract_digest("matmul-add") != contract_digest("vector-add")


def test_matmul_reference_checks_addend_edges_and_reduction_tail():
    correct = [[list(row) for row in case] for case in matmul_references()]
    assert verify_matmul_outputs(correct)
    feedback = matmul_feedback(correct)
    assert all(row["passed"] for row in feedback)
    assert [row["output_tiles"] for row in feedback] == [1, 12, 30]
    assert not verify_matmul_outputs(correct[:-1])
    for replacement in (float("nan"), float("inf"), True, 12345.0):
        broken = copy.deepcopy(correct)
        broken[-1][-1][-1] = replacement
        assert not verify_matmul_outputs(broken)
        assert not matmul_feedback(broken)[-1]["passed"]
    m, n, k = MATMUL_SHAPES[-1]
    a, b, c = matmul_inputs(m, n, k, len(MATMUL_SHAPES))
    no_addend = copy.deepcopy(correct)
    for i in range(m):
        for j in range(n):
            no_addend[-1][i][j] -= c[i][j]
    assert not verify_matmul_outputs(no_addend)
    no_k_tail = copy.deepcopy(correct)
    for i in range(m):
        for j in range(n):
            no_k_tail[-1][i][j] -= a[i][-1] * b[-1][j]
    assert not verify_matmul_outputs(no_k_tail)


def test_matmul_driver_uses_trusted_binding_and_selected_operator():
    vector = driver_source("vector-add")
    matrix = driver_source("matmul-add")
    compile(matrix, "matmul-driver", "exec")
    assert '"-lc10", "binding.cpp"' in matrix
    assert "rsi_pi_matmul.matmul_add" in matrix
    assert "rsi_pi_smoke.vector_add" not in matrix
    assert "rsi_pi_smoke.vector_add" in vector
    assert "TODO" in starter_source("matmul-add")
    with pytest.raises(ValueError):
        driver_source("unregistered")


def test_task_metadata_prompt_and_source_are_recorded_without_paid_calls(tmp_path, monkeypatch):
    import benchmarks.pi_coding.experiment as experiment
    monkeypatch.setattr(experiment, "LOCAL", tmp_path / "local")
    env = tmp_path / "providers.env"
    env.write_text("DEEPSEEK_API_KEY=fake-for-offline-test\n")
    env.chmod(0o600)
    args = parse_args(BASE_ARGS + ["--task", "matmul-add", "--env-file", str(env),
                                  "--budget-file", str(tmp_path / "budget.json"), "Prefer compact tiling."])
    seen = []
    monkeypatch.setattr(experiment, "pi_attempt", lambda provider, gateway, tools, prompt, run_dir, trace, **kw: seen.append(prompt))
    assert run(args) == 2  # No candidate validation means no fabricated pass.
    result_path = next((experiment.LOCAL / "runs").glob("*/result.json"))
    result = json.loads(result_path.read_text())
    assert result["task"] == "matmul-add"
    assert result["task_contract_sha256"] == contract_digest("matmul-add")
    assert result["user_prompt"] == "Prefer compact tiling."
    assert result["requests"] == []
    assert "Prefer compact tiling." in seen[0]
    assert "(257,321,193)" in seen[0]
    assert "vector-add" not in seen[0]
    assert (result_path.parent / "prompt.txt").read_text() == seen[0]


def test_matmul_provider_only_never_reports_a_pass(tmp_path):
    (tmp_path / "kernel.cpp").write_text(starter_source("matmul-add"))
    result = Validator(tmp_path, None, task="matmul-add").validate()
    assert result["task"] == "matmul-add"
    assert result["status"] == "infrastructure_unavailable"
    assert result["passed"] is False


def test_final_unvalidated_write_gets_independent_validation(tmp_path):
    (tmp_path / "kernel.cpp").write_text("first")
    validator = Validator(tmp_path, None, task="matmul-add")
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 20)
    tools.execute("write_candidate", {"source": "first"})
    older = tools.execute("validate_candidate", {})
    older.update(status="passed", passed=True)  # A pass applies only to that digest.
    tools.execute("write_candidate", {"source": "changed-final-revision"})
    trace = []
    assert final_status(tools, validator, trace) == "infrastructure_unavailable"
    assert len(validator.results) == 2
    assert trace[-1]["type"] == "host_final_validation"
    assert final_status(tools, validator, trace) == "infrastructure_unavailable"
    assert len(validator.results) == 2  # Do not repeat an existing verdict.


def test_final_validation_respects_exhausted_cycle_budget(tmp_path):
    (tmp_path / "kernel.cpp").write_text("first")
    validator = Validator(tmp_path, None)
    tools = CandidateTools(tmp_path, validator, deadline=time.monotonic() + 20)
    for _ in range(3):
        tools.execute("validate_candidate", {})
    tools.execute("write_candidate", {"source": "unvalidated-final-revision"})
    assert final_status(tools, validator, []) == "unverified"
    assert len(validator.results) == 3


def test_streamed_provider_error_is_recorded_and_redacted(tmp_path, monkeypatch):
    import requests
    original = requests.post
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_lines(self):
            yield b'data: {"type":"error","code":"insufficient_quota","message":"No credits; fake-provider-key"}'
            yield b''
    monkeypatch.setattr("benchmarks.pi_coding.gateway.requests.post", lambda *a, **kw: Response())
    with Gateway("openai", "fake-provider-key", Budget(tmp_path / "budget.json"), "offline") as gate:
        original(gate.url + "/responses", json={"model": "gpt-5.6-sol", "input": [
            {"role": "user", "content": "hi"}], "stream": True},
            headers={"Authorization": "Bearer " + gate.token}, timeout=5)
    assert gate.records[0]["status"] == "provider_error"
    assert gate.records[0]["provider_error"]["code"] == "insufficient_quota"
    assert "fake-provider-key" not in gate.records[0]["provider_error"]["message"]
