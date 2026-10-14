import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.fireworks import (
    FIREWORKS_BASE_MODEL,
    FIREWORKS_INFERENCE_URL,
    FireworksInferenceProfile,
    FireworksManagedSFT,
    FireworksSFTContract,
    complete_fireworks,
    load_fireworks_profile,
    validate_chat_jsonl,
)


def _dataset(path: Path) -> Path:
    rows = [
        {"messages": [
            {"role": "system", "content": "Write a Catlass DSL kernel."},
            {"role": "user", "content": f"Implement case {index}."},
            {"role": "assistant", "content": f"kernel {index}"},
        ]}
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, path, *, json_body=None, file_path=None):
        self.calls.append({
            "method": method,
            "path": path,
            "json_body": json_body,
            "file_path": file_path,
        })
        return self.responses.pop(0)


def test_checked_in_inference_profile_is_exact_and_drift_fails(tmp_path):
    profile = load_fireworks_profile()
    assert profile == FireworksInferenceProfile(
        profile_id="fireworks-deepseek-v4-flash-v1",
        model=FIREWORKS_BASE_MODEL,
        base_url=FIREWORKS_INFERENCE_URL,
        credential_env="FIREWORKS_API_KEY",
        max_tokens=32768,
        temperature=0.0,
        top_p=1.0,
    )
    assert len(profile.fingerprint) == 64

    payload = profile.as_dict()
    payload["model"] = "accounts/fireworks/models/deepseek-v4-flash-0731"
    source = tmp_path / "drift.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact DeepSeek V4 Flash base"):
        load_fireworks_profile(source)


def test_inference_uses_direct_openai_compatible_route_and_ignored_key():
    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            seen["request"] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="candidate")
            )])

    result = complete_fireworks(
        "system", "user",
        history=[{"role": "assistant", "content": "prior"}],
        environment={"FIREWORKS_API_KEY": "top-secret"},
        client_factory=FakeClient,
    )
    assert result.text == "candidate"
    assert seen["client"] == {
        "api_key": "top-secret", "base_url": FIREWORKS_INFERENCE_URL,
    }
    assert seen["request"] == {
        "model": FIREWORKS_BASE_MODEL,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "prior"},
            {"role": "user", "content": "user"},
        ],
        "max_tokens": 32768,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    assert "top-secret" not in json.dumps(result.provenance)


def test_inference_requires_credential_before_client_construction():
    with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
        complete_fireworks("system", "user", environment={}, client_factory=lambda **_: None)


def test_sft_contract_is_fixed_and_seed_is_explicitly_local_only():
    contract = FireworksSFTContract()
    assert contract.request_fields("dataset", "output") == {
        "baseModel": FIREWORKS_BASE_MODEL,
        "dataset": "dataset",
        "outputModel": "output",
        "epochs": 1,
        "learningRate": 1e-5,
        "maxContextLength": 32768,
        "loraRank": 32,
    }
    assert contract.seed == 42
    assert contract.seed_api_support is False
    for change, message in [
        ({"lora_rank": 16}, "rank"),
        ({"epochs": 2}, "epochs"),
        ({"learning_rate": 2e-5}, "learning rate"),
        ({"max_context_length": 16384}, "context"),
        ({"seed": 7}, "seed"),
    ]:
        with pytest.raises(ValueError, match=message):
            FireworksSFTContract(**change)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"messages": [{"role": "user", "content": "x"}]}] * 2, "at least 3"),
        ([{"messages": "bad"}] * 3, "messages"),
        ([{"messages": [{"role": "developer", "content": "x"}]}] * 3, "role"),
    ],
)
def test_dataset_validation_fails_closed(tmp_path, rows, message):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_chat_jsonl(path)


def test_dataset_validation_returns_non_secret_content_identity(tmp_path):
    path = _dataset(tmp_path / "train.jsonl")
    result = validate_chat_jsonl(path)
    assert result.example_count == 3
    assert result.byte_count == path.stat().st_size
    assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_dataset_validation_accepts_documented_tool_traces_and_weights(tmp_path):
    rows = [{
        "tools": [{
            "type": "function",
            "function": {
                "name": "compile_kernel", "description": "Compile a candidate",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"},
                }},
            },
        }],
        "messages": [
            {"role": "user", "content": "Compile it."},
            {"role": "assistant", "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "compile_kernel",
                             "arguments": "{\"path\":\"kernel.py\"}"},
            }], "weight": 1},
            {"role": "tool", "tool_call_id": "call-1",
             "name": "compile_kernel", "content": "ok", "weight": 0},
            {"role": "assistant", "content": "Compilation passed."},
        ],
    }] * 3
    path = tmp_path / "tools.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert validate_chat_jsonl(path).example_count == 3


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ({"role": "assistant", "tool_calls": []}, "tool_calls"),
        ({"role": "assistant", "tool_calls": [{"type": "function",
          "function": {"name": "compile", "arguments": 3}}]}, "arguments"),
        ({"role": "tool", "content": "ok"}, "tool_call_id or name"),
        ({"role": "tool", "tool_call_id": "call-1"}, "content"),
        ({"role": "assistant", "content": "x", "weight": 0.5}, "weight"),
    ],
)
def test_dataset_validation_rejects_malformed_tool_messages(tmp_path, message, error):
    path = tmp_path / "bad-tools.jsonl"
    row = {"messages": [{"role": "user", "content": "x"}, message]}
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(3)), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        validate_chat_jsonl(path)


def test_dataset_validation_rejects_malformed_top_level_tools(tmp_path):
    path = tmp_path / "bad-tools.jsonl"
    row = {"tools": [{"type": "function", "function": {"name": ""}}],
           "messages": [{"role": "user", "content": "x"}]}
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(3)), encoding="utf-8")
    with pytest.raises(ValueError, match="tools"):
        validate_chat_jsonl(path)


def test_billable_submission_requires_two_explicit_opt_ins(tmp_path):
    dataset = _dataset(tmp_path / "train.jsonl")
    client = FireworksManagedSFT(FakeTransport([]), environment={})
    with pytest.raises(RuntimeError, match="--submit"):
        client.submit(
            dataset, account_id="acct", dataset_id="rsi-data",
            output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=False,
        )
    with pytest.raises(RuntimeError, match="RSI_ALLOW_FIREWORKS_SFT"):
        client.submit(
            dataset, account_id="acct", dataset_id="rsi-data",
            output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=True,
        )


def test_complete_mocked_managed_sft_lifecycle_and_provenance(tmp_path):
    dataset = _dataset(tmp_path / "train.jsonl")
    dataset_name = "accounts/acct/datasets/rsi-data"
    output_model = "accounts/acct/models/rsi-v4-flash"
    job_name = "accounts/acct/supervisedFineTuningJobs/rsi-sft"
    transport = FakeTransport([
        {"name": dataset_name, "exampleCount": "3", "format": "CHAT"},
        {"id": dataset_name, "filename": "train.jsonl", "bytes": dataset.stat().st_size},
        {
            "name": job_name, "state": "JOB_STATE_PENDING",
            "dataset": dataset_name, "outputModel": output_model,
            **FireworksSFTContract().request_fields(dataset_name, output_model),
        },
        {
            "name": job_name, "state": "JOB_STATE_RUNNING",
            "dataset": dataset_name, "outputModel": output_model,
            **FireworksSFTContract().request_fields(dataset_name, output_model),
        },
        {
            "name": job_name, "state": "JOB_STATE_COMPLETED",
            "dataset": dataset_name, "outputModel": output_model,
            "baseModel": FIREWORKS_BASE_MODEL, "epochs": 1,
            "learningRate": 1e-5, "maxContextLength": 32768, "loraRank": 32,
            "completedTime": "2026-10-14T12:00:00Z",
        },
        {"name": output_model, "baseModel": FIREWORKS_BASE_MODEL,
         "createTime": "2026-10-14T12:00:00Z"},
    ])
    provenance_path = tmp_path / "provenance.json"
    result = FireworksManagedSFT(
        transport, environment={"RSI_ALLOW_FIREWORKS_SFT": "1"}, poll_seconds=0,
    ).run(
        dataset, account_id="acct", dataset_id="rsi-data",
        output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=True,
        provenance_path=provenance_path,
    )

    assert result["state"] == "JOB_STATE_COMPLETED"
    assert result["job_name"] == job_name
    assert result["trained_model"] == output_model
    assert result["checkpoint_id"] == output_model
    assert result["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert result["contract"]["seed"] == 42
    assert result["contract"]["seed_api_support"] is False
    assert json.loads(provenance_path.read_text()) == result
    assert [call["method"] for call in transport.calls] == [
        "POST", "POST", "POST", "GET", "GET", "GET",
    ]
    assert transport.calls[0]["json_body"] == {
        "datasetId": "rsi-data",
        "dataset": {"displayName": "rsi-data", "exampleCount": "3",
                    "userUploaded": {}, "format": "CHAT"},
    }
    assert transport.calls[1]["file_path"] == dataset
    assert transport.calls[2]["json_body"] == {
        "dataset": dataset_name,
        "displayName": "rsi-sft",
        "outputModel": output_model,
        "baseModel": FIREWORKS_BASE_MODEL,
        "epochs": 1,
        "learningRate": 1e-5,
        "maxContextLength": 32768,
        "loraRank": 32,
    }


@pytest.mark.parametrize("state", ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"])
def test_terminal_failure_never_resolves_a_model(tmp_path, state):
    dataset = _dataset(tmp_path / "train.jsonl")
    dataset_name = "accounts/acct/datasets/rsi-data"
    output_model = "accounts/acct/models/rsi-v4-flash"
    job_name = "accounts/acct/supervisedFineTuningJobs/rsi-sft"
    transport = FakeTransport([
        {"name": dataset_name}, {"id": dataset_name},
        {"name": job_name, "state": state, "dataset": dataset_name,
         "outputModel": output_model, "baseModel": FIREWORKS_BASE_MODEL,
         "epochs": 1, "learningRate": 1e-5, "maxContextLength": 32768,
         "loraRank": 32},
    ])
    client = FireworksManagedSFT(
        transport, environment={"RSI_ALLOW_FIREWORKS_SFT": "1"}, poll_seconds=0,
    )
    with pytest.raises(RuntimeError, match=state):
        client.run(
            dataset, account_id="acct", dataset_id="rsi-data",
            output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=True,
            provenance_path=tmp_path / "provenance.json",
        )
    assert len(transport.calls) == 3


def test_provider_echo_drift_is_rejected_before_polling(tmp_path):
    dataset = _dataset(tmp_path / "train.jsonl")
    dataset_name = "accounts/acct/datasets/rsi-data"
    output_model = "accounts/acct/models/rsi-v4-flash"
    transport = FakeTransport([
        {"name": dataset_name}, {"id": dataset_name},
        {"name": "accounts/acct/supervisedFineTuningJobs/rsi-sft",
         "state": "JOB_STATE_PENDING", "dataset": dataset_name,
         "outputModel": output_model, "baseModel": FIREWORKS_BASE_MODEL,
         "epochs": 1, "learningRate": 1e-5, "maxContextLength": 32768,
         "loraRank": 16},
    ])
    with pytest.raises(RuntimeError, match="loraRank"):
        FireworksManagedSFT(
            transport, environment={"RSI_ALLOW_FIREWORKS_SFT": "1"}, poll_seconds=0,
        ).run(
            dataset, account_id="acct", dataset_id="rsi-data",
            output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=True,
            provenance_path=tmp_path / "provenance.json",
        )


def test_cli_defaults_to_non_billable_dry_run(tmp_path, capsys):
    from run_a5_fireworks_sft import main

    dataset = _dataset(tmp_path / "train.jsonl")
    assert main([
        str(dataset), "--dataset-id", "rsi-data",
        "--output-model-id", "rsi-v4-flash", "--job-id", "rsi-sft",
        "--provenance", str(tmp_path / "never-written.json"),
    ]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["mode"] == "dry-run"
    assert preview["base_model"] == FIREWORKS_BASE_MODEL
    assert preview["contract"]["lora_rank"] == 32
    assert not (tmp_path / "never-written.json").exists()
