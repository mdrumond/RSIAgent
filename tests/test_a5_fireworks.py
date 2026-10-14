import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.fireworks import (
    FIREWORKS_BASE_MODEL,
    FIREWORKS_INFERENCE_URL,
    FireworksInferenceProfile,
    FireworksHTTPTransport,
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
    with pytest.raises(TypeError):
        result.provenance["generation"]["temperature"] = 1.0
    with pytest.raises(TypeError):
        result.provenance["model"] = "drift"


def test_inference_requires_credential_before_client_construction():
    with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
        complete_fireworks("system", "user", environment={}, client_factory=lambda **_: None)


def test_http_transport_retries_only_retryable_get_failures():
    import requests

    class Response:
        def __init__(self, status, value=None):
            self.status_code = status
            self.value = value or {"state": "JOB_STATE_RUNNING"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)

        def json(self):
            return self.value

    class Session:
        def __init__(self, values):
            self.values = list(values)
            self.calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            value = self.values.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

    session = Session([requests.Timeout(), Response(429), Response(503), Response(200)])
    transport = FireworksHTTPTransport(
        "secret", session=session, get_retries=3, retry_backoff=0,
    )
    assert transport.request("GET", "/v1/job")["state"] == "JOB_STATE_RUNNING"
    assert session.calls == 4

    post_session = Session([requests.Timeout(), Response(200)])
    transport = FireworksHTTPTransport(
        "secret", session=post_session, get_retries=3, retry_backoff=0,
    )
    with pytest.raises(requests.Timeout):
        transport.request("POST", "/v1/job", json_body={})
    assert post_session.calls == 1


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


@pytest.mark.parametrize(
    ("weights", "error"),
    [([1.0, None, 0.5], "every row"), ([float("inf")] * 3, "finite"),
     ([True] * 3, "numeric")],
)
def test_dataset_validation_enforces_consistent_finite_sample_weights(
    tmp_path, weights, error,
):
    rows = []
    for weight in weights:
        row = {"messages": [{"role": "user", "content": "x"}]}
        if weight is not None:
            row["weight"] = weight
        rows.append(row)
    path = tmp_path / "weights.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        validate_chat_jsonl(path)


def test_dataset_validation_accepts_consistent_root_sample_weights(tmp_path):
    rows = [
        {"weight": value, "messages": [{"role": "user", "content": "x"}]}
        for value in (1.0, 0.0, -0.5)
    ]
    path = tmp_path / "weights.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert validate_chat_jsonl(path).example_count == 3


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (("arguments", "not-json"), "JSON"),
        (("id", ""), "ID"),
        (("name", "undeclared"), "declared"),
    ],
)
def test_dataset_validation_rejects_invalid_tool_calls(tmp_path, mutation, error):
    field, value = mutation
    call = {"id": "call-1", "type": "function", "function": {
        "name": "compile_kernel", "arguments": "{}",
    }}
    if field in call:
        call[field] = value
    else:
        call["function"][field] = value
    row = {
        "tools": [{"type": "function", "function": {"name": "compile_kernel"}}],
        "messages": [{"role": "assistant", "tool_calls": [call]}],
    }
    path = tmp_path / "bad-call.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(3)), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        validate_chat_jsonl(path)


@pytest.mark.parametrize("kind", ["duplicate", "unknown-result", "wrong-name"])
def test_dataset_validation_matches_results_to_unique_pending_calls(tmp_path, kind):
    calls = [{"id": "call-1", "type": "function", "function": {
        "name": "compile_kernel", "arguments": "{}",
    }}]
    if kind == "duplicate":
        calls.append(dict(calls[0]))
    result = {"role": "tool", "tool_call_id": "call-1",
              "name": "compile_kernel", "content": "ok"}
    if kind == "unknown-result":
        result["tool_call_id"] = "missing"
    if kind == "wrong-name":
        result["name"] = "other"
    row = {
        "tools": [{"type": "function", "function": {"name": "compile_kernel"}}],
        "messages": [{"role": "assistant", "tool_calls": calls}, result],
    }
    path = tmp_path / "bad-result.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(3)), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate|pending|name"):
        validate_chat_jsonl(path)


def test_billable_submission_requires_two_explicit_opt_ins(tmp_path):
    dataset = _dataset(tmp_path / "train.jsonl")
    client = FireworksManagedSFT(FakeTransport([]), environment={})
    with pytest.raises(RuntimeError, match="--submit"):
        client.submit(
            dataset, account_id="acct", dataset_id="rsi-data",
            output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=False,
            provenance_path=tmp_path / "provenance.json",
        )
    with pytest.raises(RuntimeError, match="RSI_ALLOW_FIREWORKS_SFT"):
        client.submit(
            dataset, account_id="acct", dataset_id="rsi-data",
            output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=True,
            provenance_path=tmp_path / "provenance.json",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"poll_seconds": -1}, "poll_seconds"),
        ({"poll_seconds": math.inf}, "poll_seconds"),
        ({"poll_seconds": math.nan}, "poll_seconds"),
        ({"max_polls": 0}, "max_polls"),
        ({"max_polls": 1.5}, "max_polls"),
    ],
)
def test_poll_configuration_fails_before_any_provider_mutation(kwargs, message):
    transport = FakeTransport([])
    with pytest.raises(ValueError, match=message):
        FireworksManagedSFT(transport, **kwargs)
    assert transport.calls == []


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
        {"name": output_model, "peftDetails": {"baseModel": FIREWORKS_BASE_MODEL},
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
    assert transport.calls[1]["file_path"].read_bytes() == dataset.read_bytes()
    assert transport.calls[1]["file_path"] != dataset
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


def test_submission_snapshots_exact_bytes_and_rejects_upload_size_drift(tmp_path):
    dataset = _dataset(tmp_path / "train.jsonl")
    original = dataset.read_bytes()
    dataset_name = "accounts/acct/datasets/rsi-data"

    class MutatingTransport(FakeTransport):
        def request(self, method, path, *, json_body=None, file_path=None):
            if len(self.calls) == 0:
                dataset.write_text("mutated after validation\n", encoding="utf-8")
            if file_path is not None:
                assert file_path != dataset
                assert file_path.read_bytes() == original
            return super().request(method, path, json_body=json_body, file_path=file_path)

    transport = MutatingTransport([
        {"name": dataset_name},
        {"id": dataset_name, "bytes": len(original) + 1},
    ])
    client = FireworksManagedSFT(
        transport, environment={"RSI_ALLOW_FIREWORKS_SFT": "1"}, poll_seconds=0,
    )
    with pytest.raises(RuntimeError, match="byte count"):
        client.run(
            dataset, account_id="acct", dataset_id="rsi-data",
            output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=True,
            provenance_path=tmp_path / "provenance.json",
        )
    assert len(transport.calls) == 2


def test_interrupted_poll_persists_identity_and_observe_never_posts(tmp_path):
    dataset = _dataset(tmp_path / "train.jsonl")
    dataset_name = "accounts/acct/datasets/rsi-data"
    output_model = "accounts/acct/models/rsi-v4-flash"
    job_name = "accounts/acct/supervisedFineTuningJobs/rsi-sft"
    fields = FireworksSFTContract().request_fields(dataset_name, output_model)
    provenance = tmp_path / "provenance.json"
    submit_transport = FakeTransport([
        {"name": dataset_name}, {"id": dataset_name, "bytes": dataset.stat().st_size},
        {"name": job_name, "state": "JOB_STATE_PENDING", **fields},
        {"name": job_name, "state": "JOB_STATE_RUNNING", **fields},
    ])
    with pytest.raises(TimeoutError):
        FireworksManagedSFT(
            submit_transport, environment={"RSI_ALLOW_FIREWORKS_SFT": "1"},
            poll_seconds=0, max_polls=1,
        ).run(
            dataset, account_id="acct", dataset_id="rsi-data",
            output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=True,
            provenance_path=provenance,
        )
    provisional = json.loads(provenance.read_text())
    assert provisional["lifecycle"] == "submitted"
    assert provisional["job_name"] == job_name
    assert provisional["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()

    observe_transport = FakeTransport([
        {"name": job_name, "state": "JOB_STATE_COMPLETED", **fields},
        {"name": output_model, "peftDetails": {"baseModel": FIREWORKS_BASE_MODEL}},
    ])
    result = FireworksManagedSFT(observe_transport, poll_seconds=0).observe(
        account_id="acct", job_id="rsi-sft", output_model_id="rsi-v4-flash",
        provenance_path=provenance,
    )
    assert result["lifecycle"] == "completed"
    assert {call["method"] for call in observe_transport.calls} == {"GET"}


@pytest.mark.parametrize("state", ["JOB_STATE_COMPLETED", "JOB_STATE_FAILED"])
def test_final_allowed_get_returns_terminal_state_instead_of_timeout(tmp_path, state):
    dataset = _dataset(tmp_path / "train.jsonl")
    dataset_name = "accounts/acct/datasets/rsi-data"
    output_model = "accounts/acct/models/rsi-v4-flash"
    job_name = "accounts/acct/supervisedFineTuningJobs/rsi-sft"
    fields = FireworksSFTContract().request_fields(dataset_name, output_model)
    responses = [
        {"name": dataset_name}, {"id": dataset_name, "bytes": dataset.stat().st_size},
        {"name": job_name, "state": "JOB_STATE_PENDING", **fields},
        {"name": job_name, "state": state, **fields},
    ]
    if state == "JOB_STATE_COMPLETED":
        responses.append({"name": output_model,
                          "peftDetails": {"baseModel": FIREWORKS_BASE_MODEL}})
    client = FireworksManagedSFT(
        FakeTransport(responses), environment={"RSI_ALLOW_FIREWORKS_SFT": "1"},
        poll_seconds=0, max_polls=1,
    )
    if state == "JOB_STATE_COMPLETED":
        assert client.run(
            dataset, account_id="acct", dataset_id="rsi-data",
            output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=True,
            provenance_path=tmp_path / "result.json",
        )["state"] == state
    else:
        with pytest.raises(RuntimeError, match=state):
            client.run(
                dataset, account_id="acct", dataset_id="rsi-data",
                output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=True,
                provenance_path=tmp_path / "result.json",
            )


def test_run_has_explicit_keyword_only_public_contract():
    parameters = inspect.signature(FireworksManagedSFT.run).parameters.values()
    assert all(parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in parameters)


@pytest.mark.parametrize(
    "model",
    [
        {"name": "accounts/acct/models/rsi-v4-flash"},
        {"name": "accounts/acct/models/rsi-v4-flash",
         "peftDetails": {"baseModel": "accounts/fireworks/models/other"}},
    ],
)
def test_completed_model_requires_exact_peft_base(tmp_path, model):
    dataset = _dataset(tmp_path / "train.jsonl")
    dataset_name = "accounts/acct/datasets/rsi-data"
    output_model = "accounts/acct/models/rsi-v4-flash"
    job_name = "accounts/acct/supervisedFineTuningJobs/rsi-sft"
    fields = FireworksSFTContract().request_fields(dataset_name, output_model)
    transport = FakeTransport([
        {"name": dataset_name}, {"id": dataset_name, "bytes": dataset.stat().st_size},
        {"name": job_name, "state": "JOB_STATE_COMPLETED", **fields}, model,
    ])
    with pytest.raises(RuntimeError, match="PEFT base"):
        FireworksManagedSFT(
            transport, environment={"RSI_ALLOW_FIREWORKS_SFT": "1"}, poll_seconds=0,
        ).run(
            dataset, account_id="acct", dataset_id="rsi-data",
            output_model_id="rsi-v4-flash", job_id="rsi-sft", submit=True,
            provenance_path=tmp_path / "provenance.json",
        )


def test_fireworks_surface_is_exported_from_a5kernels_facade():
    from benchmarks import a5kernels

    assert a5kernels.FireworksManagedSFT is FireworksManagedSFT
    assert a5kernels.FIREWORKS_BASE_MODEL == FIREWORKS_BASE_MODEL


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


def test_cli_observe_existing_uses_get_only_resume_surface(tmp_path, capsys, monkeypatch):
    import run_a5_fireworks_sft as cli

    seen = {}

    class Client:
        def __init__(self, transport, **kwargs):
            seen["transport"] = transport

        def observe(self, **kwargs):
            seen["observe"] = kwargs
            return {"lifecycle": "completed"}

    monkeypatch.setenv("FIREWORKS_API_KEY", "secret")
    monkeypatch.setattr(cli, "FireworksHTTPTransport", lambda key: ("transport", key))
    monkeypatch.setattr(cli, "FireworksManagedSFT", Client)
    provenance = tmp_path / "provenance.json"
    assert cli.main([
        "--observe-existing", "--account-id", "acct", "--job-id", "rsi-sft",
        "--output-model-id", "rsi-v4-flash", "--provenance", str(provenance),
    ]) == 0
    assert seen["transport"] == ("transport", "secret")
    assert seen["observe"] == {
        "account_id": "acct", "job_id": "rsi-sft",
        "output_model_id": "rsi-v4-flash", "provenance_path": provenance,
    }
    assert json.loads(capsys.readouterr().out)["lifecycle"] == "completed"


def test_cli_loads_env_before_account_default_and_credentials(tmp_path, capsys, monkeypatch):
    import run_a5_fireworks_sft as cli

    env_file = tmp_path / ".env"
    env_file.write_text(
        "FIREWORKS_API_KEY=from-file\nFIREWORKS_ACCOUNT_ID=env-account\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RSIAGENT_ENV_FILE", str(env_file))
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_ACCOUNT_ID", raising=False)
    seen = {}

    class Client:
        def __init__(self, transport, **kwargs):
            seen["transport"] = transport

        def observe(self, **kwargs):
            seen["observe"] = kwargs
            return {"lifecycle": "completed"}

    monkeypatch.setattr(cli, "FireworksHTTPTransport", lambda key: ("transport", key))
    monkeypatch.setattr(cli, "FireworksManagedSFT", Client)
    assert cli.main([
        "--observe-existing", "--job-id", "rsi-sft",
        "--output-model-id", "rsi-v4-flash",
        "--provenance", str(tmp_path / "provenance.json"),
    ]) == 0
    assert seen["transport"] == ("transport", "from-file")
    assert seen["observe"]["account_id"] == "env-account"
    assert json.loads(capsys.readouterr().out)["lifecycle"] == "completed"
