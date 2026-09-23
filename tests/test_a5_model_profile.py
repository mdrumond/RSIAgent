import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.model_profile import (
    A5ModelProfile,
    GenerationParameters,
    CANONICAL_MODEL,
    CANONICAL_PROVIDER,
    complete_a5,
    load_a5_model_profile,
)


def canonical_mapping():
    return {
        "profile_id": "openai-gpt-5.6-sol-v1",
        "model": CANONICAL_MODEL,
        "provider": CANONICAL_PROVIDER,
        "allow_fallbacks": False,
        "require_parameters": True,
        "generation": {
            "max_tokens": 32768,
            "temperature": 0.0,
            "top_p": 1.0,
            "reasoning_effort": "high",
            "allow_truncation_retry": False,
        },
    }


def test_checked_in_profile_is_canonical_and_stable():
    profile = load_a5_model_profile()

    assert profile == A5ModelProfile.from_mapping(canonical_mapping())
    assert len(profile.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        profile.model = "another/model"


def test_completion_uses_exact_route_and_records_request_provenance():
    calls = []

    def fake_client(*args, **kwargs):
        calls.append((args, kwargs))
        return "kernel candidate"

    completion = complete_a5(
        "system prompt", "user prompt", history=[{"role": "assistant", "content": "x"}],
        client=fake_client,
    )

    assert completion.text == "kernel candidate"
    assert calls == [(
        (CANONICAL_MODEL, "system prompt", "user prompt"),
        {
            "history": [{"role": "assistant", "content": "x"}],
            "max_tokens": 32768,
            "temperature": 0.0,
            "top_p": 1.0,
            "reasoning_effort": "high",
            "allow_truncation_retry": False,
            "provider_order": [CANONICAL_PROVIDER],
            "provider_allow_fallbacks": False,
            "provider_require_parameters": True,
        },
    )]
    evidence = completion.provenance.as_dict()
    assert evidence["model"] == CANONICAL_MODEL
    assert evidence["provider"] == CANONICAL_PROVIDER
    assert evidence["allow_fallbacks"] is False
    assert evidence["generation"] == canonical_mapping()["generation"]
    with pytest.raises(TypeError):
        completion.provenance.generation["temperature"] = 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", "openai/gpt-5.6-astra", "unsupported A5 model"),
        ("model", "gpt-5.6-sol", "unsupported A5 model"),
        ("provider", "Auto", "unsupported A5 provider"),
        ("allow_fallbacks", True, "fallback must be disabled"),
        ("require_parameters", False, "must require parameter support"),
    ],
)
def test_unsupported_aliases_and_routes_fail_before_client(field, value, message):
    data = canonical_mapping()
    data[field] = value
    calls = []

    with pytest.raises(ValueError, match=message):
        complete_a5("s", "u", profile=A5ModelProfile.from_mapping(data), client=calls.append)
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens", 65536),
        ("temperature", 0.1),
        ("top_p", 0.95),
        ("reasoning_effort", "medium"),
        ("allow_truncation_retry", True),
    ],
)
def test_generation_drift_is_rejected(field, value):
    data = canonical_mapping()
    data["generation"][field] = value
    with pytest.raises(ValueError, match=field):
        A5ModelProfile.from_mapping(data)


def test_profile_file_with_unknown_fields_is_rejected(tmp_path):
    data = canonical_mapping()
    data["model_alias"] = "astra"
    source = tmp_path / "profile.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="extra=.*model_alias"):
        load_a5_model_profile(source)


def test_direct_construction_cannot_bypass_profile_controls():
    with pytest.raises(ValueError, match="temperature"):
        GenerationParameters(32768, 1.0, 1.0, "high")
    with pytest.raises(ValueError, match="unsupported A5 model"):
        replace(load_a5_model_profile(), model="openai/gpt-5.6-astra")
    with pytest.raises(ValueError, match="generation"):
        replace(load_a5_model_profile(), generation=canonical_mapping()["generation"])
    with pytest.raises(FrozenInstanceError):
        load_a5_model_profile().generation.max_tokens = 1000


@pytest.mark.parametrize("truncated", [False, True])
def test_default_client_wire_route_and_controls_stay_fixed(monkeypatch, truncated):
    from llm import client

    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            provider="OpenAI", choices=[SimpleNamespace(
                finish_reason="length" if truncated else "stop",
                message=SimpleNamespace(content="" if truncated else "candidate"),
            )],
        )

    monkeypatch.setattr(client, "_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    result = complete_a5("system", "user")
    assert result.text == ("" if truncated else "candidate")
    assert requests == [{
        "model": CANONICAL_MODEL,
        "messages": [{"role": "system", "content": "system"},
                     {"role": "user", "content": "user"}],
        "max_tokens": 32768, "temperature": 0.0, "top_p": 1.0,
        "extra_body": {
            "reasoning": {"effort": "high"},
            "provider": {"order": ["OpenAI"], "allow_fallbacks": False,
                         "require_parameters": True},
        },
    }]
    assert result.provenance.generation["allow_truncation_retry"] is False
    assert result.provenance.profile_sha256 == load_a5_model_profile().fingerprint
