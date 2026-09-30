from dataclasses import replace

import pytest

from benchmarks.a3_model_profiles import (
    DEEPSEEK_BASE_URL,
    OPENROUTER_BASE_URL,
    A3Completion,
    A3ModelProfile,
    complete_a3,
    load_a3_model_profile,
)
from benchmarks.a3_experiments import BackendModel


def test_registered_profiles_have_exact_routes_and_stable_fingerprints():
    gpt = load_a3_model_profile(BackendModel.GPT_5_6_SOL)
    deepseek = load_a3_model_profile(BackendModel.DEEPSEEK_FLASH)

    assert (gpt.model, gpt.base_url, gpt.credential_env) == (
        "openai/gpt-5.6-sol", OPENROUTER_BASE_URL, "OPENROUTER_API_KEY"
    )
    assert (deepseek.model, deepseek.base_url, deepseek.credential_env) == (
        "deepseek-flash", DEEPSEEK_BASE_URL, "DEEPSEEK_API_KEY"
    )
    assert gpt.fingerprint == "e443eb94240d22e6a5baa4f008141eb1dcc2c05dbd1b59a2c82a9ec5633c0cac"
    assert deepseek.fingerprint == "e9b6c97cbaca17596ead9fc2ab470e3df41e6b89e845410cb2871f62b4a94b7b"
    assert not gpt.allow_fallbacks and not deepseek.allow_fallbacks


@pytest.mark.parametrize("alias", ["gpt-5.6-sol", "deepseek-v4-flash", "astra", "deepseek"])
def test_loader_rejects_aliases(alias):
    with pytest.raises(ValueError, match="registered BackendModel"):
        load_a3_model_profile(alias)


def test_profiles_reject_route_or_generation_mutation():
    profile = load_a3_model_profile(BackendModel.DEEPSEEK_FLASH)
    with pytest.raises(ValueError, match="canonical A3 model profile"):
        replace(profile, base_url="https://proxy.invalid")
    with pytest.raises(ValueError, match="canonical A3 model profile"):
        replace(profile, generation={"max_tokens": 1})


def test_fake_transport_captures_exact_provider_specific_requests():
    calls = []

    def transport(*, base_url, api_key, request):
        calls.append({"base_url": base_url, "api_key": api_key, "request": request})
        return "candidate"

    environment = {"OPENROUTER_API_KEY": "gpt-secret", "DEEPSEEK_API_KEY": "ds-secret"}
    for model in (BackendModel.GPT_5_6_SOL, BackendModel.DEEPSEEK_FLASH):
        result = complete_a3(
            "system", "user", model=model, environ=environment, transport=transport
        )
        assert result.text == "candidate"
        assert result.provenance["profile_sha256"] == load_a3_model_profile(model).fingerprint
        assert result.provenance["allow_fallbacks"] is False

    assert calls == [
        {
            "base_url": OPENROUTER_BASE_URL,
            "api_key": "gpt-secret",
            "request": {
                "model": "openai/gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
                "max_tokens": 32768,
                "temperature": 0.0,
                "top_p": 1.0,
                "reasoning_effort": "high",
                "extra_body": {"provider": {
                    "order": ["OpenAI"], "allow_fallbacks": False,
                    "require_parameters": True,
                }},
            },
        },
        {
            "base_url": DEEPSEEK_BASE_URL,
            "api_key": "ds-secret",
            "request": {
                "model": "deepseek-flash",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
                "max_tokens": 32768,
                "top_p": 1.0,
                "reasoning_effort": "high",
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        },
    ]


def test_credentials_are_required_from_the_selected_provider_only():
    transport = lambda **_kwargs: "ok"
    complete_a3(
        "s", "u", model=BackendModel.GPT_5_6_SOL,
        environ={"OPENROUTER_API_KEY": "gpt"}, transport=transport,
    )
    complete_a3(
        "s", "u", model=BackendModel.DEEPSEEK_FLASH,
        environ={"DEEPSEEK_API_KEY": "deepseek"}, transport=transport,
    )
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        complete_a3(
            "s", "u", model=BackendModel.DEEPSEEK_FLASH,
            environ={"OPENROUTER_API_KEY": "wrong-provider"}, transport=transport,
        )


def test_profile_json_rejects_extra_alias_field():
    profile = load_a3_model_profile(BackendModel.DEEPSEEK_FLASH)
    payload = profile.as_dict()
    payload["model_alias"] = "deepseek"
    with pytest.raises(ValueError, match="exact schema"):
        A3ModelProfile.from_mapping(payload)


def test_completion_provenance_is_deeply_immutable_and_caller_isolated():
    caller_owned = {
        "generation": {"max_tokens": 32768, "tags": ["native", "a3"]},
        "route": ["deepseek"],
    }
    completion = A3Completion("candidate", caller_owned)

    caller_owned["generation"]["max_tokens"] = 1
    caller_owned["generation"]["tags"].append("changed")
    caller_owned["route"].append("fallback")
    assert completion.as_dict()["provenance"] == {
        "generation": {"max_tokens": 32768, "tags": ["native", "a3"]},
        "route": ["deepseek"],
    }
    with pytest.raises(TypeError):
        completion.provenance["generation"]["max_tokens"] = 1
    with pytest.raises(AttributeError):
        completion.provenance["generation"]["tags"].append("changed")
    exported = completion.as_dict()
    exported["provenance"]["generation"]["max_tokens"] = 2
    assert completion.provenance["generation"]["max_tokens"] == 32768


def test_transport_request_mutation_cannot_change_recorded_provenance():
    def mutating_transport(*, request, **_kwargs):
        request["reasoning_effort"] = "low"
        request["extra_body"]["thinking"]["type"] = "disabled"
        return "candidate"

    result = complete_a3(
        "system", "user", model=BackendModel.DEEPSEEK_FLASH,
        environ={"DEEPSEEK_API_KEY": "secret"}, transport=mutating_transport,
    )
    assert result.as_dict()["provenance"]["generation"] == {
        "max_tokens": 32768,
        "top_p": 1.0,
        "reasoning_effort": "high",
        "temperature": None,
        "thinking": True,
    }
