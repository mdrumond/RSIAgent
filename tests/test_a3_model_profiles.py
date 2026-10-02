from dataclasses import replace
from types import SimpleNamespace

import pytest

import benchmarks.a3_model_profiles as model_module

from benchmarks.a3_model_profiles import (
    DEEPSEEK_BASE_URL,
    OPENAI_BASE_URL,
    A3Completion,
    A3ModelProfile,
    A3ProviderResponseError,
    A3TransportResult,
    complete_a3,
    load_a3_model_profile,
)
from benchmarks.a3_experiments import BackendModel


def test_registered_profiles_have_exact_routes_and_stable_fingerprints():
    gpt = load_a3_model_profile(BackendModel.GPT_5_6_SOL)
    deepseek = load_a3_model_profile(BackendModel.DEEPSEEK_FLASH)

    assert (gpt.model, gpt.base_url, gpt.credential_env) == (
        "gpt-5.6-sol", OPENAI_BASE_URL, "OPENAI_API_KEY"
    )
    assert (deepseek.model, deepseek.base_url, deepseek.credential_env) == (
        "deepseek-flash", DEEPSEEK_BASE_URL, "DEEPSEEK_API_KEY"
    )
    assert gpt.fingerprint == "e58ce9883286271c419262dd66d88d8d8c1bcab3b5da683eadc43891bf6b4ce8"
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens", 32768.0),
        ("top_p", True),
        ("temperature", False),
        ("thinking", 1),
    ],
)
def test_profile_generation_rejects_python_equal_wrong_types(field, value):
    payload = load_a3_model_profile(BackendModel.GPT_5_6_SOL).as_dict()
    payload["generation"][field] = value

    with pytest.raises(ValueError, match="generation field types"):
        A3ModelProfile.from_mapping(payload)


def test_profile_rejects_boolean_equivalent_fallback_control():
    payload = load_a3_model_profile(BackendModel.GPT_5_6_SOL).as_dict()
    payload["allow_fallbacks"] = 0

    with pytest.raises(ValueError, match="profile field types"):
        A3ModelProfile.from_mapping(payload)


def test_fake_transport_captures_exact_provider_specific_requests():
    calls = []

    def transport(*, base_url, api_key, request):
        calls.append({"base_url": base_url, "api_key": api_key, "request": request})
        return A3TransportResult("candidate", 17)

    environment = {"OPENAI_API_KEY": "gpt-secret", "DEEPSEEK_API_KEY": "ds-secret"}
    for model in (BackendModel.GPT_5_6_SOL, BackendModel.DEEPSEEK_FLASH):
        result = complete_a3(
            "system", "user", model=model, environ=environment, transport=transport
        )
        assert result.text == "candidate"
        assert result.completion_tokens == 17
        assert result.provenance["profile_sha256"] == load_a3_model_profile(model).fingerprint
        assert result.provenance["allow_fallbacks"] is False

    assert calls == [
        {
            "base_url": OPENAI_BASE_URL,
            "api_key": "gpt-secret",
            "request": {
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
                "max_completion_tokens": 32768,
                "reasoning_effort": "high",
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
    transport = lambda **_kwargs: A3TransportResult("ok", 1)
    complete_a3(
        "s", "u", model=BackendModel.GPT_5_6_SOL,
        environ={"OPENAI_API_KEY": "gpt"}, transport=transport,
    )
    complete_a3(
        "s", "u", model=BackendModel.DEEPSEEK_FLASH,
        environ={"DEEPSEEK_API_KEY": "deepseek"}, transport=transport,
    )
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        complete_a3(
            "s", "u", model=BackendModel.DEEPSEEK_FLASH,
            environ={"OPENAI_API_KEY": "wrong-provider"}, transport=transport,
        )


def test_direct_openai_ignores_openrouter_credentials():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        complete_a3(
            "s", "u", model=BackendModel.GPT_5_6_SOL,
            environ={"OPENROUTER_API_KEY": "must-not-be-used"},
            transport=lambda **_kwargs: A3TransportResult("ok", 1),
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
    completion = A3Completion("candidate", 9, caller_owned)

    caller_owned["generation"]["max_tokens"] = 1
    caller_owned["generation"]["tags"].append("changed")
    caller_owned["route"].append("fallback")
    assert completion.as_dict() == {
        "text": "candidate",
        "completion_tokens": 9,
        "provenance": {
        "generation": {"max_tokens": 32768, "tags": ["native", "a3"]},
        "route": ["deepseek"],
        },
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
        return A3TransportResult("candidate", 12)

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


@pytest.mark.parametrize(
    "transport_result",
    ["candidate", None, A3TransportResult("candidate", 0), A3TransportResult("candidate", True)],
)
def test_live_completion_rejects_missing_or_invalid_provider_usage(transport_result):
    with pytest.raises(A3ProviderResponseError) as raised:
        complete_a3(
            "system", "user", model=BackendModel.DEEPSEEK_FLASH,
            environ={"DEEPSEEK_API_KEY": "secret"},
            transport=lambda **_kwargs: transport_result,
        )
    assert raised.value.code in {
        "invalid-content-or-usage", "invalid-transport-result",
    }


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_live_completion_rejects_empty_or_whitespace_provider_content(text):
    with pytest.raises(A3ProviderResponseError) as raised:
        complete_a3(
            "system", "user", model=BackendModel.DEEPSEEK_FLASH,
            environ={"DEEPSEEK_API_KEY": "secret"},
            transport=lambda **_kwargs: A3TransportResult(text, 7),
        )
    assert raised.value.code == "invalid-content-or-usage"


def test_openai_compatible_response_rejects_missing_content(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        usage=SimpleNamespace(completion_tokens=23),
    )
    completions = SimpleNamespace(create=lambda **_request: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setitem(
        __import__("sys").modules, "openai",
        SimpleNamespace(OpenAI=lambda **_route: client),
    )

    with pytest.raises(A3ProviderResponseError) as raised:
        model_module._openai_transport(
            base_url=DEEPSEEK_BASE_URL,
            api_key="secret",
            request={"model": "deepseek-flash"},
        )
    assert raised.value.code == "invalid-content-or-usage"


def test_openai_compatible_response_parses_provider_completion_usage(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="candidate"))],
        usage=SimpleNamespace(completion_tokens=23),
    )
    completions = SimpleNamespace(create=lambda **_request: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setitem(
        __import__("sys").modules, "openai",
        SimpleNamespace(OpenAI=lambda **_route: client),
    )

    assert model_module._openai_transport(
        base_url=DEEPSEEK_BASE_URL, api_key="secret", request={"model": "deepseek-flash"}
    ) == A3TransportResult("candidate", 23)
