"""Closed, provider-specific model routes for A3 kernel experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from benchmarks.a3_experiments import BackendModel


OPENAI_BASE_URL = "https://api.openai.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_PROFILE_DIR = Path(__file__).resolve().parents[1] / "config" / "a3"
_PROFILE_FILES = {
    BackendModel.GPT_5_6_SOL: "openai-gpt-5.6-sol.json",
    BackendModel.DEEPSEEK_FLASH: "deepseek-flash.json",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("A3 model profile requires its exact schema")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class A3Generation:
    max_tokens: int
    top_p: float
    reasoning_effort: str
    temperature: float | None
    thinking: bool

    def __post_init__(self) -> None:
        if (
            type(self.max_tokens) is not int
            or type(self.top_p) is not float
            or type(self.reasoning_effort) is not str
            or (
                self.temperature is not None
                and type(self.temperature) is not float
            )
            or type(self.thinking) is not bool
        ):
            raise ValueError("A3 generation field types must be exact")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "A3Generation":
        _exact_keys(
            value,
            {"max_tokens", "top_p", "reasoning_effort", "temperature", "thinking"},
        )
        return cls(**value)


@dataclass(frozen=True)
class A3ModelProfile:
    profile_id: str
    backend_model: BackendModel
    model: str
    provider: str
    base_url: str
    credential_env: str
    allow_fallbacks: bool
    generation: A3Generation

    def __post_init__(self) -> None:
        if (
            type(self.profile_id) is not str
            or type(self.backend_model) is not BackendModel
            or type(self.model) is not str
            or type(self.provider) is not str
            or type(self.base_url) is not str
            or type(self.credential_env) is not str
            or type(self.allow_fallbacks) is not bool
            or type(self.generation) is not A3Generation
        ):
            raise ValueError("canonical A3 model profile field types must be exact")
        if not isinstance(self.backend_model, BackendModel) or not isinstance(
            self.generation, A3Generation
        ):
            raise ValueError("canonical A3 model profile requires typed fields")
        expected = _CANONICAL_VALUES[self.backend_model]
        if self.as_dict() != expected:
            raise ValueError("profile does not match its canonical A3 model profile")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "A3ModelProfile":
        _exact_keys(
            value,
            {
                "profile_id", "backend_model", "model", "provider", "base_url",
                "credential_env", "allow_fallbacks", "generation",
            },
        )
        try:
            backend_model = BackendModel(value["backend_model"])
        except (TypeError, ValueError) as exc:
            raise ValueError("profile backend_model is not registered") from exc
        return cls(
            profile_id=value["profile_id"],
            backend_model=backend_model,
            model=value["model"],
            provider=value["provider"],
            base_url=value["base_url"],
            credential_env=value["credential_env"],
            allow_fallbacks=value["allow_fallbacks"],
            generation=A3Generation.from_mapping(value["generation"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "backend_model": self.backend_model.value,
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "credential_env": self.credential_env,
            "allow_fallbacks": self.allow_fallbacks,
            "generation": asdict(self.generation),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def request(self, system: str, user: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "reasoning_effort": self.generation.reasoning_effort,
        }
        if self.backend_model is BackendModel.GPT_5_6_SOL:
            request["max_completion_tokens"] = self.generation.max_tokens
        else:
            request["max_tokens"] = self.generation.max_tokens
            request["top_p"] = self.generation.top_p
            request["extra_body"] = {"thinking": {"type": "enabled"}}
        return request


_CANONICAL_VALUES: dict[BackendModel, dict[str, Any]] = {
    BackendModel.GPT_5_6_SOL: {
        "profile_id": "openai-gpt-5.6-sol-v1",
        "backend_model": "openai/gpt-5.6-sol",
        "model": "gpt-5.6-sol",
        "provider": "OpenAI",
        "base_url": OPENAI_BASE_URL,
        "credential_env": "OPENAI_API_KEY",
        "allow_fallbacks": False,
        "generation": {
            "max_tokens": 32768, "top_p": 1.0, "reasoning_effort": "high",
            "temperature": None, "thinking": False,
        },
    },
    BackendModel.DEEPSEEK_FLASH: {
        "profile_id": "deepseek-flash-native-v1",
        "backend_model": "deepseek-flash",
        "model": "deepseek-flash",
        "provider": "DeepSeek",
        "base_url": DEEPSEEK_BASE_URL,
        "credential_env": "DEEPSEEK_API_KEY",
        "allow_fallbacks": False,
        "generation": {
            "max_tokens": 32768, "top_p": 1.0, "reasoning_effort": "high",
            "temperature": None, "thinking": True,
        },
    },
}


def load_a3_model_profile(model: BackendModel) -> A3ModelProfile:
    if not isinstance(model, BackendModel):
        raise ValueError("model must be a registered BackendModel; aliases are rejected")
    path = _PROFILE_DIR / _PROFILE_FILES[model]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load registered A3 model profile: {path}") from exc
    return A3ModelProfile.from_mapping(payload)


@dataclass(frozen=True)
class A3Completion:
    text: str
    completion_tokens: int
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_completion_values(
            self.text, self.completion_tokens, label="completion"
        )
        object.__setattr__(self, "provenance", _deep_freeze(self.provenance))

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "completion_tokens": self.completion_tokens,
            "provenance": _json_value(self.provenance),
        }


@dataclass(frozen=True)
class A3TransportResult:
    text: str
    completion_tokens: int


def _validate_completion_values(
    text: Any, completion_tokens: Any, *, label: str
) -> None:
    if (
        type(text) is not str
        or not text.strip()
        or type(completion_tokens) is not int
        or completion_tokens < 1
    ):
        raise ValueError(
            f"{label} requires non-empty text and positive provider token usage"
        )


def _validated_transport_result(text: Any, completion_tokens: Any) -> A3TransportResult:
    _validate_completion_values(text, completion_tokens, label="transport result")
    return A3TransportResult(text, completion_tokens)


Transport = Callable[..., A3TransportResult]


def _openai_transport(
    *, base_url: str, api_key: str, request: Mapping[str, Any]
) -> A3TransportResult:
    from openai import OpenAI

    response = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
        **dict(request)
    )
    try:
        return _validated_transport_result(
            response.choices[0].message.content or "",
            response.usage.completion_tokens,
        )
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("provider response is missing completion token usage") from exc


def complete_a3(
    system: str,
    user: str,
    *,
    model: BackendModel,
    environ: Mapping[str, str] | None = None,
    transport: Transport | None = None,
) -> A3Completion:
    profile = load_a3_model_profile(model)
    environment = os.environ if environ is None else environ
    api_key = environment.get(profile.credential_env, "")
    if not api_key:
        raise ValueError(f"missing required credential {profile.credential_env}")
    request = profile.request(system, user)
    result = (transport or _openai_transport)(
        base_url=profile.base_url, api_key=api_key, request=request
    )
    if not isinstance(result, A3TransportResult):
        raise TypeError("transport must return A3TransportResult with provider token usage")
    return A3Completion(
        text=result.text,
        completion_tokens=result.completion_tokens,
        provenance={
            "profile_id": profile.profile_id,
            "profile_sha256": profile.fingerprint,
            "model": profile.model,
            "provider": profile.provider,
            "base_url": profile.base_url,
            "credential_env": profile.credential_env,
            "allow_fallbacks": profile.allow_fallbacks,
            "generation": asdict(profile.generation),
        },
    )


__all__ = [
    "A3Completion", "A3Generation", "A3ModelProfile", "A3TransportResult", "DEEPSEEK_BASE_URL",
    "OPENAI_BASE_URL", "complete_a3", "load_a3_model_profile",
]
