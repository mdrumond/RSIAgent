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


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_PROFILE_DIR = Path(__file__).resolve().parents[1] / "config" / "a3"
_PROFILE_FILES = {
    BackendModel.GPT_5_6_SOL: "openai-gpt-5.6-sol.json",
    BackendModel.DEEPSEEK_FLASH: "deepseek-flash.json",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("A3 model profile requires its exact schema")


@dataclass(frozen=True)
class A3Generation:
    max_tokens: int
    top_p: float
    reasoning_effort: str
    temperature: float | None
    thinking: bool

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
            "max_tokens": self.generation.max_tokens,
            "top_p": self.generation.top_p,
            "reasoning_effort": self.generation.reasoning_effort,
        }
        if self.backend_model is BackendModel.GPT_5_6_SOL:
            request["temperature"] = self.generation.temperature
            request["extra_body"] = {
                "provider": {
                    "order": [self.provider],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
            }
        else:
            request["extra_body"] = {"thinking": {"type": "enabled"}}
        return request


_CANONICAL_VALUES: dict[BackendModel, dict[str, Any]] = {
    BackendModel.GPT_5_6_SOL: {
        "profile_id": "openai-gpt-5.6-sol-v1",
        "backend_model": "openai/gpt-5.6-sol",
        "model": "openai/gpt-5.6-sol",
        "provider": "OpenAI",
        "base_url": OPENROUTER_BASE_URL,
        "credential_env": "OPENROUTER_API_KEY",
        "allow_fallbacks": False,
        "generation": {
            "max_tokens": 32768, "top_p": 1.0, "reasoning_effort": "high",
            "temperature": 0.0, "thinking": False,
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
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


Transport = Callable[..., str]


def _openai_transport(*, base_url: str, api_key: str, request: Mapping[str, Any]) -> str:
    from openai import OpenAI

    response = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
        **dict(request)
    )
    return response.choices[0].message.content or ""


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
    text = (transport or _openai_transport)(
        base_url=profile.base_url, api_key=api_key, request=request
    )
    return A3Completion(
        text=text,
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
    "A3Completion", "A3Generation", "A3ModelProfile", "DEEPSEEK_BASE_URL",
    "OPENROUTER_BASE_URL", "complete_a3", "load_a3_model_profile",
]
