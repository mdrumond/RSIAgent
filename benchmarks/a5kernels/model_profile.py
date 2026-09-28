"""Fail-closed model routing for A5 kernel experiments.

The general RSIAgent role configuration remains deliberately separate.  An A5
experiment opts into this module and receives one immutable OpenRouter route;
misspellings and convenient aliases are rejected before a provider call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


CANONICAL_PROFILE_ID = "openai-gpt-5.6-sol-v1"
CANONICAL_MODEL = "openai/gpt-5.6-sol"
CANONICAL_PROVIDER = "OpenAI"
_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "a5"
    / "openai-gpt-5.6-sol.json"
)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"invalid {label} fields: missing={missing}, extra={extra}")


@dataclass(frozen=True)
class GenerationParameters:
    """Generation controls supported by the existing OpenRouter client."""

    max_tokens: int
    temperature: float
    top_p: float
    reasoning_effort: str
    allow_truncation_retry: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens != 32768:
            raise ValueError("the A5 profile requires max_tokens=32768")
        if type(self.temperature) not in (int, float) or self.temperature != 0.0:
            raise ValueError("the A5 profile requires temperature=0.0")
        if type(self.top_p) not in (int, float) or self.top_p != 1.0:
            raise ValueError("the A5 profile requires top_p=1.0")
        if self.reasoning_effort != "high":
            raise ValueError("the A5 profile requires reasoning_effort='high'")
        if self.allow_truncation_retry is not False:
            raise ValueError("the A5 profile requires allow_truncation_retry=False")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GenerationParameters":
        if not isinstance(value, Mapping):
            raise ValueError("generation must be an object")
        _exact_keys(
            value,
            {"max_tokens", "temperature", "top_p", "reasoning_effort",
             "allow_truncation_retry"},
            "generation",
        )
        result = cls(
            max_tokens=value["max_tokens"],
            temperature=value["temperature"],
            top_p=value["top_p"],
            reasoning_effort=value["reasoning_effort"],
            allow_truncation_retry=value["allow_truncation_retry"],
        )
        return result

    def request_values(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class A5ModelProfile:
    """One approved model and one exact OpenRouter provider route."""

    profile_id: str
    model: str
    provider: str
    allow_fallbacks: bool
    require_parameters: bool
    generation: GenerationParameters

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "A5ModelProfile":
        if not isinstance(value, Mapping):
            raise ValueError("model profile must be an object")
        _exact_keys(
            value,
            {
                "profile_id",
                "model",
                "provider",
                "allow_fallbacks",
                "require_parameters",
                "generation",
            },
            "profile",
        )
        result = cls(
            profile_id=value["profile_id"],
            model=value["model"],
            provider=value["provider"],
            allow_fallbacks=value["allow_fallbacks"],
            require_parameters=value["require_parameters"],
            generation=GenerationParameters.from_mapping(value["generation"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not isinstance(self.generation, GenerationParameters):
            raise ValueError("generation must be GenerationParameters")
        self.generation.validate()
        if self.profile_id != CANONICAL_PROFILE_ID:
            raise ValueError(f"unsupported A5 profile_id: {self.profile_id!r}")
        if self.model != CANONICAL_MODEL:
            raise ValueError(f"unsupported A5 model: {self.model!r}")
        if self.provider != CANONICAL_PROVIDER:
            raise ValueError(f"unsupported A5 provider: {self.provider!r}")
        if self.allow_fallbacks is not False:
            raise ValueError("A5 provider fallback must be disabled")
        if self.require_parameters is not True:
            raise ValueError("A5 provider must require parameter support")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def request_kwargs(self) -> dict[str, Any]:
        """Return an independent request mapping accepted by ``llm.client.chat``."""
        return {
            **self.generation.request_values(),
            "provider_order": [self.provider],
            "provider_allow_fallbacks": self.allow_fallbacks,
            "provider_require_parameters": self.require_parameters,
        }


@dataclass(frozen=True)
class A5RunProvenance:
    """The complete model route and generation request for one completion."""

    profile_id: str
    profile_sha256: str
    model: str
    provider: str
    allow_fallbacks: bool
    require_parameters: bool
    generation: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Prevent a caller from mutating recorded parameters after the run.
        object.__setattr__(self, "generation", MappingProxyType(dict(self.generation)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "model": self.model,
            "provider": self.provider,
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "generation": dict(self.generation),
        }


@dataclass(frozen=True)
class A5Completion:
    text: str
    provenance: A5RunProvenance


CompletionClient = Callable[..., str]


def load_a5_model_profile(path: Path | None = None) -> A5ModelProfile:
    source = _PROFILE_PATH if path is None else Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load A5 model profile: {source}") from exc
    return A5ModelProfile.from_mapping(value)


def complete_a5(
    system: str,
    user: str,
    *,
    history: list | None = None,
    profile: A5ModelProfile | None = None,
    client: CompletionClient | None = None,
) -> A5Completion:
    """Issue one fixed-route completion and return its request provenance.

    Provenance describes the requested route and controls, not provider response
    metadata. Truncated output is returned without changing generation controls.

    ``client`` is injectable so profile validation and request construction are
    testable without credentials or a network.  The production default is the
    repository's existing OpenRouter client.
    """
    selected = load_a5_model_profile() if profile is None else profile
    selected.validate()
    if client is None:
        from llm.client import chat

        client = chat
    text = client(
        selected.model,
        system,
        user,
        history=history,
        **selected.request_kwargs(),
    )
    provenance = A5RunProvenance(
        profile_id=selected.profile_id,
        profile_sha256=selected.fingerprint,
        model=selected.model,
        provider=selected.provider,
        allow_fallbacks=selected.allow_fallbacks,
        require_parameters=selected.require_parameters,
        generation=selected.generation.request_values(),
    )
    return A5Completion(text=text, provenance=provenance)
