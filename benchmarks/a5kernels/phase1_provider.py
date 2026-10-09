"""Direct, closed provider routes for the A5 Phase 1 experiment cells."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Mapping, Sequence

from benchmarks.a5kernels.phase1_experiments import A5BackendModel, A5ModelIdentity
from benchmarks.a5kernels.phase1_live import InfrastructureFailure
from benchmarks.a5kernels.trial import (
    _ACTION_NUDGE,
    _SYSTEM_PROMPT,
    ActorOutcome,
    ActionOutcome,
    TrialAction,
)


_DIRECT_VALUES = {
    A5BackendModel.GPT_5_6_SOL: {
        "identity": {
            "backend_model": "openai/gpt-5.6-sol",
            "profile_id": "openai-gpt-5.6-sol-v1",
            "provider": "OpenAI",
            "route": "direct:https://api.openai.com/v1",
            "credential_env": "OPENAI_API_KEY",
        },
        "model": "gpt-5.6-sol",
        "base_url": "https://api.openai.com/v1",
        "max_tokens": 32768,
    },
    A5BackendModel.DEEPSEEK_FLASH: {
        "identity": {
            "backend_model": "deepseek-flash",
            "profile_id": "deepseek-flash-native-v1",
            "provider": "DeepSeek",
            "route": "direct:https://api.deepseek.com",
            "credential_env": "DEEPSEEK_API_KEY",
        },
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "max_tokens": 8192,
    },
}

_PROVIDER_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class DirectModelProfile:
    identity: A5ModelIdentity
    model: str
    base_url: str
    max_tokens: int

    def __post_init__(self) -> None:
        expected = _DIRECT_VALUES[self.identity.backend_model]
        if (
            self.identity.as_dict() != expected["identity"]
            or self.model != expected["model"]
            or self.base_url != expected["base_url"]
            or self.max_tokens != expected["max_tokens"]
        ):
            raise ValueError("direct profile does not match its registered identity")

    def request(self, messages: Sequence[Mapping[str, str]]) -> dict[str, object]:
        request: dict[str, object] = {
            "model": self.model,
            "messages": [dict(item) for item in messages],
            "reasoning_effort": "high",
        }
        if self.identity.backend_model is A5BackendModel.GPT_5_6_SOL:
            request["max_completion_tokens"] = self.max_tokens
        else:
            request.update({
                "max_tokens": self.max_tokens,
                "top_p": 1.0,
                "extra_body": {"thinking": {"type": "disabled"}},
            })
        return request


def direct_profile(identity: A5ModelIdentity) -> DirectModelProfile:
    if not isinstance(identity, A5ModelIdentity):
        raise TypeError("identity must be a registered A5 model identity")
    values = _DIRECT_VALUES[identity.backend_model]
    return DirectModelProfile(
        identity, values["model"], values["base_url"], values["max_tokens"]
    )


@dataclass(frozen=True)
class DirectCompletion:
    text: str
    completion_tokens: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or not self.text.strip()
            or type(self.completion_tokens) is not int
            or self.completion_tokens < 1
        ):
            raise ValueError("provider completion must contain text and token usage")


Transport = Callable[..., DirectCompletion]


def _is_openai_transport_failure(exc: Exception) -> bool:
    """Recognize optional OpenAI SDK connection failures when it is installed."""

    try:
        from openai import APIConnectionError, APITimeoutError
    except ImportError:
        return False
    return isinstance(exc, (APIConnectionError, APITimeoutError))


def openai_compatible_transport(*, base_url, api_key, request) -> DirectCompletion:
    from openai import OpenAI

    response = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=_PROVIDER_TIMEOUT_SECONDS,
        max_retries=0,
    ).chat.completions.create(
        **dict(request)
    )
    try:
        return DirectCompletion(
            response.choices[0].message.content or "",
            response.usage.completion_tokens,
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        raise ValueError("provider response is missing content or token usage") from None


class DirectCompletionProvider:
    """Hold credentials privately and issue only exact registered direct requests."""

    def __init__(
        self,
        environment: Mapping[str, str],
        *,
        transport: Transport = openai_compatible_transport,
    ) -> None:
        credentials = {}
        for values in _DIRECT_VALUES.values():
            name = values["identity"]["credential_env"]
            value = environment.get(name, "")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing required credential {name}")
            credentials[name] = value
        self._credentials = credentials
        self._transport = transport

    def complete(
        self,
        profile: DirectModelProfile,
        messages: Sequence[Mapping[str, str]],
    ) -> DirectCompletion:
        if profile != direct_profile(profile.identity):
            raise ValueError("provider profile is not canonical")
        try:
            result = self._transport(
                base_url=profile.base_url,
                api_key=self._credentials[profile.identity.credential_env],
                request=profile.request(messages),
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = (
                isinstance(exc, (ConnectionError, TimeoutError))
                or _is_openai_transport_failure(exc)
                or status in {408, 409, 425, 429}
                or isinstance(status, int) and 500 <= status <= 599
            )
            if retryable:
                raise InfrastructureFailure(
                    "direct provider transport unavailable"
                ) from exc
            raise
        if not isinstance(result, DirectCompletion):
            raise TypeError("provider transport returned an invalid completion")
        return result


class DirectActorDriver:
    """Run the existing restricted A5 action protocol over one direct profile."""

    def __init__(self, provider: DirectCompletionProvider, *, max_turns: int = 40):
        if type(max_turns) is not int or not 1 <= max_turns <= 40:
            raise ValueError("max_turns must be in [1, 40]")
        self.provider = provider
        self.max_turns = max_turns

    def __call__(
        self,
        instruction: str,
        *,
        profile: DirectModelProfile,
        workspace,
        context,
        turn_parser: Callable[[str], TrialAction | None],
        action_executor: Callable[[TrialAction], ActionOutcome],
    ) -> ActorOutcome:
        del workspace, context
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        tokens = 0
        for turn in range(1, self.max_turns + 1):
            completion = self.provider.complete(profile, messages)
            tokens += completion.completion_tokens
            messages.append({"role": "assistant", "content": completion.text})
            action = turn_parser(completion.text)
            if action is None:
                observation = json.dumps({
                    "host": {"event": "invalid-action", "required": _ACTION_NUDGE}
                }, sort_keys=True, separators=(",", ":"))
            else:
                outcome = action_executor(action)
                observation = outcome.observation
                if outcome.terminal:
                    return ActorOutcome(outcome.status, turn, tokens)
            messages.append({"role": "user", "content": observation})
        return ActorOutcome("turn-budget-exhausted", self.max_turns, tokens)


__all__ = [
    "DirectActorDriver",
    "DirectCompletion",
    "DirectCompletionProvider",
    "DirectModelProfile",
    "direct_profile",
    "openai_compatible_transport",
]
