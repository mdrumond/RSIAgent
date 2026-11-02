"""Direct, closed provider routes for the A5 Phase 1 experiment cells."""

from __future__ import annotations

from dataclasses import dataclass
from copy import copy
import math
import time
from typing import Callable, Mapping, Sequence
from benchmarks.ascend.agent_context import (
    ContextBudget, DEFAULT_CONTEXT_BUDGETS,
)

from benchmarks.a5kernels.phase1_experiments import A5BackendModel, A5ModelIdentity
from benchmarks.a5kernels.trial import (
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

_PROVIDER_TIMEOUT_SECONDS = 600
_MAX_DISTILLATION_ATTEMPTS = 3
_MAX_DISTILLATION_ARTIFACT_BYTES = 64_000
_MAX_DISTILLATION_SINGLE_ARTIFACT_BYTES = 32_000
_DISTILLATION_PAYLOAD_ROLES = {
    "host-verification", "performance-study", "profiling",
}

DISTILLATION_PROMPT = """The host has now returned the terminal result. Distill only
reusable knowledge supported by the admitted artifacts below. Respond with exactly
one JSON object and no prose:
{"distillation":{"no_new_knowledge":false,"lessons":[{"category":"api|shape|control-flow|data-movement|performance|failure-repair|correctness|unknown","claim":"...","applicability":"...","preconditions":"...","limitations":"...","evidence_artifact_sha256":["lowercase sha256"],"failure_signature":null,"repair":null,"confidence":"low|medium|high"}]}}
Set no_new_knowledge to true and lessons to [] when the project established no new
reusable knowledge. Host facts remain host-owned. Compilation and profiling alone
do not establish correctness. Cite only hashes in this admitted artifact map:"""


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


class _EmptyTextCompletion(ValueError):
    """A rejected textual completion whose raw text and usage remain available."""

    def __init__(self, text, completion_tokens):
        if (not isinstance(text, str) or text.strip()
                or type(completion_tokens) is not int or completion_tokens < 1):
            raise ValueError("empty completion requires text and positive token usage")
        self.text, self.completion_tokens = text, completion_tokens
        super().__init__("provider completion must contain non-empty text")


@dataclass(frozen=True)
class DirectCompletion:
    text: str
    completion_tokens: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or type(self.completion_tokens) is not int
            or self.completion_tokens < 1
        ):
            raise ValueError("provider completion must contain text and token usage")
        if not self.text.strip():
            raise _EmptyTextCompletion(self.text, self.completion_tokens)


Transport = Callable[..., DirectCompletion]


class ProviderRejectionError(RuntimeError):
    """Provider rejection (final after retry exhaustion), without raw secrets."""

    def __init__(self, *, code: str, status_code: int, request_id: str | None):
        self.code, self.status_code, self.request_id = code, status_code, request_id
        super().__init__(f"provider rejected request: {code} (HTTP {status_code})")

    def evidence(self):
        return {"exception_type": type(self).__name__, "code": self.code,
                "status_code": self.status_code, "request_id": self.request_id}


def openai_compatible_transport(*, base_url, api_key, request) -> DirectCompletion:
    """Historical transport identity; production calls must use native Pi."""
    raise RuntimeError('direct HTTP completion was replaced by native Pi sessions')


class DirectCompletionProvider:
    """Hold registered credentials/profiles for native Pi composition."""

    def __init__(
        self,
        environment: Mapping[str, str],
        *,
        transport: Transport = openai_compatible_transport,
        context_budgets: Mapping[str, ContextBudget] | None = None,
        invalid_prompt_retries: int = 3,
        invalid_prompt_backoff_seconds: float = 120,
        sleeper: Callable[[float], None] = time.sleep,
        retry_max_seconds: float = 480,
        clock=None,
        models: Sequence[A5BackendModel] | None = None,
    ) -> None:
        selected = tuple(_DIRECT_VALUES) if models is None else tuple(models)
        if not selected or any(model not in _DIRECT_VALUES for model in selected):
            raise ValueError("provider models must use registered identities")
        credentials = {}
        for model in selected:
            values = _DIRECT_VALUES[model]
            name = values["identity"]["credential_env"]
            value = environment.get(name, "")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing required credential {name}")
            credentials[name] = value
        self._credentials = credentials
        self._transport = transport
        if type(invalid_prompt_retries) is not int or invalid_prompt_retries < 0:
            raise ValueError("invalid_prompt_retries must be a nonnegative integer")
        if (type(invalid_prompt_backoff_seconds) not in (float, int)
                or not math.isfinite(invalid_prompt_backoff_seconds)
                or invalid_prompt_backoff_seconds < 0):
            raise ValueError("invalid_prompt_backoff_seconds must be nonnegative")
        self.invalid_prompt_retries = invalid_prompt_retries
        self.invalid_prompt_backoff_seconds = invalid_prompt_backoff_seconds
        from benchmarks.ascend.time_budget import deadline_after
        deadline_after(retry_max_seconds)
        self.retry_max_seconds = retry_max_seconds
        self.clock = clock or time.monotonic
        self._sleep = sleeper
        self.context_budgets = dict(DEFAULT_CONTEXT_BUDGETS)
        if context_budgets is not None:
            if set(context_budgets) - set(self.context_budgets) or not all(
                isinstance(value, ContextBudget) for value in context_budgets.values()
            ):
                raise ValueError("context budgets must use registered model identities")
            self.context_budgets.update(context_budgets)

    def coding_environment(self):
        """Host-only credential copy for native Pi model configuration."""
        return dict(self._credentials)

    def context_budget(self, profile):
        return self.context_budgets[profile.identity.backend_model.value]

    def complete(self, profile, messages, *, attempt_observer=None, max_seconds=None):
        """Explicitly injected fixture transport only; no production HTTP fallback."""
        if self._transport is openai_compatible_transport:
            raise RuntimeError('production completion requires a native Pi session')
        if profile != direct_profile(profile.identity):
            raise ValueError('provider profile is not canonical')
        if profile.identity.credential_env not in self._credentials:
            raise ValueError('provider credential is not configured for this model')
        if max_seconds is not None and max_seconds <= 0:
            raise TimeoutError('fixture completion elapsed allowance exhausted')
        result = self._transport(base_url=profile.base_url,
            api_key=self._credentials[profile.identity.credential_env], request=profile.request(messages))
        if not isinstance(result, DirectCompletion):
            raise TypeError('injected fixture transport returned an invalid completion')
        return result


from benchmarks.ascend.actor_control import actor_lease


class DirectActorDriver:
    """Run the existing restricted A5 action protocol over one direct profile."""

    def __init__(
        self, provider: DirectCompletionProvider, *, max_turns: int = 40, max_seconds: float = 3600, clock=None,
        trace_event: Callable[[str, Mapping[str, object]], None] | None = None,
        system_prompt: str = _SYSTEM_PROMPT, canonical=None, canonical_skills=None, canonical_budgets=None,
    ):
        from benchmarks.ascend.time_budget import deadline_after
        deadline_after(max_seconds)
        self.max_seconds = max_seconds
        self.clock = clock or time.monotonic
        self.provider = provider
        self.max_turns = max_turns
        if trace_event is not None and not callable(trace_event):
            raise TypeError("trace_event must be callable")
        self.trace_event = trace_event
        self.system_prompt = system_prompt
        if canonical is None:
            from rsi.domain_bindings import kernel_adoption
            canonical = kernel_adoption()
        self.canonical = canonical
        self.canonical_skills, self.canonical_budgets = canonical_skills, canonical_budgets

    def _trace(self, kind: str, payload: Mapping[str, object]) -> None:
        if self.trace_event is not None:
            self.trace_event(kind, dict(payload))

    def with_trace(self, trace_event):
        """Bind a project-local observer without mutating this shared driver."""
        return type(self)(self.provider, max_turns=self.max_turns, max_seconds=self.max_seconds, clock=self.clock,
                          trace_event=trace_event, system_prompt=self.system_prompt, canonical=self.canonical,
                          canonical_skills=self.canonical_skills, canonical_budgets=self.canonical_budgets)

    def with_time_budget(self, seconds):
        """Bind provider calls to one request without mutating a shared driver."""
        from benchmarks.ascend.time_budget import deadline_after
        if seconds <= 0:
            raise TimeoutError("Actor provider time allowance exhausted")
        driver = copy(self)
        driver.max_seconds = min(self.max_seconds, seconds)
        driver._provider_deadline = min(deadline_after(driver.max_seconds, clock=self.clock),
                                        getattr(self, '_provider_deadline', float('inf')))
        return driver

    @actor_lease
    def __call__(self, instruction: str, *, profile: DirectModelProfile,
                 workspace, context, turn_parser: Callable[[str], TrialAction | None],
                 action_executor: Callable[[TrialAction], ActionOutcome]) -> ActorOutcome:
        from rsi.domain_execution import require_adoption
        return require_adoption(self.canonical).run(self, instruction=instruction,
            profile=profile, workspace=workspace, context=context,
            turn_parser=turn_parser, action_executor=action_executor)

    def distill(
        self,
        outcome: ActorOutcome,
        *,
        admitted_artifacts: Mapping[str, str] | None = None,
        host_summaries: Sequence[Mapping[str, object]] = (),
    ) -> ActorOutcome:
        """Use the target binding's retained canonical history and charged clock."""
        from rsi.domain_execution import require_adoption, CanonicalConfigurationError
        binding = require_adoption(self.canonical)
        if not callable(binding.distill):
            raise CanonicalConfigurationError('canonical target binding lacks terminal distillation')
        return binding.distill(self, outcome, admitted_artifacts=admitted_artifacts,
                               host_summaries=host_summaries)


__all__ = [
    "DirectActorDriver",
    "DISTILLATION_PROMPT",
    "DirectCompletion",
    "DirectCompletionProvider",
    "DirectModelProfile",
    "direct_profile",
    "openai_compatible_transport",
]
