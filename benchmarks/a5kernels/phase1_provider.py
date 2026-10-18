"""Direct, closed provider routes for the A5 Phase 1 experiment cells."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceKind, EvidenceLedger
from benchmarks.a5kernels.phase1_experiments import A5BackendModel, A5ModelIdentity
from benchmarks.a5kernels.phase1_live import InfrastructureFailure
from benchmarks.a5kernels.phase1_memory import parse_agent_distillation
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


class _ProviderEvidenceRecorder:
    """Persist secret-free exact requests and raw provider text for one trial."""

    def __init__(self, workspace) -> None:
        self.enabled = workspace is not None
        if not self.enabled:
            return
        root = Path(workspace).parent
        self.artifacts = ContentAddressedArtifacts(root / "artifacts")
        self.ledger_path = root / "evidence.jsonl"

    def request(self, profile, messages, *, stage: str, turn: int) -> None:
        if not self.enabled:
            return
        digest = self.artifacts.put_json(profile.request(messages))
        self._append("provider-request", digest, stage=stage, turn=turn)

    def response(self, text: str, *, stage: str, turn: int) -> None:
        if not self.enabled:
            return
        digest = self.artifacts.put_bytes(text.encode("utf-8"))
        self._append("provider-response", digest, stage=stage, turn=turn)

    def failure(self, exc: Exception, *, stage: str, turn: int) -> None:
        if not self.enabled:
            return
        digest = self.artifacts.put_json({"exception_type": type(exc).__name__})
        self._append("provider-response-failure", digest, stage=stage, turn=turn)

    def observation(self, text: str, *, turn: int) -> None:
        if not self.enabled:
            return
        digest = self.artifacts.put_bytes(text.encode("utf-8"))
        self._append("host-observation", digest, origin="actor", turn=turn)

    def _append(self, role: str, digest: str, **fields: object) -> None:
        EvidenceLedger(self.ledger_path).append(EvidenceKind.ARTIFACT, {
            "role": role, "sha256": digest, **fields,
        })

    def bounded_payloads(self, admitted: Mapping[str, str]) -> dict[str, object]:
        if not self.enabled:
            return {}
        remaining = _MAX_DISTILLATION_ARTIFACT_BYTES
        payloads = {}
        for digest, role in sorted(admitted.items()):
            if role not in _DISTILLATION_PAYLOAD_ROLES:
                continue
            try:
                content = self.artifacts.read_bytes(digest)
            except (FileNotFoundError, ValueError):
                payloads[digest] = {"role": role, "retained": False}
                continue
            entry = {"role": role, "size_bytes": len(content), "retained": True}
            if len(content) > _MAX_DISTILLATION_SINGLE_ARTIFACT_BYTES or len(content) > remaining:
                entry["payload_omitted"] = "bounded-context-limit"
            else:
                try:
                    entry["payload"] = json.loads(content)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    entry["payload"] = content.decode("utf-8", errors="replace")
                remaining -= len(content)
            payloads[digest] = entry
        return payloads


@dataclass(frozen=True)
class _DirectActorContinuation:
    profile: DirectModelProfile
    messages: tuple[Mapping[str, str], ...]
    recorder: _ProviderEvidenceRecorder
    total_turns: int


class DirectActorDriver:
    """Run the existing restricted A5 action protocol over one direct profile."""

    def __init__(
        self, provider: DirectCompletionProvider, *, max_turns: int = 40,
        trace_event: Callable[[str, Mapping[str, object]], None] | None = None,
    ):
        if type(max_turns) is not int or not 1 <= max_turns <= 40:
            raise ValueError("max_turns must be in [1, 40]")
        self.provider = provider
        self.max_turns = max_turns
        if trace_event is not None and not callable(trace_event):
            raise TypeError("trace_event must be callable")
        self.trace_event = trace_event

    def _trace(self, kind: str, payload: Mapping[str, object]) -> None:
        if self.trace_event is not None:
            self.trace_event(kind, dict(payload))

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
        del context
        recorder = _ProviderEvidenceRecorder(workspace)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        for message in messages:
            self._trace("message", message)
        tokens = 0
        turn = 0
        ordinary_turns = 0
        while ordinary_turns < self.max_turns:
            turn += 1
            recorder.request(profile, messages, stage="action", turn=turn)
            completion = self.provider.complete(profile, messages)
            recorder.response(completion.text, stage="action", turn=turn)
            tokens += completion.completion_tokens
            messages.append({"role": "assistant", "content": completion.text})
            self._trace("message", messages[-1])
            action = turn_parser(completion.text)
            if action is None:
                observation = json.dumps({
                    "host": {"event": "invalid-action", "required": _ACTION_NUDGE}
                }, sort_keys=True, separators=(",", ":"))
                recorder.observation(observation, turn=turn)
            else:
                outcome = action_executor(action)
                observation = outcome.observation
                if not outcome.budget_exempt:
                    ordinary_turns += 1
                try:
                    host = json.loads(observation).get("host", {})
                except (AttributeError, json.JSONDecodeError):
                    host = {}
                failed = (
                    host.get("passed") is False
                    or host.get("status") in {"failed", "error"}
                    or "fail" in str(host.get("event", "")).lower()
                )
                self._trace("tool", {
                    "name": action.kind,
                    "status": "failed" if failed else "observed",
                    "observation": observation,
                })
                if outcome.terminal:
                    messages.append({"role": "user", "content": observation})
                    self._trace("message", messages[-1])
                    self._trace("terminal", {
                        "status": outcome.status,
                        "admitted_artifacts": dict(outcome.admitted_artifacts),
                    })
                    return ActorOutcome(
                        outcome.status, turn, tokens,
                        admitted_artifacts=dict(outcome.admitted_artifacts),
                        continuation=_DirectActorContinuation(
                            profile, tuple(dict(item) for item in messages), recorder, turn
                        ),
                    )
            messages.append({"role": "user", "content": observation})
            self._trace("message", messages[-1])
            if action is None:
                ordinary_turns += 1
        self._trace("terminal", {"status": "turn-budget-exhausted"})
        return ActorOutcome("turn-budget-exhausted", turn, tokens)

    def distill(
        self,
        outcome: ActorOutcome,
        *,
        admitted_artifacts: Mapping[str, str] | None = None,
        host_summaries: Sequence[Mapping[str, object]] = (),
    ) -> ActorOutcome:
        """Resume the exact terminal Actor history after all host work completes."""

        continuation = outcome.continuation
        if (not isinstance(continuation, _DirectActorContinuation)
                or outcome.distillation is not None):
            raise ValueError("Actor outcome is not awaiting terminal distillation")
        admitted = dict(outcome.admitted_artifacts)
        if admitted_artifacts is not None:
            admitted.update(admitted_artifacts)
        messages = [dict(item) for item in continuation.messages]
        context = {
            "admitted_artifacts": admitted,
            "artifact_payloads": continuation.recorder.bounded_payloads(admitted),
            "host_summaries": [dict(item) for item in host_summaries[:16]],
            "terminal_host_observation": continuation.messages[-1]["content"],
        }
        messages.append({
            "role": "user",
            "content": DISTILLATION_PROMPT + "\n" + json.dumps(
                context, sort_keys=True, separators=(",", ":")
            ),
        })
        self._trace("message", messages[-1])
        tokens = outcome.tokens or 0
        for correction in range(_MAX_DISTILLATION_ATTEMPTS):
            turn = continuation.total_turns + correction + 1
            continuation.recorder.request(
                continuation.profile, messages, stage="distillation", turn=turn,
            )
            try:
                completion = self.provider.complete(continuation.profile, messages)
            except InfrastructureFailure as exc:
                continuation.recorder.failure(
                    exc, stage="distillation", turn=turn,
                )
                return outcome
            continuation.recorder.response(
                completion.text, stage="distillation", turn=turn,
            )
            self._trace("message", {"role": "assistant", "content": completion.text})
            tokens += completion.completion_tokens
            try:
                distillation = parse_agent_distillation(completion.text, admitted)
            except ValueError as exc:
                if correction + 1 == _MAX_DISTILLATION_ATTEMPTS:
                    raise
                messages.extend((
                    {"role": "assistant", "content": completion.text},
                    {"role": "user", "content": (
                        "The previous distillation was invalid: " + str(exc) + ". "
                        "Correct only the structured distillation JSON. Do not propose "
                        "or rerun kernel actions."
                    )},
                ))
                continue
            return ActorOutcome(
                outcome.status, outcome.iterations, tokens, outcome.wall_time_s,
                completion.text, distillation, admitted, None,
            )
        raise AssertionError("bounded distillation loop did not terminate")


__all__ = [
    "DirectActorDriver",
    "DISTILLATION_PROMPT",
    "DirectCompletion",
    "DirectCompletionProvider",
    "DirectModelProfile",
    "direct_profile",
    "openai_compatible_transport",
]
