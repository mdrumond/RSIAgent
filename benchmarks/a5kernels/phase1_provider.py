"""Direct, closed provider routes for the A5 Phase 1 experiment cells."""

from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence
from benchmarks.ascend.agent_context import (
    ContextBudget, ContextBudgetError, DEFAULT_CONTEXT_BUDGETS, WorkingContext, request_bytes,
)

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceKind, EvidenceLedger, canonical_digest
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

_PROVIDER_TIMEOUT_SECONDS = 600
_LOGGER = logging.getLogger(__name__)
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
            response.choices[0].message.content,
            response.usage.completion_tokens,
        )
    except _EmptyTextCompletion:
        raise
    except (AttributeError, IndexError, TypeError, ValueError):
        raise ValueError("provider response is missing content or token usage") from None


class DirectCompletionProvider:
    """Hold credentials privately and issue only exact registered direct requests."""

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
        """Host-only credential copy for the Pi gateway; never passed to Pi."""
        return dict(self._credentials)

    def context_budget(self, profile):
        return self.context_budgets[profile.identity.backend_model.value]

    def complete(
        self,
        profile: DirectModelProfile,
        messages: Sequence[Mapping[str, str]],
        *, attempt_observer=None,
    ) -> DirectCompletion:
        if profile != direct_profile(profile.identity):
            raise ValueError("provider profile is not canonical")
        if profile.identity.credential_env not in self._credentials:
            raise ValueError("provider credential is not configured for this model")
        if request_bytes(profile.request(messages)) > self.context_budget(profile).hard:
            raise ContextBudgetError("direct provider request exceeds context hard byte budget")
        request = profile.request(messages)
        request_sha256 = canonical_digest(request)
        deadline = self.clock() + self.retry_max_seconds
        attempt = 0
        while True:
            attempt += 1
            if attempt_observer is not None:
                attempt_observer("request", {"attempt": attempt, "request": deepcopy(request),
                                             "request_sha256": request_sha256})
            try:
                return self._complete_once(profile, request)
            except ProviderRejectionError as exc:
                retry = self.clock() + self.invalid_prompt_backoff_seconds < deadline
                delay = self.invalid_prompt_backoff_seconds if retry else 0
                rejection = {"attempt": attempt, **exc.evidence(),
                             "request_sha256": request_sha256,
                             "retry": retry, "backoff_seconds": delay}
                # Curriculum/KDB callers need diagnostics too, even without an
                # Actor evidence observer. Never log exception bodies or keys.
                _LOGGER.warning("provider prompt rejection: %s", json.dumps(rejection, sort_keys=True))
                if attempt_observer is not None:
                    attempt_observer("rejection", rejection)
                if not retry:
                    raise
                self._sleep(delay)

    def _complete_once(self, profile, request):
        try:
            result = self._transport(
                base_url=profile.base_url,
                api_key=self._credentials[profile.identity.credential_env],
                request=deepcopy(request),
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None)
            code = getattr(exc, "code", None)
            if isinstance(body, dict):
                error = body.get("error", body)
                if isinstance(error, dict):
                    code = error.get("code", code)
            if status == 400 and code == "invalid_prompt":
                request_id = getattr(exc, "request_id", None)
                # Retain only an ID, never arbitrary exception/body text.
                if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", request_id):
                    request_id = None
                if request_id and any(secret in request_id for secret in self._credentials.values()):
                    request_id = None
                raise ProviderRejectionError(code=code, status_code=status,
                                             request_id=request_id) from None
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
        payload = exc.evidence() if isinstance(exc, ProviderRejectionError) else {"exception_type": type(exc).__name__}
        digest = self.artifacts.put_json(payload)
        self._append("provider-response-failure", digest, stage=stage, turn=turn)

    def attempt(self, event, payload, *, stage, turn):
        if not self.enabled:
            return
        digest = self.artifacts.put_json(payload.get("request", payload))
        role = "provider-request" if event == "request" else "provider-attempt-rejection"
        self._append(role, digest, stage=stage, turn=turn,
                     attempt=payload["attempt"], request_sha256=payload["request_sha256"])

    def observation(self, text: str, *, turn: int) -> None:
        if not self.enabled:
            return
        digest = self.artifacts.put_bytes(text.encode("utf-8"))
        self._append("host-observation", digest, origin="actor", turn=turn)

    def _append(self, role: str, digest: str, **fields: object) -> None:
        EvidenceLedger(self.ledger_path).append(EvidenceKind.ARTIFACT, {
            "role": role, "sha256": digest, **fields,
        })

    def bounded_payloads(self, admitted: Mapping[str, str], *, max_bytes=_MAX_DISTILLATION_ARTIFACT_BYTES) -> dict[str, object]:
        if not self.enabled:
            return {}
        remaining = max_bytes
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
class _ActorContinuation:
    profile: DirectModelProfile
    messages: tuple[Mapping[str, str], ...]
    recorder: _ProviderEvidenceRecorder
    working_context: WorkingContext
    total_turns: int
    deadline: float | None = None


class DirectActorDriver:
    """Run the existing restricted A5 action protocol over one direct profile."""

    def __init__(
        self, provider: DirectCompletionProvider, *, max_turns: int = 40, max_seconds: float = 3600, clock=None,
        trace_event: Callable[[str, Mapping[str, object]], None] | None = None,
        system_prompt: str = _SYSTEM_PROMPT,
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

    def _trace(self, kind: str, payload: Mapping[str, object]) -> None:
        if self.trace_event is not None:
            self.trace_event(kind, dict(payload))

    def with_trace(self, trace_event):
        """Bind a project-local observer without mutating this shared driver."""
        return type(self)(self.provider, max_turns=self.max_turns, max_seconds=self.max_seconds, clock=self.clock,
                          trace_event=trace_event, system_prompt=self.system_prompt)

    def _prepare_context(self, working, profile, recorder, *, turn):
        tokens = 0

        def summarize(messages):
            nonlocal tokens
            self._request(recorder, profile, messages, stage="compaction", turn=turn)
            try:
                completion = self._complete(recorder, profile, messages, stage="compaction", turn=turn)
            except _EmptyTextCompletion as exc:
                # Only compaction treats empty textual output as correctable.
                # Ordinary Actor/provider calls still reject this ValueError.
                completion = exc
            except Exception as exc:
                recorder.failure(exc, stage="compaction", turn=turn)
                if isinstance(exc, ProviderRejectionError):
                    raise
                raise ContextBudgetError("work-log summarization failed; original evidence retained") from exc
            recorder.response(completion.text, stage="compaction", turn=turn)
            tokens += completion.completion_tokens
            self._trace("tool", {"name": "context-compaction", "status": "observed",
                "messages": messages, "work_log": completion.text,
                "completion_tokens": completion.completion_tokens})
            return completion.text

        messages = working.prepare(profile, self.provider.context_budget(profile), summarize)
        return messages, tokens

    def _request(self, recorder, profile, messages, *, stage, turn):
        self._trace("tool", {"name": "provider-request", "stage": stage, "turn": turn,
                             "request": profile.request(messages)})

    def _complete(self, recorder, profile, messages, *, stage, turn):
        def observe(event, payload):
            recorder.attempt(event, payload, stage=stage, turn=turn)
            self._trace("tool", {"name": "provider-attempt-" + event,
                                "stage": stage, "turn": turn, **payload})
        return self.provider.complete(profile, messages, attempt_observer=observe)

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
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
        ]
        for message in messages:
            self._trace("message", message)
        working = WorkingContext(messages)
        from benchmarks.ascend.actor_control import ActorConversation, ActorStopped
        conversation = ActorConversation(workspace, instruction, working, action_executor)
        tokens = 0
        turn = conversation.turn
        ordinary_turns = 0
        deadline = self.clock() + self.max_seconds
        while self.clock() < deadline:
            try:
                conversation.boundary()
            except ActorStopped:
                return ActorOutcome('user-stopped', turn, tokens)
            turn += 1
            messages, summary_tokens = self._prepare_context(working, profile, recorder, turn=turn)
            tokens += summary_tokens
            self._request(recorder, profile, messages, stage="action", turn=turn)
            try:
                completion = self._complete(recorder, profile, messages, stage="action", turn=turn)
            except Exception as exc:
                recorder.failure(exc, stage="action", turn=turn)
                raise
            recorder.response(completion.text, stage="action", turn=turn)
            tokens += completion.completion_tokens
            messages.append({"role": "assistant", "content": completion.text})
            working.append(messages[-1])
            self._trace("message", messages[-1])
            if conversation.control.pending():
                working.append({'role': 'user', 'content':
                    'The preceding proposed action was not executed because user input arrived.'})
                conversation.save(turn)
                continue
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
                    working.append(messages[-1])
                    self._trace("message", messages[-1])
                    conversation.save(turn)
                    self._trace("terminal", {
                        "status": outcome.status,
                        "admitted_artifacts": dict(outcome.admitted_artifacts),
                    })
                    return ActorOutcome(
                        outcome.status, turn, tokens,
                        admitted_artifacts=dict(outcome.admitted_artifacts),
                        continuation=_ActorContinuation(
                            profile, tuple(dict(item) for item in messages), recorder, working, turn, deadline
                        ),
                    )
            messages.append({"role": "user", "content": observation})
            working.append(messages[-1])
            self._trace("message", messages[-1])
            conversation.save(turn)
            if action is None:
                ordinary_turns += 1
        self._trace("terminal", {"status": "time-budget-exhausted"})
        return ActorOutcome("time-budget-exhausted", turn, tokens)

    def distill(
        self,
        outcome: ActorOutcome,
        *,
        admitted_artifacts: Mapping[str, str] | None = None,
        host_summaries: Sequence[Mapping[str, object]] = (),
    ) -> ActorOutcome:
        """Resume the exact terminal Actor history after all host work completes."""

        continuation = outcome.continuation
        if (not isinstance(continuation, _ActorContinuation)
                or outcome.distillation is not None):
            raise ValueError("Actor outcome is not awaiting terminal distillation")
        admitted = dict(outcome.admitted_artifacts)
        if admitted_artifacts is not None:
            admitted.update(admitted_artifacts)
        messages = [dict(item) for item in continuation.messages]
        context = {
            "admitted_artifacts": admitted,
            "artifact_payloads": continuation.recorder.bounded_payloads(
                admitted, max_bytes=self.provider.context_budget(continuation.profile).target // 4,
            ),
            "host_summaries": [dict(item) for item in host_summaries[:16]],
            "terminal_host_observation": continuation.messages[-1]["content"],
        }
        if continuation.recorder.enabled:
            from benchmarks.ascend.agent_presentation import compact_observation
            context["host_summaries"] = [compact_observation(dict(item), continuation.recorder.artifacts)
                                         for item in host_summaries[:16]]
            if request_bytes(context) > self.provider.context_budget(continuation.profile).target // 2:
                context = compact_observation({"event": "terminal-distillation-context", **context},
                                              continuation.recorder.artifacts)
                context["admitted_artifacts"] = admitted
        messages.append({
            "role": "user",
            "content": DISTILLATION_PROMPT + "\n" + json.dumps(
                context, sort_keys=True, separators=(",", ":")
            ),
        })
        self._trace("message", messages[-1])
        working = deepcopy(continuation.working_context)
        # Terminal schema and admitted citation identities must survive repairs:
        # they are instructions, not history that a work log may fold away.
        working.pinned.append(dict(messages[-1]))
        tokens = outcome.tokens or 0
        deadline = continuation.deadline or self.clock() + self.max_seconds
        correction = -1
        while self.clock() < deadline:
            correction += 1
            turn = continuation.total_turns + correction + 1
            messages, summary_tokens = self._prepare_context(
                working, continuation.profile, continuation.recorder, turn=turn,
            )
            tokens += summary_tokens
            self._request(continuation.recorder,
                continuation.profile, messages, stage="distillation", turn=turn,
            )
            try:
                completion = self._complete(continuation.recorder, continuation.profile,
                                            messages, stage="distillation", turn=turn)
            except InfrastructureFailure as exc:
                continuation.recorder.failure(
                    exc, stage="distillation", turn=turn,
                )
                return replace(outcome, tokens=tokens)
            except Exception as exc:
                continuation.recorder.failure(exc, stage="distillation", turn=turn)
                raise
            continuation.recorder.response(
                completion.text, stage="distillation", turn=turn,
            )
            self._trace("message", {"role": "assistant", "content": completion.text})
            tokens += completion.completion_tokens
            try:
                distillation = parse_agent_distillation(completion.text, admitted)
            except ValueError as exc:
                if self.clock() >= deadline:
                    raise
                messages.extend((
                    {"role": "assistant", "content": completion.text},
                    {"role": "user", "content": (
                        "The previous distillation was invalid: " + str(exc) + ". "
                        "Correct only the structured distillation JSON. Do not propose "
                        "or rerun kernel actions."
                    )},
                ))
                working.tail = [dict(message) for message in messages[-2:]]
                self._trace("message", messages[-1])
                continue
            return ActorOutcome(
                outcome.status, outcome.iterations, tokens, outcome.wall_time_s,
                completion.text, distillation, admitted, None,
            )
        self._trace("terminal", {"status": "distillation-time-budget-exhausted"})
        return replace(outcome, tokens=tokens)


__all__ = [
    "DirectActorDriver",
    "DISTILLATION_PROMPT",
    "DirectCompletion",
    "DirectCompletionProvider",
    "DirectModelProfile",
    "direct_profile",
    "openai_compatible_transport",
]
