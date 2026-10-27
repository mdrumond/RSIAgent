"""Bounded Actor bridge for adaptive, host-verified Phase-1 projects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from pathlib import Path
import re
from typing import Callable, Mapping, Protocol

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.failure_cases import retain_phase1_failure
from benchmarks.ascend.agent_presentation import (
    artifact_page_presentation, compact_observation, memory_presentation, mismatch_summary,
)
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_adaptive import (
    AdaptiveProjectRequest,
    AdaptiveProjectResult,
)
from benchmarks.a5kernels.phase1_curriculum import FailureMode
from benchmarks.a5kernels.phase1_memory import (
    AgentDistillation,
    HostFact,
    Phase1ProjectMemory,
)
from benchmarks.a5kernels.knowledge_agent import Citation
from benchmarks.a5kernels.phase1_project_executor import (
    PreparedProject,
    ProjectCompileResult,
    ProjectExecutionResult,
    ProjectExecutor,
    _tensor_payload, _expected_payload,
)
from benchmarks.a5kernels.phase1_project import EvidenceClassification
from benchmarks.a5kernels.trial import (
    ActionOutcome,
    ActorOutcome,
    TrialAction,
    parse_trial_action,
)


class AdaptiveActorDriver(Protocol):
    """The action and terminal-distillation subset used by this adapter."""

    def __call__(
        self, instruction: str, *, profile: object, workspace: Path,
        context: Path, turn_parser: object, action_executor: object,
    ) -> ActorOutcome: ...

    def distill(
        self, outcome: ActorOutcome, *,
        admitted_artifacts: Mapping[str, str] | None = None,
        host_summaries: tuple[Mapping[str, object], ...] = (),
    ) -> ActorOutcome: ...


@dataclass(frozen=True)
class AdaptiveActionContext:
    request: AdaptiveProjectRequest
    prepared: PreparedProject
    source: str | None
    latest_execution: ProjectExecutionResult | None
    artifacts: ContentAddressedArtifacts


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class AdaptiveActionResult:
    """Durable treatment output kept separate from execution authority."""

    observation: Mapping[str, object]
    citations: tuple[Citation, ...] = ()
    host_facts: tuple[HostFact, ...] = ()
    admitted_artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Mapping):
            raise TypeError("adaptive action observation must be a mapping")
        canonical_bytes(self.observation)
        if type(self.citations) is not tuple or not all(
            isinstance(item, Citation) for item in self.citations
        ):
            raise TypeError("adaptive action citations must contain Citation values")
        if any(
            not isinstance(citation.collection, str)
            or not citation.collection
            or not isinstance(citation.path, str)
            or not citation.path
            or type(citation.start_line) is not int
            or type(citation.end_line) is not int
            or citation.start_line < 1
            or citation.end_line < citation.start_line
            or not isinstance(citation.chunk_hash, str)
            or _SHA256.fullmatch(citation.chunk_hash) is None
            for citation in self.citations
        ):
            raise ValueError("adaptive action citations must retain exact locations")
        if type(self.host_facts) is not tuple or not all(
            isinstance(item, HostFact) for item in self.host_facts
        ):
            raise TypeError("adaptive action host facts must contain HostFact values")
        admitted = self.admitted_artifacts
        if not isinstance(admitted, Mapping) or any(
            not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or not isinstance(role, str)
            or not role.strip()
            or role != role.strip()
            for digest, role in admitted.items()
        ):
            raise ValueError("adaptive action admissions must map SHA-256 to roles")
        missing = {
            fact.evidence_sha256 for fact in self.host_facts
        } - set(admitted)
        if missing:
            raise ValueError("adaptive action host facts require admitted evidence")
        if any(
            admitted[fact.evidence_sha256] != fact.category
            for fact in self.host_facts
        ):
            raise ValueError("adaptive action host fact role must match its category")
        object.__setattr__(self, "admitted_artifacts", dict(admitted))


ActionDependency = Callable[
    [Mapping[str, object], AdaptiveActionContext], AdaptiveActionResult
]


def disabled_knowledge_action(
    _payload: Mapping[str, object], _context: AdaptiveActionContext,
) -> AdaptiveActionResult:
    return AdaptiveActionResult({
        "event": "knowledge-query", "available": False,
        "reason": "disabled-by-treatment",
    })


def disabled_profile_action(
    _payload: Mapping[str, object], _context: AdaptiveActionContext,
) -> AdaptiveActionResult:
    return AdaptiveActionResult({
        "event": "profile", "available": False,
        "reason": "disabled-by-treatment",
    })


def disabled_diagnostic_action(_payload, _context) -> AdaptiveActionResult:
    return AdaptiveActionResult({"event": "diagnostic", "available": False,
                                 "reason": "diagnostic-backend-not-bound"})


def _same_prepared(left: PreparedProject, right: PreparedProject) -> bool:
    return (
        left.spec == right.spec
        and left.inputs == right.inputs
        and left.expected == right.expected
        and left.family == right.family
        and left.abi == right.abi
        and left.instruction == right.instruction
    )


class _ProjectActions:
    def __init__(
        self,
        executor: ProjectExecutor,
        request: AdaptiveProjectRequest,
        prepared: PreparedProject,
        artifacts: ContentAddressedArtifacts,
        *,
        max_actions: int,
        max_seconds: float,
        max_source_bytes: int,
        query_knowledge: ActionDependency,
        request_profile: ActionDependency,
        request_timing: ActionDependency,
        request_diagnostic: ActionDependency = disabled_diagnostic_action,
    ) -> None:
        self.executor = executor
        self.request = request
        self.prepared = prepared
        self.artifacts = artifacts
        self.read_artifacts = artifacts
        self.max_actions = max_actions
        from benchmarks.ascend.time_budget import deadline_after
        self.deadline = deadline_after(max_seconds)
        self.max_source_bytes = max_source_bytes
        self.query_knowledge = query_knowledge
        self.request_profile = request_profile
        self.request_timing = request_timing
        self.request_diagnostic = request_diagnostic
        self.actions: list[str] = []
        self.metered_actions = 0
        self.source: str | None = None
        self.source_sha256: str | None = None
        self.latest: ProjectExecutionResult | None = None
        self.latest_source_sha256: str | None = None
        self.admitted: dict[str, str] = {}
        self.citations: list[Citation] = []
        self.host_facts: list[HostFact] = []
        self.observed_failures: dict[FailureMode, int] = {}
        self.latest_run_at: int | None = None
        self.failure_context = None
        if request.failure_context is not None:
            from benchmarks.ascend.failure_handoff import FailureContext
            self.failure_context = FailureContext(request.failure_context, request.spec,
                                                  prepared.inputs, prepared.expected)
            self.failure_context.require_repair_evidence(request.spec)
            if self.failure_context.compatible:
                # Historical authenticated negative evidence predates every action here.
                self.observed_failures[request.spec.failure_mode] = -1

    def __call__(self, action: TrialAction) -> ActionOutcome:
        from benchmarks.ascend.execution import process_deadline
        with process_deadline(self.deadline):
            return self._execute_action(action)

    def restore_source(self, source):
        """Restore editable source only; verification authority must be rebuilt."""
        result = self._write(TrialAction('write', {'content': source}))
        if self.source != source:
            raise ValueError(result.observation)

    def _execute_action(self, action: TrialAction) -> ActionOutcome:
        if not isinstance(action, TrialAction):
            raise TypeError("adaptive projects accept only TrialAction values")
        self.actions.append(action.kind)
        if time.monotonic() >= self.deadline:
            return self._observation({"error": "time budget exhausted"}, True, "budget")
        if action.kind == "propose":
            return self._observation({"event": "proposal-recorded"})
        if action.kind == "read-artifact":
            try:
                page = artifact_page_presentation(action.payload, self.read_artifacts)
            except (ValueError, FileNotFoundError) as exc:
                return self._observation({"error": str(exc)})
            return ActionOutcome(canonical_bytes({"host": page}).decode("ascii"),
                                 False, "continue")
        if action.kind == "write":
            return self._write(action)
        if action.kind == "compile":
            return self._compile()
        if action.kind == "run":
            return self._run()
        if action.kind == "query-knowledge":
            return self._dependency(self.query_knowledge, action.payload)
        if action.kind == "request-profile":
            return self._measurement(self.request_profile, action.payload)
        if action.kind == "request-timing":
            return self._measurement(self.request_timing, action.payload)
        if action.kind in {"request-debug", "request-sanitizer"}:
            if self.source is None:
                return self._observation({"error": "write candidate source first"})
            return self._dependency(self.request_diagnostic, {
                "kind": action.kind, **action.payload,
            })
        if action.kind == "submit":
            return self._submit()
        return self._observation({
            "error": f"{action.kind} is not bound for this project",
            "available": ["propose", "write", "compile", "run", "submit"],
        })

    def _write(self, action: TrialAction) -> ActionOutcome:
        source = action.payload["content"]
        if not isinstance(source, str) or not source.strip():
            return self._observation({"error": "candidate source must be non-empty"})
        encoded = source.encode("utf-8")
        if len(encoded) > self.max_source_bytes:
            return self._observation({"error": "candidate source exceeds byte limit"})
        self.source = source
        self.source_sha256 = self.artifacts.put_bytes(encoded)
        self.admitted[self.source_sha256] = "candidate-source"
        self.latest = None
        self.latest_source_sha256 = None
        # Earlier verified variants remain available for paired comparisons, but
        # cannot satisfy the final-source measurement gates.
        for digest, role in tuple(self.admitted.items()):
            if role in {"performance-study", "profiling"}:
                self.admitted[digest] = "comparison-" + role
        return self._observation({
            "event": "candidate-written", "source_sha256": self.source_sha256,
        })

    def _compile(self) -> ActionOutcome:
        if self.source is None:
            return self._observation({"error": "write candidate source first"})
        result: ProjectCompileResult = self.executor.compile(
            self.request.spec, self.source
        )
        if not _same_prepared(result.prepared, self.prepared):
            raise ValueError("compilation rematerialized different project inputs or oracle")
        digest = self.artifacts.put_json(result.artifact)
        if digest != result.artifact_sha256:
            raise ValueError("compile artifact hash does not match its payload")
        self.admitted[digest] = "compile"
        if not result.observation.passed:
            self.observed_failures[FailureMode.COMPILE] = len(self.actions)
            retain_phase1_failure(self.artifacts, self.request.spec, self.source_sha256,
                "compile-failure", result.observation.diagnostic, {
                    "diagnostics": result.artifact,
                    "inputs": {"tensors": [_tensor_payload(x) for x in self.prepared.inputs]},
                    "expected": _expected_payload(self.prepared.expected)},
                self.prepared.expected.dtype)
        return self._observation({
            "event": "host-compile", "source_sha256": self.source_sha256,
            "passed": result.observation.passed,
            "diagnostic": result.observation.diagnostic,
            "artifact": {"sha256": digest, "payload": result.artifact},
        })

    def _run(self) -> ActionOutcome:
        if self.source is None:
            return self._observation({"error": "write candidate source first"})
        result = self.executor.execute(self.request.spec, self.source)
        if not _same_prepared(result.prepared, self.prepared):
            raise ValueError("execution rematerialized different project inputs or oracle")
        self.latest = result
        self.latest_source_sha256 = self.source_sha256
        self.latest_run_at = len(self.actions)
        if result.evidence.classification is EvidenceClassification.COMPILE_FAILURE:
            self.observed_failures[FailureMode.COMPILE] = self.latest_run_at
        elif result.evidence.classification is EvidenceClassification.HOST_MISMATCH:
            self.observed_failures[FailureMode.HOST_MISMATCH] = self.latest_run_at
        retained = {}
        for name, payload in result.artifacts.items():
            digest = self.artifacts.put_json(payload)
            if digest != result.artifact_sha256[name]:
                raise ValueError("execution artifact hash does not match its payload")
            role = (
                "host-verification"
                if name == "classification" and result.evidence.verified
                else f"execution-{name}"
            )
            self.admitted[digest] = role
            retained[name] = {"sha256": digest, "payload": payload}
        if not result.evidence.verified:
            retain_phase1_failure(self.artifacts, self.request.spec, self.source_sha256,
                result.evidence.classification.value, result.evidence.detail, result.artifacts,
                self.prepared.expected.dtype)
        # Source intent cannot establish that a device print executed. This
        # backend exposes correctness artifacts, but no retained print trace.
        return self._observation({
            "event": "host-execution", "requested_action": "run",
            "source_sha256": self.source_sha256,
            "classification": result.evidence.classification.value,
            "detail": result.evidence.detail,
            "numerical_summary": mismatch_summary(
                result.artifacts["expected"]["storage"],
                result.artifacts["observed"]["output"],
            ),
            "artifacts": retained,
        })

    def _measurement(
        self, dependency: ActionDependency, payload: Mapping[str, object],
    ) -> ActionOutcome:
        if self.latest is None or not self.latest.evidence.verified or (
            self.latest_source_sha256 != self.source_sha256
        ):
            return self._observation({
                "error": "measurement requires HOST_VERIFIED evidence for current source; run first"
            })
        return self._dependency(dependency, payload)

    def _dependency(
        self, dependency: ActionDependency, payload: Mapping[str, object],
    ) -> ActionOutcome:
        result = dependency(payload, AdaptiveActionContext(
            self.request, self.prepared, self.source, self.latest, self.artifacts,
        ))
        if not isinstance(result, AdaptiveActionResult):
            raise TypeError("adaptive action dependency must return AdaptiveActionResult")
        for digest, role in result.admitted_artifacts.items():
            self.artifacts.read_bytes(digest)
            existing = self.admitted.get(digest)
            if existing is not None and existing not in {role, "comparison-" + role}:
                raise ValueError("adaptive action changed an admitted artifact role")
            self.admitted[digest] = role
        for citation in result.citations:
            if citation not in self.citations:
                self.citations.append(citation)
        for fact in result.host_facts:
            if fact not in self.host_facts:
                self.host_facts.append(fact)
        return self._observation(result.observation)

    def _submit(self) -> ActionOutcome:
        if (
            self.latest is None
            or self.latest_source_sha256 != self.source_sha256
            or not self.latest.evidence.verified
        ):
            return self._observation({
                "error": "submit requires HOST_VERIFIED evidence for current source"
            })
        missing = []
        mode = self.request.spec.failure_mode
        if mode is not FailureMode.NONE and (
            mode not in self.observed_failures
            or self.latest_run_at is None
            or self.latest_run_at <= self.observed_failures[mode]
        ):
            missing.append(f"observed-{mode.value}")
        if (
            self.request.spec.timing_lengths
            and "performance-study" not in self.admitted.values()
        ):
            missing.append("performance-study")
        if "sync-performance" in self.request.spec.learning_focus:
            measured = {"current": set(), "comparison": set()}
            for digest, role in self.admitted.items():
                if role not in {"profiling", "performance-study",
                                "comparison-profiling", "comparison-performance-study"}:
                    continue
                payload = self.artifacts.read_json(digest)
                candidate = payload.get("candidate_sha256") if isinstance(payload, Mapping) else None
                if not isinstance(candidate, str) or not _SHA256.fullmatch(candidate):
                    continue
                bucket = "comparison" if role.startswith("comparison-") else "current"
                measured[bucket].add(candidate)
            if (self.source_sha256 not in measured["current"]
                    or not measured["comparison"] - {self.source_sha256}):
                missing.append(
                    "sync-performance requires measurements of two distinct verified sources: "
                    "retain one comparison variant, then verify and measure the final source"
                )
        if self.request.spec.profile_metric is not None and not any(
            fact.category == "profiling"
            and self.admitted.get(fact.evidence_sha256) == "profiling"
            for fact in self.host_facts
        ):
            missing.append("profiling")
        if missing:
            return self._observation({
                "error": "submit requires retained project evidence",
                "missing": missing,
            })
        return self._observation({
            "event": "host-verified-submit",
            "classification": self.latest.evidence.classification.value,
            "source_sha256": self.source_sha256,
            "classification_sha256": self.latest.artifact_sha256["classification"],
        }, True, "host-verified")

    def _observation(
        self, payload: Mapping[str, object], terminal: bool = False, status: str = "continue",
    ) -> ActionOutcome:
        return ActionOutcome(
            canonical_bytes({"host": compact_observation(payload, self.artifacts)}).decode("ascii"), terminal, status,
        )


class _ObservedActions:
    """Keep live host state and deadlines visible through trace instrumentation."""
    def __init__(self, actions, observe):
        self.actions, self.observe = actions, observe

    def __getattr__(self, name):
        return getattr(self.actions, name)

    def __call__(self, action):
        return self.observe(action)


class AdaptiveActorProjectExecutor:
    """Drive bounded candidate repair, then emit only distilled verified memory."""

    def __init__(
        self, executor: ProjectExecutor, actor: AdaptiveActorDriver, profile: object,
        *, max_actions: int = 40, max_source_bytes: int = 128_000,
        max_seconds: float = 3600,
        query_knowledge: ActionDependency | None = None,
        request_profile: ActionDependency | None = None,
        request_timing: ActionDependency | None = None,
        request_diagnostic: ActionDependency | None = None,
        immutable_context: bytes = b"",
        diagnostic_instructions: str | None = None,
        trace_recorder_factory=None,
        retain_semantic_failure: bool = False,
    ) -> None:
        if not callable(actor):
            raise TypeError("actor must be callable")
        if type(max_source_bytes) is not int or max_source_bytes < 1:
            raise ValueError("max_source_bytes must be positive")
        if not isinstance(immutable_context, bytes):
            raise TypeError("immutable context must be bytes")
        try:
            immutable_context.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("immutable context must be UTF-8") from exc
        self.executor = executor
        self.actor = actor
        self.profile = profile
        self.max_actions = max_actions
        self.max_seconds = max_seconds
        self.max_source_bytes = max_source_bytes
        self.query_knowledge = (
            disabled_knowledge_action if query_knowledge is None else query_knowledge
        )
        self.request_profile = (
            disabled_profile_action if request_profile is None else request_profile
        )
        self.request_timing = (
            disabled_profile_action if request_timing is None else request_timing
        )
        self.request_diagnostic = (
            disabled_diagnostic_action if request_diagnostic is None else request_diagnostic
        )
        self.immutable_context = immutable_context
        self.diagnostic_instructions = diagnostic_instructions
        self.trace_recorder_factory = trace_recorder_factory
        self.retain_semantic_failure = retain_semantic_failure
        if not all(callable(action) for action in (
            self.query_knowledge, self.request_profile, self.request_timing,
            self.request_diagnostic,
        )):
            raise TypeError("adaptive action dependencies must be callable")

    def __call__(self, request: AdaptiveProjectRequest) -> AdaptiveProjectResult:
        if not isinstance(request, AdaptiveProjectRequest):
            raise TypeError("request must be AdaptiveProjectRequest")
        if type(request.publish_learning) is not bool:
            raise TypeError("publish_learning must be boolean")
        started = time.monotonic()
        remaining = getattr(request, "remaining_seconds", None)
        if remaining is not None and remaining <= 0:
            raise RuntimeError("project time budget exhausted")
        max_seconds = min(self.max_seconds, remaining) if remaining is not None else self.max_seconds
        executor = self.executor
        if request.failure_context is not None and request.spec.failure_mode is not FailureMode.NONE:
            from benchmarks.ascend.failure_handoff import FailureContext
            failure = FailureContext(request.failure_context, request.spec)
            failure.require_repair_evidence(request.spec)
            executor = executor.bind_case(request.spec, failure.inputs)
        prepared = executor.prepare(request.spec)
        if (
            prepared.spec != request.spec
            or prepared.inputs != request.inputs
            or prepared.expected != request.oracle
        ):
            raise ValueError("request spec, inputs, or oracle differ from rematerialized project")
        root = Path(request.branch_root)
        artifacts = ContentAddressedArtifacts(root / "artifacts")
        memory_sha256 = artifacts.put_bytes(request.pre_wave_memory)
        actions = _ProjectActions(
            executor, request, prepared, artifacts,
            max_actions=self.max_actions, max_seconds=max(0.001, max_seconds-(time.monotonic()-started)), max_source_bytes=self.max_source_bytes,
            query_knowledge=self.query_knowledge,
            request_profile=self.request_profile,
            request_timing=self.request_timing,
            request_diagnostic=self.request_diagnostic,
        )
        actions.deadline = min(actions.deadline, started+max_seconds)
        from benchmarks.ascend.agent_presentation import MemoryArtifactReader
        actions.read_artifacts = MemoryArtifactReader(request, artifacts)
        if actions.failure_context is not None:
            from benchmarks.ascend.failure_handoff import FailureArtifactReader
            actions.read_artifacts = FailureArtifactReader(actions.read_artifacts, actions.failure_context)
        diagnostics = ("Diagnostics: {\"request-debug\":{}} or "
                   "{\"request-sanitizer\":{\"tool\":\"memcheck\"}} "
                   "(also racecheck/initcheck/synccheck). Available before verification. "
                   "These retain diagnostics, never HOST_VERIFIED or timing. Tool exit zero "
                   "does not establish Catlass instruction coverage; inspect raw report. "
                   "Unavailable tools are reported explicitly. Do not deliberately run "
                   "hanging kernels; bounded termination does not guarantee device recovery.\n")
        opening = (diagnostics if self.diagnostic_instructions is None else self.diagnostic_instructions)
        if actions.failure_context is not None:
            opening += ("\nLinked observed failures (unverified context, not reusable memory): "
                        + json.dumps(actions.failure_context.presentation(), sort_keys=True)
                        + "\nRead the supplied source, exact inputs, expected and observed results using "
                        "read-artifact. A compatible attached case satisfies the negative-example "
                        "requirement; repair it and host-verify your current source. Do not invent "
                        "a new mismatch. Incompatible cases are context only.\n")
        opening += prepared.instruction + "\nPre-wave memory (immutable, readable index): " + json.dumps(
            memory_presentation(request.pre_wave_memory, artifacts),
            sort_keys=True, separators=(",", ":"))
        opening += ('\nInspect full retained evidence or memory with '
                    '{"read-artifact":{"sha256":"<hash>","offset":0,"limit":8192}}. '
                    'Offsets and limits are bytes; follow next_offset for more. '
                    'Observation previews do not replace authoritative host verification.')
        opening += (
            '\nAfter host verification of the current source, request timing with '
            '{"request-timing":{}} and profiling with {"request-profile":{}}. '
            'The host selects timing lengths and profiler metric from the project spec. '
            'Inspect returned raw measurements and diagnostics; compile success is not correctness. '
            'Rewriting source requires new verification and new measurements before submit. '
            'Report measurement scope and uncertainty when distilling performance lessons.'
        )
        if self.immutable_context:
            opening += (
                "\nProduction context (immutable, identical for every project): "
                + self.immutable_context.decode("utf-8")
            )
        recorder = self.trace_recorder_factory(request) if self.trace_recorder_factory else None
        actor = self.actor
        execute_action = actions
        if recorder is not None:
            def observe(kind, payload):
                if kind == "message":
                    recorder.message(**payload)
                elif kind == "tool":
                    recorder.tool_event(str(payload["name"]), payload)

            actor = actor.with_trace(observe)

            def observe_action(action):
                outcome = actions(action)
                if action.kind == "write" and actions.source is not None:
                    recorder.candidate(actions.source, candidate_id=actions.source_sha256)
                if action.kind == "compile" and actions.source is not None:
                    host = json.loads(outcome.observation)["host"]
                    if "artifact" in host and not host["passed"]:
                        recorder.validation(
                            candidate_id=actions.source_sha256,
                            validator="host-compile", status="failed",
                            evidence_sha256=host["artifact"]["sha256"],
                            candidate_sha256=actions.source_sha256,
                            details={"compile_only": True},
                        )
                if action.kind == "run" and actions.latest is not None:
                    result = actions.latest
                    recorder.validation(
                        candidate_id=actions.source_sha256, validator="host-execution",
                        status="passed" if result.evidence.verified else "failed",
                        evidence_sha256=result.artifact_sha256["classification"],
                        candidate_sha256=actions.source_sha256,
                        details={"classification": result.evidence.classification.value},
                    )
                return outcome
            execute_action = _ObservedActions(actions, observe_action)
        try:
            outcome = actor(
                opening, profile=self.profile, workspace=root / "actor",
                context=artifacts.objects / memory_sha256[:2] / memory_sha256[2:],
                turn_parser=parse_trial_action, action_executor=execute_action,
            )
        except Exception as exc:
            if recorder is not None:
                from benchmarks.ascend.agent_context import ContextBudgetError
                from benchmarks.a5kernels.phase1_provider import ProviderRejectionError
                context_abort = isinstance(exc, ContextBudgetError)
                provider_abort = isinstance(exc, ProviderRejectionError)
                abort_details = ({"abort_status": "context-budget-exhausted"} if context_abort else
                                 {"abort_status": "provider-rejected", **exc.evidence()} if provider_abort else {})
                digest = artifacts.put_json(exc.evidence() if provider_abort else
                                            {"exception": type(exc).__name__, "detail": str(exc)})
                recorder.validation(
                    candidate_id=actions.source_sha256 or "no-candidate", validator="actor-abort",
                    status="failed" if context_abort or provider_abort else "infrastructure-failure", evidence_sha256=digest,
                    candidate_sha256=actions.source_sha256 or "0" * 64,
                    details=abort_details,
                )
                recorder.finish()
            raise
        result = actions.latest
        if not isinstance(outcome, ActorOutcome):
            raise RuntimeError("Actor returned an invalid outcome")
        from collections import Counter
        action_counts = {"write": 0, "compile": 0, "run": 0, **Counter(actions.actions)}
        if (self.retain_semantic_failure and result is not None
                and not result.evidence.verified
                and actions.latest_source_sha256 == actions.source_sha256):
            # Retain host facts for selection, never distill or publish failed
            # candidates as successful knowledge. Infrastructure exceptions
            # escape the action loop before reaching this branch.
            if recorder is not None:
                recorder.finish()
            memory = Phase1ProjectMemory(
                request.spec, actions.source_sha256, tuple(actions.actions),
                (*actions.host_facts, HostFact("diagnostic", result.evidence.detail,
                                              result.artifact_sha256["classification"])),
                no_new_knowledge=True,
                admitted_artifact_sha256=tuple(sorted(actions.admitted)),
                artifact_store="artifacts",
            )
            return AdaptiveProjectResult(memory, result.evidence, canonical_bytes({
                "classification": result.evidence.classification.value,
                "detail": result.evidence.detail, "source_sha256": actions.source_sha256,
                "actor_status": outcome.status,
                "iterations": outcome.iterations, "tokens": outcome.tokens,
                "action_counts": action_counts,
                "metered_actions": actions.metered_actions,
                "artifacts": dict(result.artifact_sha256),
            }), result.evidence.detail.strip() or "Host execution failed")
        if (
            not isinstance(outcome, ActorOutcome)
            or outcome.status != "host-verified"
            or result is None
            or not result.evidence.verified
            or actions.latest_source_sha256 != actions.source_sha256
        ):
            if recorder is not None:
                if actions.latest is None or actions.latest.evidence.verified:
                    digest = artifacts.put_json({"status": getattr(outcome, "status", "invalid-outcome")})
                    recorder.validation(
                        candidate_id=actions.source_sha256 or "no-candidate", validator="actor-terminal",
                        status="failed", evidence_sha256=digest,
                        candidate_sha256=actions.source_sha256 or "0" * 64,
                        details={"abort_status": "completed-unverified"},
                    )
                recorder.finish()
            from benchmarks.a5kernels.phase1_wave_runtime import SemanticProjectFailure
            raise SemanticProjectFailure(
                f"Actor ended without host-verified current source ({outcome.status}); "
                f"observed action counts: {json.dumps(action_counts, sort_keys=True)}", {
                "actor_status": getattr(outcome, "status", "invalid-outcome"),
                "iterations": getattr(outcome, "iterations", None),
                "tokens": getattr(outcome, "tokens", None),
                "source_sha256": actions.source_sha256,
                "actions": list(actions.actions),
                "action_counts": action_counts,
                "metered_actions": actions.metered_actions,
                "host_facts": [asdict(fact) for fact in actions.host_facts],
                "admitted_artifact_sha256": sorted(actions.admitted),
            })
        if recorder is not None:
            recorder.supervised_source(actions.source)
            recorder.finish()
        facts = (*actions.host_facts, HostFact(
            "host-verification", result.evidence.detail,
            result.artifact_sha256["classification"],
        ))
        if request.publish_learning:
            learned = self.actor.distill(
                outcome, admitted_artifacts=actions.admitted,
                host_summaries=tuple(asdict(fact) for fact in facts),
            )
            if not isinstance(learned.distillation, AgentDistillation):
                raise RuntimeError("terminal project omitted structured distillation")
            if learned.terminal_response is None:
                raise RuntimeError("terminal project omitted exact distillation response")
            distillation = learned.distillation
            artifacts.put_bytes(learned.terminal_response.encode("utf-8"))
        else:
            # Subject campaigns delegate learning to an independent verifier.
            # The factual result is still compatible with legacy wave readers.
            distillation = AgentDistillation(True, (), tuple(sorted(actions.admitted)))
        assert actions.source_sha256 is not None
        memory = Phase1ProjectMemory(
            request.spec, actions.source_sha256, tuple(actions.actions), facts,
            kdb_citations=tuple(actions.citations),
            lessons=distillation.lessons,
            no_new_knowledge=distillation.no_new_knowledge,
            admitted_artifact_sha256=distillation.admitted_artifact_sha256,
            artifact_store="artifacts",
        )
        evidence_payload = canonical_bytes({
            "classification": result.evidence.classification.value,
            "detail": result.evidence.detail,
            "source_sha256": actions.source_sha256,
            "artifacts": dict(result.artifact_sha256),
        })
        return AdaptiveProjectResult(
            memory, result.evidence, evidence_payload,
            f"{request.spec.project_id} passed host verification",
        )


__all__ = [
    "ActionDependency", "AdaptiveActionContext", "AdaptiveActionResult",
    "AdaptiveActorDriver",
    "AdaptiveActorProjectExecutor",
    "disabled_knowledge_action", "disabled_profile_action",
]
