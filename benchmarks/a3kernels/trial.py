"""A3-owned model action loop with injected host authorities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

from benchmarks.a3_experiments import (
    A3ExperimentCell,
    BackendModel,
    KnowledgeMode,
    ProfilingGuidance,
)
from benchmarks.a3_model_profiles import (
    A3Completion, A3ModelProfile, A3ProviderResponseError,
)
from benchmarks.a3kernels.candidate import (
    CANDIDATE_SOURCE_CONTRACT,
    CandidateCompilation,
)
from benchmarks.a3kernels.knowledge_agent import KnowledgeQuery, KnowledgeResult
from benchmarks.a3kernels.phase1_evidence import (
    EvidenceKind, EvidenceLedger, canonical_digest,
)
from benchmarks.a3kernels.phase1_memory import (
    AgentInterpretation,
    HostFact,
    Phase1LearningJournal,
    ProjectMemory,
)
from benchmarks.a3kernels.phase1_protocol import FailedEvidence, VerifiedResult
from benchmarks.a3kernels.phase1_protocol import A3_EXECUTION_PROFILES
from benchmarks.a3kernels.phase1_registry import CurriculumProposal
from benchmarks.a3kernels.profiling import (
    A3ProfilingSession,
    CandidateBinding,
    CompactProfileResult,
    ProfileMetric,
    ProfileRequest,
    ProfilingTreatment,
    StudyDimensions,
)


@dataclass(frozen=True)
class Action:
    kind: str
    source: str | None = None
    query: str | None = None
    limit: int | None = None
    metric: str | None = None
    interpretation: str | None = None
    supports: tuple[str, ...] = ()


_ACTION_FIELDS = {
    "write_source": {"action", "source"},
    "compile": {"action"},
    "run": {"action"},
    "query": {"action", "query", "limit"},
    "profile": {"action", "metric"},
    "submit": {"action", "interpretation", "supports"},
}
_KNOWLEDGE_RESULT_LIMIT = 5
_KNOWLEDGE_TEXT_LIMIT = 2048
_PROFILE_PAIR_LIMIT = 16
_PROFILE_NAME_LIMIT = 128
_MODEL_DIAGNOSTIC_LIMIT = 2048
_ATTEMPT_COMPLETION_SCHEMA = "a3-attempt-completion-v1"
_COMPLETION_CALL_LIMITS = {
    ProfilingGuidance.WITHOUT_GUIDANCE: 3,
    ProfilingGuidance.WITH_GUIDANCE: 4,
}
_COMPLETION_STATES = frozenset({
    "compile-required", "compile-retry", "run-required", "run-retry",
    "profile-required", "submit-ready",
})


def parse_action(text: str) -> Action:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("action must be one JSON object") from exc
    if not isinstance(value, dict) or type(value.get("action")) is not str:
        raise ValueError("action must select a registered action")
    kind = value["action"]
    if kind not in _ACTION_FIELDS:
        raise ValueError("action must select a registered action")
    expected_fields = sorted(_ACTION_FIELDS[kind])
    received_fields = sorted(value)
    if received_fields != expected_fields:
        raise ValueError(
            "action fields do not match the registered schema: "
            f"expected={expected_fields!r}; received={received_fields!r}"
        )
    if kind == "write_source":
        if type(value["source"]) is not str or not value["source"].strip():
            raise ValueError("write_source action requires non-empty source")
        return Action(kind, source=value["source"])
    if kind == "query":
        try:
            query = KnowledgeQuery(value["query"], value["limit"])
        except (TypeError, ValueError) as exc:
            raise ValueError("query action is invalid") from exc
        return Action(kind, query=query.query, limit=query.limit)
    if kind == "profile":
        if value["metric"] != ProfileMetric.PIPE_UTILIZATION.value:
            raise ValueError("profile action requires PipeUtilization")
        return Action(kind, metric=value["metric"])
    if kind == "submit":
        supports = value["supports"]
        if (
            type(value["interpretation"]) is not str
            or not value["interpretation"].strip()
            or not isinstance(supports, list)
            or any(type(item) is not str for item in supports)
        ):
            raise ValueError("submit action requires interpretation and supports")
        return Action(
            kind,
            interpretation=value["interpretation"].strip(),
            supports=tuple(supports),
        )
    return Action(kind)


@dataclass(frozen=True)
class TrialBudgets:
    max_turns: int = 24
    max_tokens: int = 32768

    def __post_init__(self) -> None:
        if (
            type(self.max_turns) is not int
            or not 1 <= self.max_turns <= 100
            or type(self.max_tokens) is not int
            or not 1 <= self.max_tokens <= 1_000_000
        ):
            raise ValueError("trial budgets are outside registered bounds")

    @classmethod
    def for_model(cls, model: BackendModel) -> "TrialBudgets":
        if not isinstance(model, BackendModel):
            raise TypeError("default trial budgets require a registered backend model")
        if model is BackendModel.DEEPSEEK_FLASH:
            return cls(max_turns=24, max_tokens=65536)
        return cls(max_turns=24, max_tokens=32768)


def trial_protocol_sha256() -> str:
    model_budgets = {}
    for model in BackendModel:
        budgets = TrialBudgets.for_model(model)
        model_budgets[model.value] = {
            "max_turns": budgets.max_turns,
            "max_tokens": budgets.max_tokens,
        }
    return canonical_digest({
        "schema": "a3-trial-protocol-v2",
        "verification_feedback_schema": "a3-verification-mismatch-v1",
        "attempt_completion": {
            "schema": _ATTEMPT_COMPLETION_SCHEMA,
            "source_frozen": True,
            "allowed_operations": ["compile", "run", "profile", "submit"],
            "call_limits": {
                treatment.value: limit
                for treatment, limit in _COMPLETION_CALL_LIMITS.items()
            },
            "token_ceiling": "absolute",
            "rewrite_required": "terminate",
        },
        "model_budgets": model_budgets,
    })


@dataclass(frozen=True)
class TrialResult:
    status: str
    turns: int
    tokens: int
    verified: VerifiedResult | None
    profile_result: CompactProfileResult | None
    knowledge_queries: int
    failures: tuple[str, ...]
    memory_entry_sha256: str | None = None
    completion_turns: int = 0


Actor = Callable[[A3ModelProfile, str], A3Completion]


class A3TrialLoop:
    def __init__(
        self,
        *,
        cell: A3ExperimentCell,
        proposal: CurriculumProposal,
        profile: A3ModelProfile,
        actor: Actor,
        candidate,
        knowledge,
        profiler: A3ProfilingSession,
        evidence: EvidenceLedger,
        memory: Phase1LearningJournal,
        workdir: Path,
        budgets: TrialBudgets | None = None,
        timing_dimensions: StudyDimensions | None = None,
        execution_profile: str = "gz-a3",
    ) -> None:
        if not isinstance(cell, A3ExperimentCell):
            raise TypeError("trial requires an A3ExperimentCell")
        if not isinstance(proposal, CurriculumProposal):
            raise TypeError("trial requires a CurriculumProposal")
        if not isinstance(profile, A3ModelProfile) or profile.backend_model is not cell.backend_model:
            raise ValueError("model profile must exactly match the experiment cell")
        expected_kdb = cell.knowledge is KnowledgeMode.WITH_KDB
        if type(getattr(knowledge, "enabled", None)) is not bool or knowledge.enabled != expected_kdb:
            raise ValueError("Knowledge Agent availability must exactly match the cell")
        if memory.cell_id != cell.cell_id:
            raise ValueError("memory must be isolated to the experiment cell")
        self.cell, self.proposal, self.profile = cell, proposal, profile
        self.actor, self.candidate, self.knowledge, self.profiler = actor, candidate, knowledge, profiler
        self.evidence, self.memory, self.workdir = evidence, memory, Path(workdir)
        self.budgets = budgets if budgets is not None else TrialBudgets.for_model(cell.backend_model)
        if execution_profile not in A3_EXECUTION_PROFILES:
            raise ValueError("trial execution_profile must be a registered A3 profile")
        self.execution_profile = execution_profile
        if timing_dimensions is not None and not isinstance(timing_dimensions, StudyDimensions):
            raise TypeError("timing_dimensions must be StudyDimensions")
        self.timing_dimensions = timing_dimensions

    def run(self) -> TrialResult:
        source = None
        compilation = None
        verified = None
        profile_result = None
        timing_result = None
        consumed_turns, tokens, failures, observations = (
            self._recover_provider_response_failures()
        )
        actions: list[str] = []
        compile_attempt_id: str | None = None
        run_attempt_id: str | None = None
        rewrite_source: str | None = None
        queries = 0
        completion_turns = 0
        completion_started = False
        completion_limit = _COMPLETION_CALL_LIMITS[self.cell.profiling]
        last_turn = consumed_turns
        if tokens > self.budgets.max_tokens:
            failures.append("token budget exhausted")
            return TrialResult(
                "budget-exhausted", consumed_turns, tokens, verified,
                profile_result, queries, tuple(failures),
                completion_turns=completion_turns,
            )
        for turn in range(
            consumed_turns + 1,
            self.budgets.max_turns + completion_limit + 1,
        ):
            in_completion = turn > self.budgets.max_turns
            if in_completion:
                state = self._trial_state(
                    source, compilation, verified, profile_result,
                    rewrite_source, compile_attempt_id, run_attempt_id,
                )
                if not completion_started:
                    if state not in _COMPLETION_STATES:
                        break
                    completion_started = True
                    self._record_completion_boundary(
                        source=source, state=state, call_limit=completion_limit,
                    )
                elif state not in _COMPLETION_STATES:
                    break
                completion_turns += 1
            last_turn = turn
            try:
                completion = self.actor(
                    self.profile,
                    self._context(
                        observations, source, compilation, verified,
                        profile_result, rewrite_source, compile_attempt_id,
                        run_attempt_id,
                        completion_phase=in_completion,
                        completion_calls_remaining=(
                            completion_limit - completion_turns + 1
                            if in_completion else None
                        ),
                    ),
                )
            except A3ProviderResponseError as exc:
                tokens += exc.completion_tokens or 0
                self._retain_provider_response_failure(
                    turn, exc, failures, observations
                )
                if tokens > self.budgets.max_tokens:
                    failures.append("token budget exhausted")
                    return TrialResult(
                        "budget-exhausted", turn, tokens, verified,
                        profile_result, queries, tuple(failures),
                        completion_turns=completion_turns,
                    )
                continue
            if not isinstance(completion, A3Completion):
                raise TypeError("actor must return A3Completion")
            tokens += completion.completion_tokens
            if tokens > self.budgets.max_tokens:
                failures.append("token budget exhausted")
                return TrialResult(
                    "budget-exhausted", turn, tokens, verified,
                    profile_result, queries, tuple(failures),
                    completion_turns=completion_turns,
                )
            try:
                if completion.provenance.get("profile_sha256") != self.profile.fingerprint:
                    raise ValueError("completion provenance does not match the cell model")
                action = parse_action(completion.text)
                state = self._trial_state(
                    source, compilation, verified, profile_result, rewrite_source,
                    compile_attempt_id, run_attempt_id,
                )
                if action.kind not in self._allowed_action_kinds(
                    state, completion_phase=in_completion,
                ):
                    raise ValueError(f"{action.kind} is not valid in {state}")
                if (
                    action.kind == "write_source"
                    and rewrite_source is not None
                    and action.source == rewrite_source
                ):
                    raise ValueError(
                        "candidate source must change after deterministic failure"
                    )
                actions.append(action.kind)
                self._record_action(turn, action)
                if action.kind == "write_source":
                    source, compilation, verified, profile_result = action.source, None, None, None
                    compile_attempt_id = None
                    run_attempt_id = None
                    rewrite_source = None
                    timing_result = None
                    observations.append("candidate source updated")
                elif action.kind == "compile":
                    if source is None:
                        raise ValueError("write_source is required before compile")
                    if compile_attempt_id is None:
                        compile_attempt_id = f"turn-{turn}"
                    verified = profile_result = timing_result = None
                    compile_result = self.candidate.compile(
                        source, self.workdir, request_id=self.cell.cell_id,
                        attempt_id=compile_attempt_id, project_id=self.proposal.project_id,
                        length=self._length(), padded_length=self._padded_length(),
                        block_count=self._block_count(), seed=0,
                        execution_profile=self.execution_profile,
                    )
                    if isinstance(compile_result, FailedEvidence):
                        if not self._retryable_failure("compile", compile_result):
                            compile_attempt_id = None
                            rewrite_source = source
                        self._retain_candidate_failure(
                            turn, "compile", compile_result, failures, observations
                        )
                        continue
                    if not isinstance(compile_result, CandidateCompilation):
                        raise TypeError("candidate compiler returned an invalid result")
                    compilation = compile_result
                    compile_attempt_id = None
                    rewrite_source = None
                    observations.append(self._authoritative_observation(
                        kind="compile",
                        evidence_sha256=compilation.attestation_sha256,
                        source_fingerprint=compilation.plan.source_fingerprint,
                        execution_id=compilation.plan.execution_id,
                    ))
                elif action.kind == "run":
                    if source is None or compilation is None:
                        raise ValueError("successful compile is required before run")
                    verified = profile_result = timing_result = None
                    if run_attempt_id is None:
                        run_attempt_id = f"turn-{turn}"
                    result = self.candidate.run(
                        source, self.workdir, request_id=self.cell.cell_id,
                        attempt_id=run_attempt_id, project_id=self.proposal.project_id,
                        length=self._length(), padded_length=self._padded_length(),
                        block_count=self._block_count(), seed=0,
                        execution_profile=self.execution_profile,
                    )
                    if isinstance(result, FailedEvidence):
                        if not self._retryable_failure("run", result):
                            run_attempt_id = None
                            rewrite_source = source
                        self._retain_candidate_failure(
                            turn, "run", result, failures, observations
                        )
                        continue
                    if not isinstance(result, VerifiedResult):
                        raise TypeError("candidate runner returned an invalid result")
                    run_attempt_id = None
                    if result.source_fingerprint != compilation.plan.source_fingerprint:
                        raise ValueError("candidate source identity changed after compile")
                    if not result.passed:
                        rewrite_source = source
                        self._retain_verification_failure(
                            turn, result, failures, observations
                        )
                        continue
                    verified = result
                    rewrite_source = None
                    observations.append(self._authoritative_observation(
                        kind="host-verification",
                        evidence_sha256=result.evidence_sha256,
                        source_fingerprint=result.source_fingerprint,
                        execution_id=result.execution_id,
                    ))
                elif action.kind == "query":
                    queries += 1
                    results = self.knowledge.query(KnowledgeQuery(action.query, action.limit))
                    if not self.knowledge.enabled:
                        observations.append("knowledge disabled")
                    else:
                        selected = results[:_KNOWLEDGE_RESULT_LIMIT]
                        if any(not isinstance(item, KnowledgeResult) for item in selected):
                            raise TypeError("Knowledge Agent returned an invalid result")
                        observations.append({
                            "knowledge_results": [
                                {
                                    "citation": asdict(item.citation),
                                    "text": item.text[:_KNOWLEDGE_TEXT_LIMIT],
                                }
                                for item in selected
                            ],
                            "returned_count": len(results),
                            "included_count": len(selected),
                            "text_truncated": any(
                                len(item.text) > _KNOWLEDGE_TEXT_LIMIT
                                for item in selected
                            ),
                        })
                elif action.kind == "profile":
                    if verified is None:
                        raise ValueError("verified candidate is required before profile")
                    request = ProfileRequest(
                        CandidateBinding(verified.execution_id, verified.source_fingerprint),
                        StudyDimensions(self._length(), self._block_count(), 5, 20),
                        ProfileMetric(action.metric),
                        treatment=(
                            ProfilingTreatment.ON
                            if self.cell.profiling is ProfilingGuidance.WITH_GUIDANCE
                            else ProfilingTreatment.OFF
                        ),
                        execution_profile=self.execution_profile,
                    )
                    profile_result = self.profiler.profile(request)
                    observations.append(
                        "profiling disabled" if profile_result is None
                        else self._profile_observation(profile_result)
                    )
                elif action.kind == "submit":
                    failure = self._submit_gate(source, compilation, verified, profile_result)
                    if failure:
                        raise ValueError(failure)
                    assert verified is not None
                    if self.timing_dimensions is not None and timing_result is None:
                        timing_result = self.profiler.time(
                            CandidateBinding(verified.execution_id, verified.source_fingerprint),
                            self.timing_dimensions,
                        )
                    facts = [
                        HostFact("compile", "candidate compiled with the host-owned fixture", compilation.attestation_sha256),
                        HostFact("host-verification", "candidate output passed host verification", verified.evidence_sha256),
                    ]
                    if profile_result is not None:
                        facts.append(HostFact(
                            "profiling", self._profile_statement(profile_result),
                            profile_result.evidence_sha256,
                        ))
                    if timing_result is not None:
                        facts.append(HostFact("profiling", "host timing samples captured", timing_result.evidence_sha256))
                    interpretation = AgentInterpretation(
                        action.interpretation, action.supports
                    )
                    project_memory = ProjectMemory(
                        self.proposal, self.proposal.project_id, self.cell.cell_id,
                        self.memory.lineage_id, verified.source_fingerprint, tuple(actions),
                        tuple(facts), (interpretation,),
                    )
                    entry = self.memory.append(project_memory)
                    self.evidence.append(EvidenceKind.RESULT, self._identity({"verified": verified, "memory_entry_sha256": entry["entry_sha256"]}))
                    return TrialResult(
                        "passed", turn, tokens, verified, profile_result,
                        queries, tuple(failures), entry["entry_sha256"],
                        completion_turns,
                    )
            except (TypeError, ValueError, RuntimeError) as exc:
                failure = str(exc)
                failures.append(failure)
                observations.append("failure: " + failure)
                self.evidence.append(EvidenceKind.FAILURE, self._identity({"turn": turn, "detail": failure}))
        return TrialResult(
            "budget-exhausted", last_turn, tokens, verified, profile_result,
            queries, tuple(failures), completion_turns=completion_turns,
        )

    def _recover_provider_response_failures(
        self,
    ) -> tuple[int, int, list[str], list[object]]:
        retained = [
            entry for entry in self.evidence.entries
            if entry.payload.get("cell_id") == self.cell.cell_id
            and entry.payload.get("project_id") == self.proposal.project_id
            and entry.payload.get("execution_profile") == self.execution_profile
        ]
        if not retained:
            return 0, 0, [], []
        if any(
            entry.kind != EvidenceKind.FAILURE.value
            or "provider_response_failure" not in entry.payload
            for entry in retained
        ):
            raise ValueError(
                "provider failure recovery requires only leading retained failures"
            )
        tokens = 0
        failures: list[str] = []
        observations: list[object] = []
        for expected_turn, entry in enumerate(retained, 1):
            if entry.payload.get("turn") != expected_turn:
                raise ValueError(
                    "provider failure recovery turns are not consecutive"
                )
            structured = entry.payload["provider_response_failure"]
            if not isinstance(structured, Mapping) or set(structured) != {
                "code", "completion_tokens",
            }:
                raise ValueError("provider failure recovery evidence is invalid")
            failure = A3ProviderResponseError(
                structured["code"],
                completion_tokens=structured["completion_tokens"],
            )
            tokens += failure.completion_tokens or 0
            failures.append(f"provider response failure: {failure.code}")
            observations.append({
                "provider_response_failure": {
                    "code": failure.code,
                    "completion_tokens": failure.completion_tokens,
                }
            })
        return len(retained), tokens, failures, observations

    def _retain_provider_response_failure(
        self,
        turn: int,
        failure: A3ProviderResponseError,
        failures: list[str],
        observations: list[object],
    ) -> None:
        detail = f"provider response failure: {failure.code}"
        structured = {
            "code": failure.code,
            "completion_tokens": failure.completion_tokens,
        }
        failures.append(detail)
        observations.append({"provider_response_failure": structured})
        self.evidence.append(
            EvidenceKind.FAILURE,
            self._identity({
                "turn": turn,
                "provider_response_failure": structured,
            }),
        )

    def _submit_gate(self, source, compilation, verified, profile_result) -> str | None:
        if (
            source is None
            or not isinstance(compilation, CandidateCompilation)
            or verified is None
        ):
            return "source, compile, and host verification are required before submit"
        if verified.source_fingerprint != compilation.plan.source_fingerprint:
            return "candidate source identity changed before submit"
        if self.cell.profiling is ProfilingGuidance.WITH_GUIDANCE and profile_result is None:
            return "required profile is missing before submit"
        return None

    def _length(self) -> int:
        value = dict(self.proposal.parameters).get("length", 32)
        return value if type(value) is int else 32

    def _block_count(self) -> int:
        value = dict(self.proposal.parameters).get("block_count", 1)
        return value if type(value) is int else 1

    def _padded_length(self) -> int:
        return ((self._length() + 63) // 64) * 64

    def _retain_candidate_failure(
        self,
        turn: int,
        operation: str,
        failure: FailedEvidence,
        failures: list[str],
        observations: list[object],
    ) -> None:
        detail = f"{operation} failed: {failure.detail}"
        failures.append(detail)
        observations.append({
            "candidate_failure": self._compact_failure(
                stage=failure.stage,
                error_type=failure.error_type,
                detail=failure.detail,
                source_fingerprint=failure.source_fingerprint,
                execution_id=failure.execution_id,
                attestation_sha256=failure.attestation_sha256,
                next_action=(
                    "retry"
                    if self._retryable_failure(operation, failure)
                    else "write_source"
                ),
            )
        })
        self.evidence.append(
            EvidenceKind.FAILURE,
            self._identity({"turn": turn, "failed_evidence": failure}),
        )

    def _retain_verification_failure(
        self,
        turn: int,
        result: VerifiedResult,
        failures: list[str],
        observations: list[object],
    ) -> None:
        detail = (
            "host verification failed: "
            f"exit_code={result.exit_code}, max_abs_error={result.max_abs_error}, "
            f"tolerance={result.tolerance}"
        )
        failures.append(detail)
        observations.append({
            "failed_verification": {
                "diagnostic": detail,
                "max_abs_error": result.max_abs_error,
                "tolerance": result.tolerance,
                "mismatch": (
                    asdict(result.mismatch)
                    if result.mismatch is not None else None
                ),
                "source_fingerprint": result.source_fingerprint,
                "execution_id": result.execution_id,
                "evidence_sha256": result.evidence_sha256,
                "attestation_sha256": result.attestation_sha256,
                "next_action": "write_source",
            }
        })
        self.evidence.append(
            EvidenceKind.FAILURE,
            self._identity({"turn": turn, "failed_verification": result}),
        )

    @staticmethod
    def _retryable_failure(operation: str, failure: FailedEvidence) -> bool:
        return failure.stage == "prepare" or (
            failure.error_type == "RuntimeError"
            and (
                (operation == "compile" and failure.stage == "compile")
                or (operation == "run" and failure.stage == "execute")
            )
        )

    @staticmethod
    def _compact_failure(
        *, stage: str, error_type: str, detail: str,
        source_fingerprint: str, execution_id: str,
        attestation_sha256: str, next_action: str,
    ) -> dict[str, object]:
        lines = [" ".join(line.split()) for line in detail.splitlines()]
        lines = [line for line in lines if line]
        preferred = [
            line for line in lines
            if "error:" in line.lower() or "fatal:" in line.lower()
        ]
        selected = preferred or lines
        unique = list(dict.fromkeys(selected))
        normalized = "\n".join(lines)
        diagnostic = "\n".join(unique)[:_MODEL_DIAGNOSTIC_LIMIT]
        return {
            "stage": stage,
            "error_type": error_type,
            "diagnostic": diagnostic,
            "detail_truncated": diagnostic != normalized,
            "source_fingerprint": source_fingerprint,
            "execution_id": execution_id,
            "attestation_sha256": attestation_sha256,
            "next_action": next_action,
        }

    @staticmethod
    def _authoritative_observation(
        *, kind: str, evidence_sha256: str, source_fingerprint: str,
        execution_id: str,
    ) -> dict[str, object]:
        return {"authoritative_evidence": {
            "kind": kind,
            "status": "passed",
            "evidence_sha256": evidence_sha256,
            "source_fingerprint": source_fingerprint,
            "execution_id": execution_id,
        }}

    def _identity(self, payload: Mapping[str, object]) -> dict[str, object]:
        return {
            "target": "Ascend910B4", "language": "ascend-c",
            "execution_profile": self.execution_profile,
            "runtime": "native-ascend-c",
            "cell_id": self.cell.cell_id, "project_id": self.proposal.project_id,
            **payload,
        }

    def _record_action(self, turn: int, action: Action) -> None:
        payload: dict[str, object] = {"turn": turn, "action": action.kind}
        if action.kind == "write_source":
            assert action.source is not None
            payload["source_sha256"] = self._source_sha256(action.source)
        self.evidence.append(
            EvidenceKind.ACTION,
            self._identity(payload),
        )

    @staticmethod
    def _source_sha256(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _record_completion_boundary(
        self, *, source: str | None, state: str, call_limit: int,
    ) -> None:
        assert source is not None
        allowed = self._allowed_action_kinds(state, completion_phase=True)
        self.evidence.append(
            EvidenceKind.PLAN,
            self._identity({
                "attempt_completion_boundary": {
                    "nominal_max_turns": self.budgets.max_turns,
                    "call_limit": call_limit,
                    "trial_state": state,
                    "source_sha256": self._source_sha256(source),
                    "allowed_actions": list(allowed),
                }
            }),
        )

    @staticmethod
    def _profile_pairs(values) -> list[list[object]]:
        return [
            [str(name)[:_PROFILE_NAME_LIMIT], float(value)]
            for name, value in values[:_PROFILE_PAIR_LIMIT]
        ]

    @classmethod
    def _profile_observation(cls, result: CompactProfileResult) -> dict[str, object]:
        return {"profile_evidence": {
            "metric": result.metric.value,
            "metric_values": cls._profile_pairs(result.metric_values),
            "timeline": cls._profile_pairs(result.timeline),
            "evidence_sha256": result.evidence_sha256,
            "source_fingerprint": result.source_fingerprint,
        }}

    @classmethod
    def _profile_statement(cls, result: CompactProfileResult) -> str:
        def render(values) -> str:
            return ",".join(f"{name}={value:.12g}" for name, value in values)
        metrics = render(cls._profile_pairs(result.metric_values))
        timeline = render(cls._profile_pairs(result.timeline))
        return f"raw {result.metric.value}: metrics[{metrics}] timeline[{timeline}]"

    @staticmethod
    def _eligible_submit_supports(
        observations: list[object], source_fingerprint: str | None,
    ) -> list[str]:
        supports = []
        for observation in observations:
            if not isinstance(observation, Mapping):
                continue
            for key in ("authoritative_evidence", "profile_evidence"):
                evidence = observation.get(key)
                if not isinstance(evidence, Mapping):
                    continue
                if evidence.get("source_fingerprint") != source_fingerprint:
                    continue
                digest = evidence.get("evidence_sha256")
                if isinstance(digest, str) and digest not in supports:
                    supports.append(digest)
        return supports

    def _trial_state(
        self, source, compilation, verified, profile_result, rewrite_source,
        compile_attempt_id=None, run_attempt_id=None,
    ) -> str:
        if source is None:
            return "source-required"
        if rewrite_source is not None:
            return "rewrite-required"
        if not isinstance(compilation, CandidateCompilation):
            return (
                "compile-retry" if compile_attempt_id is not None
                else "compile-required"
            )
        if verified is None:
            return "run-retry" if run_attempt_id is not None else "run-required"
        if (
            self.cell.profiling is ProfilingGuidance.WITH_GUIDANCE
            and profile_result is None
        ):
            return "profile-required"
        return "submit-ready"

    def _allowed_action_kinds(
        self, state: str, *, completion_phase: bool = False,
    ) -> tuple[str, ...]:
        core = {
            "source-required": ("write_source",),
            "rewrite-required": ("write_source",),
            "compile-required": ("compile",),
            "compile-retry": ("compile",),
            "run-required": ("run",),
            "run-retry": ("run",),
            "profile-required": ("profile",),
            "submit-ready": ("submit", "write_source"),
        }[state]
        if completion_phase:
            return tuple(kind for kind in core if kind != "write_source")
        if self.cell.knowledge is KnowledgeMode.WITH_KDB:
            return (*core, "query")
        return core

    def _context(
        self, observations: list[object], source: str | None,
        compilation=None, verified=None, profile_result=None,
        rewrite_source=None, compile_attempt_id=None, run_attempt_id=None,
        *, completion_phase: bool = False,
        completion_calls_remaining: int | None = None,
    ) -> str:
        state = self._trial_state(
            source, compilation, verified, profile_result, rewrite_source,
            compile_attempt_id, run_attempt_id,
        )
        source_fingerprint = (
            compilation.plan.source_fingerprint
            if isinstance(compilation, CandidateCompilation) else None
        )
        allowed = self._allowed_action_kinds(
            state, completion_phase=completion_phase,
        )
        context = {
                "action_contract": {
                    "profile_metric": ProfileMetric.PIPE_UTILIZATION.value,
                    "eligible_submit_supports": self._eligible_submit_supports(
                        observations, source_fingerprint
                    ),
                    "trial_state": state,
                    "required_next_action": self._required_next_action(state),
                },
                "allowed_actions": {
                    kind: sorted(_ACTION_FIELDS[kind]) for kind in sorted(allowed)
                },
                "cell": self.cell.as_dict(),
                "candidate_source_contract": CANDIDATE_SOURCE_CONTRACT.as_dict(),
                "current_candidate_source": source,
                "lineage_id": self.memory.lineage_id,
                "memory": json.loads(self.memory.project_context()),
                "observations": observations,
                "policy": {
                    "logical_length": self._length(),
                    "padded_length": self._padded_length(),
                    "block_count": self._block_count(),
                    "profiling_treatment": self.cell.profiling.value,
                },
                "proposal": self.proposal.as_dict(),
                "source_slot": CANDIDATE_SOURCE_CONTRACT.source_slot,
            }
        if completion_phase:
            context["attempt_completion"] = {
                "active": True,
                "call_limit": _COMPLETION_CALL_LIMITS[self.cell.profiling],
                "calls_remaining": completion_calls_remaining,
                "source_frozen": True,
            }
        return json.dumps(
            context,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _required_next_action(self, state: str) -> str | None:
        return {
            "rewrite-required": "write_source",
            "compile-required": "compile",
            "run-required": "run",
            "profile-required": "profile",
        }.get(state)


__all__ = ["A3TrialLoop", "Action", "TrialBudgets", "TrialResult", "parse_action"]
