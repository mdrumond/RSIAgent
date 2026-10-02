"""A3-owned model action loop with injected host authorities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

from benchmarks.a3_experiments import (
    A3ExperimentCell,
    BackendModel,
    KnowledgeMode,
    ProfilingGuidance,
)
from benchmarks.a3_model_profiles import A3Completion, A3ModelProfile
from benchmarks.a3kernels.candidate import (
    CANDIDATE_SOURCE_CONTRACT,
    CandidateCompilation,
)
from benchmarks.a3kernels.knowledge_agent import KnowledgeQuery, KnowledgeResult
from benchmarks.a3kernels.phase1_evidence import EvidenceKind, EvidenceLedger
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


def parse_action(text: str) -> Action:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("action must be one JSON object") from exc
    if not isinstance(value, dict) or type(value.get("action")) is not str:
        raise ValueError("action must select a registered action")
    kind = value["action"]
    if kind not in _ACTION_FIELDS or set(value) != _ACTION_FIELDS[kind]:
        raise ValueError("action fields do not match the registered schema")
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
    max_turns: int = 12
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
        max_tokens = 65536 if model is BackendModel.DEEPSEEK_FLASH else 32768
        return cls(max_tokens=max_tokens)


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
        failures: list[str] = []
        observations: list[object] = []
        actions: list[str] = []
        tokens = queries = 0
        for turn in range(1, self.budgets.max_turns + 1):
            completion = self.actor(self.profile, self._context(observations, source))
            if not isinstance(completion, A3Completion):
                raise TypeError("actor must return A3Completion")
            tokens += completion.completion_tokens
            if tokens > self.budgets.max_tokens:
                failures.append("token budget exhausted")
                return TrialResult("budget-exhausted", turn, tokens, verified, profile_result, queries, tuple(failures))
            try:
                if completion.provenance.get("profile_sha256") != self.profile.fingerprint:
                    raise ValueError("completion provenance does not match the cell model")
                action = parse_action(completion.text)
                actions.append(action.kind)
                self._record_action(turn, action)
                if action.kind == "write_source":
                    source, compilation, verified, profile_result = action.source, None, None, None
                    observations.append("candidate source updated")
                elif action.kind == "compile":
                    if source is None:
                        raise ValueError("write_source is required before compile")
                    compilation = self.candidate.compile(
                        source, self.workdir, request_id=self.cell.cell_id,
                        attempt_id=f"turn-{turn}", project_id=self.proposal.project_id,
                        length=self._length(), padded_length=self._padded_length(),
                        block_count=self._block_count(), seed=0,
                        execution_profile=self.execution_profile,
                    )
                    if isinstance(compilation, FailedEvidence):
                        self._retain_candidate_failure(
                            turn, "compile", compilation, failures, observations
                        )
                        continue
                    if not isinstance(compilation, CandidateCompilation):
                        raise TypeError("candidate compiler returned an invalid result")
                    observations.append("host compile passed")
                elif action.kind == "run":
                    if source is None or compilation is None:
                        raise ValueError("successful compile is required before run")
                    result = self.candidate.run(
                        source, self.workdir, request_id=self.cell.cell_id,
                        attempt_id=f"turn-{turn}", project_id=self.proposal.project_id,
                        length=self._length(), padded_length=self._padded_length(),
                        block_count=self._block_count(), seed=0,
                        execution_profile=self.execution_profile,
                    )
                    if isinstance(result, FailedEvidence):
                        self._retain_candidate_failure(
                            turn, "run", result, failures, observations
                        )
                        continue
                    if not isinstance(result, VerifiedResult) or not result.passed:
                        raise ValueError("host verification did not pass")
                    if result.source_fingerprint != compilation.plan.source_fingerprint:
                        raise ValueError("candidate source identity changed after compile")
                    verified = result
                    observations.append("host verification passed")
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
                    return TrialResult("passed", turn, tokens, verified, profile_result, queries, tuple(failures), entry["entry_sha256"])
            except (TypeError, ValueError, RuntimeError) as exc:
                failure = str(exc)
                failures.append(failure)
                observations.append("failure: " + failure)
                self.evidence.append(EvidenceKind.FAILURE, self._identity({"turn": turn, "detail": failure}))
        return TrialResult("budget-exhausted", self.budgets.max_turns, tokens, verified, profile_result, queries, tuple(failures))

    def _submit_gate(self, source, compilation, verified, profile_result) -> str | None:
        if source is None or compilation is None or verified is None:
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
        structured = asdict(failure)
        observations.append({"failed_evidence": structured})
        self.evidence.append(
            EvidenceKind.FAILURE,
            self._identity({"turn": turn, "failed_evidence": failure}),
        )

    def _identity(self, payload: Mapping[str, object]) -> dict[str, object]:
        return {
            "target": "Ascend910B4", "language": "ascend-c",
            "execution_profile": self.execution_profile,
            "runtime": "native-ascend-c",
            "cell_id": self.cell.cell_id, "project_id": self.proposal.project_id,
            **payload,
        }

    def _record_action(self, turn: int, action: Action) -> None:
        self.evidence.append(
            EvidenceKind.ACTION,
            self._identity({"turn": turn, "action": action.kind}),
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
        }}

    @classmethod
    def _profile_statement(cls, result: CompactProfileResult) -> str:
        def render(values) -> str:
            return ",".join(f"{name}={value:.12g}" for name, value in values)
        metrics = render(cls._profile_pairs(result.metric_values))
        timeline = render(cls._profile_pairs(result.timeline))
        return f"raw {result.metric.value}: metrics[{metrics}] timeline[{timeline}]"

    def _context(self, observations: list[object], source: str | None) -> str:
        return json.dumps(
            {
                "allowed_actions": {
                    kind: sorted(fields) for kind, fields in sorted(_ACTION_FIELDS.items())
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
            },
            sort_keys=True,
            separators=(",", ":"),
        )


__all__ = ["A3TrialLoop", "Action", "TrialBudgets", "TrialResult", "parse_action"]
