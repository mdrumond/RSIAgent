"""Typed A3/Ascend-C Phase 2 contracts over the generic evolution loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Any

from benchmarks.a3_experiments import (
    A3ExperimentCell,
    A3Language,
    A3Target,
    BackendModel,
    KnowledgeMode,
    ProfilingGuidance,
    ProgrammingLevel,
)
from benchmarks.a3kernels.phase1_evidence import EvidenceKind, EvidenceLedger
from benchmarks.a3kernels.phase1_registry import CurriculumProposal
from core.self_evolving_loop import (
    ActorLearning,
    EvolutionResult,
    EvolutionStatus,
    SelfEvolvingLoopHooks,
    TargetVerdict,
    TargetVerification,
    run_self_evolving_loop,
)


MAX_PRACTICE_PROJECTS = 8
MAX_TARGET_ATTEMPTS = 9
_SHA = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class A3Phase2ProtocolError(RuntimeError):
    """A Phase 2 agent violated an A3-owned protocol contract."""


class A3Phase2InfrastructureError(RuntimeError):
    """Infrastructure prevented a grounded semantic verdict."""


class A3Phase2BudgetStop(RuntimeError):
    """The bounded Phase 2 lineage cannot begin another action."""


@dataclass(frozen=True)
class A3Phase2Identity:
    cell_id: str
    backend_model: BackendModel
    knowledge: KnowledgeMode
    profiling: ProfilingGuidance
    programming_level: ProgrammingLevel = ProgrammingLevel.TILED
    target: str = "Ascend910B4"
    language: str = "ascend-c"
    runtime: str = "native-ascend-c"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.backend_model, BackendModel)
            or not isinstance(self.knowledge, KnowledgeMode)
            or not isinstance(self.profiling, ProfilingGuidance)
            or self.programming_level is not ProgrammingLevel.TILED
            or self.target != "Ascend910B4"
            or self.language != "ascend-c"
            or self.runtime != "native-ascend-c"
            or not isinstance(self.cell_id, str)
            or _SAFE_ID.fullmatch(self.cell_id) is None
        ):
            raise ValueError("Phase 2 identity must be tiled A3 native Ascend C")

    @classmethod
    def from_cell(cls, cell: A3ExperimentCell) -> "A3Phase2Identity":
        if (
            not isinstance(cell, A3ExperimentCell)
            or cell.target is not A3Target.A3
            or cell.language is not A3Language.ASCEND_C
            or cell.programming_level is not ProgrammingLevel.TILED
        ):
            raise ValueError("Phase 2 requires a registered tiled A3 Ascend C cell")
        return cls(
            cell.cell_id,
            cell.backend_model,
            cell.knowledge,
            cell.profiling,
            cell.programming_level,
        )

    def evidence_fields(self) -> dict[str, str]:
        return {
            "cell_id": self.cell_id,
            "target": self.target,
            "language": self.language,
            "runtime": self.runtime,
            "backend_model": self.backend_model.value,
            "knowledge": self.knowledge.value,
            "profiling": self.profiling.value,
            "programming_level": self.programming_level.value,
        }


@dataclass(frozen=True)
class OwnedAgent:
    identity: A3Phase2Identity
    instance_id: str
    value: Any

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, A3Phase2Identity)
            or not isinstance(self.instance_id, str)
            or _SAFE_ID.fullmatch(self.instance_id) is None
        ):
            raise ValueError("owned agent requires A3 identity and a stable instance ID")


@dataclass(frozen=True)
class GroundedTarget:
    identity: A3Phase2Identity
    verdict: TargetVerdict
    report: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        try:
            verdict = TargetVerdict(self.verdict)
        except (TypeError, ValueError) as exc:
            raise ValueError("target verdict must be PASS or FAIL") from exc
        if verdict not in (TargetVerdict.PASS, TargetVerdict.FAIL):
            raise ValueError("target verdict must be grounded PASS or FAIL")
        object.__setattr__(self, "verdict", verdict)
        if (
            not isinstance(self.identity, A3Phase2Identity)
            or not isinstance(self.report, str)
            or not self.report.strip()
            or _SHA.fullmatch(self.evidence_sha256) is None
        ):
            raise ValueError("grounded target evidence is invalid")


@dataclass(frozen=True)
class GroundedLearning:
    identity: A3Phase2Identity
    memory: Any
    diagnosis: str
    grounding_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, A3Phase2Identity)
            or not isinstance(self.diagnosis, str)
            or not self.diagnosis.strip()
            or _SHA.fullmatch(self.grounding_sha256) is None
        ):
            raise ValueError("grounded learning evidence is invalid")


class CurriculumStatus(str, Enum):
    READY = "ready"
    PRACTICE = "practice"


@dataclass(frozen=True)
class CurriculumDecision:
    identity: A3Phase2Identity
    status: CurriculumStatus
    reason: str
    project: CurriculumProposal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, A3Phase2Identity):
            raise ValueError("Curriculum decision requires A3 identity")
        if not isinstance(self.status, CurriculumStatus):
            raise ValueError("Curriculum must decide ready or practice")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Curriculum decision reason must be nonempty")
        if (self.status is CurriculumStatus.PRACTICE) != isinstance(
            self.project, CurriculumProposal
        ):
            raise ValueError("practice requires exactly one registered A3 project")

    @classmethod
    def ready(cls, identity: A3Phase2Identity, reason: str) -> "CurriculumDecision":
        return cls(identity, CurriculumStatus.READY, reason)

    @classmethod
    def practice(
        cls, identity: A3Phase2Identity, project: CurriculumProposal, reason: str
    ) -> "CurriculumDecision":
        return cls(identity, CurriculumStatus.PRACTICE, reason, project)


@dataclass(frozen=True)
class PracticeResult:
    identity: A3Phase2Identity
    project_id: str
    memory: Any
    evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, A3Phase2Identity)
            or _SHA.fullmatch(self.project_id) is None
            or _SHA.fullmatch(self.evidence_sha256) is None
        ):
            raise ValueError("practice result evidence is invalid")


@dataclass(frozen=True)
class A3Phase2Hooks:
    fresh_target_environment: Callable[[A3Phase2Identity, int], OwnedAgent]
    fresh_target_verifier: Callable[[A3Phase2Identity, OwnedAgent, int], OwnedAgent]
    fresh_target_actor: Callable[[A3Phase2Identity, Any, int], OwnedAgent]
    actor_work: Callable[[A3Phase2Identity, OwnedAgent, OwnedAgent, OwnedAgent, str], Any]
    verify_target: Callable[[A3Phase2Identity, OwnedAgent, OwnedAgent, str, Any], GroundedTarget]
    learn_target: Callable[[A3Phase2Identity, OwnedAgent, OwnedAgent, str, Any, GroundedTarget, Any], GroundedLearning]
    curriculum_review: Callable[[A3Phase2Identity, str, GroundedTarget, GroundedLearning, Any, int], CurriculumDecision]
    run_practice: Callable[[A3Phase2Identity, str, CurriculumProposal, Any, int], PracticeResult]
    release_target_environment: Callable[[A3Phase2Identity, OwnedAgent], None]


@dataclass(frozen=True)
class A3Phase2Result:
    identity: A3Phase2Identity
    output: Any
    environment: OwnedAgent
    verdict: TargetVerdict
    report: str
    memory: Any
    target_attempts: int
    practice_projects: int
    termination: str
    evidence_sha256: str


class A3Phase2Adapter:
    """Validate A3 agent contracts while delegating sequencing to the core loop."""

    def __init__(
        self,
        identity: A3Phase2Identity,
        hooks: A3Phase2Hooks,
        evidence_path: str | Path,
    ) -> None:
        if not isinstance(identity, A3Phase2Identity):
            raise TypeError("identity must be A3Phase2Identity")
        if not isinstance(hooks, A3Phase2Hooks):
            raise TypeError("hooks must be A3Phase2Hooks")
        self.identity = identity
        self.hooks = hooks
        self.evidence = EvidenceLedger(evidence_path)

    def _require_identity(self, value: Any, label: str) -> None:
        if getattr(value, "identity", None) != self.identity:
            raise A3Phase2ProtocolError(f"{label} identity drift")

    def _event(
        self, event: str, details: dict[str, Any], kind: EvidenceKind = EvidenceKind.ACTION
    ) -> None:
        self.evidence.append(
            kind,
            {
                **self.identity.evidence_fields(),
                "phase": "phase2",
                "event": event,
                **details,
            },
        )

    def run(self, target_direction: str, memory: Any) -> A3Phase2Result:
        seen_targets: set[str] = set()
        seen_projects: set[str] = set()
        current_verification: GroundedTarget | None = None
        current_learning: GroundedLearning | None = None
        practice_count = 0

        self._event(
            "PHASE2_STARTED",
            {
                "stop_policy": "curriculum_review",
                "max_practice_projects": MAX_PRACTICE_PROJECTS,
                "max_target_attempts": MAX_TARGET_ATTEMPTS,
            },
            EvidenceKind.PLAN,
        )

        def environment(_direction: str, attempt: int) -> OwnedAgent:
            if attempt > MAX_TARGET_ATTEMPTS:
                self._event(
                    "PHASE2_BUDGET_STOPPED",
                    {"target_attempts": attempt - 1, "practice_projects": practice_count},
                    EvidenceKind.RESULT,
                )
                raise A3Phase2BudgetStop("nine target attempts exhausted")
            item = self.hooks.fresh_target_environment(self.identity, attempt)
            self._require_identity(item, "target environment")
            if not isinstance(item, OwnedAgent) or item.instance_id in seen_targets:
                raise A3Phase2ProtocolError("target environment is not fresh")
            seen_targets.add(item.instance_id)
            return item

        def verifier(env: OwnedAgent, _direction: str, attempt: int) -> OwnedAgent:
            item = self.hooks.fresh_target_verifier(self.identity, env, attempt)
            self._require_identity(item, "target verifier")
            if not isinstance(item, OwnedAgent):
                raise A3Phase2ProtocolError("target verifier contract is invalid")
            return item

        def actor(active_memory: Any, attempt: int) -> OwnedAgent:
            item = self.hooks.fresh_target_actor(self.identity, active_memory, attempt)
            self._require_identity(item, "target actor")
            if not isinstance(item, OwnedAgent):
                raise A3Phase2ProtocolError("target actor contract is invalid")
            return item

        def work(
            active_actor: OwnedAgent, active_verifier: OwnedAgent,
            env: OwnedAgent, direction: str,
        ) -> Any:
            return self.hooks.actor_work(
                self.identity, active_actor, active_verifier, env, direction
            )

        def verify(
            active_verifier: OwnedAgent, env: OwnedAgent,
            direction: str, output: Any,
        ) -> TargetVerification:
            nonlocal current_verification
            item = self.hooks.verify_target(
                self.identity, active_verifier, env, direction, output
            )
            self._require_identity(item, "target verification")
            if not isinstance(item, GroundedTarget):
                raise A3Phase2ProtocolError("target verification contract is invalid")
            current_verification = item
            return TargetVerification(item.verdict, item.report)

        def learn(
            active_actor: OwnedAgent, env: OwnedAgent, direction: str,
            output: Any, verdict: TargetVerdict, report: str, active_memory: Any,
        ) -> ActorLearning:
            nonlocal current_learning
            verified = current_verification
            if verified is None or verified.verdict is not verdict or verified.report != report:
                raise A3Phase2ProtocolError("learning is not bound to target verification")
            item = self.hooks.learn_target(
                self.identity, active_actor, env, direction, output, verified, active_memory
            )
            self._require_identity(item, "target learning")
            if (
                not isinstance(item, GroundedLearning)
                or item.grounding_sha256 != verified.evidence_sha256
            ):
                raise A3Phase2ProtocolError("learning is not grounded by target evidence")
            current_learning = item
            return ActorLearning(item.memory, item.diagnosis)

        def evolve(
            direction: str, verdict: TargetVerdict, report: str,
            diagnosis: str, active_memory: Any, evolution: int,
        ) -> EvolutionResult:
            nonlocal practice_count
            verified, learned = current_verification, current_learning
            if (
                verified is None
                or learned is None
                or verified.verdict is not verdict
                or verified.report != report
                or learned.diagnosis != diagnosis
            ):
                raise A3Phase2ProtocolError("Curriculum input is not grounded learning")
            decision = self.hooks.curriculum_review(
                self.identity, direction, verified, learned, active_memory, evolution
            )
            self._require_identity(decision, "Curriculum decision")
            if not isinstance(decision, CurriculumDecision):
                raise A3Phase2ProtocolError("Curriculum decision contract is invalid")
            self._event(
                "CURRICULUM_DECIDED",
                {"evolution": evolution, "decision": decision.status.value,
                 "reason": decision.reason},
            )
            if decision.status is CurriculumStatus.READY:
                return EvolutionResult(
                    EvolutionStatus.READY_FOR_RETRY, active_memory, 0, decision.reason
                )
            assert decision.project is not None
            project_id = decision.project.project_id
            if practice_count >= MAX_PRACTICE_PROJECTS:
                self._event(
                    "PHASE2_BUDGET_STOPPED",
                    {"target_attempts": len(seen_targets),
                     "practice_projects": practice_count},
                    EvidenceKind.RESULT,
                )
                raise A3Phase2BudgetStop("eight practice projects exhausted")
            if project_id in seen_projects:
                raise A3Phase2ProtocolError("focused practice project must be fresh")
            practiced = self.hooks.run_practice(
                self.identity, direction, decision.project, active_memory, evolution
            )
            self._require_identity(practiced, "practice result")
            if not isinstance(practiced, PracticeResult) or practiced.project_id != project_id:
                raise A3Phase2ProtocolError("practice result does not match its project")
            seen_projects.add(project_id)
            practice_count += 1
            self._event(
                "PRACTICE_COMPLETED",
                {"evolution": evolution, "project_id": project_id,
                 "practice_projects": practice_count,
                 "evidence_sha256": practiced.evidence_sha256,
                 "fresh_target_required": True},
                EvidenceKind.RESULT,
            )
            status = (
                EvolutionStatus.STALLED
                if practice_count == MAX_PRACTICE_PROJECTS
                else EvolutionStatus.READY_FOR_RETRY
            )
            return EvolutionResult(status, practiced.memory, 1, decision.reason)

        core_hooks = SelfEvolvingLoopHooks(
            fresh_target_environment=environment,
            fresh_target_verifier=verifier,
            fresh_target_actor=actor,
            actor_work=work,
            verifier_verify=verify,
            actor_learn=learn,
            evolve=evolve,
            learn_on_pass=True,
            curriculum_after_pass=True,
            release_target_environment=lambda env: self.hooks.release_target_environment(
                self.identity, env
            ),
            on_event=lambda event, payload: self._event(event, payload),
        )
        try:
            result = run_self_evolving_loop(target_direction, memory, core_hooks)
        except A3Phase2InfrastructureError as exc:
            self._event(
                "PHASE2_INFRASTRUCTURE_EXCEPTION",
                {"error_type": type(exc).__name__, "error": str(exc)},
                EvidenceKind.FAILURE,
            )
            raise
        except A3Phase2BudgetStop:
            raise
        except A3Phase2ProtocolError as exc:
            self._event(
                "PHASE2_PROTOCOL_EXCEPTION",
                {"error_type": type(exc).__name__, "error": str(exc)},
                EvidenceKind.FAILURE,
            )
            raise
        except Exception as exc:
            self._event(
                "PHASE2_PROTOCOL_EXCEPTION",
                {"error_type": type(exc).__name__, "error": str(exc)},
                EvidenceKind.FAILURE,
            )
            raise

        termination = result.termination
        if practice_count == MAX_PRACTICE_PROJECTS:
            termination = "practice_budget_final_target"
        self._event(
            "PHASE2_COMPLETED",
            {"target_attempts": result.target_cycles,
             "practice_projects": result.practice_projects,
             "verdict": result.verifier_verdict.value,
             "termination": termination},
            EvidenceKind.RESULT,
        )
        if not isinstance(result.environment, OwnedAgent):
            raise A3Phase2ProtocolError("terminal target environment contract is invalid")
        return A3Phase2Result(
            self.identity,
            result.output,
            result.environment,
            result.verifier_verdict,
            result.verifier_report,
            result.memory,
            result.target_cycles,
            result.practice_projects,
            termination,
            self.evidence.head_sha256,
        )


__all__ = [
    "A3Phase2Adapter", "A3Phase2BudgetStop", "A3Phase2Hooks",
    "A3Phase2Identity", "A3Phase2InfrastructureError", "A3Phase2ProtocolError",
    "A3Phase2Result", "CurriculumDecision", "CurriculumStatus",
    "GroundedLearning", "GroundedTarget", "MAX_PRACTICE_PROJECTS",
    "MAX_TARGET_ATTEMPTS", "OwnedAgent", "PracticeResult",
]
