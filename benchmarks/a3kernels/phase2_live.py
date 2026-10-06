"""Live integration boundary for one A3 Phase 2 experiment lineage.

The runner owns treatment admission, immutable memory, retry bookkeeping, and
authoritative learning.  Model, KDB, profiling, and native execution remain in
the existing injected services so qualification can use fakes without creating
a second execution stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Callable

from benchmarks.a3_experiments import A3ExperimentCell
from benchmarks.a3kernels.live_composition import AuthoritativeResultStore
from benchmarks.a3kernels.phase1_evidence import EvidenceKind, EvidenceLedger
from benchmarks.a3kernels.phase1_memory import HostFact
from benchmarks.a3kernels.phase1_registry import CurriculumProposal
from benchmarks.a3kernels.phase2_memory import (
    Phase1Snapshot, Phase2Learning, Phase2LearningJournal, phase2_cell_pairs,
)
from benchmarks.a3kernels.phase2_protocol import (
    A3Phase2Adapter, A3Phase2Hooks, A3Phase2Identity,
    A3Phase2InfrastructureError, A3Phase2Result, CurriculumDecision,
    GroundedLearning, GroundedTarget, OwnedAgent, PracticeResult,
)
from benchmarks.a3kernels.phase2_target import (
    PHASE2_TARGET_SUITE_SHA256, Phase2TargetEvidence,
)
from core.self_evolving_loop import TargetVerdict


_MAX_INFRASTRUCTURE_RETRIES = 3
_INFRASTRUCTURE_BACKOFF_SECONDS = 120


@dataclass(frozen=True)
class PracticeExecution:
    """One existing Phase 1 composition result exposed to Phase 2."""

    project_id: str
    candidate_sha256: str
    evidence_sha256: str
    success: bool
    statement: str

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("practice success must be bool")
        if not self.statement.strip():
            raise ValueError("practice statement must be non-empty")


@dataclass(frozen=True)
class Phase2LiveDependencies:
    """Treatment-bound services composed from the existing A3 live stack."""

    cell: A3ExperimentCell
    execution_profile: str
    target_source: Callable[[A3Phase2Identity, dict[str, object], int], str]
    target_verify: Callable[[str, str, str, Path], Phase2TargetEvidence]
    curriculum: Callable[
        [A3Phase2Identity, GroundedTarget, GroundedLearning,
         dict[str, object], int], CurriculumDecision
    ]
    practice: Callable[
        [A3Phase2Identity, CurriculumProposal, dict[str, object], int],
        PracticeExecution
    ]

    def __post_init__(self) -> None:
        if self.execution_profile not in {"bz-a3-1", "bz-a3-2"}:
            raise ValueError("Phase 2 live execution requires a BZ-A3 profile")
        if not isinstance(self.cell, A3ExperimentCell):
            raise TypeError("Phase 2 live dependencies require an experiment cell")


def phase2_cells() -> tuple[A3ExperimentCell, ...]:
    """Return the canonical eight tiled treatment cells."""
    return tuple(destination for _source, destination in phase2_cell_pairs())


class Phase2LiveRunner:
    """Execute a treatment-isolated Phase 2 lineage over host-owned evidence."""

    def __init__(
        self, *, root: Path, snapshot: Phase1Snapshot,
        dependencies: Phase2LiveDependencies,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = Path(root).resolve()
        self.snapshot = snapshot
        self.dependencies = dependencies
        self.sleeper = sleeper
        self._retry_ordinal = 0
        self._target_serial = 0
        self._validate_treatment()

    def _validate_treatment(self) -> None:
        self.snapshot.verify()
        cell = self.dependencies.cell
        if cell not in phase2_cells():
            raise ValueError("Phase 2 live cell is not a registered tiled treatment")
        expected = {
            "destination_cell_id": cell.cell_id,
            "model": cell.backend_model.value,
            "knowledge": cell.knowledge.value,
            "profiling": cell.profiling.value,
        }
        if any(self.snapshot.manifest.get(key) != value
               for key, value in expected.items()):
            raise ValueError("Phase 1 snapshot and Phase 2 treatment do not match")

    def run(self, target_direction: str) -> A3Phase2Result:
        self.root.mkdir(parents=True, exist_ok=True)
        authority = AuthoritativeResultStore(self.root / "authority.jsonl")
        journal = Phase2LearningJournal(
            self.root / "learning.jsonl", self.snapshot, authority,
        )
        evidence_path = self.root / "evidence.jsonl"
        EvidenceLedger.recover_incomplete_tail(evidence_path)

        for retry in range(_MAX_INFRASTRUCTURE_RETRIES + 1):
            self._retry_ordinal = retry
            adapter = A3Phase2Adapter(
                A3Phase2Identity.from_cell(self.dependencies.cell),
                self._hooks(authority, journal), evidence_path,
            )
            try:
                result = adapter.run(target_direction, journal.context())
                self.snapshot.verify()
                return result
            except A3Phase2InfrastructureError as exc:
                EvidenceLedger(evidence_path).append(EvidenceKind.FAILURE, {
                    "phase": "phase2", "event": "INFRASTRUCTURE_UNVERIFIED",
                    "retry_ordinal": retry, "maximum_retries": 3,
                    "error_type": type(exc).__name__, "detail": str(exc),
                    "semantic_verdict": None,
                })
                if retry == _MAX_INFRASTRUCTURE_RETRIES:
                    raise
                EvidenceLedger(evidence_path).append(EvidenceKind.PLAN, {
                    "phase": "phase2", "event": "INFRASTRUCTURE_RETRY_SCHEDULED",
                    "retry_ordinal": retry + 1,
                    "backoff_seconds": _INFRASTRUCTURE_BACKOFF_SECONDS,
                })
                self.sleeper(_INFRASTRUCTURE_BACKOFF_SECONDS)
        raise AssertionError("unreachable")

    def _hooks(
        self, authority: AuthoritativeResultStore,
        journal: Phase2LearningJournal,
    ) -> A3Phase2Hooks:
        identity = A3Phase2Identity.from_cell(self.dependencies.cell)

        def fresh(kind: str, attempt: int, value: Any) -> OwnedAgent:
            return OwnedAgent(
                identity, f"{kind}-r{self._retry_ordinal}-t{attempt}", value,
            )

        def environment(_identity, attempt):
            self._target_serial += 1
            workdir = self.root / "targets" / f"r{self._retry_ordinal}-t{attempt}"
            workdir.mkdir(parents=True, exist_ok=True)
            return fresh("environment", attempt, workdir)

        def actor(_identity, memory, attempt):
            return fresh("actor", attempt, memory)

        def verifier(_identity, _environment, attempt):
            return fresh("verifier", attempt, None)

        def work(_identity, active_actor, _verifier, _environment, _direction):
            source = self.dependencies.target_source(
                identity, active_actor.value, self._target_serial,
            )
            if not isinstance(source, str) or not source.strip():
                raise ValueError("target actor must produce non-empty Ascend C source")
            return source

        def verify(_identity, _verifier, active_environment, _direction, source):
            serial = self._target_serial
            request_id = (
                f"{self.snapshot.snapshot_id[:16]}-r{self._retry_ordinal}-t{serial}"
            )
            attempt_id = f"target-r{self._retry_ordinal}-t{serial}"
            result = self.dependencies.target_verify(
                source, request_id, attempt_id, active_environment.value,
            )
            if not isinstance(result, Phase2TargetEvidence):
                raise TypeError("target verifier must return Phase2TargetEvidence")
            if result.request_id != request_id or result.attempt_id != attempt_id:
                raise ValueError("target evidence is not bound to the live attempt")
            if result.execution_profile != self.dependencies.execution_profile:
                raise ValueError("target evidence changed the execution treatment")
            result.write(active_environment.value / "target-evidence.json")
            return GroundedTarget(
                identity,
                TargetVerdict.PASS if result.passed else TargetVerdict.FAIL,
                "all held-out cases passed" if result.passed
                else "one or more held-out cases failed",
                result.attestation_sha256,
            )

        def learn(_identity, _actor, environment, _direction, source,
                  grounded, _memory):
            target = Phase2TargetEvidence.read(
                environment.value / "target-evidence.json"
            )
            candidate = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if candidate != target.candidate_sha256:
                raise ValueError("persisted target evidence changed candidate identity")
            authority.register(
                "host-verification", grounded.evidence_sha256,
                PHASE2_TARGET_SUITE_SHA256, candidate,
                grounded.verdict is TargetVerdict.PASS,
            )
            journal.append(Phase2Learning(
                "target-verdict", grounded.report,
                PHASE2_TARGET_SUITE_SHA256, candidate,
                (HostFact(
                    "host-verification", grounded.report,
                    grounded.evidence_sha256,
                    success=grounded.verdict is TargetVerdict.PASS,
                ),),
            ))
            return GroundedLearning(
                identity, journal.context(), grounded.report,
                grounded.evidence_sha256,
            )

        def curriculum(_identity, _direction, grounded, learning, memory, evolution):
            return self.dependencies.curriculum(
                identity, grounded, learning, memory, evolution,
            )

        def practice(_identity, _direction, proposal, memory, evolution):
            completed = self.dependencies.practice(
                identity, proposal, memory, evolution,
            )
            if completed.project_id != proposal.project_id:
                raise ValueError("practice result does not match requested project")
            authority.register(
                "runtime", completed.evidence_sha256, completed.project_id,
                completed.candidate_sha256, completed.success,
            )
            journal.append(Phase2Learning(
                "practice", completed.statement, completed.project_id,
                completed.candidate_sha256,
                (HostFact(
                    "runtime", completed.statement, completed.evidence_sha256,
                    success=completed.success,
                ),),
            ))
            return PracticeResult(
                identity, proposal.project_id, journal.context(),
                completed.evidence_sha256,
            )

        return A3Phase2Hooks(
            fresh_target_environment=environment,
            fresh_target_verifier=verifier,
            fresh_target_actor=actor,
            actor_work=work,
            verify_target=verify,
            learn_target=learn,
            curriculum_review=curriculum,
            run_practice=practice,
            release_target_environment=lambda _identity, _environment: None,
        )


__all__ = [
    "Phase2LiveDependencies", "Phase2LiveRunner", "PracticeExecution",
    "phase2_cells",
]
