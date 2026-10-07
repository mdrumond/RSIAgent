"""Live integration boundary for one A3 Phase 2 experiment lineage.

The runner owns treatment admission, immutable memory, retry bookkeeping, and
authoritative learning.  Model, KDB, profiling, and native execution remain in
the existing injected services so qualification can use fakes without creating
a second execution stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable

from benchmarks.a3_experiments import A3ExperimentCell
from benchmarks.a3kernels.live_composition import AuthoritativeResultStore
from benchmarks.a3kernels.phase1_evidence import (
    EvidenceKind, EvidenceLedger, canonical_bytes, canonical_digest,
)
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
    PHASE2_TARGET_SUITE_SHA256, Phase2CandidateAdmissionEvidence,
    Phase2Evidence, Phase2TargetEvidence, read_phase2_evidence,
)
from core.self_evolving_loop import SelfEvolvingStart, TargetVerdict


_MAX_INFRASTRUCTURE_RETRIES = 3
_INFRASTRUCTURE_BACKOFF_SECONDS = 120
_RUN_STATE_SCHEMA = "a3-phase2-live-run-state-v1"
_RESUMABLE_PHASES = frozenset({"source", "evidence", "learning"})
_TARGET_DIRECTORY = re.compile(r"r[0-3]-t([1-9][0-9]*)")
_MAX_DIAGNOSTIC_DETAIL = 512


def _target_diagnostic(result: Phase2Evidence) -> str:
    """Render a bounded diagnostic containing only host target evidence."""
    if isinstance(result, Phase2CandidateAdmissionEvidence):
        return json.dumps({
            "schema": "a3-phase2-target-diagnostic-v1",
            "verdict": "FAIL",
            "failed_case_count": 1,
            "failed_cases": [{
                "stage": result.failure_stage,
                "error_type": result.error_type,
                "detail": result.detail[:_MAX_DIAGNOSTIC_DETAIL],
            }],
        }, sort_keys=True, separators=(",", ":"))
    failed = []
    for item in result.cases:
        if item.passed:
            continue
        case = {"case_id": item.case.case_id, "stage": item.failure_stage}
        for name in ("error_type", "max_abs_error"):
            value = getattr(item, name)
            if value is not None:
                case[name] = value
        if item.detail is not None:
            case["detail"] = item.detail[:_MAX_DIAGNOSTIC_DETAIL]
        mismatch = getattr(item.authority, "mismatch", None)
        if mismatch is not None:
            case["mismatch"] = asdict(mismatch)
        failed.append(case)
    return json.dumps({
        "schema": "a3-phase2-target-diagnostic-v1",
        "verdict": "PASS" if result.passed else "FAIL",
        "failed_case_count": len(failed),
        "failed_cases": failed,
    }, sort_keys=True, separators=(",", ":"))


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
    target_verify: Callable[[str, str, str, Path], Phase2Evidence]
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
        self._state: dict[str, object] | None = None
        self._resume_available = False
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

        self._prepare_recovery(target_direction)
        assert self._state is not None
        first_retry = int(self._state["retry_ordinal"])

        for retry in range(first_retry, _MAX_INFRASTRUCTURE_RETRIES + 1):
            self._retry_ordinal = retry
            adapter = A3Phase2Adapter(
                A3Phase2Identity.from_cell(self.dependencies.cell),
                self._hooks(authority, journal), evidence_path,
            )
            try:
                start, completed_projects = self._resume_boundary(journal)
                result = adapter.run(
                    target_direction, journal.context(), start=start,
                    completed_practice_projects=completed_projects,
                )
                self._checkpoint(phase="complete")
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
                    self._checkpoint(phase="infrastructure-exhausted")
                    raise
                self._checkpoint(
                    phase="retry-wait", retry_ordinal=retry + 1,
                    infrastructure_failures=retry + 1,
                )
                EvidenceLedger(evidence_path).append(EvidenceKind.PLAN, {
                    "phase": "phase2", "event": "INFRASTRUCTURE_RETRY_SCHEDULED",
                    "retry_ordinal": retry + 1,
                    "backoff_seconds": _INFRASTRUCTURE_BACKOFF_SECONDS,
                })
                self.sleeper(_INFRASTRUCTURE_BACKOFF_SECONDS)
                self._checkpoint(phase="retry-ready")
        raise AssertionError("unreachable")

    @property
    def _state_path(self) -> Path:
        return self.root / "run-state.json"

    def _prepare_recovery(self, target_direction: str) -> None:
        direction_sha256 = hashlib.sha256(target_direction.encode("utf-8")).hexdigest()
        state = self._read_state()
        serials = [
            int(match.group(1))
            for path in (self.root / "targets").glob("r*-t*")
            if (match := _TARGET_DIRECTORY.fullmatch(path.name)) is not None
        ]
        self._target_serial = max(serials, default=0)
        if state is None:
            self._state = {
                "schema": _RUN_STATE_SCHEMA,
                "snapshot_id": self.snapshot.snapshot_id,
                "cell_id": self.dependencies.cell.cell_id,
                "target_direction_sha256": direction_sha256,
                "retry_ordinal": 0,
                "infrastructure_failures": 0,
                "serial": self._target_serial,
                "core_attempt": 0,
                "phase": "ready",
                "request_id": None,
                "attempt_id": None,
                "source_sha256": None,
                "evidence_sha256": None,
                "learning_entry_sha256": None,
            }
            self._checkpoint()
            return
        if (
            state["snapshot_id"] != self.snapshot.snapshot_id
            or state["cell_id"] != self.dependencies.cell.cell_id
            or state["target_direction_sha256"] != direction_sha256
        ):
            raise ValueError("persisted Phase 2 run state changed lineage")
        self._state = state
        self._target_serial = max(self._target_serial, int(state["serial"]))
        if state["phase"] == "retry-wait":
            self.sleeper(_INFRASTRUCTURE_BACKOFF_SECONDS)
            self._checkpoint(phase="retry-ready")
        self._resume_available = state["phase"] in _RESUMABLE_PHASES

    def _read_state(self) -> dict[str, object] | None:
        if not self._state_path.exists():
            return None
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            digest = state.pop("state_sha256")
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("invalid Phase 2 run state") from exc
        required = {
            "schema", "snapshot_id", "cell_id", "target_direction_sha256",
            "retry_ordinal", "infrastructure_failures", "serial", "core_attempt",
            "phase", "request_id", "attempt_id", "source_sha256",
            "evidence_sha256", "learning_entry_sha256",
        }
        if (
            set(state) != required
            or state["schema"] != _RUN_STATE_SCHEMA
            or digest != canonical_digest(state)
            or type(state["retry_ordinal"]) is not int
            or not 0 <= state["retry_ordinal"] <= _MAX_INFRASTRUCTURE_RETRIES
            or type(state["serial"]) is not int
            or state["serial"] < 0
        ):
            raise ValueError("invalid Phase 2 run state")
        return state

    def _checkpoint(self, **updates: object) -> None:
        assert self._state is not None
        self._state = {**self._state, **updates}
        value = {**self._state, "state_sha256": canonical_digest(self._state)}
        with tempfile.NamedTemporaryFile(
            "wb", dir=self.root, prefix=".phase2-state-", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self._state_path)

    def _attempt_directory(self) -> Path:
        assert self._state is not None
        return self.root / "targets" / (
            f"r{self._state['retry_ordinal']}-t{self._state['serial']}"
        )

    def _resume_boundary(
        self, journal: Phase2LearningJournal,
    ) -> tuple[SelfEvolvingStart | None, tuple[str, ...]]:
        assert self._state is not None
        core_attempt = int(self._state["core_attempt"])
        if core_attempt <= 1 or self._state["phase"] in {"ready", "complete"}:
            return None, ()
        projects = tuple(
            str(entry["learning"]["project_id"])
            for entry in journal.read()
            if entry["learning"]["kind"] == "practice"
        )
        if not projects:
            raise ValueError("resumed target attempt is missing completed practice")
        return SelfEvolvingStart(
            target_cycles=core_attempt - 1,
            evolutions=core_attempt - 1,
            practice_projects=len(projects),
            final_after_stall=len(projects) == 8,
        ), projects

    @staticmethod
    def _publish_source(path: Path, source: str) -> None:
        value = source.encode("utf-8")
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if path.read_bytes() != value:
                raise RuntimeError("persisted Phase 2 source conflicts") from None
            return
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())

    def _hooks(
        self, authority: AuthoritativeResultStore,
        journal: Phase2LearningJournal,
    ) -> A3Phase2Hooks:
        identity = A3Phase2Identity.from_cell(self.dependencies.cell)

        def fresh(kind: str, value: Any) -> OwnedAgent:
            assert self._state is not None
            return OwnedAgent(
                identity,
                f"{kind}-r{self._state['retry_ordinal']}-t{self._state['serial']}",
                value,
            )

        def environment(_identity, attempt):
            if self._resume_available:
                self._resume_available = False
                workdir = self._attempt_directory()
            else:
                self._target_serial += 1
                request_id = (
                    f"{self.snapshot.snapshot_id[:16]}-r{self._retry_ordinal}"
                    f"-t{self._target_serial}"
                )
                self._checkpoint(
                    retry_ordinal=self._retry_ordinal,
                    serial=self._target_serial,
                    core_attempt=attempt,
                    phase="allocated",
                    request_id=request_id,
                    attempt_id=f"target-r{self._retry_ordinal}-t{self._target_serial}",
                    source_sha256=None,
                    evidence_sha256=None,
                    learning_entry_sha256=None,
                )
                workdir = self._attempt_directory()
            workdir.mkdir(parents=True, exist_ok=True)
            return fresh("environment", workdir)

        def actor(_identity, memory, attempt):
            return fresh("actor", memory)

        def verifier(_identity, _environment, attempt):
            return fresh("verifier", None)

        def work(_identity, active_actor, _verifier, _environment, _direction):
            source_path = self._attempt_directory() / "candidate.ascendc"
            assert self._state is not None
            if self._state["phase"] in _RESUMABLE_PHASES:
                source = source_path.read_text(encoding="utf-8")
                if hashlib.sha256(source.encode("utf-8")).hexdigest() != self._state[
                    "source_sha256"
                ]:
                    raise ValueError("persisted Phase 2 source changed")
                return source
            source = self.dependencies.target_source(
                identity, active_actor.value, self._target_serial,
            )
            if not isinstance(source, str) or not source.strip():
                raise ValueError("target actor must produce non-empty Ascend C source")
            self._publish_source(source_path, source)
            self._checkpoint(
                phase="source",
                source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )
            return source

        def verify(_identity, _verifier, active_environment, _direction, source):
            assert self._state is not None
            request_id = str(self._state["request_id"])
            attempt_id = str(self._state["attempt_id"])
            evidence_path = active_environment.value / "target-evidence.json"
            if evidence_path.exists():
                result = read_phase2_evidence(evidence_path)
            else:
                result = self.dependencies.target_verify(
                    source, request_id, attempt_id, active_environment.value,
                )
            if not isinstance(
                result, (Phase2TargetEvidence, Phase2CandidateAdmissionEvidence)
            ):
                raise TypeError(
                    "target verifier must return authenticated target evidence"
                )
            if result.request_id != request_id or result.attempt_id != attempt_id:
                raise ValueError("target evidence is not bound to the live attempt")
            if result.execution_profile != self.dependencies.execution_profile:
                raise ValueError("target evidence changed the execution treatment")
            candidate = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if result.candidate_sha256 != candidate or candidate != self._state[
                "source_sha256"
            ]:
                raise ValueError("target evidence changed the persisted source")
            if (
                self._state["evidence_sha256"] is not None
                and self._state["evidence_sha256"] != result.attestation_sha256
            ):
                raise ValueError("target evidence changed the persisted checkpoint")
            result.write(evidence_path)
            self._checkpoint(
                phase="evidence", evidence_sha256=result.attestation_sha256,
            )
            return GroundedTarget(
                identity,
                TargetVerdict.PASS if result.passed else TargetVerdict.FAIL,
                _target_diagnostic(result),
                result.attestation_sha256,
            )

        def learn(_identity, _actor, environment, _direction, source,
                  grounded, _memory):
            target = read_phase2_evidence(
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
            learning = Phase2Learning(
                "target-verdict", grounded.report,
                PHASE2_TARGET_SUITE_SHA256, candidate,
                (HostFact(
                    "host-verification", grounded.report,
                    grounded.evidence_sha256,
                    success=grounded.verdict is TargetVerdict.PASS,
                ),),
            )
            matching = []
            for entry in journal.read():
                value = entry["learning"]
                facts = value["host_facts"]
                if any(
                    fact["evidence_sha256"] == grounded.evidence_sha256
                    for fact in facts
                ):
                    matching.append(entry)
            if matching:
                if len(matching) != 1 or matching[0]["learning"] != json.loads(
                    canonical_bytes(asdict(learning))
                ):
                    raise ValueError("persisted Phase 2 learning conflicts")
                entry = matching[0]
            else:
                entry = journal.append(learning)
            self._checkpoint(
                phase="learning",
                learning_entry_sha256=entry["entry_sha256"],
            )
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
