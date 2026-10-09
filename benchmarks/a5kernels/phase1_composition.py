"""Deterministic live composition for the gate-qualified A5 Phase 1 cells."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable, Protocol, Sequence

from benchmarks.a5kernels.bz import RuntimeUnavailableError
from benchmarks.a5kernels.evidence import (
    EvidenceKind,
    EvidenceLedger,
    canonical_bytes,
    canonical_digest,
)
from benchmarks.a5kernels.knowledge_agent import (
    KnowledgeAgent,
    ProgressiveMemoryJournal,
)
from benchmarks.a5kernels.phase1_experiments import (
    A5Phase1Cell,
    KnowledgeMode,
    PENDING_PROFILER_REASON,
    ProfilingGuidance,
    ProgrammingGuideIdentity,
    build_cells,
)
from benchmarks.a5kernels.phase1_live import (
    BACKOFF_SECONDS,
    MAX_INFRASTRUCTURE_RETRIES,
    GateReport,
    InfrastructureFailure,
    require_gate,
)
from benchmarks.a5kernels.phase1_memory import (
    Phase1LearningJournal,
    Phase1ProjectMemory,
)
from benchmarks.a5kernels.phase1_registry import (
    DEFAULT_PROPOSALS,
    CurriculumProposal,
    dry_run_plan,
)
from benchmarks.a5kernels.phase1_runtime import Phase1ProjectRuntime


PLAN_SCHEMA = "a5-catlass-phase1-live-plan-v1"
TERMINAL_SCHEMA = "a5-catlass-phase1-cell-terminal-v1"
REPORT_SCHEMA = "a5-catlass-phase1-live-report-v1"
A5_PROFILING_GUIDANCE = (
    "After a host-correct run, request bounded BZ-A5 msprof evidence to guide "
    "performance work. Treat timing and counters as measurements, never as "
    "correctness; only host verification establishes correctness."
)


class CellExecutionStatus(str, Enum):
    RUNNABLE = "runnable"
    PENDING = "pending"


@dataclass(frozen=True)
class CellDisposition:
    """Host decision made before a cell can reach treatment dependencies."""

    status: CellExecutionStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CellExecutionStatus):
            raise TypeError("cell execution status must be registered")
        if self.status is CellExecutionStatus.RUNNABLE and self.reason is not None:
            raise ValueError("runnable cells cannot have a pending reason")
        if self.status is CellExecutionStatus.PENDING and not (
            isinstance(self.reason, str) and self.reason.strip()
        ):
            raise ValueError("pending cells require a reason")

    @classmethod
    def runnable(cls) -> "CellDisposition":
        return cls(CellExecutionStatus.RUNNABLE)

    @classmethod
    def pending(cls, reason: str) -> "CellDisposition":
        return cls(CellExecutionStatus.PENDING, reason)


def runnable_cell(cell: A5Phase1Cell) -> CellDisposition:
    """Apply the registry readiness decision before treatment dependencies."""

    return (
        CellDisposition.runnable()
        if cell.runnable
        else CellDisposition.pending(PENDING_PROFILER_REASON)
    )


@dataclass(frozen=True)
class LiveProjectRequest:
    """Host-selected inputs visible to one isolated project agent."""

    cell: A5Phase1Cell
    proposal: CurriculumProposal
    ordinal: int
    guide_text: str
    guide_sha256: str
    qualification_sha256: str
    memory_context: str
    memory_path: Path


class ProjectExecutor(Protocol):
    def __call__(
        self,
        request: LiveProjectRequest,
        runtime: Phase1ProjectRuntime,
        knowledge: KnowledgeAgent,
        profiling_guidance: str | None,
        ledger: EvidenceLedger,
    ) -> Phase1ProjectMemory: ...


@dataclass(frozen=True)
class LiveDependencies:
    """Injected provider/runtime seams; host treatment selection stays local."""

    execute: ProjectExecutor
    knowledge_factory: Callable[[A5Phase1Cell, Path], KnowledgeAgent]

    def __post_init__(self) -> None:
        if not callable(self.execute) or not callable(self.knowledge_factory):
            raise TypeError("live dependencies must be callable")


class Phase1CellComposition:
    """Execute runnable cells and report pending registered treatments."""

    def __init__(
        self,
        root: Path,
        guide_path: Path,
        guide: ProgrammingGuideIdentity,
        qualification: GateReport | None,
        dependencies: LiveDependencies,
        *,
        proposals: Sequence[CurriculumProposal] = DEFAULT_PROPOSALS,
        cell_disposition: Callable[[A5Phase1Cell], CellDisposition] = runnable_cell,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if qualification is None:
            raise ValueError("a complete Phase 1 qualification report is required")
        if not isinstance(guide, ProgrammingGuideIdentity):
            raise TypeError("guide must be a ProgrammingGuideIdentity")
        if not isinstance(dependencies, LiveDependencies):
            raise TypeError("dependencies must be LiveDependencies")
        require_gate(qualification, guide)
        try:
            guide_bytes = Path(guide_path).read_bytes()
            guide_text = guide_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError("unable to read admitted guide content") from exc
        if hashlib.sha256(guide_bytes).hexdigest() != guide.guide_sha256:
            raise ValueError("guide content does not match admitted guide identity")

        plan = dry_run_plan(proposals)
        canonical = tuple(
            CurriculumProposal.from_mapping({
                key: row[key]
                for key in ("family", "parameters", "hypothesis", "evidence_preset")
            })
            for row in plan["projects"]
        )
        if tuple(proposals) != canonical or not canonical:
            raise ValueError("projects must use non-empty canonical registry order")

        self.root = Path(root)
        self.guide = guide
        self.guide_text = guide_text
        self.qualification = qualification
        self.dependencies = dependencies
        self.proposals = canonical
        self.cells = build_cells(guide)
        dispositions = tuple(
            runnable_cell(cell) if not cell.runnable else cell_disposition(cell)
            for cell in self.cells
        )
        if any(not isinstance(item, CellDisposition) for item in dispositions):
            raise TypeError("cell_disposition must return CellDisposition values")
        self._cell_dispositions = dict(zip(self.cells, dispositions, strict=True))
        self.sleeper = sleeper

    @classmethod
    def one_project_smoke(
        cls,
        root: Path,
        guide_path: Path,
        guide: ProgrammingGuideIdentity,
        qualification: GateReport | None,
        dependencies: LiveDependencies,
        *,
        cell_disposition: Callable[[A5Phase1Cell], CellDisposition] = runnable_cell,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> "Phase1CellComposition":
        """Build the resumable smoke over exactly the first registered project."""

        return cls(
            root,
            guide_path,
            guide,
            qualification,
            dependencies,
            proposals=(DEFAULT_PROPOSALS[0],),
            cell_disposition=cell_disposition,
            sleeper=sleeper,
        )

    def plan(self) -> dict[str, object]:
        """Return the complete stable live plan without invoking a provider."""

        plan = {
            "schema": PLAN_SCHEMA,
            "guide": self.guide.as_dict(),
            "qualification_sha256": self.qualification.report_sha256,
            "infrastructure_retry": {
                "max_retries": MAX_INFRASTRUCTURE_RETRIES,
                "backoff_seconds": BACKOFF_SECONDS,
            },
            "cells": [
                cell.as_dict() for cell in self.cells if self._is_runnable(cell)
            ],
            "projects": dry_run_plan(self.proposals)["projects"],
        }
        pending = self._pending_treatment_cells()
        if pending:
            plan["pending_treatment_cells"] = pending
        return plan

    def run(self) -> tuple[dict[str, object], ...]:
        self._ensure_plan()
        records = []
        for cell in self.cells:
            if not self._is_runnable(cell):
                continue
            terminal = self._load_terminal(cell)
            if terminal is None:
                terminal = self._run_cell(cell)
            records.append(terminal)
        return tuple(records)

    def resume(self) -> tuple[dict[str, object], ...]:
        return self.run()

    def report(self) -> dict[str, object]:
        records = []
        for cell in self.cells:
            if not self._is_runnable(cell):
                continue
            terminal = self._load_terminal(cell)
            if terminal is not None:
                records.append(terminal)
        completed = {row["cell_id"] for row in records}
        runnable = [cell for cell in self.cells if self._is_runnable(cell)]
        pending_treatments = self._pending_treatment_cells()
        complete = len(records) == len(runnable)
        report = {
            "schema": REPORT_SCHEMA,
            "status": (
                "complete-with-pending-treatments"
                if complete and pending_treatments
                else "complete" if complete else "pending"
            ),
            "guide": self.guide.as_dict(),
            "qualification_sha256": self.qualification.report_sha256,
            "completed_cell_ids": [row["cell_id"] for row in records],
            "pending_cell_ids": [
                cell.cell_id for cell in runnable if cell.cell_id not in completed
            ],
            "records": records,
        }
        if pending_treatments:
            report["pending_treatment_cells"] = pending_treatments
        return report

    def _is_runnable(self, cell: A5Phase1Cell) -> bool:
        return self._cell_dispositions[cell].status is CellExecutionStatus.RUNNABLE

    def _pending_treatment_cells(self) -> list[dict[str, object]]:
        return [
            {**cell.as_dict(), "reason": disposition.reason or ""}
            for cell in self.cells
            if (disposition := self._cell_dispositions[cell]).status
            is CellExecutionStatus.PENDING
        ]

    def evidence(self, cell_id: str) -> EvidenceLedger:
        registered = {cell.cell_id for cell in self.cells}
        if cell_id not in registered:
            raise ValueError("cell_id is not registered in this Phase 1 plan")
        return EvidenceLedger(self.root / "cells" / cell_id / "evidence.jsonl")

    def _run_cell(self, cell: A5Phase1Cell) -> dict[str, object]:
        cell_root = self.root / "cells" / cell.cell_id
        ledger = EvidenceLedger(cell_root / "evidence.jsonl")
        journal = Phase1LearningJournal(cell_root / "memory.jsonl", self.proposals)
        state = journal.resume_state()
        for ordinal in range(state.completed_projects + 1, len(self.proposals) + 1):
            proposal = self.proposals[ordinal - 1]
            memory_path = cell_root / "agent-memory" / f"project-{ordinal:02d}"
            memory_path.mkdir(parents=True, exist_ok=True)
            request = LiveProjectRequest(
                cell=cell,
                proposal=proposal,
                ordinal=ordinal,
                guide_text=self.guide_text,
                guide_sha256=self.guide.guide_sha256,
                qualification_sha256=self.qualification.report_sha256,
                memory_context=journal.project_context(),
                memory_path=memory_path,
            )
            runtime = Phase1ProjectRuntime.from_proposal(proposal)
            ledger.append(EvidenceKind.REQUEST, {
                "cell": cell.as_dict(),
                "ordinal": ordinal,
                "proposal": proposal.as_dict(),
                "guide_sha256": self.guide.guide_sha256,
                "qualification_sha256": self.qualification.report_sha256,
            })
            memory = self._execute_with_retries(
                request, runtime, ledger,
            )
            self._validate_memory(cell, proposal, memory)
            ledger.append(EvidenceKind.RESULT, {
                "cell_id": cell.cell_id,
                "ordinal": ordinal,
                "memory": memory.as_dict(),
            })
            journal.append(memory)

        state = journal.resume_state()
        if state.completed_projects != len(self.proposals):
            raise RuntimeError("cell ended before every registered project committed")
        body = {
            "schema": TERMINAL_SCHEMA,
            "status": "passed",
            "cell_id": cell.cell_id,
            "model": cell.model.as_dict(),
            "guide_sha256": self.guide.guide_sha256,
            "qualification_sha256": self.qualification.report_sha256,
            "completed_projects": state.completed_projects,
            "memory_head_sha256": state.head_sha256,
            "evidence_head_sha256": ledger.head_sha256,
        }
        terminal = {**body, "record_sha256": canonical_digest(body)}
        self._atomic_write(self._terminal_path(cell), terminal)
        return terminal

    def _execute_with_retries(
        self,
        request: LiveProjectRequest,
        runtime: Phase1ProjectRuntime,
        ledger: EvidenceLedger,
    ) -> Phase1ProjectMemory:
        failures = 0
        while True:
            knowledge = self._knowledge(request.cell, request.memory_path)
            guidance = (
                A5_PROFILING_GUIDANCE
                if request.cell.profiling is ProfilingGuidance.WITH_GUIDANCE
                else None
            )
            try:
                return self.dependencies.execute(
                    request, runtime, knowledge, guidance, ledger,
                )
            except (InfrastructureFailure, RuntimeUnavailableError):
                if failures >= MAX_INFRASTRUCTURE_RETRIES:
                    raise
                failures += 1
                ledger.append(EvidenceKind.RESULT, {
                    "event": "infrastructure-retry",
                    "cell_id": request.cell.cell_id,
                    "ordinal": request.ordinal,
                    "retry": failures,
                    "backoff_seconds": BACKOFF_SECONDS,
                })
                self.sleeper(BACKOFF_SECONDS)
            finally:
                knowledge.close()

    def _knowledge(self, cell: A5Phase1Cell, memory_path: Path) -> KnowledgeAgent:
        if cell.knowledge is KnowledgeMode.WITHOUT_KDB:
            return KnowledgeAgent(
                enabled=False,
                journal=ProgressiveMemoryJournal(memory_path / "knowledge.jsonl"),
            )
        knowledge = self.dependencies.knowledge_factory(cell, memory_path)
        if not isinstance(knowledge, KnowledgeAgent) or not knowledge.enabled:
            raise ValueError("KDB-on cells require the enabled Knowledge Agent boundary")
        return knowledge

    @staticmethod
    def _validate_memory(
        cell: A5Phase1Cell,
        proposal: CurriculumProposal,
        memory: Phase1ProjectMemory,
    ) -> None:
        if not isinstance(memory, Phase1ProjectMemory) or memory.proposal != proposal:
            raise ValueError("project executor returned memory for a different proposal")
        if cell.knowledge is KnowledgeMode.WITHOUT_KDB and memory.kdb_citations:
            raise ValueError("KDB-off cells cannot commit KDB citations")

    def _ensure_plan(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "plan.json"
        expected = canonical_bytes(self.plan()) + b"\n"
        if path.exists():
            if path.read_bytes() != expected:
                raise ValueError("existing live plan does not match this run")
            return
        path.write_bytes(expected)

    def _load_terminal(self, cell: A5Phase1Cell) -> dict[str, object] | None:
        path = self._terminal_path(cell)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("unable to load cell terminal") from exc
        expected = {
            "schema", "status", "cell_id", "model", "guide_sha256",
            "qualification_sha256", "completed_projects", "memory_head_sha256",
            "evidence_head_sha256", "record_sha256",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("invalid cell terminal fields")
        digest = value["record_sha256"]
        body = {key: item for key, item in value.items() if key != "record_sha256"}
        if (
            digest != canonical_digest(body)
            or value["schema"] != TERMINAL_SCHEMA
            or value["status"] != "passed"
            or value["cell_id"] != cell.cell_id
            or value["model"] != cell.model.as_dict()
            or value["guide_sha256"] != self.guide.guide_sha256
            or value["qualification_sha256"] != self.qualification.report_sha256
            or value["completed_projects"] != len(self.proposals)
        ):
            raise ValueError("cell terminal does not match the live plan")
        journal = Phase1LearningJournal(
            self.root / "cells" / cell.cell_id / "memory.jsonl", self.proposals,
        )
        ledger = self.evidence(cell.cell_id)
        if (
            journal.resume_state().head_sha256 != value["memory_head_sha256"]
            or ledger.head_sha256 != value["evidence_head_sha256"]
        ):
            raise ValueError("cell terminal does not match retained evidence")
        return value

    def _terminal_path(self, cell: A5Phase1Cell) -> Path:
        return self.root / "cells" / cell.cell_id / "terminal.json"

    @staticmethod
    def _atomic_write(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(canonical_bytes(value) + b"\n")
        os.replace(temporary, path)


__all__ = [
    "A5_PROFILING_GUIDANCE",
    "CellDisposition",
    "CellExecutionStatus",
    "LiveDependencies",
    "LiveProjectRequest",
    "Phase1CellComposition",
    "runnable_cell",
]
