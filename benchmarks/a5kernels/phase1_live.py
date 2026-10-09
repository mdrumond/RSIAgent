"""Fail-closed qualification gate for A5 Catlass Phase 1 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable, Iterable, Mapping, Sequence

from benchmarks.a5kernels.bz import RuntimeUnavailableError
from benchmarks.a5kernels.catlass_harness import (
    CATLASS_REVISION,
    HarnessContract,
    HarnessResult,
)
from benchmarks.a5kernels.phase1_experiments import (
    A5ModelIdentity,
    MODEL_IDENTITIES,
    PENDING_PROFILER_REASON,
    ProgrammingGuideIdentity,
    build_cells,
)
from benchmarks.a5kernels.protocol import canonical_hash


GATE_SCHEMA = "a5-catlass-phase1-qualification-v1"
MAX_SEMANTIC_REVISIONS = 3
MAX_INFRASTRUCTURE_RETRIES = 3
BACKOFF_SECONDS = 120
REQUIRED_HOST_PASSES = 2
_AGENTS_PER_PROVIDER = 3
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class InfrastructureFailure(RuntimeError):
    """A retryable provider or remote-execution failure."""


@dataclass(frozen=True)
class QualificationAgent:
    agent_id: str
    ordinal: int
    model: A5ModelIdentity

    def __post_init__(self) -> None:
        if _SAFE_ID.fullmatch(self.agent_id) is None:
            raise ValueError("qualification agent_id is invalid")
        if self.ordinal not in range(1, _AGENTS_PER_PROVIDER + 1):
            raise ValueError("qualification agent ordinal is invalid")
        if self.model not in MODEL_IDENTITIES:
            raise ValueError("qualification model is not registered")

    def as_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "ordinal": self.ordinal,
            "model": self.model.as_dict(),
        }


@dataclass(frozen=True)
class QualificationTask:
    task_id: str
    agent: QualificationAgent
    contract: HarnessContract

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "agent": self.agent.as_dict(),
            "contract": self.contract.value,
        }


@dataclass(frozen=True)
class QualificationRevision:
    revision: int
    source_sha256: str
    source_infrastructure_retries: int
    host_infrastructure_retries: tuple[int, ...]
    host_results: tuple[HarnessResult, ...]
    outcome: str

    def as_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "source_sha256": self.source_sha256,
            "source_infrastructure_retries": self.source_infrastructure_retries,
            "host_infrastructure_retries": list(self.host_infrastructure_retries),
            "host_results": [asdict(item) for item in self.host_results],
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class QualificationRecord:
    task: QualificationTask
    guide_sha256: str
    catlass_revision: str
    revisions: tuple[QualificationRevision, ...]
    admitted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task.as_dict(),
            "guide_sha256": self.guide_sha256,
            "catlass_revision": self.catlass_revision,
            "revisions": [item.as_dict() for item in self.revisions],
            "admitted": self.admitted,
        }


@dataclass(frozen=True)
class GateReport:
    schema: str
    guide: ProgrammingGuideIdentity
    records: tuple[QualificationRecord, ...]
    ready: bool
    report_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "guide": self.guide.as_dict(),
            "records": [item.as_dict() for item in self.records],
            "ready": self.ready,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "report_sha256": self.report_sha256}


class QualificationBlocked(RuntimeError):
    def __init__(self, task: QualificationTask,
                 revisions: Sequence[QualificationRevision]):
        super().__init__(
            f"{task.task_id} exhausted {MAX_SEMANTIC_REVISIONS} semantic revisions"
        )
        self.task = task
        self.revisions = tuple(revisions)


SourceProvider = Callable[
    [QualificationAgent, HarnessContract, int, tuple[QualificationRevision, ...]],
    str,
]


def qualification_agents() -> tuple[QualificationAgent, ...]:
    agents = []
    for model in MODEL_IDENTITIES:
        prefix = model.provider.lower()
        for ordinal in range(1, _AGENTS_PER_PROVIDER + 1):
            agents.append(QualificationAgent(f"{prefix}-{ordinal:02d}", ordinal, model))
    return tuple(agents)


def qualification_tasks() -> tuple[QualificationTask, ...]:
    return tuple(
        QualificationTask(
            f"{agent.agent_id}-{contract.value}", agent, contract,
        )
        for agent in qualification_agents()
        for contract in HarnessContract
    )


class Phase1GateRunner:
    """Ask six isolated agents for source and grade it only on the host."""

    def __init__(self, guide: ProgrammingGuideIdentity, harness,
                 source_provider: SourceProvider, *, sleep=time.sleep):
        if not isinstance(guide, ProgrammingGuideIdentity):
            raise ValueError("guide must be an admitted programming guide")
        self.guide = guide
        self.harness = harness
        self.source_provider = source_provider
        self.sleep = sleep

    def run(self) -> GateReport:
        records = tuple(self._run_task(task) for task in qualification_tasks())
        body = {
            "schema": GATE_SCHEMA,
            "guide": self.guide.as_dict(),
            "records": [item.as_dict() for item in records],
            "ready": True,
        }
        report = GateReport(
            GATE_SCHEMA, self.guide, records, True, canonical_hash(body),
        )
        require_gate(report, self.guide)
        return report

    def _run_task(self, task: QualificationTask) -> QualificationRecord:
        history: list[QualificationRevision] = []
        for revision in range(1, MAX_SEMANTIC_REVISIONS + 1):
            source, source_retries = self._retry(
                lambda: self.source_provider(
                    task.agent, task.contract, revision, tuple(history)
                )
            )
            if not isinstance(source, str) or not source.strip():
                source = ""
            source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
            host_results: list[HarnessResult] = []
            host_retries: list[int] = []
            outcome = "semantic-fail"
            try:
                for pass_number in range(1, REQUIRED_HOST_PASSES + 1):
                    attempt_id = f"{task.task_id}-r{revision}-host{pass_number}"
                    host_result, retries = self._retry(
                        lambda attempt_id=attempt_id: self.harness.run(
                            source, task.contract, attempt_id
                        )
                    )
                    host_results.append(host_result)
                    host_retries.append(retries)
                if all(
                    item.host_verdict == "PASS"
                    and item.exit_code == 0
                    and item.contract == task.contract.value
                    and item.source_sha256 == source_sha
                    and item.catlass_revision == CATLASS_REVISION
                    and item.execution_profile == "bz-a5"
                    for item in host_results
                ):
                    outcome = "admitted"
            except ValueError:
                outcome = "candidate-rejected"
            entry = QualificationRevision(
                revision, source_sha, source_retries, tuple(host_retries),
                tuple(host_results), outcome,
            )
            history.append(entry)
            if outcome == "admitted":
                return QualificationRecord(
                    task, self.guide.guide_sha256, CATLASS_REVISION,
                    tuple(history), True,
                )
        raise QualificationBlocked(task, history)

    def _retry(self, operation):
        failures = 0
        while True:
            try:
                return operation(), failures
            except (InfrastructureFailure, RuntimeUnavailableError):
                if failures >= MAX_INFRASTRUCTURE_RETRIES:
                    raise
                failures += 1
                self.sleep(BACKOFF_SECONDS)


def dry_run_manifest(guide: ProgrammingGuideIdentity) -> dict[str, object]:
    cells = build_cells(guide)
    runnable = tuple(cell for cell in cells if cell.runnable)
    pending = tuple(cell for cell in cells if not cell.runnable)
    return {
        "schema": "a5-catlass-phase1-dry-run-v2",
        "guide": guide.as_dict(),
        "qualification": {
            "agents": [item.as_dict() for item in qualification_agents()],
            "tasks": [item.as_dict() for item in qualification_tasks()],
            "task_count": len(qualification_tasks()),
            "required_host_passes": REQUIRED_HOST_PASSES,
            "max_semantic_revisions": MAX_SEMANTIC_REVISIONS,
            "infrastructure_retry": {
                "max_retries": MAX_INFRASTRUCTURE_RETRIES,
                "backoff_seconds": BACKOFF_SECONDS,
            },
        },
        "phase1": {
            "cells": [cell.as_dict() for cell in runnable],
            "pending_treatment_cells": [
                {**cell.as_dict(), "reason": PENDING_PROFILER_REASON}
                for cell in pending
            ],
        },
    }


def require_gate(report: GateReport, guide: ProgrammingGuideIdentity) -> None:
    expected = qualification_tasks()
    if report.schema != GATE_SCHEMA or report.ready is not True:
        raise ValueError("Phase 1 qualification gate is not ready")
    if report.guide != guide:
        raise ValueError("Phase 1 qualification guide does not match")
    if len(report.records) != 18:
        raise ValueError("Phase 1 qualification requires exactly 18 records")
    if tuple(item.task for item in report.records) != expected:
        raise ValueError("Phase 1 qualification tasks are incomplete or reordered")
    for record in report.records:
        if (
            not record.admitted
            or record.guide_sha256 != guide.guide_sha256
            or record.catlass_revision != CATLASS_REVISION
            or not 1 <= len(record.revisions) <= MAX_SEMANTIC_REVISIONS
        ):
            raise ValueError("Phase 1 qualification record is not admitted")
        for index, revision in enumerate(record.revisions, 1):
            if (
                revision.revision != index
                or _SHA256.fullmatch(revision.source_sha256) is None
                or revision.source_infrastructure_retries
                not in range(MAX_INFRASTRUCTURE_RETRIES + 1)
                or len(revision.host_infrastructure_retries)
                != len(revision.host_results)
                or any(
                    retry not in range(MAX_INFRASTRUCTURE_RETRIES + 1)
                    for retry in revision.host_infrastructure_retries
                )
                or revision.outcome
                not in {"semantic-fail", "candidate-rejected", "admitted"}
                or (index < len(record.revisions) and revision.outcome == "admitted")
            ):
                raise ValueError("Phase 1 qualification attempt lineage is invalid")
        final = record.revisions[-1]
        if final.outcome != "admitted" or len(final.host_results) != 2:
            raise ValueError("Phase 1 qualification lacks two host PASS results")
        for pass_number, result in enumerate(final.host_results, 1):
            if result.catlass_revision != CATLASS_REVISION:
                raise ValueError("Phase 1 qualification Catlass revision drifted")
            if (
                result.host_verdict != "PASS" or result.exit_code != 0
                or result.execution_profile != "bz-a5"
                or result.contract != record.task.contract.value
                or result.source_sha256 != final.source_sha256
                or result.attempt_id != (
                    f"{record.task.task_id}-r{final.revision}-host{pass_number}"
                )
                or not isinstance(result.session_handle, str)
                or not result.session_handle.startswith("bz-a5:")
                or result.record_sha256 != canonical_hash({
                    key: value for key, value in asdict(result).items()
                    if key != "record_sha256"
                })
            ):
                raise ValueError("Phase 1 qualification host evidence is not PASS")
    if report.report_sha256 != canonical_hash(report.body()):
        raise ValueError("Phase 1 qualification report digest is invalid")


def phase1_manifest(
    guide: ProgrammingGuideIdentity,
    report: GateReport | None,
    *,
    completed_cell_ids: Iterable[str] = (),
) -> dict[str, object]:
    if report is None:
        raise ValueError("a complete Phase 1 qualification gate is required")
    require_gate(report, guide)
    cells = build_cells(guide)
    runnable = tuple(cell for cell in cells if cell.runnable)
    pending_treatments = tuple(cell for cell in cells if not cell.runnable)
    completed = tuple(completed_cell_ids)
    registered = {cell.cell_id for cell in runnable}
    if len(set(completed)) != len(completed) or not set(completed) <= registered:
        raise ValueError(
            "completed cell IDs must be unique registered runnable Phase 1 cells"
        )
    complete = len(completed) == len(runnable)
    return {
        "schema": "a5-catlass-phase1-live-manifest-v2",
        "status": "complete-with-pending-treatments" if complete else "ready",
        "guide": guide.as_dict(),
        "qualification_sha256": report.report_sha256,
        "completed_cells": list(completed),
        "pending_cells": [
            cell.as_dict() for cell in runnable
            if cell.cell_id not in set(completed)
        ],
        "pending_treatment_cells": [
            {**cell.as_dict(), "reason": PENDING_PROFILER_REASON}
            for cell in pending_treatments
        ],
    }


def write_gate_report(report: GateReport, path: Path) -> None:
    require_gate(report, report.guide)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_gate_report(path: Path) -> GateReport:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        _exact_keys(
            value,
            {"schema", "guide", "records", "ready", "report_sha256"},
            "qualification report",
        )
        guide = ProgrammingGuideIdentity(**value["guide"])
        records = tuple(_record_from_dict(item) for item in value["records"])
        report = GateReport(
            value["schema"], guide, records, value["ready"],
            value["report_sha256"],
        )
    except (
        KeyError, TypeError, ValueError, StopIteration, OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("unable to load Phase 1 qualification report") from exc
    if report.report_sha256 != canonical_hash(report.body()):
        raise ValueError("Phase 1 qualification report digest is invalid")
    require_gate(report, guide)
    return report


def _record_from_dict(value: Mapping[str, object]) -> QualificationRecord:
    _exact_keys(
        value,
        {"task", "guide_sha256", "catlass_revision", "revisions", "admitted"},
        "qualification record",
    )
    task_value = value["task"]
    _exact_keys(task_value, {"task_id", "agent", "contract"}, "qualification task")
    agent_value = task_value["agent"]
    _exact_keys(agent_value, {"agent_id", "ordinal", "model"}, "qualification agent")
    model_value = agent_value["model"]
    model = next(
        item for item in MODEL_IDENTITIES
        if item.as_dict() == model_value
    )
    agent = QualificationAgent(
        agent_value["agent_id"], agent_value["ordinal"], model,
    )
    task = QualificationTask(
        task_value["task_id"], agent, HarnessContract(task_value["contract"]),
    )
    revisions = []
    revision_keys = {
        "revision", "source_sha256", "source_infrastructure_retries",
        "host_infrastructure_retries", "host_results", "outcome",
    }
    for item in value["revisions"]:
        _exact_keys(item, revision_keys, "qualification revision")
        revisions.append(QualificationRevision(
            item["revision"], item["source_sha256"],
            item["source_infrastructure_retries"],
            tuple(item["host_infrastructure_retries"]),
            tuple(HarnessResult(**result) for result in item["host_results"]),
            item["outcome"],
        ))
    return QualificationRecord(
        task, value["guide_sha256"], value["catlass_revision"], tuple(revisions),
        value["admitted"],
    )


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"invalid {label} fields")


__all__ = [
    "BACKOFF_SECONDS", "GateReport", "InfrastructureFailure",
    "MAX_INFRASTRUCTURE_RETRIES", "MAX_SEMANTIC_REVISIONS",
    "Phase1GateRunner", "QualificationAgent", "QualificationBlocked",
    "QualificationRecord", "QualificationRevision", "QualificationTask",
    "REQUIRED_HOST_PASSES", "dry_run_manifest", "load_gate_report",
    "phase1_manifest", "qualification_agents", "qualification_tasks",
    "require_gate", "write_gate_report",
]
