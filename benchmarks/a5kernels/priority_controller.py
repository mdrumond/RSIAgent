"""Deterministic resource-aware orchestration for the A5 experiment campaign."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import fcntl
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from benchmarks.a5kernels.evidence import canonical_bytes


STATE_SCHEMA = "a5-priority-controller-state-v1"
REPORT_SCHEMA = "a5-priority-controller-report-v1"


class ControllerBusy(RuntimeError):
    """Another reconciler owns the campaign controller lock."""


class JobKind(str, Enum):
    PHASE1_RESUME = "phase1-resume"
    PHASE1_RUN = "phase1-run"
    PHASE2_QUALIFY = "phase2-qualify"
    PHASE2_RUN = "phase2-run"


@dataclass(frozen=True)
class Job:
    kind: JobKind
    cell_id: str

    @property
    def job_id(self) -> str:
        return f"{self.kind.value}:{self.cell_id}"


@dataclass(frozen=True)
class RetainedAssignment:
    job: Job
    physical_device: int
    worker_handle: str
    start_identity: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.job, Job)
            or isinstance(self.physical_device, bool)
            or not isinstance(self.physical_device, int)
            or not 1 <= self.physical_device <= 7
            or not isinstance(self.worker_handle, str) or not self.worker_handle
            or isinstance(self.start_identity, bool)
            or not isinstance(self.start_identity, int) or self.start_identity < 0
        ):
            raise ValueError("invalid retained assignment")

    def as_dict(self) -> dict[str, object]:
        return {
            "job_kind": self.job.kind.value,
            "cell_id": self.job.cell_id,
            "physical_device": self.physical_device,
            "worker_handle": self.worker_handle,
            "start_identity": self.start_identity,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RetainedAssignment":
        if set(value) != {
            "job_kind", "cell_id", "physical_device", "worker_handle",
            "start_identity",
        }:
            raise ValueError("invalid controller state assignment")
        try:
            job = Job(JobKind(value["job_kind"]), value["cell_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid controller state assignment") from exc
        device, handle, identity = (
            value["physical_device"], value["worker_handle"], value["start_identity"]
        )
        if (
            not isinstance(job.cell_id, str) or not job.cell_id
            or isinstance(device, bool) or not isinstance(device, int) or not 1 <= device <= 7
            or not isinstance(handle, str) or not handle
            or isinstance(identity, bool) or not isinstance(identity, int) or identity < 0
        ):
            raise ValueError("invalid controller state assignment")
        return cls(job, device, handle, identity)


@dataclass(frozen=True)
class Observation:
    state: str
    exit_code: int | None = None

    def __post_init__(self) -> None:
        if self.state not in {"active", "completed", "incomplete"}:
            raise ValueError("invalid worker observation state")
        if self.state == "active" and self.exit_code is not None:
            raise ValueError("active worker observation cannot have an exit code")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise ValueError("worker exit code must be an integer")

    @classmethod
    def active(cls) -> "Observation":
        return cls("active")

    @classmethod
    def completed(cls, exit_code: int = 0) -> "Observation":
        return cls("completed", exit_code)

    @classmethod
    def incomplete(cls, exit_code: int | None = None) -> "Observation":
        return cls("incomplete", exit_code)


class WorkerPool(Protocol):
    def recover(self) -> Sequence[RetainedAssignment]: ...
    def observe(self, assignment: RetainedAssignment) -> Observation: ...
    def start(self, job: Job, device: int, worker_root: Path,
              command: Sequence[str]) -> RetainedAssignment: ...


@dataclass(frozen=True)
class CampaignReport:
    status: str
    active: tuple[RetainedAssignment, ...]
    incomplete: tuple[Mapping[str, object], ...]
    phase1_completed: tuple[str, ...]
    phase2_terminal: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "status": self.status,
            "active": [item.as_dict() for item in self.active],
            "incomplete": list(self.incomplete),
            "phase1_completed": list(self.phase1_completed),
            "phase2_terminal": list(self.phase2_terminal),
        }


class CampaignController:
    """Reconcile one bounded scheduling pass under a single campaign lock."""

    def __init__(
        self, *, root: Path, phase1, phase2_root: Path, destinations: Sequence,
        inventory, workers: WorkerPool, command: Callable[[Job, int], Sequence[str]],
        phase1_progress: Callable[[str], bool], snapshot: Callable[[object], Path],
        qualified: Callable[[object], bool],
        phase2_terminal: Callable[[object], Mapping[str, object] | None],
    ) -> None:
        self.root = Path(root)
        self.phase1 = phase1
        self.phase2_root = Path(phase2_root)
        self.destinations = tuple(destinations)
        self.inventory = inventory
        self.workers = workers
        self.command = command
        self.phase1_progress = phase1_progress
        self.snapshot = snapshot
        self.qualified = qualified
        self.phase2_terminal = phase2_terminal
        self.state_path = self.root / "controller-state.json"

    @contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "controller.lock").open("a+b") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ControllerBusy("A5 campaign controller is already active") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def reconcile(self) -> CampaignReport:
        with self.lock():
            active, incomplete = self._load_state()
            recovered = tuple(self.workers.recover())
            by_handle = {item.worker_handle: item for item in active}
            for assignment in recovered:
                retained = by_handle.get(assignment.worker_handle)
                if retained is not None and retained != assignment:
                    raise ValueError("recovered worker changed retained identity")
                by_handle[assignment.worker_handle] = assignment
            active = list(by_handle.values())
            retained = []
            for assignment in active:
                observation = self.workers.observe(assignment)
                if observation.state == "active":
                    retained.append(assignment)
                elif observation.state in {"completed", "incomplete"}:
                    if observation.state == "incomplete" or observation.exit_code != 0:
                        incomplete.append({
                            "job_id": assignment.job.job_id,
                            "physical_device": assignment.physical_device,
                            "worker_handle": assignment.worker_handle,
                            "exit_code": observation.exit_code,
                            "classification": "infrastructure-incomplete",
                        })
                else:
                    raise ValueError("worker returned an invalid observation")

            destination_by_source = {
                item.source_cell_id: item for item in self.destinations
            }
            all_cell_ids = {cell.cell_id for cell in self.phase1.cells}
            if not set(destination_by_source) <= all_cell_ids:
                raise ValueError("Phase 2 destinations do not match Phase 1 cells")
            runnable_cells = tuple(
                cell for cell in self.phase1.cells
                if cell.cell_id in destination_by_source
            )
            phase1_completed = tuple(sorted(
                cell.cell_id for cell in runnable_cells
                if self.phase1.load_terminal(cell) is not None
            ))
            completed_set = set(phase1_completed)
            for source_id in phase1_completed:
                self.snapshot(destination_by_source[source_id])

            terminal_ids = tuple(sorted(
                item.cell_id for item in self.destinations
                if self.phase2_terminal(item) is not None
            ))
            active_ids = {item.job.job_id for item in retained}
            jobs = []
            for cell in sorted(runnable_cells, key=lambda item: item.cell_id):
                if cell.cell_id not in completed_set:
                    kind = (JobKind.PHASE1_RESUME if self.phase1_progress(cell.cell_id)
                            else JobKind.PHASE1_RUN)
                    jobs.append(Job(kind, cell.cell_id))
            for item in sorted(self.destinations, key=lambda value: value.cell_id):
                if item.source_cell_id in completed_set and not self.qualified(item):
                    jobs.append(Job(JobKind.PHASE2_QUALIFY, item.cell_id))
            terminal_set = set(terminal_ids)
            for item in sorted(self.destinations, key=lambda value: value.cell_id):
                if (item.source_cell_id in completed_set and self.qualified(item)
                        and item.cell_id not in terminal_set):
                    jobs.append(Job(JobKind.PHASE2_RUN, item.cell_id))
            jobs = [item for item in jobs if item.job_id not in active_ids]

            snapshot = self.inventory.profiling_snapshot()
            occupied = {item.physical_device for item in retained}
            idle = sorted(
                item.physical_device for item in snapshot.eligible_devices
                if item.physical_device not in occupied
            )
            for device, job in zip(idle, jobs):
                worker_root = self.root / "workers" / job.kind.value / job.cell_id
                worker_root.mkdir(parents=True, exist_ok=True)
                command = tuple(self.command(job, device))
                if not command or any(not isinstance(item, str) or not item for item in command):
                    raise ValueError("worker command must be a non-empty argv sequence")
                assignment = self.workers.start(job, device, worker_root, command)
                if assignment.job != job or assignment.physical_device != device:
                    raise ValueError("worker assignment changed scheduled identity")
                retained.append(assignment)

            retained.sort(key=lambda item: item.physical_device)
            if len({item.physical_device for item in retained}) != len(retained):
                raise ValueError("duplicate active device in controller state")
            if len({item.job.job_id for item in retained}) != len(retained):
                raise ValueError("duplicate active job in controller state")
            self._write_state(retained, incomplete)
            complete = len(phase1_completed) == len(runnable_cells) and (
                len(terminal_ids) == len(self.destinations)
            )
            report = CampaignReport(
                "complete" if complete else "incomplete", tuple(retained),
                tuple(incomplete), phase1_completed, terminal_ids,
            )
            self._write(self.root / "report.json", report.as_dict())
            return report

    def _load_state(self):
        if not self.state_path.exists():
            return [], []
        try:
            value = json.loads(self.state_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid controller state") from exc
        if not isinstance(value, dict) or set(value) != {"schema", "active", "incomplete"}:
            raise ValueError("invalid controller state")
        if value["schema"] != STATE_SCHEMA or not isinstance(value["active"], list) \
                or not isinstance(value["incomplete"], list):
            raise ValueError("invalid controller state")
        active = [RetainedAssignment.from_mapping(item) for item in value["active"]]
        if len({item.physical_device for item in active}) != len(active):
            raise ValueError("duplicate active device in controller state")
        if len({item.job.job_id for item in active}) != len(active):
            raise ValueError("duplicate active job in controller state")
        fields = {
            "job_id", "physical_device", "worker_handle", "exit_code",
            "classification",
        }
        if any(
            not isinstance(item, dict) or set(item) != fields
            or not isinstance(item["job_id"], str) or not item["job_id"]
            or isinstance(item["physical_device"], bool)
            or not isinstance(item["physical_device"], int)
            or not 1 <= item["physical_device"] <= 7
            or not isinstance(item["worker_handle"], str) or not item["worker_handle"]
            or item["classification"] != "infrastructure-incomplete"
            or (item["exit_code"] is not None and (
                isinstance(item["exit_code"], bool)
                or not isinstance(item["exit_code"], int)
            ))
            for item in value["incomplete"]
        ):
            raise ValueError("invalid controller state")
        return active, list(value["incomplete"])

    def _write_state(self, active, incomplete) -> None:
        self._write(self.state_path, {
            "schema": STATE_SCHEMA,
            "active": [item.as_dict() for item in active],
            "incomplete": incomplete,
        })

    @staticmethod
    def _write(path: Path, value: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(canonical_bytes(value) + b"\n")
        os.replace(temporary, path)


__all__ = [
    "CampaignController", "CampaignReport", "ControllerBusy", "Job", "JobKind",
    "Observation", "RetainedAssignment", "WorkerPool",
]
