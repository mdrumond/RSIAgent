"""Adaptive A5 Phase-1 selection, wave execution, and atomic learning."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping

from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec, WaveSpec
from benchmarks.a5kernels.phase1_curriculum_selector import (
    CurriculumAgent,
    CurriculumContext,
    CurriculumSession,
    PriorOutcome,
    ProjectInformation,
    TerminalDecision,
    WaveDecision,
)
from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory
from benchmarks.a5kernels.phase1_project import (
    ExecutionEvidence,
    HostTensor,
    ReferenceOutput,
)
from benchmarks.a5kernels.phase1_wave_runtime import (
    LiveWaveCoordinator,
    WaveProjectRequest,
    WaveProjectResult,
)


STATE_SCHEMA = "a5-phase1-adaptive-state-v2"
PENDING_SCHEMA = "a5-phase1-adaptive-pending-v1"
MEMORY_SCHEMA = "a5-phase1-adaptive-memory-v1"
DELTA_SCHEMA = "a5-phase1-adaptive-memory-delta-v1"


def _validated_summary(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
    ):
        raise ValueError("adaptive result summary must be trimmed and bounded")
    return value


@dataclass(frozen=True)
class AdaptiveProjectRequest:
    """Explicit adapter input for one Production project execution."""

    spec: ProjectSpec
    authored_ordinal: int
    pre_wave_memory: bytes
    branch_root: Path
    inputs: tuple[HostTensor, ...]
    oracle: ReferenceOutput


@dataclass(frozen=True)
class AdaptiveProjectResult:
    """One evidence-layered project result before wave publication."""

    memory: Phase1ProjectMemory
    evidence: ExecutionEvidence
    evidence_payload: bytes
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.memory, Phase1ProjectMemory):
            raise TypeError("adaptive result memory must be Phase1ProjectMemory")
        if not isinstance(self.evidence, ExecutionEvidence):
            raise TypeError("adaptive result evidence must be ExecutionEvidence")
        if not isinstance(self.evidence_payload, bytes) or not self.evidence_payload:
            raise TypeError("adaptive result evidence payload must be non-empty bytes")
        _validated_summary(self.summary)


AdaptiveProjectExecutor = Callable[[AdaptiveProjectRequest], AdaptiveProjectResult]


class AdaptivePhase1Coordinator:
    """Select waves from published memory and atomically publish verified deltas."""

    def __init__(
        self,
        root: Path,
        objective: str,
        initial_memory: bytes,
        curriculum: CurriculumAgent,
        execute_project: AdaptiveProjectExecutor,
        *,
        max_parallel: int,
        project_target: int = 64,
    ) -> None:
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be non-empty")
        if not isinstance(initial_memory, bytes):
            raise TypeError("initial memory must be immutable bytes")
        if not isinstance(curriculum, CurriculumAgent):
            raise TypeError("curriculum must be CurriculumAgent")
        if not callable(execute_project):
            raise TypeError("project executor must be callable")
        if type(max_parallel) is not int or max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        if type(project_target) is not int or project_target < 1:
            raise ValueError("project_target must be positive")
        self.project_target = project_target
        self.root = Path(root)
        self.objective = objective.strip()
        self.initial_memory = initial_memory
        self.curriculum = curriculum
        self.execute_project = execute_project
        self.max_parallel = max_parallel

    def run(self) -> dict[str, object]:
        state = {**self._load_state(), "project_target": self.project_target}
        self._atomic_json(self.root / "state.json", state)
        while True:
            if state["terminal"] is not None:
                return self._report(state)
            if len(state["prior_outcomes"]) >= self.project_target:
                return self._report(state)
            snapshot = self._memory_snapshot(state)
            pending = self._load_pending(state)
            if pending is None:
                decision, session = self.curriculum.select(
                    self._context(state, snapshot), self._session(state),
                )
                if isinstance(decision, TerminalDecision):
                    state = {
                        **state,
                        "session": self._session_dict(session),
                        "terminal": {
                            "status": decision.status,
                            "reason": decision.reason,
                        },
                    }
                    self._atomic_json(self.root / "state.json", state)
                    return self._report(state)
                assert isinstance(decision, WaveDecision)
                pending = {
                    "schema": PENDING_SCHEMA,
                    "base_wave_index": state["wave_index"],
                    "base_memory_sha256": hashlib.sha256(snapshot).hexdigest(),
                    "wave": decision.wave.as_dict(),
                    "session": self._session_dict(session),
                }
                self._atomic_json(self.root / "pending.json", pending)
            wave = WaveSpec.from_mapping(pending["wave"])
            coordinator = LiveWaveCoordinator(
                self.root / "waves" / wave.wave_id,
                wave,
                snapshot,
                self._execute,
                self.max_parallel,
            )
            consolidation = coordinator.resume()
            deltas = tuple(
                self._read_delta(wave.wave_id, commit.project_id)
                for commit in consolidation.commits
            )
            state = self._publish(state, wave, pending, deltas)
            self._atomic_json(self.root / "state.json", state)
            (self.root / "pending.json").unlink(missing_ok=True)

    def resume(self) -> dict[str, object]:
        return self.run()

    def report(self) -> dict[str, object]:
        return self._report(self._load_state())

    @classmethod
    def read_retained_report(
        cls, root: Path, objective: str, initial_memory: bytes,
    ) -> dict[str, object]:
        """Read an existing campaign without constructing live dependencies."""

        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be non-empty")
        if not isinstance(initial_memory, bytes):
            raise TypeError("initial memory must be immutable bytes")
        reader = object.__new__(cls)
        reader.project_target = 64
        reader.root = Path(root)
        reader.objective = objective.strip()
        reader.initial_memory = initial_memory
        path = reader.root / "state.json"
        if not path.is_file():
            raise ValueError("retained adaptive state does not exist")
        state = reader._read_json(path)
        reader._validate_state(state)
        reader.project_target = state.get("project_target", 64)
        return reader._report(state)

    def _execute(self, request: WaveProjectRequest) -> WaveProjectResult:
        result = self.execute_project(AdaptiveProjectRequest(
            request.assignment.spec,
            request.assignment.authored_ordinal,
            request.pre_wave_memory,
            request.branch_root,
            request.inputs,
            request.oracle,
        ))
        if not isinstance(result, AdaptiveProjectResult):
            raise TypeError("project adapter must return AdaptiveProjectResult")
        if result.memory.project_spec != request.assignment.spec:
            raise ValueError("adaptive result memory must retain the executed project spec")
        delta = canonical_bytes({
            "schema": DELTA_SCHEMA,
            "project_id": request.assignment.project_id,
            "summary": result.summary,
            "memory_base64": base64.b64encode(
                canonical_bytes(result.memory.as_dict())
            ).decode("ascii"),
        })
        return WaveProjectResult(delta, result.evidence, result.evidence_payload)

    def _publish(self, state, wave, pending, deltas):
        memories = list(state["memory_bytes_base64"])
        outcomes = list(state["prior_outcomes"])
        coverage = list(state["coverage"])
        for project, delta in zip(wave.projects, deltas, strict=True):
            memories.append(delta["memory_base64"])
            outcomes.append({
                "project_id": project.spec.project_id,
                "spec": project.spec.as_dict(),
                "verified": True,
                "summary": delta["summary"],
            })
            for capability in project.spec.capabilities:
                if capability.value not in coverage:
                    coverage.append(capability.value)
        return {
            **state,
            "wave_index": state["wave_index"] + 1,
            "coverage": coverage,
            "prior_outcomes": outcomes,
            "memory_bytes_base64": memories,
            "session": pending["session"],
            "completed_waves": [*state["completed_waves"], wave.wave_id],
        }

    def _read_delta(self, wave_id: str, project_id: str) -> dict[str, object]:
        path = self.root / "waves" / wave_id / "branches" / project_id / "memory-delta.bin"
        try:
            value = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("verified project memory delta is unavailable") from exc
        if not isinstance(value, Mapping) or set(value) != {
            "schema", "project_id", "summary", "memory_base64",
        } or value["schema"] != DELTA_SCHEMA or value["project_id"] != project_id:
            raise ValueError("verified project memory delta is invalid")
        try:
            memory_bytes = base64.b64decode(value["memory_base64"], validate=True)
            memory = Phase1ProjectMemory.from_mapping(json.loads(memory_bytes))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("verified project memory bytes are invalid") from exc
        if canonical_bytes(memory.as_dict()) != memory_bytes:
            raise ValueError("verified project memory bytes are not canonical")
        return {
            "memory_base64": base64.b64encode(memory_bytes).decode("ascii"),
            "summary": _validated_summary(value["summary"]),
        }

    def _context(self, state, snapshot):
        return CurriculumContext(
            self.objective,
            state["wave_index"],
            hashlib.sha256(snapshot).hexdigest(),
            tuple(self._capability(value) for value in state["coverage"]),
            tuple(
                PriorOutcome(
                    value["project_id"],
                    ProjectInformation.from_project(value["spec"]),
                    value["verified"],
                    value["summary"],
                )
                for value in state["prior_outcomes"]
            ),
            max(0, self.project_target - len(state["prior_outcomes"])),
        )

    @staticmethod
    def _capability(value):
        from benchmarks.a5kernels.phase1_curriculum import Capability

        return Capability(value)

    def _memory_snapshot(self, state) -> bytes:
        return canonical_bytes({
            "schema": MEMORY_SCHEMA,
            "initial_memory_base64": base64.b64encode(self.initial_memory).decode("ascii"),
            "project_memory_base64": state["memory_bytes_base64"],
        })

    def _load_state(self) -> dict[str, object]:
        path = self.root / "state.json"
        if not path.exists():
            state = {
                "schema": STATE_SCHEMA,
                "objective": self.objective,
                "initial_memory_sha256": hashlib.sha256(self.initial_memory).hexdigest(),
                "wave_index": 1,
                "coverage": [],
                "prior_outcomes": [],
                "memory_bytes_base64": [],
                "session": self._session_dict(CurriculumSession()),
                "completed_waves": [],
                "terminal": None,
            }
            self._atomic_json(path, state)
            return state
        state = self._read_json(path)
        self._validate_state(state)
        return state

    def _validate_state(self, state) -> None:
        if state.get("schema") != STATE_SCHEMA or state.get("objective") != self.objective:
            raise ValueError("adaptive state does not match objective")
        if state.get("initial_memory_sha256") != hashlib.sha256(self.initial_memory).hexdigest():
            raise ValueError("adaptive state initial memory does not match")

    def _load_pending(self, state):
        path = self.root / "pending.json"
        if not path.exists():
            return None
        pending = self._read_json(path)
        if pending.get("schema") != PENDING_SCHEMA:
            raise ValueError("invalid pending adaptive wave")
        if pending.get("base_wave_index") < state["wave_index"]:
            path.unlink()
            return None
        snapshot = self._memory_snapshot(state)
        if (
            pending.get("base_wave_index") != state["wave_index"]
            or pending.get("base_memory_sha256") != hashlib.sha256(snapshot).hexdigest()
        ):
            raise ValueError("pending wave does not match published memory")
        return pending

    @staticmethod
    def _session(state):
        value = state["session"]
        return CurriculumSession(tuple(value["messages"]), value["turns"])

    @staticmethod
    def _session_dict(session):
        return {"messages": [dict(item) for item in session.messages], "turns": session.turns}

    def _report(self, state):
        snapshot = self._memory_snapshot(state)
        terminal = state["terminal"]
        pending = self._pending_report(state)
        memory = [
            json.loads(base64.b64decode(value, validate=True))
            for value in state["memory_bytes_base64"]
        ]
        return {
            "status": (
                "RECOVERY_REQUIRED" if pending is not None
                else "TARGET_REACHED" if len(state["prior_outcomes"]) >= self.project_target
                else "RUNNING" if terminal is None
                else terminal["status"]
            ),
            "reason": None if terminal is None else terminal["reason"],
            "wave_index": state["wave_index"],
            "project_target": self.project_target,
            "successful_projects": len(state["prior_outcomes"]),
            "unverified_projects": sum(row["status"] == "unverified" for row in pending["projects"]) if pending else 0,
            "coverage_gaps": self._coverage_gaps(state),
            "coverage": list(state["coverage"]),
            "published_projects": [row["project_id"] for row in state["prior_outcomes"]],
            "memory": memory,
            "memory_sha256": hashlib.sha256(snapshot).hexdigest(),
            "pending_wave": pending,
        }

    @staticmethod
    def _coverage_gaps(state):
        from benchmarks.a5kernels.phase1_curriculum import LEARNING_FOCI
        covered = {focus for row in state["prior_outcomes"]
                   for focus in row["spec"].get("learning_focus", ())}
        return [focus for focus in LEARNING_FOCI if focus not in covered]

    def _pending_report(self, state):
        pending = self._load_pending(state)
        if pending is None:
            return None
        wave = WaveSpec.from_mapping(pending["wave"])
        projects = []
        for project in wave.projects:
            path = (
                self.root / "waves" / wave.wave_id / "branches"
                / project.spec.project_id / "outcome.json"
            )
            status = "unfinished"
            if path.exists():
                outcome = self._read_json(path)
                if (
                    outcome.get("project_id") != project.spec.project_id
                    or type(outcome.get("verified")) is not bool
                ):
                    raise ValueError("pending project outcome is invalid")
                status = "verified" if outcome["verified"] else "unverified"
            projects.append({
                "project_id": project.spec.project_id,
                "status": status,
            })
        return {"wave_id": wave.wave_id, "projects": projects}

    @staticmethod
    def _read_json(path):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to read adaptive state: {path.name}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"adaptive state is not an object: {path.name}")
        return value

    @staticmethod
    def _atomic_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(canonical_bytes(value) + b"\n")
        os.replace(temporary, path)


__all__ = [
    "AdaptivePhase1Coordinator", "AdaptiveProjectExecutor",
    "AdaptiveProjectRequest", "AdaptiveProjectResult",
]
