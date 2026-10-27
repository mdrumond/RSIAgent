"""Adaptive A5 Phase-1 selection, wave execution, and atomic learning."""

from __future__ import annotations

import base64
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import math
import time
import os
from pathlib import Path
from typing import Callable, Mapping

from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec, WaveSpec
from benchmarks.a5kernels.phase1_curriculum_selector import (
    CurriculumAgent,
    CurriculumContext,
    CurriculumSession,
    CurriculumSelectionError,
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


def retained_failure_outcome(root: Path, wave_id: str, project: ProjectSpec) -> dict:
    """Concise selection feedback, backed by the retained branch evidence."""
    branch = root / "waves" / wave_id / "branches" / project.project_id
    if not (branch / "outcome.json").exists():
        raw = (branch / "interruption.json").read_bytes()
        interruption = json.loads(raw)
        return {"project_id": project.project_id, "spec": project.as_dict(), "verified": False,
                "summary": interruption["summary"],
                "evidence": {"path": str(branch.relative_to(root)),
                             "sha256": hashlib.sha256(raw).hexdigest(),
                             "classification": interruption["classification"]}}
    try:
        evidence = json.loads((branch / "execution-evidence.bin").read_bytes())
    except (ValueError, UnicodeDecodeError):
        evidence = {}
    if not isinstance(evidence, dict):
        evidence = {}
    outcome = json.loads((branch / "outcome.json").read_bytes())
    delta = json.loads((branch / "memory-delta.bin").read_bytes())
    source = evidence.get("source_sha256")
    if delta.get("schema") == DELTA_SCHEMA:
        source = source or json.loads(base64.b64decode(delta["memory_base64"], validate=True))["source_revision"]
    summary = str(delta.get("summary") or evidence.get("summary") or evidence.get("detail")
                  or "Completed attempt without host verification").strip()
    return {"project_id": project.project_id, "spec": project.as_dict(), "verified": False,
            "summary": summary[:1200].rstrip(),
            "evidence": {"path": str(branch.relative_to(root)),
                         "sha256": outcome["evidence_sha256"],
                         "source_sha256": source,
                         "classification": outcome["classification"],
                         "iterations": evidence.get("iterations"), "tokens": evidence.get("tokens"),
                         "actor_status": evidence.get("actor_status"),
                         "action_counts": evidence.get("action_counts"),
                         "metered_actions": evidence.get("metered_actions")}}


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
    memory_root: Path | None = None
    remaining_seconds: float | None = None
    publish_learning: bool = True
    failure_context: Mapping[str, object] | None = None


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
        project_target: int | None = None,
        max_projects: int | None = None,
        max_seconds: float = 7200,
        clock: Callable[[], float] = time.monotonic,
        allow_partial: bool = False,
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
        if project_target is not None and (type(project_target) is not int or project_target < 1):
            raise ValueError("project_target must be positive")
        self.project_target = project_target
        if project_target is None:
            retained = self.root_state_target(root)
            self.project_target = retained
        self.root = Path(root)
        self.objective = objective.strip()
        self.initial_memory = initial_memory
        self.curriculum = copy(curriculum)
        self.curriculum.require_fresh_project_ids = True
        self.execute_project = execute_project
        self.max_parallel = max_parallel
        if max_projects is not None and (type(max_projects) is not int or max_projects < 1):
            raise ValueError("max_projects must be a positive integer or None")
        self.max_projects = max_projects  # Legacy metadata; never an execution limit.
        if isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        self.max_seconds, self.clock = max_seconds, clock
        self.failure_feedback = None
        self.allow_partial = allow_partial
        self.environment_check = None
        self._checked_environment = None

        self.automatic_remediation = None

    @staticmethod
    def root_state_target(root):
        path = Path(root) / "state.json"
        return json.loads(path.read_bytes()).get("project_target", 64) if path.exists() else 64

    def run(self) -> dict[str, object]:
        return self._run()

    def _run(self, *, wave_limit=None, approved_only=False) -> dict[str, object]:
        state = self._load_state()
        pending = self._load_pending(state)
        if (pending is not None and len(state["prior_outcomes"]) + len(WaveSpec.from_mapping(pending["wave"]).projects) > self.project_target
                and self.automatic_remediation is None):
            raise ValueError("retained wave exceeds configured project target")
        if state.get("project_target", self.project_target) != self.project_target:
            raise ValueError("use continue_campaign to change the retained project target")
        if (self.allow_partial and pending is not None and
                (state["terminal"] is not None or state["elapsed_seconds"] >= self.max_seconds)):
            return self.finalize_pending_wave()
        if (self.automatic_remediation is None and (state["terminal"] is not None or len(state["prior_outcomes"]) >= self.project_target
                or (self.allow_partial and pending is None and self._partial_closed(state)))):
            verifier = getattr(self, 'usefulness_verifier', None)
            if verifier is not None and verifier.enabled:
                def reconcile():
                    self._assess_committed_memory(state)
                    return self._report(state)
                return self._with_allocation(reconcile)
            return self._report(state)
        return self._with_allocation(lambda: self._run_allocated(wave_limit=wave_limit, approved_only=approved_only))

    def _with_allocation(self, operation):
        return self._exclusive(lambda: self._run_allocation(operation))

    def _exclusive(self, operation):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "allocation.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("campaign allocation is already active") from None
            return operation()

    def _run_allocation(self, operation):
        self._checked_environment = None
        state = self._load_state()
        if state.get("allocation_active"):
            # Exclusive ownership proves the previous interval has no owner.
            state["elapsed_seconds"] = max(self.max_seconds, state["elapsed_seconds"])
            state["allocation_active"] = False
            state["allocation_interrupted"] = True
            self._atomic_json(self.root / "state.json", state)
        started = self.clock()
        self._deadline = started + max(0, self.max_seconds - state["elapsed_seconds"])
        # An unobserved interruption reserves the whole remaining allocation.
        state["allocation_active"] = True
        self._atomic_json(self.root / "state.json", state)
        try:
            result = operation()
        finally:
            # Reload after publication; exceptions retain the exact pending wave.
            current = self._read_json(self.root / "state.json")
            current["elapsed_seconds"] += max(0, self.clock() - started)
            current["allocation_active"] = False
            self._atomic_json(self.root / "state.json", current)
            self._deadline = None
        result["elapsed_seconds"] = current["elapsed_seconds"]
        result["remaining_seconds"] = max(0, self.max_seconds-current["elapsed_seconds"])
        return result

    def _remaining_seconds(self):
        deadline = getattr(self, "_deadline", None)
        if deadline is None:
            return self.max_seconds
        return max(0, deadline - self.clock())

    def _time_exhausted(self, state):
        if self._remaining_seconds() > 0:
            return False
        state["terminal"] = {"status": "TIME_BUDGET_EXHAUSTED",
                             "reason": "Campaign elapsed-time allocation exhausted."}
        self._atomic_json(self.root / "state.json", state)
        return True

    def _run_allocated(self, *, wave_limit=None, approved_only=False) -> dict[str, object]:
        state = self._load_state()
        self._assess_committed_memory(state)
        if self.failure_feedback is not None:
            self.failure_feedback.record(state.get('failure_investigations', []), state)
        pending = self._load_pending(state)
        if pending is not None and (
            len(state["prior_outcomes"]) + len(WaveSpec.from_mapping(pending["wave"]).projects)
            > self.project_target and self.automatic_remediation is None
        ):
            raise ValueError("retained wave exceeds configured project target")
        if state.get("project_target", self.project_target) != self.project_target:
            raise ValueError("use continue_campaign to change the retained project target")
        state = {**state, "project_target": self.project_target}
        self._atomic_json(self.root / "state.json", state)
        executed_waves = 0
        while True:
            if self._time_exhausted(state):
                return self._report(state)
            remediate = self.automatic_remediation
            obligations = remediate.refresh(state) if remediate else False
            if obligations:
                remediate.retry_ready(state)
                obligations = remediate.refresh(state)
            if state["terminal"] is not None and not obligations:
                return self._report(state)
            if len(state["prior_outcomes"]) >= self.project_target and not obligations:
                return self._report(state)
            if self._time_exhausted(state):
                return self._deadline_report(state)
            snapshot = self._memory_snapshot(state)
            pending = self._load_pending(state)
            if pending is None:
                if approved_only:
                    raise ValueError("no approved wave is pending")
                try:
                    with remediate.active() if remediate else nullcontext():
                        decision, session = self._select(
                            self._context(state, snapshot), self._session(state),
                            remaining_projects=max(1, remediate.wave_slots(state) if obligations else
                                                   self.project_target - len(state["prior_outcomes"])),
                        )
                except ValueError:
                    if remediate and remediate.selection_remaining() <= 0:
                        continue
                    raise
                if self._time_exhausted(state):
                    return self._deadline_report(state)
                if remediate and remediate.selection_remaining() <= 0:
                    state = {**state, 'session': self._session_dict(session)}
                    self._atomic_json(self.root / 'state.json', state)
                    continue
                if isinstance(decision, TerminalDecision):
                    state = {
                        **state,
                        "session": self._session_dict(session),
                        "terminal": {
                            "status": decision.status,
                            "reason": decision.reason,
                        },
                    }
                    if decision.failure_investigations:
                        state['failure_investigations'] = list(decision.failure_investigations)
                    self._atomic_json(self.root / "state.json", state)
                    if self.failure_feedback is not None:
                        self.failure_feedback.record(decision.failure_investigations, state)
                    if remediate and obligations:
                        remediate.refresh(state)
                        remediate.retry_ready(state)
                        if remediate.refresh(state):
                            state = {**state, 'terminal': None}
                            self._atomic_json(self.root / 'state.json', state)
                            continue
                    return self._report(state)
                assert isinstance(decision, WaveDecision)
                if self._time_exhausted(state):
                    return self._deadline_report(state)
                pending = {
                    "schema": PENDING_SCHEMA,
                    "base_wave_index": state["wave_index"],
                    "base_memory_sha256": hashlib.sha256(snapshot).hexdigest(),
                    "wave": decision.wave.as_dict(),
                    "session": self._session_dict(session),
                }
                if decision.failure_investigations:
                    pending['failure_investigations'] = list(decision.failure_investigations)
                self._atomic_json(self.root / "pending.json", pending)
            wave = WaveSpec.from_mapping(pending["wave"])
            if self.failure_feedback is not None:
                self.failure_feedback.record(pending.get('failure_investigations', []), state)
            if remediate:
                # Preserve the admitted wave, but apply ready repairs first.
                remediate.refresh(state)
                remediate.retry_ready(state)
                remediate.refresh(state)
            if self._time_exhausted(state):
                return self._deadline_report(state)
            coordinator = LiveWaveCoordinator(
                self.root / "waves" / wave.wave_id,
                wave,
                snapshot,
                self._execute,
                self.max_parallel,
                allow_semantic_failures=True, allow_partial=self.allow_partial,
                remaining_seconds=self._remaining_seconds if self.allow_partial else None,
                before_execute=lambda: self._check_environment(state),
            )
            consolidation = coordinator.resume()
            state = self._finish_wave(state, wave, pending, consolidation,
                "deadline" if self._remaining_seconds() <= 0 else "settled")
            self._assess_committed_memory(state)
            executed_waves += 1
            if self.allow_partial and self._partial_closed(state):
                return self._report(state)
            if wave_limit is not None and executed_waves >= wave_limit:
                return self._report(state)

    def _finish_wave(self, state, wave, pending, consolidation, reason):
        committed = {commit.project_id for commit in consolidation.commits}
        deltas = tuple(self._read_delta(wave.wave_id, project.spec.project_id)
                       if project.spec.project_id in committed else None for project in wave.projects)
        state = self._publish(state, wave, pending, deltas)
        if self.allow_partial:
            skipped = [p.spec.project_id for p in wave.projects if p.spec.project_id not in committed]
            state["wave_results"] = [*state.get("wave_results", []), {
                "wave_id": wave.wave_id, "status": "complete" if not skipped else "partial" if committed else "empty",
                "accepted_projects": [c.project_id for c in consolidation.commits],
                "skipped_projects": skipped, "closure_reason": reason,
            }]
        self._atomic_json(self.root / "state.json", state)
        (self.root / "pending.json").unlink(missing_ok=True)
        if self.failure_feedback is not None:
            self.failure_feedback.sync(state)
        return state

    def finalize_pending_wave(self):
        """Publish retained finalized results only, without extending an elapsed allocation."""
        if not self.allow_partial:
            raise ValueError("retained-only finalization requires partial-wave acceptance")
        return self._exclusive(self._finalize_pending_wave)

    def _finalize_pending_wave(self):
        state = self._load_state()
        if state.get("allocation_active"):
            state.update(elapsed_seconds=max(self.max_seconds, state["elapsed_seconds"]),
                         allocation_active=False, allocation_interrupted=True)
            self._atomic_json(self.root / "state.json", state)
        pending = self._load_pending(state)
        if pending is None:
            return self._report(state)
        return self._close_retained(state, pending)

    def _deadline_report(self, state):
        pending = self._load_pending(state) if self.allow_partial else None
        return self._close_retained(state, pending) if pending is not None else self._report(state)

    def _close_retained(self, state, pending):
        wave = WaveSpec.from_mapping(pending["wave"])
        coordinator = LiveWaveCoordinator(self.root / "waves" / wave.wave_id, wave,
            self._memory_snapshot(state), self._execute, self.max_parallel,
            allow_semantic_failures=True, allow_partial=True)
        consolidation = coordinator.finalize_retained()
        return self._report(self._finish_wave(state, wave, pending, consolidation,
            "deadline" if state["terminal"] and state["terminal"]["status"] == "TIME_BUDGET_EXHAUSTED" else "recovery"))

    def propose_wave(self):
        """Charge curriculum selection to the campaign allocation."""
        return self._with_allocation(self._propose_wave)

    def _propose_wave(self):
        """Select a wave without admitting it for execution."""
        state = self._load_state()
        if self._load_pending(state) is not None:
            raise ValueError("a pending wave already exists")
        if state["terminal"] is not None or len(state["prior_outcomes"]) >= self.project_target:
            raise ValueError("campaign cannot accept another proposal")
        if self._time_exhausted(state):
            return {"terminal": state["terminal"], "session": state["session"],
                    "base_wave_index": state["wave_index"],
                    "base_memory_sha256": hashlib.sha256(self._memory_snapshot(state)).hexdigest()}
        snapshot = self._memory_snapshot(state)
        decision, session = self._select(
            self._context(state, snapshot), self._session(state),
            remaining_projects=self.project_target - len(state["prior_outcomes"]),
        )
        if self._time_exhausted(state):
            return {"terminal": state["terminal"], "session": self._session_dict(session),
                    "base_wave_index": state["wave_index"],
                    "base_memory_sha256": hashlib.sha256(snapshot).hexdigest()}
        if isinstance(decision, TerminalDecision):
            return {"terminal": {"status": decision.status, "reason": decision.reason},
                    "base_wave_index": state["wave_index"],
                    "base_memory_sha256": hashlib.sha256(snapshot).hexdigest(),
                    "session": self._session_dict(session),
                    **({'failure_investigations': list(decision.failure_investigations)}
                       if decision.failure_investigations else {})}
        return {"schema": PENDING_SCHEMA, "base_wave_index": state["wave_index"],
                "base_memory_sha256": hashlib.sha256(snapshot).hexdigest(),
                "wave": decision.wave.as_dict(), "session": self._session_dict(session),
                **({'failure_investigations': list(decision.failure_investigations)}
                   if decision.failure_investigations else {})}

    def validate_proposal(self, proposal):
        state = self._load_state()
        snapshot = self._memory_snapshot(state)
        if (proposal.get("base_wave_index") != state["wave_index"]
                or proposal.get("base_memory_sha256") != hashlib.sha256(snapshot).hexdigest()):
            raise ValueError("proposal no longer matches published memory")
        if "wave" in proposal:
            WaveSpec.from_mapping(proposal["wave"])
        return state

    def approve_wave(self, proposal, selected_project_ids):
        """Admit a selected proposal subset as the coordinator's pending wave."""
        self.validate_proposal(proposal)
        selected = tuple(selected_project_ids)
        if "terminal" in proposal:
            if selected:
                raise ValueError("terminal approval requires zero project IDs")
            return proposal
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("approved project IDs must be non-empty and unique")
        wave = proposal["wave"]
        available = {row["spec"]["project_id"] for row in wave["projects"]}
        if not set(selected) <= available:
            raise ValueError("approved project ID is not in the proposal")
        if getattr(self, "approval_cardinality", None) is not None and len(selected) != self.approval_cardinality:
            raise ValueError("Phase 2 approval requires exactly one project")
        approved = {**proposal, "wave": {**wave, "projects": [
            row for row in wave["projects"] if row["spec"]["project_id"] in selected
        ]}}
        if self.failure_feedback is not None:
            rows = [row for row in proposal.get('failure_investigations', [])
                if row['status'] != 'investigating' or row['project_id'] in selected]
            state = self._load_state()
            self.failure_feedback.admit(rows, WaveSpec.from_mapping(approved['wave']),
                                       self._context(state, self._memory_snapshot(state)))
            approved['failure_investigations'] = rows
        WaveSpec.from_mapping(approved["wave"])
        retained = self._proposal_pending(proposal)
        if retained is not None:
            retained_ids = tuple(row["spec"]["project_id"] for row in retained["wave"]["projects"])
            approved_ids = tuple(row["spec"]["project_id"] for row in approved["wave"]["projects"])
            if retained_ids != approved_ids:
                raise ValueError("proposal already has a different approved subset")
            return retained
        self._atomic_json(self.root / "pending.json", approved)
        return approved

    def _proposal_pending(self, proposal):
        state = self.validate_proposal(proposal)
        pending = self._load_pending(state)
        if pending is None:
            return None
        proposed_wave, pending_wave = proposal.get("wave"), pending.get("wave")
        if proposed_wave is None or pending_wave is None:
            raise ValueError("pending wave does not belong to the retained proposal")
        proposed = {row["spec"]["project_id"]: row for row in proposed_wave["projects"]}
        pending_projects = pending_wave["projects"]
        same_wave = ({key: value for key, value in proposed_wave.items() if key != "projects"}
                     == {key: value for key, value in pending_wave.items() if key != "projects"})
        if (not same_wave or pending.get("session") != proposal.get("session")
                or not pending_projects
                or any(proposed.get(row["spec"]["project_id"]) != row
                       for row in pending_projects)):
            raise ValueError("pending wave does not belong to the retained proposal")
        return pending

    def discard_proposal(self, proposal):
        """Remove only a pending subset authenticated as part of this proposal."""
        pending = self._proposal_pending(proposal)
        if pending is not None:
            (self.root / "pending.json").unlink()

    def run_approved_wave(self):
        return self._run(wave_limit=1, approved_only=True)

    def reconcile_feedback(self):
        """Replay feedback publication from committed state without executing work."""
        if self.failure_feedback is not None:
            state = self._load_state()
            self.failure_feedback.record(state.get('failure_investigations', []), state)

    def run_approved_proposal(self, proposal):
        """Commit one approved wave or a bound terminal curriculum decision."""
        if "terminal" not in proposal:
            return self.run_approved_wave()
        state = self.validate_proposal(proposal)
        if state["terminal"] is not None:
            if state["terminal"] != proposal["terminal"]:
                raise ValueError("campaign has a different terminal decision")
            if self.failure_feedback is not None:
                self.failure_feedback.record(proposal.get('failure_investigations', []), state)
            return self._report(state)
        state = {**state, "session": proposal["session"],
                 "terminal": proposal["terminal"]}
        if proposal.get('failure_investigations'):
            state['failure_investigations'] = proposal['failure_investigations']
        self._atomic_json(self.root / "state.json", state)
        if self.failure_feedback is not None:
            self.failure_feedback.record(proposal.get('failure_investigations', []), state)
        return self._report(state)

    def resume(self) -> dict[str, object]:
        return self.run()

    def continue_campaign(self, project_target: int, *, reopen_terminal: bool = False):
        """Explicitly extend a campaign without erasing its prior stop decisions."""
        state = self._load_state()
        old_target = state.get("project_target", self.project_target)
        if type(project_target) is not int or project_target <= old_target:
            raise ValueError("continuation must raise the retained project target")
        terminal = state["terminal"]
        if terminal is not None and not reopen_terminal:
            raise ValueError("terminal curriculum requires explicit reopen_terminal")
        history = list(state.get("continuations", []))
        history.append({"old_target": old_target, "project_target": project_target,
                        "terminal": terminal,
                        "previous_status": self._report(state)["status"],
                        "successful_projects": len(state["prior_outcomes"])})
        if self._partial_closed(state):
            # Acknowledge only this immutable boundary; a later partial wave
            # must stop again until the user explicitly continues it.
            state = {**state, "continued_partial_wave": state["wave_results"][-1]["wave_id"]}
        state = {**state, "project_target": project_target, "terminal": None,
                 "continuations": history}
        self._atomic_json(self.root / "state.json", state)
        self.project_target = project_target
        return self._report(state)

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
        automatic = self.automatic_remediation
        with automatic.project_active(request) if automatic else nullcontext():
            return self._execute_bound(request)

    def _execute_bound(self, request: WaveProjectRequest) -> WaveProjectResult:
        failure_context = (self.failure_feedback.project_context(request.assignment.project_id)
                           if self.failure_feedback else None)
        inputs, oracle = request.inputs, request.oracle
        if failure_context is not None and request.assignment.spec.failure_mode.value != 'none':
            from benchmarks.ascend.failure_handoff import FailureContext
            failure = FailureContext(failure_context, request.assignment.spec)
            failure.require_repair_evidence(request.assignment.spec)
            inputs, oracle = failure.inputs, failure.expected
        result = self.execute_project(AdaptiveProjectRequest(
            request.assignment.spec,
            request.assignment.authored_ordinal,
            request.pre_wave_memory,
            request.branch_root,
            inputs,
            oracle,
            failure_context=failure_context,
            memory_root=self.root,
            remaining_seconds=(self.automatic_remediation.project_remaining(request)
                               if self.automatic_remediation else self._remaining_seconds()),
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
        failures = list(state.get("failed_outcomes", []))
        for project, delta in zip(wave.projects, deltas, strict=True):
            if delta is None:
                failures.append(retained_failure_outcome(self.root, wave.wave_id, project.spec))
                continue
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
            **({'failure_investigations': pending['failure_investigations']}
               if pending.get('failure_investigations') else {}),
            **({"failed_outcomes": failures} if failures else {}),
        }

    def _assess_committed_memory(self, state):
        verifier = getattr(self, "usefulness_verifier", None)
        if verifier is None or not verifier.enabled:
            return
        previous = []
        for wave_id in state["completed_waves"]:
            plan = self._read_json(self.root / "waves" / wave_id / "plan.json")
            wave = WaveSpec.from_mapping(plan["wave"])
            prior = self._memory_snapshot({**state, "memory_bytes_base64": previous})
            publication = self._read_json(self.root / "waves" / wave_id / "published.json")
            committed = {row["project_id"] for row in publication["commits"]}
            for project in wave.projects:
                if project.spec.project_id not in committed:
                    continue
                path = self.root / "waves" / wave_id / "branches" / project.spec.project_id / "memory-delta.bin"
                if not path.exists():
                    continue
                if not self._read_json(path.parent / "outcome.json")["verified"]:
                    continue
                delta = self._read_json(path)
                raw = base64.b64decode(delta["memory_base64"])
                memory = Phase1ProjectMemory.from_mapping(json.loads(raw))
                verifier.assess_memory(memory, prior, f"{wave_id}/{project.spec.project_id}")
                previous.append(delta["memory_base64"])

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

    def _check_environment(self, state, *, remaining_seconds=None):
        token = (state["wave_index"], getattr(self, "_deadline", None))
        if self.environment_check is not None and self._checked_environment != token:
            allowance = (lambda: min(self._remaining_seconds(), remaining_seconds())) if remaining_seconds else self._remaining_seconds
            self.environment_check(state["wave_index"], allowance)
            self._checked_environment = token

    def _select(self, context, *args, **kwargs):
        self._check_environment(self._load_state())
        context = replace(context, remaining_seconds=min(context.remaining_seconds, self._remaining_seconds()))
        try:
            return self.curriculum.select(context, *args, **kwargs)
        except CurriculumSelectionError as error:
            from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts

            digest = ContentAddressedArtifacts(self.root / "artifacts").put_json({
                "schema": "phase1-curriculum-rejection-v1",
                "error": str(error), "messages": list(error.messages),
            })
            raise ValueError(f"{error}; rejected selection artifact: {digest}") from error

    def _seed_context_state(self, state):
        """Authenticate portable copied memory; never publish it as new attempts."""
        if state.get("phase") == "adaptive-phase2":
            return state  # Phase 2 owns its authenticated seed outcomes already.
        try:
            seed = json.loads(self.initial_memory)
        except (ValueError, UnicodeDecodeError):
            return state
        if not isinstance(seed, dict) or seed.get("schema") != "ascend-phase2-curriculum-seed-v1":
            return state
        from benchmarks.ascend.phase2_curriculum import reviewed_seed

        authenticated, _ = reviewed_seed(self.root, seed["snapshot_id"])
        if json.loads(authenticated) != seed:
            raise ValueError("reviewed seed differs from initial memory")
        memories = [Phase1ProjectMemory.from_mapping(row) for row in seed["phase1_memory"]]
        if any(memory.project_spec is None for memory in memories):
            raise ValueError("copied seed requires generic project memory")
        outcomes = [{"project_id": memory.project_spec.project_id,
                     "spec": memory.project_spec.as_dict(), "verified": True,
                     "summary": "Reviewed seed project; consult pinned memory."}
                    for memory in memories]
        coverage = list(state["coverage"])
        for memory in memories:
            for capability in memory.project_spec.capabilities:
                if capability.value not in coverage:
                    coverage.append(capability.value)
        return {**state, "coverage": coverage,
                "prior_outcomes": [*outcomes, *state["prior_outcomes"]],
                "memory_bytes_base64": [
                    *(base64.b64encode(canonical_bytes(row)).decode("ascii")
                      for row in seed["phase1_memory"]), *state["memory_bytes_base64"]]}

    def _context_files(self, state):
        from benchmarks.ascend.context_files import ContextFiles
        files = ContextFiles(self.root / 'curriculum' / 'context')
        refs = []
        entries = []
        if hasattr(self, 'snapshot'):
            self.snapshot.verify()
            entries.extend(self.snapshot.memory_entries())
        entries.extend(json.loads(base64.b64decode(encoded, validate=True))
                       for encoded in state['memory_bytes_base64'])
        seen = set()
        for memory in entries:
            digest = hashlib.sha256(canonical_bytes(memory)).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            project_id = memory.get('project_spec', {}).get('project_id', memory.get('task_id', digest[:12]))
            refs.append(files.put('project-memory-' + project_id, memory))
        for row in state.get('failed_outcomes', []):
            refs.append(files.put('failed-project-' + row['project_id'], row))
        if len(refs) > 5:
            index = files.put('context-file-index', {'files': refs})
            refs = [index, *refs[-4:]]
        return tuple(refs), files.read

    def _context(self, state, snapshot):
        remaining = max(0, self.project_target - len(state["prior_outcomes"]))
        context_state = self._seed_context_state(state)
        if self.automatic_remediation and self.automatic_remediation.feedback.ids:
            remaining = max(remaining, self.automatic_remediation.wave_slots(state))
        refs, reader = self._context_files(context_state)
        return CurriculumContext(
            self.objective,
            state["wave_index"],
            hashlib.sha256(snapshot).hexdigest(),
            tuple(self._capability(value) for value in context_state["coverage"]),
            tuple(
                PriorOutcome(
                    value["project_id"],
                    ProjectInformation.from_project(value["spec"]),
                    value["verified"],
                    value["summary"][:240].rstrip(),
                )
                for value in self._effective_outcomes(context_state)
            ),
            remaining,
            self._curriculum_feedback(context_state),
            self.failure_feedback.context(state) if self.failure_feedback else (),
            remaining_seconds=(self.automatic_remediation.selection_remaining()
                               if self.automatic_remediation else self._remaining_seconds()),
            context_files=refs, artifact_reader=reader,
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
                "max_seconds": self.max_seconds,
                "elapsed_seconds": 0.0,
            }
            if self.max_projects is not None:
                state["max_projects"] = self.max_projects
            self._atomic_json(path, state)
            return state
        state = self._read_json(path)
        self._validate_state(state)
        if "elapsed_seconds" not in state:
            raise ValueError("legacy campaign has no elapsed-time accounting; start a new campaign")
        if state.get("max_seconds") != self.max_seconds:
            raise ValueError("adaptive state time allocation does not match")
        return state

    def _validate_state(self, state) -> None:
        if state.get("schema") != STATE_SCHEMA or state.get("objective") != self.objective:
            raise ValueError("adaptive state does not match objective")
        if state.get("initial_memory_sha256") != hashlib.sha256(self.initial_memory).hexdigest():
            raise ValueError("adaptive state initial memory does not match")
        elapsed = state.get("elapsed_seconds", 0)
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("adaptive state elapsed time is invalid")

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

    @staticmethod
    def _partial_closed(state):
        results = state.get("wave_results", [])
        return bool(results and results[-1]["status"] in {"partial", "empty"}
                    and state.get("continued_partial_wave") != results[-1]["wave_id"])

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
                else "WAVE_CLOSED" if terminal is None and self._partial_closed(state)
                else "RUNNING" if terminal is None
                else terminal["status"]
            ),
            "reason": None if terminal is None else terminal["reason"],
            "max_seconds": state.get("max_seconds"),
            "elapsed_seconds": state.get("elapsed_seconds"),
            "remaining_seconds": max(0, state["max_seconds"]-state["elapsed_seconds"]) if "elapsed_seconds" in state else None,
            "wave_index": state["wave_index"],
            "wave_results": state.get("wave_results", []),
            "project_target": self.project_target,
            "successful_projects": len(state["prior_outcomes"]),
            "completed_attempts": self._completed_attempts(state),
            "failed_projects": len(state.get("failed_outcomes", [])),
            "curriculum_memory": self._curriculum_memory(state),
            "unverified_projects": sum(row["status"] == "unverified" for row in pending["projects"]) if pending else 0,
            "coverage_gaps": self._coverage_gaps(state),
            "coverage": list(state["coverage"]),
            "published_projects": [row["project_id"] for row in state["prior_outcomes"]],
            "memory": memory,
            "memory_sha256": hashlib.sha256(snapshot).hexdigest(),
            "pending_wave": pending,
        }

    @classmethod
    def _coverage_gaps(cls, state):
        from benchmarks.a5kernels.phase1_curriculum import LEARNING_FOCI
        covered = {focus for row in cls._effective_outcomes(state)
                   if row["verified"]
                   for focus in row["spec"].get("learning_focus", ())}
        return [focus for focus in LEARNING_FOCI if focus not in covered]

    @staticmethod
    def _completed_attempts(state):
        return len(state["prior_outcomes"]) + len(state.get("failed_outcomes", []))

    @staticmethod
    def _effective_outcomes(state):
        """Requested tools count only when retained host facts show their use."""
        rows = []
        for row, encoded in zip(state["prior_outcomes"], state["memory_bytes_base64"], strict=True):
            memory = json.loads(base64.b64decode(encoded, validate=True))
            observed = set()
            for fact in memory["host_facts"]:
                if fact["category"] == "profiling":
                    observed.add("tool-msprof")
                statement = fact["statement"]
                if fact["category"] == "performance-study":
                    try:
                        summary = json.loads(statement)
                    except json.JSONDecodeError:
                        summary = None
                    timing = summary.get("timing", ()) if isinstance(summary, dict) else ()
                    if isinstance(timing, list) and any(
                        isinstance(row, dict) and row.get("device_kernel") for row in timing
                    ):
                        observed.add("tool-msprof")
                if statement.startswith("tool-observation:"):
                    record = json.loads(statement.removeprefix("tool-observation:"))
                    if record.get("available") and "execution" in record:
                        observed.add("tool-msdebug" if record.get("tool") == "msdebug" else "tool-mssanitizer")
                if statement.startswith("verified-debug-print:"):
                    observed.add("debug-print")
            spec = dict(row["spec"])
            focus = [item for item in spec.get("learning_focus", ())
                     if not (item.startswith("tool-") or item == "debug-print") or item in observed]
            focus += sorted(observed - set(focus))
            if focus:
                spec["learning_focus"] = focus
            else:
                spec.pop("learning_focus", None)
            lessons = memory.get("agent_lessons", [])
            yield_summary = ("no_new_knowledge" if memory.get("no_new_knowledge") else
                             "lessons: " + lessons[0]["claim"][:160] if lessons else "not-recorded")
            rows.append({**row, "spec": spec,
                         "summary": "Host verified; learning yield=" + yield_summary})
        return [*rows, *state.get("failed_outcomes", [])]

    @staticmethod
    def _curriculum_memory(state):
        rows = []
        for row, encoded in zip(state["prior_outcomes"], state["memory_bytes_base64"], strict=True):
            memory = json.loads(base64.b64decode(encoded, validate=True))
            rows.append({"project_id": row["project_id"], "verified": True,
                         "yield": ("no_new_knowledge" if memory.get("no_new_knowledge") else
                                   "lessons" if memory.get("agent_lessons") else "not-recorded"),
                         "summary": row["summary"][:1200],
                         "lesson_claims": [lesson["claim"][:400] for lesson in memory.get("agent_lessons", [])[:2]],
                         "source_sha256": memory["source_revision"],
                         "evidence_sha256": [fact["evidence_sha256"] for fact in memory["host_facts"]]})
        return [*rows, *({key: value for key, value in {**row, "yield": "failed"}.items()
                         if key != "spec"} for row in state.get("failed_outcomes", []))]

    @classmethod
    def _curriculum_feedback(cls, state):
        """Bound detailed pointers without letting old failures crowd out learning."""
        rows = cls._curriculum_memory(state)
        failed = [row for row in rows if not row["verified"]][-4:]
        verified = [row for row in rows if row["verified"]][-(8 - len(failed)):]
        return tuple([*verified, *failed])

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
