"""Adaptive Phase 2 projects seeded by an immutable reviewed Phase 1 snapshot.

The legacy attention runners remain separate compatibility entrypoints. Compose
this coordinator with the same target policy, Actor and project executor as
Phase 1; only the curriculum objective and reviewed seed differ.
"""

from __future__ import annotations

from dataclasses import replace
from copy import copy
import hashlib

from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator, PENDING_SCHEMA
from benchmarks.a5kernels.phase1_curriculum import WaveSpec
from benchmarks.a5kernels.phase1_curriculum_selector import TerminalDecision
from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory
from benchmarks.a5kernels.phase1_wave_runtime import LiveWaveCoordinator


class AdaptivePhase2Coordinator(AdaptivePhase1Coordinator):
    """Publish verified projects and adapt after each completed semantic attempt.

    ``reviewed_snapshot`` implements verify(), memory_entries() and snapshot_id
    (as do the existing admitted A3/A5 snapshots). Changing its identity requires
    a new campaign root. Executor exceptions retain the exact pending attempt;
    returned host classifications are semantic outcomes. ``trace_hook`` receives
    a deterministic event_id and retained result after each state publication.
    """

    def __init__(self, root, objective, reviewed_snapshot, curriculum,
                 execute_project, *, project_target=None, trace_hook=None):
        reviewed_snapshot.verify()
        entries = tuple(reviewed_snapshot.memory_entries())
        if not entries:
            raise ValueError("Phase 2 requires reviewed Phase 1 project memory")
        memories = tuple(Phase1ProjectMemory.from_mapping(row) for row in entries)
        if any(memory.project_spec is None for memory in memories):
            raise ValueError("Phase 2 requires generic project memory")
        self.snapshot = reviewed_snapshot
        self.seed_outcomes = [{
            "project_id": memory.project_spec.project_id,
            "spec": memory.project_spec.as_dict(), "verified": True,
            "summary": "Reviewed Phase 1 project; consult pinned seed memory.",
        } for memory in memories]
        initial = canonical_bytes({
            "schema": "ascend-phase2-curriculum-seed-v1",
            "snapshot_id": reviewed_snapshot.snapshot_id,
            "phase1_memory": list(entries),
        })
        if trace_hook is not None and not callable(trace_hook):
            raise TypeError("trace_hook must be callable")
        self.trace_hook = trace_hook
        curriculum = copy(curriculum)
        curriculum.require_fresh_project_ids = True
        super().__init__(root, objective, initial, curriculum, execute_project,
                         max_parallel=1, project_target=project_target)

    def _load_state(self):
        self.snapshot.verify()
        exists = (self.root / "state.json").exists()
        state = super()._load_state()
        if exists:
            if state.get("phase") != "adaptive-phase2":
                raise ValueError("campaign root is not an adaptive Phase 2 lineage")
            return state
        state.update(phase="adaptive-phase2", snapshot_id=self.snapshot.snapshot_id,
                     seed_outcomes=self.seed_outcomes, failed_outcomes=[])
        state["coverage"] = list(dict.fromkeys(
            capability for row in self.seed_outcomes
            for capability in row["spec"]["capabilities"]))
        self._atomic_json(self.root / "state.json", state)
        return state

    @staticmethod
    def _effective_outcomes(state):
        return [*state.get("seed_outcomes", []),
                *AdaptivePhase1Coordinator._effective_outcomes(state),
                *state.get("failed_outcomes", [])]

    def _context(self, state, snapshot):
        return replace(super()._context(state, snapshot), remaining_projects=1)

    def run(self):
        state = self._load_state()
        if state.get("project_target", self.project_target) != self.project_target:
            raise ValueError("use continue_campaign to change the retained project target")
        state = {**state, "project_target": self.project_target}
        self._atomic_json(self.root / "state.json", state)
        while state["terminal"] is None and len(state["prior_outcomes"]) < self.project_target:
            snapshot = self._memory_snapshot(state)
            pending = self._load_pending(state)
            if pending is None:
                decision, session = self.curriculum.select(
                    self._context(state, snapshot), self._session(state),
                    remaining_projects=1)
                if isinstance(decision, TerminalDecision):
                    state = {**state, "session": self._session_dict(session),
                             "terminal": {"status": decision.status, "reason": decision.reason}}
                    self._atomic_json(self.root / "state.json", state)
                    break
                pending = {
                    "schema": PENDING_SCHEMA,
                    "base_wave_index": state["wave_index"],
                    "base_memory_sha256": hashlib.sha256(snapshot).hexdigest(),
                    "wave": decision.wave.as_dict(), "session": self._session_dict(session),
                }
                self._atomic_json(self.root / "pending.json", pending)
            wave = WaveSpec.from_mapping(pending["wave"])
            if len(wave.projects) != 1:
                raise ValueError("Phase 2 selects exactly one fresh project per attempt")
            project = wave.projects[0].spec
            # Failed IDs are not admitted again even though their information
            # may be revisited with a distinct controlled experiment.
            if project.project_id in {row["project_id"] for row in state["failed_outcomes"]}:
                raise ValueError("Phase 2 project ID already has a completed attempt")
            runner = LiveWaveCoordinator(self.root / "waves" / wave.wave_id,
                                         wave, snapshot, self._execute, 1)
            outcome = runner._load_outcome(wave.assignments()[0])
            if outcome is None or outcome.verified:
                try:
                    runner.resume()
                except RuntimeError:
                    # Authenticate retained evidence before deciding that this
                    # was semantic failure rather than an infrastructure retry.
                    outcome = runner._load_outcome(wave.assignments()[0])
                    if outcome is None or outcome.verified:
                        raise
                else:
                    outcome = runner._load_outcome(wave.assignments()[0])
            delta = self._read_delta(wave.wave_id, project.project_id)
            if outcome.verified:
                state = self._publish(state, wave, pending, (delta,))
            else:
                state = {**state, "wave_index": state["wave_index"] + 1,
                         "session": pending["session"], "failed_outcomes": [
                             *state["failed_outcomes"], {
                                 "project_id": project.project_id, "spec": project.as_dict(),
                                 "verified": False, "summary": delta["summary"],
                             }]}
            self._atomic_json(self.root / "state.json", state)
            (self.root / "pending.json").unlink(missing_ok=True)
            if self.trace_hook is not None:
                self.trace_hook({"event_id": f"{self.snapshot.snapshot_id}:{wave.wave_id}",
                                 "phase": "adaptive-phase2", "project_id": project.project_id,
                                 "verified": outcome.verified, "summary": delta["summary"],
                                 "attempt_root": str(runner.root)})
        return self._report(state)

    def _report(self, state):
        return {**super()._report(state), "phase": "adaptive-phase2",
                "snapshot_id": self.snapshot.snapshot_id,
                "failed_projects": len(state.get("failed_outcomes", []))}
