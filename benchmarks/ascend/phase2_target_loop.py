"""Fixed-target Phase 2: attempt, diagnose, practice, learn, retry."""
from __future__ import annotations

import base64
import hashlib
import json
import time

from .phase2_time import Phase2TimeBudget

from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator, PENDING_SCHEMA
from benchmarks.a5kernels.phase1_curriculum import WaveProject, WaveSpec
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from benchmarks.a5kernels.phase1_wave_runtime import LiveWaveCoordinator
from benchmarks.ascend.phase2_curriculum import AdaptivePhase2Coordinator
from benchmarks.ascend.phase2_target_selector import TargetCurriculumContext


class TargetDrivenPhase2Coordinator(AdaptivePhase2Coordinator):
    """Reuse project execution and learning, with a separate immutable goal.

    Exceptions leave the exact pending branch and its retained execution handles
    intact. Every operation consumes elapsed time, including failed operations.
    """

    def __init__(self, root, objective, reviewed_snapshot, curriculum, execute_project, *,
                 target, artifacts, runtime, target_policy=None, max_seconds=7200,
                 max_practice_seconds=1800, clock=time.monotonic, available_tools=(), trace_hook=None):
        target.verify(artifacts, runtime, target_policy)
        self.target, self.artifacts, self.runtime = target, artifacts, dict(runtime)
        self.target_policy = target_policy
        self.time = Phase2TimeBudget(root, max_seconds, max_practice_seconds, clock=clock)
        self.available_tools = tuple(available_tools)
        super().__init__(root, objective, reviewed_snapshot, CurriculumAgent(lambda _: None),
                         execute_project, project_target=1, trace_hook=trace_hook)
        self.curriculum = curriculum

    def _remaining_seconds(self):
        return min(self.time.remaining, self.time.practice_remaining) if self.time.active_practice else self.time.remaining

    def _load_state(self):
        self.snapshot.verify()
        self.target.verify(self.artifacts, self.runtime, self.target_policy)
        state = AdaptivePhase1Coordinator._load_state(self)
        identity = {"target": self.target.as_dict(), "snapshot_id": self.snapshot.snapshot_id,
                    **self.time.identity,
                    "available_tools": list(self.available_tools)}
        if "phase" in state:
            if state["phase"] != "target-driven-phase2" or state.get("goal_identity") != identity:
                raise ValueError("retained Phase 2 target, runtime, seed or budgets changed")
        else:
            if (state["wave_index"] != 1 or state["prior_outcomes"]
                    or state["session"]["turns"] or (self.root / "pending.json").exists()):
                raise ValueError("existing campaigns cannot be converted to target-driven Phase 2")
            state.update(phase="target-driven-phase2", snapshot_id=self.snapshot.snapshot_id, goal_identity=identity,
                         seed_outcomes=self.seed_outcomes, failed_outcomes=[], attempts=[],
                         target_attempts=0, practice_since_target=0, next_kind="target")
            self._atomic_json(self.root / "state.json", state)
        return state

    def _memory_snapshot(self, state):
        value = json.loads(super()._memory_snapshot(state))
        value["phase2_feedback"] = state.get("attempts", [])[-2:]
        return canonical_bytes(value)

    def _context(self, state, snapshot):
        targets = [row for row in state["attempts"] if row["kind"] == "target"]
        if not targets:
            raise ValueError("execute the fixed target before curriculum review")
        return TargetCurriculumContext(self.target, targets[-1],
            tuple([*self._curriculum_feedback(state),
                   *[row for row in state["attempts"][-2:] if row["kind"] == "practice"]]),
            self.time.remaining,
            self.time.practice_remaining, self.available_tools,
            used_project_ids=tuple(sorted({row["project_id"]
                for row in [*self.seed_outcomes, *state["attempts"]]})))

    def _select_pending(self, state):
        snapshot = self._memory_snapshot(state)
        if state["next_kind"] == "target":
            kind, project, decision = "target", self.target.spec, None
            session = self._session(state)
        else:
            # Elapsed practice time forces a fixed-target retest.
            if self.time.practice_remaining <= 0:
                kind, project, decision = "target", self.target.spec, None
                session = self._session(state)
            else:
                decision, session = self.curriculum.select(self._context(state, snapshot), self._session(state))
                if decision.action in {"ready", "stalled"}:
                    latest = next(row for row in reversed(state["attempts"]) if row["kind"] == "target")
                    if decision.action == "ready" and (
                            not latest["verified"] or state["attempts"][-1]["kind"] != "target"):
                        # Practice success can never replace the mandatory retest.
                        kind, project = "target", self.target.spec
                    else:
                        state.update(session=self._session_dict(session), terminal={
                            "status": decision.action.upper(), "reason": decision.reason})
                        self._atomic_json(self.root / "state.json", state)
                        return None
                elif decision.action == "retry-target":
                    kind, project = "target", self.target.spec
                else:
                    kind, project = "practice", decision.project
                    seen = {row["project_id"] for row in [*self.seed_outcomes, *state["attempts"]]}
                    if project.project_id in seen or project.as_dict() == self.target.spec.as_dict():
                        raise ValueError("practice requires a fresh project ID")
        if self.time.remaining <= 0:
            state["terminal"] = {"status": "BUDGET_EXHAUSTED", "reason": "Phase 2 elapsed-time budget exhausted; readiness was not demonstrated."}
            self._atomic_json(self.root / "state.json", state)
            return None
        wave = WaveSpec(f"attempt-{state['wave_index']:03d}", hashlib.sha256(snapshot).hexdigest(),
                        (), (WaveProject(project),))
        pending = {"schema": PENDING_SCHEMA, "base_wave_index": state["wave_index"],
                   "base_memory_sha256": wave.pre_wave_memory_sha256,
                   "wave": wave.as_dict(), "session": self._session_dict(session),
                   "kind": kind, "decision": decision.as_dict() if decision else None}
        self._atomic_json(self.root / "pending.json", pending)
        return pending

    @staticmethod
    def _host_feedback(branch):
        from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
        from benchmarks.ascend.agent_presentation import mismatch_summary
        payload = json.loads((branch / "execution-evidence.bin").read_bytes())
        feedback = {key: str(payload[key])[:1600] for key in ("classification", "detail", "summary") if key in payload}
        store = ContentAddressedArtifacts(branch / "artifacts")
        hashes = payload.get("artifacts", {})
        if "observed" in hashes:
            observed = store.read_json(hashes["observed"])
            feedback.update({key: str(observed[key])[:1600] for key in (
                "compile_diagnostic", "runtime_diagnostic") if observed.get(key)})
            if observed.get("output") is not None and "expected" in hashes:
                expected = store.read_json(hashes["expected"])
                feedback["numerical_summary"] = mismatch_summary(expected["storage"], observed["output"])
        delta = json.loads((branch / "memory-delta.bin").read_bytes())
        facts = payload.get("host_facts", [])
        if "memory_base64" in delta:
            memory = json.loads(base64.b64decode(delta["memory_base64"], validate=True))
            facts = memory["host_facts"]
        if facts:
            feedback["host_facts"] = [{**fact, "statement": fact["statement"][:800]}
                                      for fact in facts[-8:]]
        return feedback

    def _run(self, *, wave_limit=None, approved_only=False):
        if approved_only:
            raise ValueError("target-driven Phase 2 uses fixed-target execution; approval-mode is unsupported")
        state, completed = self._load_state(), 0
        with self.time.active():
            self._assess_committed_memory(state)
        while state["terminal"] is None:
            if self.time.remaining <= 0:
                state["terminal"] = {"status": "BUDGET_EXHAUSTED", "reason": "Phase 2 elapsed-time budget exhausted"}
                self._atomic_json(self.root / "state.json", state)
                break
            with self.time.active():
                pending = self._load_pending(state) or self._select_pending(state)
            if pending is None:
                break
            wave = WaveSpec.from_mapping(pending["wave"])
            project = wave.projects[0].spec
            if pending["kind"] == "target" and project != self.target.spec:
                raise ValueError("pending target differs from the fixed goal")
            runner = LiveWaveCoordinator(self.root / "waves" / wave.wave_id, wave,
                self._memory_snapshot(state), self._execute, 1, allow_semantic_failures=True)
            with self.time.active(practice=pending["kind"] == "practice"):
                consolidation = runner.resume()
            committed = bool(consolidation.commits)
            delta = self._read_delta(wave.wave_id, project.project_id) if committed else None
            state = self._publish(state, wave, pending, (delta,))
            row = (state["prior_outcomes"] if committed else state["failed_outcomes"])[-1]
            attempt = {**row, "kind": pending["kind"], "wave_id": wave.wave_id,
                       "attempt_root": str(runner.root), "host_feedback": self._host_feedback(
                           runner.root / "branches" / project.project_id)}
            state["attempts"] = [*state["attempts"], attempt]
            if pending["kind"] == "target":
                state["target_attempts"] += 1
                state["practice_since_target"] = 0
                self.time.reset_practice()
            else:
                state["practice_since_target"] += 1
            state["next_kind"] = "review"
            self._atomic_json(self.root / "state.json", state)
            with self.time.active():
                self._assess_committed_memory(state)
            (self.root / "pending.json").unlink(missing_ok=True)
            if self.trace_hook is not None:
                self.trace_hook({"event_id": f"{self.snapshot.snapshot_id}:{wave.wave_id}",
                                 "phase": "target-driven-phase2", **attempt})
            completed += 1
            if wave_limit is not None and completed >= wave_limit:
                break
        return self._report(state)

    def run_attempt(self):
        """Execute at most one retained target/practice attempt, useful for inspection."""
        return self._run(wave_limit=1)

    def continue_campaign(self, *args, **kwargs):
        raise ValueError("target-driven budgets are fixed; start a new campaign to change them")

    def propose_wave(self):
        raise ValueError("target-driven Phase 2 does not support wave approval")

    def _report(self, state):
        terminal, pending = state["terminal"], self._pending_report(state)
        return {"phase": "target-driven-phase2", "target_id": self.target.target_id,
                "snapshot_id": self.snapshot.snapshot_id,
                "status": "RECOVERY_REQUIRED" if pending else terminal["status"] if terminal else "RUNNING",
                "reason": terminal["reason"] if terminal else None,
                "target_attempts": state["target_attempts"],
                "practice_attempts": sum(row["kind"] == "practice" for row in state["attempts"]),
                "successful_projects": len(state["prior_outcomes"]),
                "failed_projects": len(state["failed_outcomes"]),
                "completed_attempts": len(state["attempts"]), "attempts": state["attempts"],
                "curriculum_memory": self._curriculum_memory(state), "pending_wave": pending,
                "memory_sha256": hashlib.sha256(self._memory_snapshot(state)).hexdigest()}
