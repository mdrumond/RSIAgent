"""Fixed-goal learning over executable optimizer tasks, without a graph grammar."""
from __future__ import annotations

import base64
import json
import time

from .phase2_time import Phase2TimeBudget
from pathlib import Path

from optimizer.jobs import JobStore, atomic, canonical, sha
from optimizer.memory import verify_snapshot


class OptimizerPhase2Coordinator:
    def __init__(self, root, objective, snapshot, complete, execution_factory, *,
                 query_knowledge=None, max_seconds=7200, max_practice_seconds=1800, clock=time.monotonic,
                 selector=None, feedback_root=None, fixed_task=None):
        if selector is None:
            from .phase2_optimizer_tasks import select_optimizer_task
            selector = select_optimizer_task
        verify_snapshot(snapshot)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        from .phase2_feedback import FeedbackStore
        self.feedback = FeedbackStore(feedback_root or self.root / 'feedback')
        self.objective, self.seed, self.complete = objective, snapshot, complete
        self.factory, self.knowledge, self.selector = execution_factory, query_knowledge, selector
        self.fixed_task = fixed_task
        self.time = Phase2TimeBudget(self.root, max_seconds, max_practice_seconds, clock=clock)
        seed_path = self.root / "seed-memory.json"
        if seed_path.exists():
            if json.loads(seed_path.read_bytes()) != snapshot:
                raise ValueError("reviewed Phase 1 seed changed")
        else:
            atomic(seed_path, canonical(snapshot))
        self.identity = dict(snapshot_id=snapshot["snapshot_id"], objective=objective,
                             max_seconds=max_seconds, max_practice_seconds=max_practice_seconds)
        if fixed_task is not None:
            self.identity['fixed_task_sha256'] = sha(canonical(fixed_task))

    def _write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic(path, canonical(value))

    def _read(self, name):
        return json.loads((self.root / name).read_bytes())

    def _bound_executor(self, executor, *, practice=False):
        bind = getattr(executor, 'set_remaining_seconds', None)
        if bind is not None:
            remaining = self.time.remaining
            if practice:
                remaining = min(remaining, self.time.practice_remaining)
            bind(remaining)

    def _qualify(self, task, kind):
        self._write('qualification.json', dict(phase='optimizer-phase2', active=True, attempts=[], pending=dict(
            attempt_id=kind + '-' + sha(canonical(task))[:12], kind=kind, task=task)))
        try:
            executor = self.factory(self.root / self._directory(task), task, self.seed)
            self._bound_executor(executor, practice=kind.startswith("practice"))
            executor.prepare()
        except Exception:
            self.feedback.import_campaign(self.root)
            raise
        (self.root / 'qualification.json').unlink()

    def _bootstrap(self):
        path = self.root / "state.json"
        if path.exists():
            state = self._read("state.json")
            if state.get("phase") != "optimizer-phase2" or state["identity"] != self.identity:
                raise ValueError("retained optimizer Phase 2 identity changed")
            return state
        directory = self.root / "curriculum" / "goal"
        while self.time.remaining > 0:
            self._write("goal-selection.json", dict(stage="selection", proposal_count=0))
            try:
                task = self.fixed_task or self.selector(complete=self.complete, snapshot=self.seed,
                    objective=self.objective, root=directory, kind="goal",
                    query_knowledge=self.knowledge)
            except (ValueError, TypeError, KeyError) as error:
                self._write("goal-selection.json", dict(stage="failed", diagnostic=str(error)))
                raise
            self._write("goal-selection.json", dict(stage="qualification", candidate=task))
            try:
                # Torch reference is the oracle. No native teacher is constructed.
                self._qualify(task, 'qualification')
            except (ValueError, TypeError, KeyError) as error:
                if self.fixed_task is not None:
                    raise
                from .phase2_optimizer_tasks import reject_optimizer_task
                reject_optimizer_task(directory, str(error))
                continue
            break
        else:
            raise TimeoutError("Phase 2 time budget exhausted during qualification")
        self._write("goal-selection.json", dict(stage="qualified", candidate=task,
                    qualified_target_id=sha(canonical(task))))
        state = dict(schema="ascend-optimizer-phase2-v1", phase="optimizer-phase2",
                     identity=self.identity, snapshot_id=self.seed["snapshot_id"],
                     seed_file="seed-memory.json", target=task, attempts=[],
                     target_attempts=0, practice_since_target=0, next_kind="target",
                     terminal=None, memory_entries=[])
        self._write("state.json", state)
        return state

    def _memory(self, state):
        body = {k: v for k, v in self.seed.items() if k != "snapshot_id"}
        files = dict(body["files"])
        for row in state["attempts"]:
            data = canonical(row)
            files[f"phase2/attempts/{row['attempt_id']}.json"] = dict(
                sha256=sha(data), base64=base64.b64encode(data).decode())
            result = row["result"]
            for role in ("candidate", "baseline", "profiling", "result_artifact"):
                record = result.get(role)
                if record:
                    store = JobStore(self.root / row["execution_directory"] / "jobs")
                    data = store.read_artifact(result["job_id"], record["sha256"])
                    files[f"phase2/evidence/{record['sha256']}"] = dict(
                        sha256=sha(data), base64=base64.b64encode(data).decode())
        body.update(files=files, memory_entries=[*self.seed["memory_entries"], *state["memory_entries"]])
        return {**body, "snapshot_id": sha(canonical(body))}

    def _decision(self, state):
        name = f"curriculum/review-{len(state['attempts']):03d}.json"
        if (self.root / name).exists():
            return self._read(name)
        recent = [self._feedback(row) for row in state["attempts"][-3:]]
        context = dict(objective=self.objective, target=state["target"], recent_attempts=recent,
                       remaining_seconds=self.time.remaining,
                       practice_seconds_remaining=self.time.practice_remaining)
        messages = [{"role": "system", "content": (
            'You are the Phase 2 Curriculum Agent. Return JSON {"action":'
            '"ready"|"stalled"|"retry-target"|"practice","reason":"..."}. '
            'Choose focused practice only for an evidenced weakness, not arbitrary busywork. '
            'A verified practice never establishes target readiness: retry the fixed target. '
            'READY needs the latest attempt to be a host-verified target. '
            'Measurements must be host evidence; do not demand a speedup for correctness readiness.')},
            {"role": "user", "content": json.dumps(context, sort_keys=True)}]
        decision = json.loads(self.complete(messages))
        if (decision.get("action") not in {"ready", "stalled", "retry-target", "practice"}
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
            raise ValueError("invalid Phase 2 review decision")
        self._write(name, decision)
        return decision

    @staticmethod
    def _feedback(row):
        result = row["result"]
        return {**{k: row[k] for k in ("attempt_id", "kind", "task_id", "verified")},
            "feedback": {k: result.get(k) for k in ("classification", "summary", "speedup", "result_artifact")},
            "host_report": json.dumps(result.get("result"), sort_keys=True)[:12000]}

    def _select(self, state):
        pending = self.root / "pending.json"
        if pending.exists():
            value = self._read("pending.json")
            if any(row["attempt_id"] == value["attempt_id"] for row in state["attempts"]):
                pending.unlink()  # State publication completed before an interrupted cleanup.
            else:
                return value
        kind, task = "target", state["target"]
        if state["next_kind"] == "review":
            decision = self._decision(state)
            latest = state["attempts"][-1]
            if decision["action"] == "ready" and latest["kind"] == "target" and latest["verified"]:
                state["terminal"] = dict(status="READY", reason=decision["reason"])
                return None
            if decision["action"] == "stalled":
                state["terminal"] = dict(status="STALLED", reason=decision["reason"])
                return None
            if (decision["action"] == "practice"
                    and self.time.practice_remaining > 0
                    and self.time.remaining > 0):
                kind = "practice"
                directory = self.root / "curriculum" / f"practice-{len(state['attempts']):03d}"
                while self.time.remaining > 0 and self.time.practice_remaining > 0:
                    task = self.selector(complete=self.complete, snapshot=self._memory(state),
                        objective=self.objective, root=directory, kind="practice",
                        context=dict(target=state["target"], feedback=self._feedback(latest),
                        reason=decision["reason"], used_task_ids=[state["target"]["task_id"],
                            *[row["task_id"] for row in state["attempts"]]]), query_knowledge=self.knowledge)
                    try:
                        if task["task_id"] in {state["target"]["task_id"], *[r["task_id"] for r in state["attempts"]]}:
                            raise ValueError("practice must have a fresh task ID")
                        self._qualify(task, 'practice-qualification')
                    except (ValueError, TypeError, KeyError) as error:
                        from .phase2_optimizer_tasks import reject_optimizer_task
                        reject_optimizer_task(directory, str(error))
                        continue
                    break
                else:
                    kind, task = "target", state["target"]
        if self.time.remaining <= 0:
            state["terminal"] = dict(status="BUDGET_EXHAUSTED", reason="Phase 2 elapsed-time budget exhausted")
            return None
        value = dict(attempt_id=f"attempt-{len(state['attempts'])+1:03d}", kind=kind, task=task)
        self._write("pending.json", value)
        return value

    @staticmethod
    def _directory(task):
        return "tasks/" + task["task_id"] + "-" + sha(canonical(task))[:12]

    def _run(self, limit=None):
        with self.time.active():
            state = self._bootstrap()
        completed = 0
        self.feedback.import_campaign(self.root)
        while not state["terminal"]:
            if self.time.remaining <= 0:
                state["terminal"] = dict(status="BUDGET_EXHAUSTED", reason="Phase 2 elapsed-time budget exhausted")
                self._write("state.json", state)
                break
            with self.time.active():
                pending = self._select(state)
            self._write("state.json", state)
            if pending is None:
                break
            task, key = pending["task"], pending["attempt_id"]
            relative = self._directory(task)
            executor = self.factory(self.root / relative, task, self.seed)
            self._bound_executor(executor, practice=pending["kind"] == "practice")
            try:
                if getattr(executor, 'job_id', None) is not None:
                    executor.store.add_event(executor.job_id, 'phase2-prepare-request', dict(attempt_id=key))
                with self.time.active(practice=pending["kind"] == "practice"):
                    executor.prepare()
                    result = executor.attempt(key, memory_snapshot=self._memory(state),
                                              feedback=state["attempts"][-2:])
            except Exception:
                self.feedback.import_campaign(self.root)
                raise
            row = dict(attempt_id=key, kind=pending["kind"], task_id=task["task_id"],
                       task=task, result=result, verified=result["verified"], execution_directory=relative)
            # Publication precedes state advancement. Executor attempt IDs make recovery idempotent.
            self._write(f"attempts/{key}.json", row)
            state["attempts"].append(row)
            if pending["kind"] == "target":
                state["target_attempts"] += 1
                state["practice_since_target"] = 0
                self.time.reset_practice()
            else:
                state["practice_since_target"] += 1
            if row["verified"]:
                state["memory_entries"].append(dict(task_id=task["task_id"], attempt_id=key,
                    category="host-verification", statement="Candidate passed the frozen optimizer suite",
                    evidence_sha256=sha(canonical(row)), rationale=task["rationale"],
                    candidate_artifact=result.get("candidate"), result_artifact=result.get("result_artifact"),
                    interpretation=result.get("summary", "")))
            state["next_kind"] = "review"
            self._write("state.json", state)
            self.feedback.import_campaign(self.root)
            (self.root / "pending.json").unlink()
            completed += 1
            if limit is not None and completed >= limit:
                break
        return dict(phase="optimizer-phase2", status=(state["terminal"] or {}).get("status", "RUNNING"),
                    target_attempts=state["target_attempts"],
                    practice_attempts=sum(row["kind"] == "practice" for row in state["attempts"]),
                    completed_attempts=len(state["attempts"]), terminal=state["terminal"])

    def resume(self):
        return self._run()

    run = resume

    def run_attempt(self):
        return self._run(limit=1)
