"""Single-writer persisted workflow controller; evidence comes from host stores.

The resolver(kind, reference, stage, operation_id) must verify both provenance
and scope. Callers are trusted host adapters, never direct agent tool inputs.
Transport dispatch happens only after begin() durably records its identity.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import uuid

from rsi.contracts import BudgetLimits, Stage, StepEvidence


class WorkflowError(RuntimeError):
    """A transition, evidence requirement or remaining budget forbids work."""


class ActorClock:
    """Durable interval union. Reopened active intervals require host receipts.

    Timestamps are epoch seconds, not process-relative monotonic timestamps.
    A restart never guesses whether remote work continued during downtime.
    reconcile() requires an authoritative end timestamp in this controller clock
    domain. Never pass raw remote epoch timestamps: transport adapters must map
    host-local elapsed/monotonic receipts to the controller timeline (PR7).
    A numerically plausible timestamp is not proof of matching clock domains.
    """

    BILLABLE = {"thinking", "retrieval", "implementation", "debug",
                "compilation", "execution", "profiling"}
    EXCLUDED = {"setup", "transfer", "wait", "pause"}

    def __init__(self, records, save, clock, uncertainties=None, receipts=None):
        self.records, self.save, self.clock = records, save, clock
        self.uncertainties = {} if uncertainties is None else uncertainties
        self.receipts = {} if receipts is None else receipts
        self.unresolved = {key for key, value in records.items() if value[2] is None}

    def now(self):
        now = self.clock()
        starts = [record[1] for record in self.records.values()]
        ends = [record[2] for record in self.records.values() if record[2] is not None]
        if not math.isfinite(now) or now < max(starts + ends, default=now):
            raise WorkflowError("clock moved backwards; reconcile timing first")
        return now

    def start(self, category):
        if self.unresolved or self.uncertainties or category not in self.BILLABLE | self.EXCLUDED:
            raise WorkflowError("unresolved accounting or unknown activity")
        token = uuid.uuid4().hex
        self.records[token] = [category, self.now(), None]
        self.save()
        return token

    def end(self, token):
        if token in self.unresolved:
            raise WorkflowError("use authoritative reconciliation after restart")
        self.reconcile(token, self.now())

    def reconcile(self, token, ended_at):
        record = self.records[token]
        if (record[2] is not None or not math.isfinite(ended_at)
                or ended_at < record[1] or ended_at > self.clock()):
            raise WorkflowError("invalid interval receipt")
        record[2] = ended_at
        self.unresolved.discard(token)
        self.save()

    def seconds(self, category=None):
        if self.unresolved or self.uncertainties:
            raise WorkflowError("active or uncertain intervals require reconciliation")
        now = self.now()
        intervals = sorted((start, now if end is None else end)
                           for kind, start, end in self.records.values()
                           if kind == category or category is None and kind in self.BILLABLE)
        total, boundary = 0.0, float("-inf")
        for start, end in intervals:
            total += max(0, end - max(start, boundary))
            boundary = max(boundary, end)
        return total


class Workflow:
    """Durable stage machine, intended for one controller per state file.

    complete/reconcile accept host-observed results. Agent references alone
    cannot establish skill reads, actual retrievals or retrospectives. The
    resolver must bind those receipts to the given operation and stage.
    pending operation identities must be used as transport idempotency keys.
    """

    def __init__(self, path, resolver, *, source_hash, limits=None, clock=time.time):
        self.path, self.resolver, self.clock = Path(path), resolver, clock
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if source_hash != self.source_hash:
                raise WorkflowError("source differs from persisted checkpoint; reconcile source explicitly")
        else:
            self.state = dict(stage=Stage.PREFLIGHT.value, source=source_hash,
                              limits=(limits or BudgetLimits()).to_dict(), pending=None,
                              stopped=False, outcome=None, verified=None, tested=None,
                              design=0, implementation=0, executions=0, intervals={},
                              created=clock(), history=[])
            self._save()
        self._needs_reconciliation = self.pending is not None
        self.limits = BudgetLimits.from_dict(self.state["limits"])
        self.accounting = ActorClock(self.state["intervals"], self._save, clock,
                                     self.state.setdefault("timing_uncertainties", {}),
                                     self.state.setdefault("timing_receipts", {}))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w") as stream:
            json.dump(self.state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @property
    def stage(self):
        return Stage(self.state["stage"])

    @property
    def source_hash(self):
        return self.state["source"]

    @property
    def verified_source(self):
        return self.state["verified"]

    @property
    def outcome(self):
        return self.state["outcome"]

    @property
    def pending(self):
        return dict(self.state["pending"]) if self.state["pending"] else None

    def constrain_wall_deadline(self, scope, deadline):
        """Persist a host parent deadline without changing frozen task limits.

        Constraints only tighten, survive reopen and flow through remaining()
        to provider setup/thinking and backend execution timeouts.
        """
        if not scope or type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('named finite parent wall deadline required')
        deadlines = self.state.setdefault('wall_deadlines', {})
        deadlines[scope] = min(deadlines.get(scope, deadline), deadline)
        self._save()

    def remaining(self):
        now = self.accounting.now()
        if now < self.state["created"]:
            raise WorkflowError("clock precedes workflow creation")
        actor = self.limits.actor_seconds
        wall = self.limits.wall_seconds
        deadlines = list(self.state.get('wall_deadlines', {}).values())
        if wall is not None:
            deadlines.append(self.state['created'] + wall)
        return {"actor_seconds": None if actor is None else max(0, actor - self.accounting.seconds()),
                "wall_seconds": max(0, min(deadlines) - now) if deadlines else None}

    def begin(self, *, execution=False):
        if self.state["stopped"] or self.outcome or self.pending:
            raise WorkflowError("stopped, terminal or unreconciled operation")
        if any(value is not None and value <= 0 for value in self.remaining().values()):
            raise WorkflowError("time budget exhausted")
        counts = {}
        if self.stage == Stage.DESIGN:
            counts = {"design": self.limits.design_iterations}
        elif self.stage == Stage.IMPLEMENTATION:
            counts = {"implementation": self.limits.implementation_iterations}
        if execution:
            counts["executions"] = self.limits.execution_tries
        if any(cap is not None and self.state[key] >= cap for key, cap in counts.items()):
            raise WorkflowError("iteration or execution cap exhausted")
        before = {key: self.state[key] for key in counts}
        if self.stage == Stage.DESIGN:
            before["implementation"] = self.state["implementation"]
        for key in counts:
            self.state[key] += 1
        if self.stage == Stage.DESIGN:
            self.state["implementation"] = 0
        operation = uuid.uuid4().hex
        self.state["pending"] = dict(id=operation, stage=self.stage.value, handle=None,
                                      counts_before=before)
        self._save()
        return operation

    def reserve_execution(self, operation):
        """Durably admit one command; False means already reserved, never replay.

        A legacy begin(execution=True) supplies credit for its first command.
        Reservations survive failure/restart; only authoritative absence may
        justify release by a reconciliation adapter.
        """
        reservations = self.state.setdefault("execution_reservations", {})
        if operation in reservations:
            return False
        if self.state["stopped"] or self.outcome or self._needs_reconciliation:
            raise WorkflowError("stopped, terminal or unreconciled operation")
        if any(value is not None and value <= 0 for value in self.remaining().values()):
            raise WorkflowError("time budget exhausted")
        pending = self.state["pending"]
        credit = pending and "executions" in pending.get("counts_before", {})
        if not credit:
            cap = self.limits.execution_tries
            if cap is not None and self.state["executions"] >= cap:
                raise WorkflowError("execution cap exhausted")
            self.state["executions"] += 1
        else:
            # Once assigned, stage absence cannot refund a command that may run.
            del pending["counts_before"]["executions"]
        reservations[operation] = {"stage_operation": pending["id"] if pending else None}
        self._save()
        return True

    def dispatched(self, operation, handle):
        if (self._needs_reconciliation or self.state["stopped"] or not self.pending
                or self.pending["id"] != operation or self.pending["handle"]):
            raise WorkflowError("unknown or already dispatched operation")
        self.state["pending"]["handle"] = handle
        self._save()

    def stop(self):
        self._needs_reconciliation = self.pending is not None
        self.state["stopped"] = True
        self._save()
        return self.pending  # Host adapter cancels/observes this same operation.

    def resume(self):
        self.state["stopped"] = False
        self._save()  # Pending operations and timing receipts deliberately survive.

    def source_changed(self, source_hash):
        if self.pending:
            raise WorkflowError("record source changes with the operation result")
        if self.outcome:
            raise WorkflowError("terminal workflow is immutable")
        if source_hash != self.source_hash:
            self.state.update(source=source_hash, tested=None, verified=None)
            if self.stage in (Stage.DEBUG, Stage.OPTIMISATION, Stage.VALIDATION, Stage.ACCEPTANCE):
                self.state["stage"] = Stage.TEST.value
            self._save()

    def reject_operation(self, operation, hostfailure_ref, observed_source_hash):
        """Close a proven terminal admission failure, retaining all consumed budget.

        This is not an absence/refund receipt. The trusted resolver must establish
        actor closure and no host effects; unknown work must remain pending.
        """
        if (not self.pending or self.pending['id'] != operation or self.outcome
                or observed_source_hash != self.source_hash):
            raise WorkflowError('failed operation or observed source mismatch')
        if not self.resolver('failed-operation', hostfailure_ref, self.stage, operation):
            raise WorkflowError('untrusted failed-operation closure proof')
        self.accounting.seconds()  # Unresolved work/timing cannot be discarded.
        self.state['history'].append(dict(operation=operation, stage=self.stage.value,
            source_hash=self.source_hash, passed=False, rejected_operation=True,
            host_failure_ref=hostfailure_ref))
        self.state['pending'] = None
        self._needs_reconciliation = False
        self._save()

    def reconcile(self, operation, evidence=None, *, not_dispatched=False, **result):
        if not_dispatched:
            if not self.pending or self.pending["id"] != operation or evidence is not None:
                raise WorkflowError("invalid absent-operation receipt")
            if "counts_before" not in self.pending:
                raise WorkflowError("legacy reservation requires explicit counter reconciliation")
            self.state.update(self.pending["counts_before"])
            self.state["history"].append(dict(operation=operation, not_dispatched=True))
            self.state["pending"] = None
            self._needs_reconciliation = False
            self._save()
            return
        if evidence is None or evidence.operation_id != operation:
            raise WorkflowError("receipt operation mismatch")
        self.complete(evidence, **result)

    def complete(self, evidence: StepEvidence, *, passed=True, new_source_hash=None,
                 redesign=False):
        pending, stage = self.pending, self.stage
        if (not pending or pending["id"] != evidence.operation_id or evidence.stage != stage
                or evidence.source_hash != self.source_hash):
            raise WorkflowError("operation, stage or source mismatch")
        refs = [("skill", ref) for ref in evidence.skill_refs]
        if not refs:
            raise WorkflowError("skill-read evidence required")
        if stage in (Stage.DESIGN, Stage.OPTIMISATION):
            if not evidence.retrieval_refs or not evidence.retrospective_ref:
                raise WorkflowError("retrieval and retrospective required")
            refs += [("retrieval", ref) for ref in evidence.retrieval_refs]
            refs.append(("retrospective", evidence.retrospective_ref))
        if not all(self.resolver(kind, ref, stage, evidence.operation_id) for kind, ref in refs):
            raise WorkflowError("host evidence resolver rejected references")
        if stage == Stage.ACCEPTANCE and self.verified_source != self.source_hash:
            raise WorkflowError("final source lacks validation")
        if new_source_hash and stage not in (Stage.IMPLEMENTATION, Stage.DEBUG, Stage.OPTIMISATION):
            raise WorkflowError("stage cannot mutate source")
        if redesign and stage not in (Stage.DEBUG, Stage.VALIDATION):
            raise WorkflowError("redesign permitted only after debugging or validation")
        changed = new_source_hash is not None and new_source_hash != self.source_hash
        if changed:
            self.state.update(source=new_source_hash, tested=None, verified=None)
        order = list(Stage)
        next_stage = order[min(order.index(stage) + 1, len(order) - 1)]
        if stage == Stage.TEST:
            self.state["tested"] = self.source_hash if passed else None
        elif stage == Stage.DEBUG:
            if not passed:
                self.state.update(tested=None, verified=None)
            next_stage = Stage.OPTIMISATION if self.state["tested"] == self.source_hash else Stage.IMPLEMENTATION
        elif stage == Stage.OPTIMISATION:
            if not passed:
                self.state.update(tested=None, verified=None)
                next_stage = Stage.DEBUG
            else:
                next_stage = Stage.TEST if changed else Stage.VALIDATION
        elif stage == Stage.VALIDATION:
            if passed and self.state["tested"] == self.source_hash:
                self.state["verified"] = self.source_hash
            else:
                next_stage = Stage.DEBUG
                self.state["tested"] = None
        elif stage == Stage.ACCEPTANCE:
            self.state["outcome"] = "accepted" if passed else "rejected"
        elif not passed:
            next_stage = stage
        if redesign:
            next_stage = Stage.DESIGN
            self.state.update(tested=None, verified=None)
        self.state["history"].append(dict(evidence=evidence.to_dict(), passed=passed,
                                           new_source_hash=new_source_hash, redesign=redesign))
        self.state.update(stage=next_stage.value, pending=None)
        self._needs_reconciliation = False
        self._save()
