"""Opt-in, parent-owned generic project waves around the canonical actor.

Factories are picklable trusted host adapters, not actor tools. They bind the
request's physical device to the target executor and open/create ProjectRunner
using the frozen memory and skills. No provider, transport or A5 tensor contract
is implemented here. Unknown dispatches hold their leases until observation.
"""
from dataclasses import dataclass
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from rsi.python_execution import atomic_json, digest


@dataclass(frozen=True)
class WaveRequest:
    project_id: str
    root: Path
    device: str
    memory: bytes
    skills: bytes
    actor_seconds: float
    actor_elapsed: float
    reopen: bool
    dispatch_id: str
    recovery_receipt: str | None = None


def _read(path):
    return json.loads(path.read_text())


def _worker(factory, request):
    # Only the canonical runner owns stage progression and elapsed accounting.
    atomic_json(request.root.parent / 'worker.json', dict(
        dispatch_id=request.dispatch_id, pid=os.getpid()))
    from rsi.project_runner import ProjectRunner
    runner = factory(request)
    if not isinstance(runner, ProjectRunner) or runner.root.resolve() != request.root.resolve():
        raise TypeError('wave factory must return the assigned canonical ProjectRunner')
    limits = runner.workflow.limits
    if (limits.actor_seconds != 7200.0
            or runner.workflow.actor_allocation() != request.actor_seconds or any(
            getattr(limits, name) is not None for name in
            ('design_iterations', 'implementation_iterations', 'execution_tries'))):
        raise ValueError('wave requires elapsed actor budget without count stopping limits')
    if runner.workflow.accounting.seconds() < request.actor_elapsed:
        raise ValueError("runner discarded consumed elapsed allocation")
    result = runner.run()
    # Unresolved accounting must never become a settled project receipt.
    elapsed = runner.workflow.accounting.seconds()
    atomic_json(request.root.parent / 'wave-result.json', dict(
        dispatch_id=request.dispatch_id, result=result, actor_seconds=elapsed,
        settled=not runner.workflow.pending and result['status'] in
        ('accepted', 'rejected', 'budget_exhausted')))


class ProjectWave:
    """Durable exclusive device leases and an all-settled publication barrier.

    run(factory) admits fresh/explicitly reconciled projects only; reopening a
    wave never replays a pending dispatch. status() includes original workflow
    handles. reconcile() calls an observation-only host adapter while holding
    the parent lock. It must establish worker/command closure and authoritative
    elapsed accounting, including any unresolved intervals, before releasing a
    lease. Completed results are immutable and skipped on every subsequent run.

    publish() returns a durable ordered result manifest for services to consume
    transactionally by its digest. It never mutates a shared memory/KDB itself.
    """
    def __init__(self, root, projects, devices, memory, skills, *, max_parallel=1):
        projects, devices = tuple(projects), tuple(devices)
        if (not projects or len(set(projects)) != len(projects) or any(
                not isinstance(p, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', p)
                or p in ('.', '..') for p in projects)):
            raise ValueError('unique simple project IDs required')
        if (not devices or len(set(devices)) != len(devices)
                or any(not isinstance(d, str) or not d for d in devices)):
            raise ValueError('unique admitted physical device identities required')
        if type(max_parallel) is not int or not 1 <= max_parallel <= len(devices):
            raise ValueError('parallelism must fit admitted device leases')
        if not isinstance(memory, bytes) or not isinstance(skills, bytes):
            raise TypeError('wave snapshots must be immutable bytes')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan = dict(projects=list(projects), devices=list(devices),
                         memory=memory.hex(), skills=skills.hex(), max_parallel=max_parallel)
        self.path = self.root / 'wave.json'
        with self._lock():
            if self.path.exists():
                if self._load()['plan'] != self.plan:
                    raise ValueError('wave inputs differ from frozen plan')
            else:
                self._save(dict(plan=self.plan, stopped=False, publication=None,
                    projects={p: dict(status='ready', device=None, dispatch_id=None,
                        actor_seconds=0.0, allowance=7200.0, recovery=None, reopen=False)
                        for p in projects}))

    def _lock(self):
        class Lock:
            def __enter__(inner):
                inner.stream = (self.root / 'writer.lock').open('a')
                try:
                    fcntl.flock(inner.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    inner.stream.close()
                    raise RuntimeError('another wave writer owns this wave')
            def __exit__(inner, *args):
                inner.stream.close()
        return Lock()

    def _load(self):
        return _read(self.path)

    def _save(self, state):
        atomic_json(self.path, state)

    def _branch(self, project):
        return self.root / 'projects' / project

    def _request(self, project, row):
        return WaveRequest(project, self._branch(project) / 'runner', row['device'],
            bytes.fromhex(self.plan['memory']), bytes.fromhex(self.plan['skills']),
            row['allowance'], row['actor_seconds'], row['reopen'], row['dispatch_id'],
            json.dumps(row['recovery'], sort_keys=True) if row['recovery'] else None)

    def _available(self, state, row):
        used = {r['device'] for r in state['projects'].values() if r['device'] is not None}
        return [d for d in self.plan['devices'] if d not in used
                and row.get('lease_device', d) == d]

    def _retain(self, state, project, *, save=True):
        row = state['projects'][project]
        path = self._branch(project) / 'wave-result.json'
        if not path.exists():
            return False
        receipt = _read(path)
        if receipt['dispatch_id'] != row['dispatch_id']:
            return False  # An older, explicitly closed dispatch may have a receipt.
        elapsed = receipt['actor_seconds']
        if not math.isfinite(elapsed) or elapsed < row['actor_seconds']:
            raise ValueError('elapsed allocation moved backwards')
        row['actor_seconds'] = elapsed
        row['receipt'] = receipt
        if receipt['settled']:
            row.update(status='completed', device=None)
        else:
            row['status'] = 'pending'  # Hold device even after worker exit.
        if save:
            self._save(state)
        return True

    def run(self, runner_factory):
        """Drain admitted workers; a stop request prevents further admission.

        Abrupt parent loss leaves pending reservations intact. Children retain
        their own receipts; only an observer may prove an orphan worker closed.
        Factory must be importable/picklable for multiprocessing spawn.
        """
        with self._lock():
            state = self._load()
            if state['publication'] or state['stopped']:
                return self._status(state)
            context = multiprocessing.get_context('spawn')
            active = {}
            try:
                while True:
                    if (self.root / 'stop.json').exists():
                        state['stopped'] = True
                        self._save(state)
                    used = {r['device'] for r in state['projects'].values() if r['device'] is not None}
                    for project, row in state['projects'].items():
                        if (self.root / 'stop.json').exists():
                            state['stopped'] = True
                            self._save(state)
                        if state['stopped'] or len(used) >= self.plan['max_parallel']:
                            break
                        if row['status'] != 'ready':
                            continue
                        available = self._available(state, row)
                        if not available:
                            continue
                        if row['actor_seconds'] >= row['allowance']:
                            continue  # Budget exhaustion requires an authoritative result.
                        row.update(status='pending', device=available[0],
                                   lease_device=available[0], dispatch_id=uuid4().hex)
                        used.add(row['device'])
                        self._branch(project).mkdir(parents=True, exist_ok=True)
                        self._save(state)  # Reserve before any possible side effect.
                        process = context.Process(target=_worker,
                            args=(runner_factory, self._request(project, row)))
                        process.start()  # A start failure leaves the reservation pending.
                        active[project] = process
                    for project, process in list(active.items()):
                        if process.is_alive():
                            continue
                        process.join()
                        del active[project]
                        self._retain(state, project)
                    if not active:
                        used = {r["device"] for r in state["projects"].values() if r["device"] is not None}
                        if (not state["stopped"] and len(used) < self.plan["max_parallel"]
                                and any(r["status"] == "ready" and r["actor_seconds"] < r["allowance"]
                                        and self._available(state, r)
                                        for r in state["projects"].values())):
                            continue
                        break
                    time.sleep(.05)
            finally:
                # Do not terminate unknown target work or release its device.
                for process in active.values():
                    process.join()
            return self._status(state)

    def stop(self):
        """Request admission stop; live siblings drain and retain their results."""
        atomic_json(self.root / 'stop.json', dict(requested=True))

    def resume(self):
        with self._lock():
            state = self._load()
            (self.root / 'stop.json').unlink(missing_ok=True)
            state['stopped'] = False
            self._save(state)

    def reconcile(self, project, observe):
        """Observe original dispatch; never invoke runner.run or redispatch.

        observe(request, detached_row) -> {closed: True, observation: str,
        actor_seconds: float, resume_ready: bool}. The host must check the
        original controller AND target handles, reconcile accounting, and attest
        no pending workflow/command before resume_ready. A durable settled child
        receipt can instead complete the project once closure is observed.
        """
        with self._lock():
            state = self._load()
            row = state['projects'][project]
            if row['status'] != 'pending':
                raise ValueError('only pending dispatches require reconciliation')
            proof = observe(self._request(project, row), json.loads(json.dumps(row)))
            if proof.get('closed') is not True or not proof.get('observation', '').strip():
                raise ValueError('authoritative original worker/target closure required')
            elapsed = proof['actor_seconds']
            if not math.isfinite(elapsed) or elapsed < row['actor_seconds']:
                raise ValueError('authoritative elapsed allocation required')
            self._retain(state, project, save=False)
            if elapsed < row['actor_seconds']:
                raise ValueError('closure accounting precedes retained result')
            row.update(actor_seconds=elapsed, closure=proof)
            if row['status'] != 'completed' and proof.get('resume_ready') is True:
                row.update(status='ready', device=None, reopen=True)
            self._save(state)
            return self._status(state)

    def grant_harness_recovery(self, project, *, observation):
        """One logged extra 7200 seconds, only after observed harness closure.

        Host adapter must use Workflow.grant_actor_seconds with this retained
        receipt on the same runner; its frozen base limits never change.
        """
        with self._lock():
            state = self._load()
            row = state['projects'][project]
            if (row['status'] != 'ready' or not row.get('closure')
                    or row['closure'].get('harness_failure') is not True
                    or row['recovery'] is not None or not observation.strip()):
                raise ValueError('one observed harness recovery on a closed unfinished project required')
            row.update(allowance=14400.0, recovery=dict(observation=observation,
                closure=row['closure'], dispatch_id=row['dispatch_id']))
            self._save(state)

    def publish(self):
        """Commit ordered results once every sibling has a retained outcome."""
        with self._lock():
            state = self._load()
            if state['publication'] is not None:
                return state['publication']
            if any(r['status'] != 'completed' for r in state['projects'].values()):
                raise RuntimeError('wave publication requires all siblings settled')
            packet = dict(memory_sha256=digest(bytes.fromhex(self.plan['memory'])),
                skills_sha256=digest(bytes.fromhex(self.plan['skills'])),
                outcomes=[dict(r['receipt'], project_id=p, actor_seconds=r['actor_seconds'],
                               original_receipt=r['receipt']) for p, r in state['projects'].items()])
            payload = json.dumps(packet, sort_keys=True, allow_nan=False).encode()
            state['publication'] = dict(identity=digest(payload), **packet)
            self._save(state)
            return state['publication']

    def _status(self, state):
        result = json.loads(json.dumps(state))
        for project, row in result['projects'].items():
            worker_path = self._branch(project) / 'worker.json'
            if worker_path.exists():
                worker = _read(worker_path)
                if worker['dispatch_id'] == row['dispatch_id']:
                    row['worker'] = worker
            path = self._branch(project) / 'runner' / 'workflow.json'
            if path.exists():
                workflow = _read(path)
                row['workflow_pending'] = workflow.get('pending')
                row['execution_reservations'] = workflow.get('execution_reservations', {})
        return result

    def status(self):
        return self._status(self._load())
