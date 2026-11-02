"""Single-writer wave policy around the canonical runner, never another dev loop.

runner(spec, memory_snapshot, skill_snapshot, context) -> HostTaskResult.
Snapshots are retained immutable version references owned by existing stores.
All hooks receive context containing operation_id (their idempotency key),
namespace and remaining actor allocations. memory_update(memory, results, ctx)
returns SnapshotUpdate; skill_update(skills, memory, result, ctx) returns a
SnapshotUpdate naming a *candidate*, never an implicitly promoted active bundle.
SubjectMemory.accept can use operation_id as its existing admission identity.

step dispatches at most one durable operation. An exception, stop or restart
with pending work requires reconcile(operation_id, authoritative_result), or
confirmed not_dispatched=True. No hook/runner is automatically replayed.
"""
from dataclasses import asdict, dataclass, replace
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid

from rsi.contracts import CycleSpec, Mode, ProjectSpec
from rsi.runtime import WorkflowError


@dataclass(frozen=True)
class HostTaskResult:
    outcome: str
    artifact_ref: str
    actor_seconds: float
    classification: str | None = None
    terminal_failure_ref: str | None = None
    audit_ref: str | None = None

    def __post_init__(self):
        if not self.outcome or not self.artifact_ref:
            raise ValueError('host outcome and retained result reference required')
        _seconds(self.actor_seconds)
        if self.classification == 'harness_failure' and (
                self.outcome == 'accepted' or not self.terminal_failure_ref or not self.audit_ref):
            raise ValueError('harness failure requires terminal proof and scoped host audit')


@dataclass(frozen=True)
class SimplificationResult:
    """Trusted host result with retained timing evidence, never model estimates.

    The hook retains its receipt under context.operation_id before returning;
    receipt_ref identifies that host record for reconciliation after a lost return.
    actor_seconds is measured billable elapsed time, including model thinking and
    failed attempts, excluding setup/transfers. A host-only rewrite supplies zero
    explicitly. No implicit free-work default is permitted.
    """
    spec: ProjectSpec
    actor_seconds: float
    receipt_ref: str
    parent_project_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.spec, ProjectSpec):
            raise TypeError('revised ProjectSpec required')
        if not isinstance(self.receipt_ref, str) or not self.receipt_ref.strip():
            raise ValueError('retained host timing receipt required')
        _seconds(self.actor_seconds)


@dataclass(frozen=True)
class SimplificationDeferred:
    """Known elapsed exhaustion; retain the queue without inventing a child."""
    actor_seconds: float
    receipt_ref: str

    def __post_init__(self):
        _seconds(self.actor_seconds)
        if not self.receipt_ref:
            raise ValueError('retained exhaustion receipt required')


@dataclass(frozen=True)
class IsolationSnapshot:
    """Host receipt for separate mutable heads, even when CAS content is equal."""
    namespace: str
    memory: str
    skills: str
    branch_root: str
    receipt_ref: str

    def verify(self):
        import hashlib
        root = Path(self.branch_root).resolve()
        data = (root / 'isolation.json').read_bytes()
        proof = json.loads(data)
        if (hashlib.sha256(data).hexdigest() != self.receipt_ref
                or proof['namespace'] != self.namespace
                or proof['memory'] != self.memory or proof['skills'] != self.skills
                or not (root / 'memory/active.json').is_file()
                or not (root / 'learning.json').is_file()
                or (root / 'memory').resolve() == Path(proof['parent']).resolve()
                or not (root / 'memory/active.json').resolve().is_relative_to(root)
                or not (root / 'learning.json').resolve().is_relative_to(root)):
            raise ValueError('invalid mutable-branch isolation proof')


@dataclass(frozen=True)
class SnapshotUpdate:
    reference: str
    actor_seconds: float = 0.0

    def __post_init__(self):
        if not self.reference:
            raise ValueError('retained snapshot reference required')
        _seconds(self.actor_seconds)


@dataclass(frozen=True)
class RejectedSkillUpdate:
    """Host-proven semantic rejection of a finished, retained writer proposal."""
    operation_id: str
    namespace: str
    incumbent: str
    memory_ref: str
    project_ref: str
    proposal_ref: str
    rejection_ref: str
    actor_seconds: float

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != 'actor_seconds' and (not isinstance(value, str) or not value.strip()):
                raise ValueError('scoped retained rejection references required')
        _seconds(self.actor_seconds)


def _seconds(value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError('elapsed actor seconds must be finite and nonnegative')


class CycleScheduler:
    """One locked writer per state path; close() releases ownership.

    wave_limit and recovery_limit are explicit hard caps (None disables each).
    Actor time is supplied by host timing receipts, excluding setup/transfers.
    Wall time includes all elapsed time, including controller downtime.
    Inference requires isolate(memory, skills, context) -> IsolationSnapshot
    before any project runs. This hook must publish truly isolated store branches.
    simplify(spec, result, context) -> SimplificationResult is Phase1-only.
    Legacy hooks receive remaining parent/recovery elapsed limits. Independent
    projects receive fresh allocations; cycle totals remain telemetry and an
    explicit stop still prevents further dispatch.
    Phase2/inference recovery is emitted as deferred work for paired evaluation.
    """

    def __init__(self, path, spec, runner, memory_update, *, skill_update=None,
                 isolate=None, simplify=None, recover=True, wave_size=None,
                 wave_limit=None, recovery_limit=1, actor_seconds=None,
                 recovery_seconds=None, wall_seconds=None, tuning_every=5,
                 recovery_feedback=False, retry_original=None, max_failed_recovery_attempts=5,
                 clock=time.time, independent_projects=False):
        if not isinstance(spec, CycleSpec):
            raise TypeError('CycleSpec required')
        if type(independent_projects) is not bool:
            raise ValueError('independent_projects must be boolean')
        if spec.mode != Mode.PHASE1 and skill_update is None:
            raise ValueError('Phase2/inference requires injected skill_update')
        if spec.mode == Mode.INFERENCE and isolate is None:
            raise ValueError('inference requires an isolation hook')
        for value in (wave_size, wave_limit, recovery_limit, tuning_every):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError('count limits must be positive integers or None')
        for value in (actor_seconds, recovery_seconds, wall_seconds):
            if value is not None:
                _seconds(value)
        self.path, self.clock = Path(path), clock
        if type(max_failed_recovery_attempts) is not int or max_failed_recovery_attempts <= 0:
            raise ValueError('max_failed_recovery_attempts must be a positive integer')
        self.retry_original = retry_original
        self.runner, self.memory_update = runner, memory_update
        self.skill_update, self.isolate, self.simplify = skill_update, isolate, simplify
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix('.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise WorkflowError('cycle already has a writer') from None
        config = dict(spec=spec.to_dict(), recover=recover, wave_size=wave_size,
                      wave_limit=wave_limit, recovery_limit=recovery_limit,
                      actor_seconds=actor_seconds, recovery_seconds=recovery_seconds,
                      wall_seconds=wall_seconds, tuning_every=tuning_every)
        if independent_projects and spec.mode != Mode.PHASE1:
            config['max_failed_recovery_attempts'] = max_failed_recovery_attempts
        if recovery_feedback:
            config['recovery_feedback'] = True
        if independent_projects:
            config['independent_projects'] = True
        if retry_original is not None:
            config.update(deferred_recovery=True, max_failed_recovery_attempts=max_failed_recovery_attempts, recovery_limit=None)
        try:
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                if self.state['config'] != config:
                    raise ValueError('retained cycle configuration differs')
            else:
                self.state = dict(config=config, created=clock(), stopped=False,
                                  phase='isolate' if spec.mode == Mode.INFERENCE else 'original',
                                  namespace=('inference:' if spec.mode == Mode.INFERENCE else 'cycle:')
                                  + spec.cycle_id + ':' + uuid.uuid4().hex,
                                  memory=spec.baseline_kdb, skills=spec.baseline_skills,
                                  cursor=0, wave=1, wave_results=[], originals=[], recoveries=[],
                                  recovery_queue=[], pending=None, events=[], candidates=[],
                                  spent=0.0, recovery_spent=0.0, completed_operations={})
                self._save()
        except BaseException:
            self.close()
            raise
        self.spec, self.config = spec, config
        self.needs_reconciliation = self.state['pending'] is not None

    def close(self):
        if not self.lock.closed:
            self.lock.close()

    def _save(self):
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('w') as stream:
            json.dump(self.state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def stop(self):
        self.state['stopped'] = True
        self._save()
        return copy.deepcopy(self.state['pending'])

    def resume(self):
        self.state['stopped'] = False
        self._save()

    def admit_failed_parents(self, rows, receipt_ref):
        """Start a new diagnostic cycle from retained failures, never rerun parents."""
        s = self.state
        previous = s.get('failed_parent_admission')
        if previous is not None:
            if previous != dict(rows=rows, receipt_ref=receipt_ref):
                raise ValueError('failed-parent admission differs from its retained receipt')
            return
        if (self.spec.mode != Mode.PHASE1 or not self.simplify or not self.config['recover']
                or s['phase'] != 'original' or s['cursor'] or s['completed_operations']
                or s['pending'] or s['originals'] or s['recoveries'] or s['spent'] or not receipt_ref):
            raise WorkflowError('failed-parent admission requires a fresh configured Phase1 boundary')
        if not self.config.get('deferred_recovery') and not any(
                self.config[key] is not None for key in ('actor_seconds', 'recovery_seconds')):
            raise ValueError('diagnostic recovery requires an explicit finite elapsed allocation')
        failed = []
        for row in rows:
            ProjectSpec.from_dict(row['project'])
            if not row['artifact_ref'] or not row['operation_id']:
                raise ValueError('retained failed-parent result identity required')
            if row['outcome'] != 'accepted':
                failed.append(dict(spec=row['project'], original_id=row['original_id'],
                                   result=row, attempts=0))
        s['failed_parent_admission'] = copy.deepcopy(dict(rows=rows, receipt_ref=receipt_ref))
        s['recovery_queue'] = ([self._root_item(item['result']) for item in failed]
            if self.config.get('deferred_recovery') else copy.deepcopy(failed))
        s['cursor'] = len(self.spec.projects)
        s['phase'] = 'simplify' if failed else 'complete'
        if self.config.get('deferred_recovery') and failed:
            self._wave_end()
        self._save()

    def _root_item(self, row):
        spent = row.get('original_actor_seconds_total', row['actor_seconds'])
        spec = copy.deepcopy(row['project'])
        return dict(spec=spec, root_spec=copy.deepcopy(spec), original_id=row['original_id'],
            result=copy.deepcopy(row), root_result=copy.deepcopy(row), root_spent=spent,
            attempts=0, failed_attempts=0, failure_limit=self.config['max_failed_recovery_attempts'],
            status='queued', action='diagnostic', eligible_wave=self.state['wave'] + 1)

    def _root_remaining(self, item):
        limit = item.get('root_actor_limit', item['root_spec']['budgets']['actor_seconds'])
        if self.config.get('deferred_recovery'):
            return limit
        return None if limit is None else max(0., limit - item['root_spent'])

    def _eligible(self, *, unprepared=False):
        rows = []
        for item in self.state['recovery_queue']:
            if item.get('status') != 'queued' or item['eligible_wave'] > self.state['wave']:
                continue
            if not unprepared or item['action'] == 'diagnostic' and not item.get('prepared'):
                rows.append(item)
        return rows

    def _failed_attempt(self, item):
        item['failed_attempts'] += 1
        if item['failed_attempts'] >= item['failure_limit']:
            item['status'] = 'unrecoverable'
            self.state['events'].append(dict(kind='original_unrecoverable',
                original_id=item['original_id'], failed_attempts=item['failed_attempts']))

    def reopen_original(self, project_id):
        """Explicitly reopen eligibility, preserving every elapsed/attempt receipt."""
        if self.state['pending']:
            raise WorkflowError('unknown pending work must be observed before reopening')
        item = next(r for r in self.state['recovery_queue'] if r['original_id'] == project_id)
        if item.get('status') != 'unrecoverable':
            raise ValueError('only an unrecoverable original can be explicitly reopened')
        item.update(status='queued', action='diagnostic', prepared=False,
            failure_limit=item['failed_attempts'] + self.config['max_failed_recovery_attempts'],
            eligible_wave=self.state['wave'] + 1)
        self.state['events'].append(dict(kind='original_reopened', original_id=project_id,
            failed_attempts=item['failed_attempts'], failure_limit=item['failure_limit']))
        if self.state['phase'] in ('complete', 'wave_limit'):
            self._wave_end()
        self._save()

    def remaining(self):
        elapsed = self.clock() - self.state['created']
        if not math.isfinite(elapsed) or elapsed < 0:
            raise WorkflowError('invalid elapsed clock')
        wave = self.state.get('active_wave') or {}
        retained = sum(seconds for operation, seconds in wave.get('retained_actor_seconds', {}).items()
                       if operation not in self.state['completed_operations'])
        recovery_ids = {op['id'] for op in wave.get('operations', []) if op.get('recovery')}
        retained_recovery = sum(seconds for operation, seconds in wave.get('retained_actor_seconds', {}).items()
            if operation in recovery_ids and operation not in self.state['completed_operations'])
        return {name: None if self.config[name] is None else max(0, self.config[name] - spent)
                for name, spent in [('actor_seconds', self.state['spent'] + retained),
                                    ('recovery_seconds', self.state['recovery_spent'] + retained_recovery),
                                    ('wall_seconds', elapsed)]}

    def _context(self, operation):
        context = dict(operation_id=operation['id'], namespace=self.state['namespace'],
                    wave=self.state['wave'], original_project_id=operation.get('original_id'),
                    recovery=operation.get('recovery', False), root_retry=operation['kind'] == 'root_retry', remaining=self._allocation_remaining())
        if operation['kind'] == 'skill' and self.state['originals']:
            context['previous_skill_rejections'] = [copy.deepcopy(row) for row in
                self.state.get('rejected_skill_updates', {}).values()
                if row['project_ref'] == self.state['originals'][-1]['artifact_ref']]
        return context

    @property
    def independent_projects(self):
        return bool(self.config.get('independent_projects') or self.config.get('deferred_recovery'))

    def _allocation_remaining(self):
        # Independent projects own their allocation; cycle pools are telemetry.
        if self.independent_projects:
            return dict(actor_seconds=None, recovery_seconds=None, wall_seconds=None)
        return self.remaining()

    def reconcile_rejected_skill(self, result):
        """Charge observed rejection once, retaining the task and pending history.

        A subsequent normal step may create a fresh writer operation only if
        existing elapsed allocations permit it. This never creates a candidate.
        """
        if not isinstance(result, RejectedSkillUpdate):
            raise TypeError('host-proven rejected skill update required')
        row, state = asdict(result), self.state
        previous = state.get('rejected_skill_updates', {}).get(result.operation_id)
        if previous is not None:
            if previous != row:
                raise WorkflowError('rejected operation replay differs')
            return copy.deepcopy(previous)
        pending = state['pending']
        if (not pending or pending['kind'] != 'skill' or pending['id'] != result.operation_id
                or state['phase'] != 'skill' or state['namespace'] != result.namespace
                or state['skills'] != result.incumbent or state['memory'] != result.memory_ref
                or not state['originals'] or state['originals'][-1]['artifact_ref'] != result.project_ref
                or result.operation_id in state['completed_operations']):
            raise WorkflowError('rejected proposal scope differs from pending skill operation')
        state.setdefault('rejected_skill_updates', {})[result.operation_id] = row
        state['spent'] += result.actor_seconds
        state['completed_operations'][result.operation_id] = 'skill_rejected'
        state['events'].append(dict(kind='skill_proposal_rejected', **row))
        state['pending'] = None
        self.needs_reconciliation = False
        self._save()
        return copy.deepcopy(row)

    def _due(self, reason):
        if self.spec.mode == Mode.PHASE1:
            return
        count = len(self.state['originals'])
        existing = next((e for e in self.state['events'] if e['kind'] == 'tuning_due'
                         and e['original_count'] == count), None)
        if existing:
            if reason not in existing['reasons']:
                existing['reasons'].append(reason)
        else:
            self.state['events'].append(dict(kind='tuning_due', id=uuid.uuid4().hex,
                                            original_count=count, reasons=[reason]))

    @property
    def paired_retries(self):
        return self.independent_projects and self.spec.mode != Mode.PHASE1 and self.config['recover']

    def _paired_due(self):
        return next((row for row in self.state.get('paired_retry_queue', {}).values()
                     if row.get('feedback_pending') and row['eligible_wave'] <= self.state['wave']), None)

    def complete_paired_retry(self, event, result, receipt, feedback):
        """Observe the existing pair once; the candidate arm is the original retry."""
        if not self.paired_retries:
            return
        state = self.state
        completed = state.setdefault('paired_retry_receipts', {})
        if event['id'] in completed:
            if completed[event['id']] != receipt:
                raise WorkflowError('paired retry receipt changed')
            return
        arms = result['arms']
        candidate = arms['candidate']
        accepted = any(row['outcome'] == 'accepted' and row['workflow'] is True
                       and row['correctness'] is True for row in arms.values())
        harness = [row for row in arms.values() if row.get('classification') == 'harness_failure']
        if any(not row.get('terminal_failure_ref') or not row.get('audit_ref') for row in harness):
            raise ValueError('paired harness failure requires terminal proof and host audit')
        original = event['original']
        root = original['original_id']
        item = state.setdefault('paired_retry_queue', {}).setdefault(root, dict(
            original=copy.deepcopy(original), failed_attempts=0,
            original_actor_seconds=original['actor_seconds'], candidate_actor_seconds=0.))
        item['candidate_actor_seconds'] += candidate['actor_seconds']
        failed = candidate['outcome'] != 'accepted' and candidate.get('task_failure') is True and not harness
        if failed:
            item['failed_attempts'] += 1
        limit = self.config.get('max_failed_recovery_attempts', 5)
        status = ('accepted' if accepted else 'abandoned' if harness else
                  'unrecoverable' if item['failed_attempts'] >= limit else
                  'queued' if failed else 'observation_required')
        item.update(status=status, result=copy.deepcopy(feedback), feedback_pending=True,
                    eligible_wave=state['wave'] + 1, checkpoint=copy.deepcopy(feedback.get('retry_checkpoint', feedback['paired_checkpoint'])))
        state['events'].append(dict(kind='paired_retry_resolved', original_id=root,
            event_id=event['id'], receipt_ref=receipt, failed_attempts=item['failed_attempts'], status=status))
        completed[event['id']] = receipt
        if state['cursor'] >= len(self.spec.projects) and state['phase'] in ('original', 'complete'):
            self._wave_end()
        self._save()

    def _wave_end(self):
        self._due('wave_end')
        paired_pending = self.paired_retries and any(
            row.get('feedback_pending') for row in self.state.get('paired_retry_queue', {}).values())
        if (self.state['cursor'] >= len(self.spec.projects) and not paired_pending and not (self.config.get('deferred_recovery')
                and any(r.get('status') == 'queued' for r in self.state['recovery_queue']))):
            self.state['phase'] = 'complete'
        elif self.config['wave_limit'] is not None and self.state['wave'] >= self.config['wave_limit']:
            self.state['phase'] = 'wave_limit'
        else:
            self.state['wave'] += 1
            self.state['wave_results'] = []
            self.state['phase'] = 'prepare_wave' if self.config.get('deferred_recovery') else 'original'

    def _after_original(self):
        if self.config.get('deferred_recovery'):
            wave = self.state.get('active_wave') or {}
            done = {r['operation_id'] for r in self.state['wave_results']}
            if any(op['id'] not in done for op in wave.get('operations', [])):
                self.state['phase'] = 'original'
                return
            if any(r.get('prepared') for r in self._eligible()):
                self.state['phase'] = 'original'
                return
        count = len(self.state['originals'])
        if self.config['tuning_every'] and count % self.config['tuning_every'] == 0:
            self._due('cadence')
        size = self.config['wave_size']
        ended = self.state['cursor'] == len(self.spec.projects) or size and count % size == 0
        if self.spec.mode == Mode.PHASE1 and ended:
            self.state['phase'] = 'memory'
        elif ended:
            self._wave_end()
        else:
            self.state['phase'] = 'original'

    def reserve_original_wave(self, maximum):
        """Reserve authored siblings under the existing first original admission."""
        s = self.state
        if s.get('active_wave'):
            return copy.deepcopy(s['active_wave'])
        first = s['pending']
        if not first or first['kind'] not in ('original', 'recovery', 'root_retry'):
            raise WorkflowError('wave requires the first canonical original admission')
        recovery = first['kind'] == 'recovery'
        size = self.config['wave_size']
        count = min(maximum, len(self.spec.projects) - s['cursor'])
        if recovery or first['kind'] == 'root_retry':
            count = 1
        elif size:
            count = min(count, size - len(s['wave_results']))
        remaining = self._allocation_remaining()
        if count < 1 or remaining['wall_seconds'] is not None and remaining['wall_seconds'] <= 0:
            raise WorkflowError('parent elapsed allocation cannot reserve a project wave')
        candidates = []
        if self.config.get('deferred_recovery'):
            candidates.append(copy.deepcopy(first))
            if first['kind'] == 'original':
                authored = sum(r.get('kind', 'original') == 'original' for r in s['wave_results'])
                quota = (size - authored) if size else maximum
                for offset in range(1, min(quota, len(self.spec.projects) - s['cursor'])):
                    spec = self.spec.projects[s['cursor'] + offset]
                    candidates.append(dict(id=uuid.uuid4().hex, kind='original', recovery=False,
                        original_id=spec.project_id, spec=spec.to_dict()))
            for item in self._eligible():
                if not item.get('prepared') or item['original_id'] == first['original_id'] and first['kind'] != 'original':
                    continue
                candidates.append(dict(id=uuid.uuid4().hex,
                    kind='root_retry' if item['action'] == 'root_retry' else 'recovery',
                    recovery=item['action'] != 'root_retry', original_id=item['original_id'], spec=item['spec']))
        else:
            for offset in range(count):
                spec = ProjectSpec.from_dict(first['spec']) if recovery else self.spec.projects[s['cursor'] + offset]
                candidates.append(dict(id=first['id'] if not offset else uuid.uuid4().hex,
                    kind=first['kind'], recovery=recovery,
                    original_id=first['original_id'] if recovery else spec.project_id, spec=spec.to_dict()))
        operations = []
        reserved, reserved_recovery = 0., 0.
        for operation in candidates:
            spec = ProjectSpec.from_dict(operation['spec'])
            allowance = spec.budgets.actor_seconds if spec.budgets.actor_seconds is not None else 7200.
            caps = [allowance]
            if operation['kind'] == 'root_retry':
                item = next(r for r in s['recovery_queue'] if r['original_id'] == operation['original_id'])
                if self._root_remaining(item) is not None:
                    caps.append(self._root_remaining(item))
            if remaining['actor_seconds'] is not None:
                caps.append(remaining['actor_seconds'] - reserved)
            if operation['recovery'] and remaining['recovery_seconds'] is not None:
                caps.append(remaining['recovery_seconds'] - reserved_recovery)
            if self.config.get('deferred_recovery'):
                allowance = max(0., min(caps))
                if allowance == 0:
                    continue
            elif min(caps) < allowance:
                break
            reserved += allowance
            reserved_recovery += allowance if operation['recovery'] else 0.
            limits = replace(spec.budgets, actor_seconds=allowance, design_iterations=None,
                             implementation_iterations=None, execution_tries=None)
            spec = replace(spec, budgets=limits)
            operations.append(dict(operation, spec=spec.to_dict()))
        if not operations:
            raise WorkflowError('parent elapsed allocation cannot reserve a project wave')
        wave = dict(id=uuid.uuid4().hex, operations=operations, memory=s['memory'], skills=s['skills'],
                    phase='executing', wall_deadline=(s['created'] + self.config['wall_seconds']
                    if not self.independent_projects and self.config['wall_seconds'] is not None else None))
        s['pending'] = operations[0]
        s['active_wave'] = wave
        self._save()
        return copy.deepcopy(wave)

    def close_original_wave(self):
        wave = self.state.get('active_wave')
        if (not wave or self.state['pending'] or self.state['phase'] in ('memory', 'skill')
                or any(op['id'] not in self.state['completed_operations'] for op in wave['operations'])):
            return False
        self.state.setdefault('wave_history', []).append(wave)
        self.state['active_wave'] = None
        self._save()
        return True

    def step(self):
        """Dispatch once; pending failures must be reconciled, never replayed."""
        s = self.state
        if s['pending']:
            raise WorkflowError('pending operation requires reconciliation')
        if s['stopped'] or s['phase'] in ('complete', 'wave_limit'):
            return None
        if s['phase'] == 'recovery' and not s['recovery_queue']:
            self._wave_end()
            self._save()
            return None
        remaining = self._allocation_remaining()
        if self.config.get('deferred_recovery') and s['phase'] == 'prepare_wave':
            eligible = self._eligible()
            target = next((r for r in eligible if not r.get('prepared')), None)
            if target is None:
                s['phase'] = 'original'
                self._save()
                return None
            if target['action'] == 'diagnostic':
                if remaining['recovery_seconds'] == 0:
                    target['status'] = 'budget_exhausted'
                    self._save()
                    return None
                s['phase'] = 'simplify'
            else:
                s['phase'] = 'prepare_retry'
        if (self.config.get('deferred_recovery') and s['phase'] == 'original'
                and any(remaining[k] == 0 for k in ('actor_seconds', 'wall_seconds'))
                and s['wave_results']):
            s['phase'] = 'memory'
        if (self.config.get('recovery_feedback') and s['phase'] in ('simplify', 'recovery', 'recovery_memory')
                and any(value is not None and value <= 0 for value in remaining.values())):
            s['stopped'] = True
            s['events'].append(dict(kind='semantic_recovery_elapsed_exhausted', remaining=remaining))
            self._save()
            return None
        retained_original = (s['phase'] == 'original' and s.get('active_wave', {})
                             and s['active_wave']['phase'] == 'settled')
        if (not retained_original and not (self.config.get('deferred_recovery') and s['phase'] == 'memory')
                and any(remaining[k] is not None and remaining[k] <= 0
                                        for k in ('actor_seconds', 'wall_seconds'))):
            if self.config.get('deferred_recovery'):
                s['stopped'] = True
                self._save()
                return None
            raise WorkflowError('parent elapsed allocation exhausted')
        if self.paired_retries and s['phase'] == 'original' and not s.get('paired_retry_current'):
            item = self._paired_due()
            if item:
                s['paired_retry_current'] = item['original']['original_id']
                s['phase'] = 'memory'
        phase = s['phase']
        operation = dict(id=uuid.uuid4().hex, kind='memory' if phase == 'recovery_memory' else phase,
                         recovery=phase in ('recovery', 'simplify', 'recovery_memory'))
        if phase == 'recovery_memory':
            operation['feedback'] = 'recovery'
        if phase == 'original':
            wave = s.get('active_wave')
            retained = next((op for op in wave['operations']
                if op['id'] not in s['completed_operations']), None) if wave else None
            if retained:
                operation = copy.deepcopy(retained)
            else:
                authored = sum(row.get('kind', 'original') == 'original' for row in s['wave_results'])
                if (not self.config.get('deferred_recovery') or s['cursor'] < len(self.spec.projects)
                        and (not self.config['wave_size'] or authored < self.config['wave_size'])):
                    operation['spec'] = self.spec.projects[s['cursor']].to_dict()
                    operation['original_id'] = operation['spec']['project_id']
                else:
                    item = next((r for r in self._eligible() if r.get('prepared')), None)
                    if item is None:
                        if s['wave_results']:
                            s['phase'] = 'memory'
                        else:
                            self._wave_end()
                        self._save()
                        return None
                    operation.update(kind='root_retry' if item['action'] == 'root_retry' else 'recovery',
                        recovery=item['action'] != 'root_retry', spec=item['spec'], original_id=item['original_id'])
        elif phase in ('recovery', 'simplify', 'prepare_retry'):
            item = (next(r for r in self._eligible() if not r.get('prepared'))
                    if self.config.get('deferred_recovery') else s['recovery_queue'][0])
            if ((self.config['recovery_limit'] is not None and item['attempts'] >= self.config['recovery_limit'])
                    or remaining['recovery_seconds'] is not None and remaining['recovery_seconds'] <= 0):
                s['events'].append(dict(kind='recovery_exhausted', original_id=item['original_id']))
                s['recovery_queue'].pop(0)
                s['phase'] = 'simplify' if s['recovery_queue'] and self.simplify else 'recovery'
                self._save()
                return None
            operation.update(spec=item['spec'], original_id=item['original_id'])
        s['pending'] = operation
        self._save()  # Identity is durable before any external action, including hooks.
        ctx = self._context(operation)
        ctx['root_retry'] = operation['kind'] == 'root_retry'
        if phase in ('original', 'recovery', 'simplify', 'prepare_retry'):
            spec = ProjectSpec.from_dict(operation['spec'])
        if phase in ('original', 'recovery', 'simplify', 'prepare_retry') and not retained_original:
            caps = [v for v in (spec.budgets.actor_seconds, remaining['actor_seconds'],
                                remaining['recovery_seconds'] if operation['recovery'] else None) if v is not None]
            wall = [v for v in (spec.budgets.wall_seconds, remaining['wall_seconds']) if v is not None]
            spec = replace(spec, budgets=replace(spec.budgets, actor_seconds=min(caps) if caps else None,
                                                wall_seconds=min(wall) if wall else None))
        if operation['kind'] == 'root_retry':
            item = next(r for r in s['recovery_queue'] if r['original_id'] == operation['original_id'])
            cap = self._root_remaining(item)
            if cap is not None:
                spec = replace(spec, budgets=replace(spec.budgets,
                    actor_seconds=min(cap, spec.budgets.actor_seconds) if spec.budgets.actor_seconds is not None else cap))
        if self.config.get('deferred_recovery') and 'spec' in operation:
            operation['spec'] = spec.to_dict()
            self._save()
        if phase in ('original', 'recovery'):
            result = self.runner(spec, s['memory'], s['skills'], ctx)
        elif phase in ('memory', 'recovery_memory'):
            rows = s['recovery_feedback_rows'] if phase == 'recovery_memory' else (
                s['wave_results'] if self.spec.mode == Mode.PHASE1 else [
                    s['paired_retry_queue'][s['paired_retry_current']]['result']
                    if s.get('paired_retry_current') else s['originals'][-1]])
            result = self.memory_update(s['memory'], copy.deepcopy(rows), ctx)
        elif phase == 'skill':
            row = (s['paired_retry_queue'][s['paired_retry_current']]['result']
                   if s.get('paired_retry_current') else s['originals'][-1])
            result = self.skill_update(s['skills'], s['memory'], copy.deepcopy(row), ctx)
        elif phase == 'simplify':
            item = next(r for r in s['recovery_queue'] if r['original_id'] == operation['original_id'])
            result = self.simplify(spec, copy.deepcopy(item.get('root_result', item['result'])), ctx)
        elif phase == 'prepare_retry':
            item = next(r for r in s['recovery_queue'] if r['original_id'] == operation['original_id'])
            result = self.retry_original(ProjectSpec.from_dict(item['root_spec']), copy.deepcopy(item['root_result']), ctx)
        elif phase == 'isolate':
            result = self.isolate(s['memory'], s['skills'], ctx)
        else:
            raise WorkflowError('unknown scheduler phase')
        self.reconcile(operation['id'], result)
        return result

    def reconcile(self, operation_id, result=None, *, not_dispatched=False):
        s = self.state
        if operation_id in s['completed_operations']:
            raise WorkflowError('operation already reconciled')
        op = s['pending']
        if not op or op['id'] != operation_id:
            raise WorkflowError('unknown pending operation')
        if not_dispatched:
            if result is not None:
                raise ValueError('absence cannot also supply a result')
            s['pending'] = None
            self.needs_reconciliation = False
            self._save()
            return
        kind = op['kind']
        if kind in ('original', 'recovery', 'root_retry'):
            if not isinstance(result, HostTaskResult):
                raise TypeError('host task result required')
            row = dict(project=op['spec'], original_id=op['original_id'], operation_id=operation_id,
                       outcome=result.outcome, artifact_ref=result.artifact_ref, actor_seconds=result.actor_seconds)
            for key in ('classification', 'terminal_failure_ref', 'audit_ref'):
                if getattr(result, key) is not None:
                    row[key] = getattr(result, key)
            harness_failure = result.classification == 'harness_failure'
            if harness_failure:
                s['events'].append(dict(kind='project_abandoned', operation_id=operation_id,
                    original_id=op['original_id'], terminal_failure_ref=result.terminal_failure_ref,
                    audit_ref=result.audit_ref))
            if self.config.get('deferred_recovery'):
                row['kind'] = kind
            s['spent'] += result.actor_seconds
            if kind == 'original':
                s['originals'].append(row)
                s['wave_results'].append(row)
                s['cursor'] += 1
                if self.config.get('deferred_recovery') and result.outcome != 'accepted' and not harness_failure:
                    s['recovery_queue'].append(self._root_item(row))
                if self.spec.mode == Mode.PHASE1:
                    self._after_original()
                else:
                    s['phase'] = 'memory'
            elif self.config.get('deferred_recovery'):
                item = next(r for r in s['recovery_queue'] if r['original_id'] == op['original_id'])
                if kind == 'root_retry':
                    s.setdefault('root_retries', []).append(row)
                    item['root_spent'] += result.actor_seconds
                    item['root_result'] = copy.deepcopy(row)
                    item['result'] = copy.deepcopy(row)
                    if harness_failure:
                        item['status'] = 'abandoned'
                    elif result.outcome == 'accepted':
                        item['status'] = 'accepted'
                    else:
                        self._failed_attempt(item)
                    item['action'] = 'diagnostic'
                else:
                    s['recovery_spent'] += result.actor_seconds
                    s['recoveries'].append(row)
                    item['attempts'] += 1
                    if harness_failure:
                        item['status'] = 'abandoned'
                    else:
                        item['action'] = 'root_retry'
                item.update(prepared=False, spec=copy.deepcopy(item['root_spec']), eligible_wave=s['wave'] + 1)
                s['wave_results'].append(row)
                self._after_original()
            else:
                s['recovery_spent'] += result.actor_seconds
                s['recoveries'].append(row)
                item = s['recovery_queue'][0]
                item['attempts'] += 1
                item['result'] = row
                if result.outcome == 'accepted':
                    s['recovery_queue'].pop(0)
                s['phase'] = 'simplify' if s['recovery_queue'] and self.simplify else 'recovery'
                if self.config.get('recovery_feedback'):
                    s['after_recovery_memory'] = s['phase']
                    s['recovery_feedback_rows'] = [copy.deepcopy(row)]
                    s['phase'] = 'recovery_memory'
        elif kind in ('memory', 'skill'):
            if not isinstance(result, SnapshotUpdate):
                raise TypeError('retained snapshot update required')
            s['spent'] += result.actor_seconds
            if kind == 'skill':
                s['candidates'].append(dict(reference=result.reference, operation_id=operation_id))
                current = s.get('paired_retry_current')
                item = s['paired_retry_queue'][current] if current else None
                last = item['original'] if item else s['originals'][-1]
                if self.config['recover'] and last['outcome'] != 'accepted' and last.get('classification') != 'harness_failure':
                    s['events'].append(dict(kind='paired_recovery_due', id=uuid.uuid4().hex,
                                            original=copy.deepcopy(last), memory=s['memory'],
                                            incumbent=s['skills'], candidate=result.reference,
                                            **({'paired_checkpoint': item['checkpoint']} if item else {})))
                elif last['outcome'] == 'accepted':
                    s['events'].append(dict(kind='candidate_validation_due', id=uuid.uuid4().hex,
                                            original=copy.deepcopy(last), memory=s['memory'],
                                            incumbent=s['skills'], candidate=result.reference))
                if item:
                    item['feedback_pending'] = False
                    item['status'] = 'pair_pending'
                    s.pop('paired_retry_current')
                    s['phase'] = 'original'
                else:
                    self._after_original()
            else:
                s['memory'] = result.reference
                if self.config.get('deferred_recovery'):
                    self._wave_end()
                elif op.get('feedback') == 'recovery':
                    s['recovery_spent'] += result.actor_seconds
                    s['recovery_feedback_rows'] = []
                    s['phase'] = s.pop('after_recovery_memory')
                elif s.get('paired_retry_current'):
                    item = s['paired_retry_queue'][s['paired_retry_current']]
                    if item['status'] == 'queued':
                        s['phase'] = 'skill'
                    else:
                        item['feedback_pending'] = False
                        s.pop('paired_retry_current')
                        if s['cursor'] >= len(self.spec.projects) and not self._paired_due():
                            self._wave_end()
                        else:
                            s['phase'] = 'original'
                elif self.spec.mode != Mode.PHASE1:
                    s['phase'] = 'skill'
                else:
                    parents = {row['project']['project_id']: row for row in s.get('historical_failed_parents', [])}
                    parents.update({row['project']['project_id']: row for row in s['wave_results']})
                    s['recovery_queue'] = [dict(spec=r['project'], original_id=r['original_id'],
                                                result=r, attempts=0) for r in s['wave_results']
                                           if self.config['recover'] and r['outcome'] != 'accepted']
                    if self.config.get('recovery_feedback'):
                        s['recovery_queue'] = [dict(spec=r['project'], original_id=r['original_id'],
                            result=r, attempts=0) for r in parents.values()
                            if self.config['recover'] and r['outcome'] != 'accepted']
                    s['phase'] = 'simplify' if s['recovery_queue'] and self.simplify else 'recovery'
        elif kind == 'prepare_retry':
            item = next(r for r in s['recovery_queue'] if r['original_id'] == op['original_id'])
            if not isinstance(result, ProjectSpec) or result.project_id != item['root_spec']['project_id']:
                raise ValueError('original retry must retain the immutable original project identity')
            item.update(spec=result.to_dict(), prepared=True)
            s['phase'] = 'prepare_wave'
        elif kind == 'simplify':
            original = ProjectSpec.from_dict(op['spec'])
            if isinstance(result, SimplificationDeferred) and self.config.get('deferred_recovery'):
                item = next(r for r in s['recovery_queue'] if r['original_id'] == op['original_id'])
                item.update(prepared=False, eligible_wave=s['wave'] + 1)
                s['spent'] += result.actor_seconds
                s['recovery_spent'] += result.actor_seconds
                finding = copy.deepcopy(item['root_result'])
                finding.update(investigation_receipt_ref=result.receipt_ref, kind='investigation_failure')
                s['wave_results'].append(finding)
                s['events'].append(dict(kind='simplification_deferred', operation_id=operation_id,
                    receipt_ref=result.receipt_ref, actor_seconds=result.actor_seconds))
                s['phase'] = 'prepare_wave'
            elif isinstance(result, SimplificationDeferred):
                s['spent'] += result.actor_seconds
                s['recovery_spent'] += result.actor_seconds
                s['events'].append(dict(kind='simplification_deferred', operation_id=operation_id,
                    receipt_ref=result.receipt_ref, actor_seconds=result.actor_seconds))
                s['stopped'] = True
                s['completed_operations'][operation_id] = kind
                s['pending'] = None
                self.needs_reconciliation = False
                self._save()
                return
            if isinstance(result, SimplificationDeferred):
                s['completed_operations'][operation_id] = kind
                s['pending'] = None
                self.needs_reconciliation = False
                self._save()
                return
            if not isinstance(result, SimplificationResult):
                raise TypeError('host simplification result with elapsed receipt required')
            index = 0
            if result.parent_project_id is not None:
                index = next((i for i, item in enumerate(s['recovery_queue'])
                              if item.get('root_spec', item['spec'])['project_id'] == result.parent_project_id), None)
                if index is None:
                    raise ValueError('selected parent is outside the admitted failure queue')
                selected = s['recovery_queue'][index]
                original = ProjectSpec.from_dict(selected['spec'])
            revised = result.spec
            if self.config.get('recovery_feedback'):
                allowance = revised.budgets.actor_seconds
                caps = [value - result.actor_seconds for key, value in self._allocation_remaining().items()
                        if key in ('actor_seconds', 'recovery_seconds') and value is not None]
                if allowance is None or allowance <= 0 or any(allowance > value for value in caps):
                    raise ValueError('child allocation exceeds the measured remaining semantic pool')
            if revised.project_id == original.project_id:
                raise ValueError('simplification requires a new linked project identity')
            catalog = (s.get('failed_parent_admission', {}).get('rows', [])
                       + s.get('historical_failed_parents', []))
            if revised.project_id in {r['project']['project_id']
                                      for r in s['originals'] + s['recoveries'] + catalog}:
                raise ValueError('revised project identity already used')
            selected = s['recovery_queue'].pop(index)
            s['recovery_queue'].insert(0, selected)
            op['original_id'] = selected['original_id']
            s['events'].append(dict(kind='project_revised', original_id=op['original_id'],
                                    parent_id=original.project_id, revised=revised.to_dict(),
                                    operation_id=operation_id, receipt_ref=result.receipt_ref,
                                    actor_seconds=result.actor_seconds))
            s['spent'] += result.actor_seconds
            s['recovery_spent'] += result.actor_seconds
            s['recovery_queue'][0]['spec'] = revised.to_dict()
            if self.config.get('deferred_recovery'):
                s['recovery_queue'][0]['prepared'] = True
                s['phase'] = 'prepare_wave'
            else:
                s['phase'] = 'recovery'
        elif kind == 'isolate':
            if not isinstance(result, IsolationSnapshot):
                raise ValueError('host mutable-branch isolation receipt required')
            result.verify()
            if result.namespace != s['namespace']:
                raise ValueError('isolation namespace differs')
            s['isolation'] = dict(branch_root=result.branch_root, receipt_ref=result.receipt_ref)
            s['memory'], s['skills'] = result.memory, result.skills
            s['phase'] = 'original'
        s['completed_operations'][operation_id] = kind
        s['pending'] = None
        self.needs_reconciliation = False
        self._save()

    def extension_events(self):
        """Unacknowledged host extensions; tuning remains owned by PR11."""
        return copy.deepcopy([e for e in self.state['events']
            if e.get('id') and e['id'] not in self.state.get('extension_receipts', {})])

    def enqueue_tuning(self, event_id, steering):
        """Idempotent host/manual request, sharing scheduled extension accounting."""
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError('nonblank tuning request identity required')
        steering = copy.deepcopy(steering)
        if (not isinstance(steering, dict) or set(steering) - {'user_guidance', 'project_ref'}
                or any(not isinstance(v, str) for v in steering.values())):
            raise ValueError('structured tuning steering required')
        if self.spec.mode == Mode.PHASE1:
            raise WorkflowError('Phase1 cannot tune skills')
        event = dict(kind='manual_tuning', id=event_id, steering=steering)
        existing = next((e for e in self.state['events'] if e.get('id') == event_id), None)
        if existing is not None and existing != event:
            raise ValueError('tuning request identity reused')
        if existing is None:
            self.state['events'].append(event)
            self._save()
        return event_id

    def complete_extension(self, event_id, receipt_ref, *, actor_seconds=0.0, recovery=False):
        _seconds(actor_seconds)
        receipts = self.state.setdefault('extension_receipts', {})
        receipt = dict(reference=receipt_ref, actor_seconds=actor_seconds, recovery=recovery)
        if event_id in receipts:
            if receipts[event_id] != receipt:
                raise ValueError('extension receipt differs')
            return
        if not receipt_ref or not any(e.get('id') == event_id for e in self.state['events']):
            raise ValueError('retained receipt and known event required')
        receipts[event_id] = receipt
        self.state['spent'] += actor_seconds
        if recovery:
            self.state['recovery_spent'] += actor_seconds
        self._save()

    def publish_validated_skills(self, expected, reference):
        """Host coordinator calls only with Learning.active; no candidate activation."""
        if self.spec.mode == Mode.PHASE1:
            raise WorkflowError('Phase1 cannot promote skills')
        if self.state['pending'] or self.state['skills'] not in (expected, reference):
            raise WorkflowError('skill pointer changed or task running')
        if not reference:
            raise ValueError('active bundle reference required')
        self.state['skills'] = reference
        self._save()
