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

    def __post_init__(self):
        if not self.outcome or not self.artifact_ref:
            raise ValueError('host outcome and retained result reference required')
        _seconds(self.actor_seconds)


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
    Its spec and context receive remaining parent/recovery elapsed limits.
    Phase2/inference recovery is emitted as deferred work for paired evaluation.
    """

    def __init__(self, path, spec, runner, memory_update, *, skill_update=None,
                 isolate=None, simplify=None, recover=True, wave_size=None,
                 wave_limit=None, recovery_limit=1, actor_seconds=None,
                 recovery_seconds=None, wall_seconds=None, tuning_every=5,
                 recovery_feedback=False,
                 clock=time.time):
        if not isinstance(spec, CycleSpec):
            raise TypeError('CycleSpec required')
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
        if recovery_feedback:
            config['recovery_feedback'] = True
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
        if not any(self.config[key] is not None for key in ('actor_seconds', 'recovery_seconds')):
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
        s['recovery_queue'] = copy.deepcopy(failed)
        s['cursor'] = len(self.spec.projects)
        s['phase'] = 'simplify' if failed else 'complete'
        self._save()

    def remaining(self):
        elapsed = self.clock() - self.state['created']
        if not math.isfinite(elapsed) or elapsed < 0:
            raise WorkflowError('invalid elapsed clock')
        wave = self.state.get('active_wave') or {}
        retained = sum(seconds for operation, seconds in wave.get('retained_actor_seconds', {}).items()
                       if operation not in self.state['completed_operations'])
        recovering = any(op['kind'] == 'recovery' for op in wave.get('operations', []))
        return {name: None if self.config[name] is None else max(0, self.config[name] - spent)
                for name, spent in [('actor_seconds', self.state['spent'] + retained),
                                    ('recovery_seconds', self.state['recovery_spent'] + (retained if recovering else 0)),
                                    ('wall_seconds', elapsed)]}

    def _context(self, operation):
        context = dict(operation_id=operation['id'], namespace=self.state['namespace'],
                    wave=self.state['wave'], original_project_id=operation.get('original_id'),
                    recovery=operation.get('recovery', False), remaining=self.remaining())
        if operation['kind'] == 'skill' and self.state['originals']:
            context['previous_skill_rejections'] = [copy.deepcopy(row) for row in
                self.state.get('rejected_skill_updates', {}).values()
                if row['project_ref'] == self.state['originals'][-1]['artifact_ref']]
        return context

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

    def _wave_end(self):
        self._due('wave_end')
        if self.state['cursor'] >= len(self.spec.projects):
            self.state['phase'] = 'complete'
        elif self.config['wave_limit'] is not None and self.state['wave'] >= self.config['wave_limit']:
            self.state['phase'] = 'wave_limit'
        else:
            self.state['wave'] += 1
            self.state['wave_results'] = []
            self.state['phase'] = 'original'

    def _after_original(self):
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
        if not first or first['kind'] not in ('original', 'recovery'):
            raise WorkflowError('wave requires the first canonical original admission')
        recovery = first['kind'] == 'recovery'
        size = self.config['wave_size']
        count = min(maximum, len(self.spec.projects) - s['cursor'])
        if recovery:
            count = 1
        elif size:
            count = min(count, size - len(s['wave_results']))
        remaining = self.remaining()
        if count < 1 or remaining['wall_seconds'] is not None and remaining['wall_seconds'] <= 0:
            raise WorkflowError('parent elapsed allocation cannot reserve a project wave')
        operations = []
        reserved = 0.
        caps = [value for value in (remaining['actor_seconds'],
            remaining['recovery_seconds'] if recovery else None) if value is not None]
        for offset in range(count):
            spec = ProjectSpec.from_dict(first['spec']) if recovery else self.spec.projects[s['cursor'] + offset]
            allowance = spec.budgets.actor_seconds if spec.budgets.actor_seconds is not None else 7200.
            if any(reserved + allowance > cap for cap in caps):
                break
            reserved += allowance
            limits = replace(spec.budgets, actor_seconds=allowance, design_iterations=None,
                             implementation_iterations=None, execution_tries=None)
            spec = replace(spec, budgets=limits)
            operations.append(dict(id=first['id'] if not offset else uuid.uuid4().hex,
                kind=first['kind'], recovery=recovery,
                original_id=first['original_id'] if recovery else spec.project_id, spec=spec.to_dict()))
        if not operations:
            raise WorkflowError('parent elapsed allocation cannot reserve a project wave')
        wave = dict(id=uuid.uuid4().hex, operations=operations, memory=s['memory'], skills=s['skills'],
                    phase='executing', wall_deadline=(s['created'] + self.config['wall_seconds']
                    if self.config['wall_seconds'] is not None else None))
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
        remaining = self.remaining()
        if (self.config.get('recovery_feedback') and s['phase'] in ('simplify', 'recovery', 'recovery_memory')
                and any(value is not None and value <= 0 for value in remaining.values())):
            s['stopped'] = True
            s['events'].append(dict(kind='semantic_recovery_elapsed_exhausted', remaining=remaining))
            self._save()
            return None
        retained_original = (s['phase'] == 'original' and s.get('active_wave', {})
                             and s['active_wave']['phase'] == 'settled')
        if not retained_original and any(remaining[k] is not None and remaining[k] <= 0
                                        for k in ('actor_seconds', 'wall_seconds')):
            raise WorkflowError('parent elapsed allocation exhausted')
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
                operation['spec'] = self.spec.projects[s['cursor']].to_dict()
                operation['original_id'] = operation['spec']['project_id']
        elif phase in ('recovery', 'simplify'):
            item = s['recovery_queue'][0]
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
        if phase in ('original', 'recovery', 'simplify'):
            spec = ProjectSpec.from_dict(operation['spec'])
        if phase in ('original', 'recovery', 'simplify') and not retained_original:
            caps = [v for v in (spec.budgets.actor_seconds, remaining['actor_seconds'],
                                remaining['recovery_seconds'] if operation['recovery'] else None) if v is not None]
            wall = [v for v in (spec.budgets.wall_seconds, remaining['wall_seconds']) if v is not None]
            spec = replace(spec, budgets=replace(spec.budgets, actor_seconds=min(caps) if caps else None,
                                                wall_seconds=min(wall) if wall else None))
        if phase in ('original', 'recovery'):
            result = self.runner(spec, s['memory'], s['skills'], ctx)
        elif phase in ('memory', 'recovery_memory'):
            rows = s['recovery_feedback_rows'] if phase == 'recovery_memory' else (
                s['wave_results'] if self.spec.mode == Mode.PHASE1 else [s['originals'][-1]])
            result = self.memory_update(s['memory'], copy.deepcopy(rows), ctx)
        elif phase == 'skill':
            result = self.skill_update(s['skills'], s['memory'], copy.deepcopy(s['originals'][-1]), ctx)
        elif phase == 'simplify':
            item = s['recovery_queue'][0]
            result = self.simplify(spec, copy.deepcopy(item['result']), ctx)
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
        if kind in ('original', 'recovery'):
            if not isinstance(result, HostTaskResult):
                raise TypeError('host task result required')
            row = dict(project=op['spec'], original_id=op['original_id'], operation_id=operation_id,
                       outcome=result.outcome, artifact_ref=result.artifact_ref, actor_seconds=result.actor_seconds)
            s['spent'] += result.actor_seconds
            if kind == 'original':
                s['originals'].append(row)
                s['wave_results'].append(row)
                s['cursor'] += 1
                if self.spec.mode == Mode.PHASE1:
                    self._after_original()
                else:
                    s['phase'] = 'memory'
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
                last = s['originals'][-1]
                if self.config['recover'] and last['outcome'] != 'accepted':
                    s['events'].append(dict(kind='paired_recovery_due', id=uuid.uuid4().hex,
                                            original=copy.deepcopy(last), memory=s['memory'],
                                            incumbent=s['skills'], candidate=result.reference))
                elif last['outcome'] == 'accepted':
                    s['events'].append(dict(kind='candidate_validation_due', id=uuid.uuid4().hex,
                                            original=copy.deepcopy(last), memory=s['memory'],
                                            incumbent=s['skills'], candidate=result.reference))
                self._after_original()
            else:
                s['memory'] = result.reference
                if op.get('feedback') == 'recovery':
                    s['recovery_spent'] += result.actor_seconds
                    s['recovery_feedback_rows'] = []
                    s['phase'] = s.pop('after_recovery_memory')
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
        elif kind == 'simplify':
            original = ProjectSpec.from_dict(op['spec'])
            if isinstance(result, SimplificationDeferred):
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
            if not isinstance(result, SimplificationResult):
                raise TypeError('host simplification result with elapsed receipt required')
            index = 0
            if result.parent_project_id is not None:
                index = next((i for i, item in enumerate(s['recovery_queue'])
                              if item['spec']['project_id'] == result.parent_project_id), None)
                if index is None:
                    raise ValueError('selected parent is outside the admitted failure queue')
                selected = s['recovery_queue'][index]
                original = ProjectSpec.from_dict(selected['spec'])
            revised = result.spec
            if self.config.get('recovery_feedback'):
                allowance = revised.budgets.actor_seconds
                caps = [value - result.actor_seconds for key, value in self.remaining().items()
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
