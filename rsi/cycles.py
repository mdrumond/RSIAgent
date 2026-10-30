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
from dataclasses import dataclass, replace
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
class SnapshotUpdate:
    reference: str
    actor_seconds: float = 0.0

    def __post_init__(self):
        if not self.reference:
            raise ValueError('retained snapshot reference required')
        _seconds(self.actor_seconds)


def _seconds(value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError('elapsed actor seconds must be finite and nonnegative')


class CycleScheduler:
    """One locked writer per state path; close() releases ownership.

    wave_limit and recovery_limit are explicit hard caps (None disables each).
    Actor time is supplied by host timing receipts, excluding setup/transfers.
    Wall time includes all elapsed time, including controller downtime.
    Inference requires isolate(memory, skills, context) -> (memory_ref, skill_ref)
    before any project runs. This hook must publish truly isolated store branches.
    simplify(spec, result, context) -> ProjectSpec is optional and Phase1-only.
    Phase2/inference recovery is emitted as deferred work for paired evaluation.
    """

    def __init__(self, path, spec, runner, memory_update, *, skill_update=None,
                 isolate=None, simplify=None, recover=True, wave_size=None,
                 wave_limit=None, recovery_limit=1, actor_seconds=None,
                 recovery_seconds=None, wall_seconds=None, tuning_every=5,
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

    def remaining(self):
        elapsed = self.clock() - self.state['created']
        if not math.isfinite(elapsed) or elapsed < 0:
            raise WorkflowError('invalid elapsed clock')
        return {name: None if self.config[name] is None else max(0, self.config[name] - spent)
                for name, spent in [('actor_seconds', self.state['spent']),
                                    ('recovery_seconds', self.state['recovery_spent']),
                                    ('wall_seconds', elapsed)]}

    def _context(self, operation):
        return dict(operation_id=operation['id'], namespace=self.state['namespace'],
                    wave=self.state['wave'], original_project_id=operation.get('original_id'),
                    recovery=operation.get('recovery', False), remaining=self.remaining())

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

    def step(self):
        """Dispatch once; pending failures must be reconciled, never replayed."""
        s = self.state
        if s['pending']:
            raise WorkflowError('pending operation requires reconciliation')
        if s['stopped'] or s['phase'] in ('complete', 'wave_limit'):
            return None
        remaining = self.remaining()
        if any(remaining[k] is not None and remaining[k] <= 0
               for k in ('actor_seconds', 'wall_seconds')):
            raise WorkflowError('parent elapsed allocation exhausted')
        phase = s['phase']
        if phase == 'recovery' and not s['recovery_queue']:
            self._wave_end()
            self._save()
            return None
        operation = dict(id=uuid.uuid4().hex, kind=phase, recovery=phase in ('recovery', 'simplify'))
        if phase == 'original':
            operation['spec'] = self.spec.projects[s['cursor']].to_dict()
            operation['original_id'] = operation['spec']['project_id']
        elif phase in ('recovery', 'simplify'):
            item = s['recovery_queue'][0]
            if ((self.config['recovery_limit'] is not None and item['attempts'] >= self.config['recovery_limit'])
                    or remaining['recovery_seconds'] is not None and remaining['recovery_seconds'] <= 0):
                s['events'].append(dict(kind='recovery_exhausted', original_id=item['original_id']))
                s['recovery_queue'].pop(0)
                s['phase'] = 'recovery'
                self._save()
                return None
            operation.update(spec=item['spec'], original_id=item['original_id'])
        s['pending'] = operation
        self._save()  # Identity is durable before any external action, including hooks.
        ctx = self._context(operation)
        if phase in ('original', 'recovery'):
            spec = ProjectSpec.from_dict(operation['spec'])
            caps = [v for v in (spec.budgets.actor_seconds, remaining['actor_seconds'],
                                remaining['recovery_seconds'] if operation['recovery'] else None) if v is not None]
            wall = [v for v in (spec.budgets.wall_seconds, remaining['wall_seconds']) if v is not None]
            spec = replace(spec, budgets=replace(spec.budgets, actor_seconds=min(caps) if caps else None,
                                                wall_seconds=min(wall) if wall else None))
            result = self.runner(spec, s['memory'], s['skills'], ctx)
        elif phase == 'memory':
            rows = s['wave_results'] if self.spec.mode == Mode.PHASE1 else [s['originals'][-1]]
            result = self.memory_update(s['memory'], copy.deepcopy(rows), ctx)
        elif phase == 'skill':
            result = self.skill_update(s['skills'], s['memory'], copy.deepcopy(s['originals'][-1]), ctx)
        elif phase == 'simplify':
            item = s['recovery_queue'][0]
            result = self.simplify(ProjectSpec.from_dict(item['spec']), copy.deepcopy(item['result']), ctx)
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
                self._after_original()
            else:
                s['memory'] = result.reference
                if self.spec.mode != Mode.PHASE1:
                    s['phase'] = 'skill'
                else:
                    s['recovery_queue'] = [dict(spec=r['project'], original_id=r['original_id'],
                                                result=r, attempts=0) for r in s['wave_results']
                                           if self.config['recover'] and r['outcome'] != 'accepted']
                    s['phase'] = 'simplify' if s['recovery_queue'] and self.simplify else 'recovery'
        elif kind == 'simplify':
            original = ProjectSpec.from_dict(op['spec'])
            if not isinstance(result, ProjectSpec) or result.project_id == original.project_id:
                raise ValueError('simplification requires a new linked project identity')
            if result.project_id in {r['project']['project_id'] for r in s['originals'] + s['recoveries']}:
                raise ValueError('revised project identity already used')
            s['events'].append(dict(kind='project_revised', original_id=op['original_id'],
                                    parent_id=original.project_id, revised=result.to_dict()))
            s['recovery_queue'][0]['spec'] = result.to_dict()
            s['phase'] = 'recovery'
        elif kind == 'isolate':
            if (not isinstance(result, tuple) or len(result) != 2 or
                    any(not isinstance(ref, str) or not ref for ref in result) or
                    result[0] == s['memory'] or result[1] == s['skills']):
                raise ValueError('isolation must return independent memory and skill references')
            s['memory'], s['skills'] = result
            s['phase'] = 'original'
        s['completed_operations'][operation_id] = kind
        s['pending'] = None
        self.needs_reconciliation = False
        self._save()
