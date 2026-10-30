"""Serial heldout A/B trials through PairedRecovery and the canonical runner.

Host callers supply independently selected checkpoints and identities of ALL
training projects. Byte disjointness catches reuse/renaming; semantic independence
is a curriculum responsibility. No CPU suite is silently substituted for a device
curriculum. joint_limits applies per pair; parent elapsed allocation bounds the
whole round. No learning or input changes occur between arms or repetitions.
"""
from dataclasses import replace
import copy
import json
import math
from pathlib import Path
import time
import uuid

from rsi.contracts import BudgetLimits
from rsi.learning import Validation
from rsi.project_runner import ProjectRunner
from rsi.python_execution import atomic_json, digest
from rsi.recovery import PairedRecovery, _files
from rsi.runtime import WorkflowError


def case_identity(checkpoint):
    """Host identity excludes project name so renaming does not evade exclusion."""
    if not isinstance(checkpoint, ProjectRunner):
        raise ValueError('canonical checkpoint required')
    return dict(project_id=checkpoint.spec.project_id, fingerprint=digest(json.dumps(dict(
        source=_files(checkpoint.backend.project, checkpoint.store),
        verifier=digest(checkpoint.backend.verifier.read_bytes())), sort_keys=True).encode()))


def _bounded(*values):
    return min((v for v in values if v is not None), default=None)


class SkillEvaluation:
    @classmethod
    def create(cls, path, *, checkpoints, training, artifacts, memory_ref, kdb_ref,
               model_config, incumbent, candidate, joint_limits, remaining_parent,
               runner_factory, case_count=5, repetitions=2, steering=None, clock=time.time):
        created = clock()  # Include input verification and all pair staging in wall time.
        checkpoints, training = tuple(checkpoints), tuple(training)
        if any(type(n) is not int or n <= 0 for n in (case_count, repetitions)):
            raise ValueError('positive case count and repetitions required')
        if len(checkpoints) != case_count:
            raise ValueError('explicit independent heldout set must match case_count')
        identities = [case_identity(c) for c in checkpoints]
        if any(set(t) != {'project_id', 'fingerprint'} or not all(isinstance(v, str) and v for v in t.values())
               for t in training):
            raise ValueError('host training identities required')
        for key in ('project_id', 'fingerprint'):
            heldout = [c[key] for c in identities]
            if len(set(heldout)) != len(heldout) or set(heldout) & {t[key] for t in training}:
                raise ValueError('heldout duplicates or overlaps training')
        path = Path(path)
        if path.exists():
            raise ValueError('evaluation exists; reopen retained round')
        # Guidance is retained as structured steering, never injected into one arm.
        steering = json.loads(json.dumps(steering or {}, allow_nan=False))
        if not isinstance(steering, dict) or set(steering) - {'user_guidance', 'project_ref'}:
            raise ValueError('supported structured steering: user_guidance, project_ref')
        if any(not isinstance(v, str) for v in steering.values()):
            raise ValueError('steering values must be strings')
        count = case_count * repetitions
        actor_cap = remaining_parent.get('actor_seconds')
        for value in (actor_cap, remaining_parent.get('wall_seconds')):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise WorkflowError('parent elapsed allocation exhausted or invalid')
        limits = replace(joint_limits,
            actor_seconds=_bounded(joint_limits.actor_seconds, actor_cap / count if actor_cap is not None else None),
            wall_seconds=_bounded(joint_limits.wall_seconds, remaining_parent.get('wall_seconds') / count
                                  if remaining_parent.get('wall_seconds') is not None else None))
        BudgetLimits.from_dict(limits.to_dict())
        plan = dict(incumbent=incumbent, candidate=candidate, training=list(training), cases=identities,
            case_count=case_count, repetitions=repetitions, steering=steering,
            parent=copy.deepcopy(remaining_parent), created=created, pairs=[])
        # Persist all frozen paired inputs BEFORE the first actor dispatch.
        for i, checkpoint in enumerate(checkpoints):
            for repetition in range(repetitions):
                pair_path = path.parent / (path.stem + '-pairs') / f'{i}-{repetition}.json'
                pair = PairedRecovery.create(pair_path, checkpoint, artifacts=artifacts,
                    memory_ref=memory_ref, kdb_ref=kdb_ref, model_config=model_config,
                    incumbent=incumbent, candidate=candidate, joint_limits=limits,
                    remaining_parent={k: remaining_parent.get(k) for k in ('actor_seconds', 'wall_seconds')},
                    runner_factory=runner_factory, clock=clock)
                plan['pairs'].append(dict(path=str(pair_path.resolve()), frozen=pair.state['frozen'],
                                          case=i, repetition=repetition))
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, dict(plan=artifacts.put_json(plan), operation_id=uuid.uuid4().hex))
        return cls(path, artifacts=artifacts, runner_factory=runner_factory, clock=clock)

    def __init__(self, path, *, artifacts, runner_factory, clock=time.time):
        self.path, self.store, self.factory, self.clock = Path(path), artifacts, runner_factory, clock
        self.state = json.loads(self.path.read_text())
        self.plan = self.store.read_json(self.state['plan'])

    def pair(self, index):
        row = self.plan['pairs'][index]
        pair = PairedRecovery(row['path'], artifacts=self.store,
                              runner_factory=self.factory, clock=self.clock)
        if pair.state['frozen'] != row['frozen']:
            raise WorkflowError('retained paired inputs changed')
        wall = self.plan['parent'].get('wall_seconds')
        if wall is not None:
            # The same original absolute bound reaches the current arm's setup,
            # provider and commands. Pair owns persistence and only tightens;
            # resume/observation never grants a fresh relative allowance.
            pair.bind_parent_wall_deadline(self.plan['created'] + wall)
        return pair

    def step(self):
        result = self.result()
        if result is not None:
            return result
        wall = self.plan['parent'].get('wall_seconds')
        elapsed = self.clock() - self.plan['created']
        if not math.isfinite(elapsed) or elapsed < 0 or wall is not None and elapsed >= wall:
            raise WorkflowError('evaluation wall allocation exhausted or invalid; preserve pending arms')
        for i in range(len(self.plan['pairs'])):
            pair = self.pair(i)
            if pair.result() is None:
                pair.step()  # Unknown pending arm raises, never automatically replays.
                return self.result()

    def reconcile(self, *, not_dispatched=False):
        """Explicitly observe the SAME retained arm, never call runner.run here."""
        for i in range(len(self.plan['pairs'])):
            pair = self.pair(i)
            for arm, row in pair.state['arms'].items():
                if row['status'] == 'pending':
                    return pair.reconcile(arm, not_dispatched=not_dispatched)
        raise WorkflowError('no pending trial to observe')

    def result(self):
        pairs = [self.pair(i).result() for i in range(len(self.plan['pairs']))]
        if any(p is None for p in pairs):
            return None
        # Never trade a regression on one heldout for aggregate performance gains.
        # Acceptance on a failed checkpoint measures error avoidance/recovery;
        # performance only breaks ties after workflow and correctness.
        comparisons, eligible = [], True
        for pair in pairs:
            scores = {}
            for arm, row in pair['arms'].items():
                accepted = row['outcome'] == 'accepted'
                history = [h for h in self.store.read_json(row['workflow_receipt'])['history']
                           if h.get('evidence')]
                failures = [i for i, h in enumerate(history) if not h['passed'] and h['evidence']['stage']
                            in ('implementation', 'test', 'debug', 'optimisation')]
                recovered = row['correctness'] is True and all(any(
                    j > i and h['passed'] and h['evidence']['stage'] in ('test', 'validation')
                    for j, h in enumerate(history)) for i in failures)
                scores[arm] = (row['workflow'] is True, row['correctness'] is True,
                               recovered, row['performance'] == 'met')
                if arm == 'candidate':
                    eligible &= accepted and all(scores[arm][:3]) and row['performance'] not in ('not_measured', 'shortfall')
            comparison = ((scores['candidate'] > scores['incumbent']) -
                          (scores['candidate'] < scores['incumbent']))
            if pair['decision'] == 'inconclusive':
                # A closed provider/setup failure is not evidence against skills.
                # A genuine both-pass tie may coexist with another beneficial
                # case, but an unobserved comparator must block round promotion.
                comparison = 0
                eligible &= all(row['outcome'] == 'accepted' and all(scores[arm][:3])
                    and row['performance'] not in ('not_measured', 'shortfall')
                    for arm, row in pair['arms'].items())
            elif pair['decision'] == 'regression':
                comparison = -1
            comparisons.append(comparison)
        decision = ('regression' if any(c < 0 for c in comparisons) else
                    'benefit' if eligible and any(c > 0 for c in comparisons) else 'inconclusive')
        evidence = dict(plan=self.state['plan'], pairs=[p['evidence_ref'] for p in pairs],
            decision=decision, steering=self.plan['steering'],
            actor_seconds=sum(p['actor_seconds'] for p in pairs))
        cap = self.plan['parent'].get('actor_seconds')
        if cap is not None and evidence['actor_seconds'] > cap:
            evidence['decision'] = 'inconclusive'
        return dict(evidence, evidence_ref=self.store.put_json(evidence))

    def validation(self, incumbent, candidate, cases, operation_id):
        """Learning.validate adapter admits only this complete, beneficial round."""
        result = self.result()
        if result is None:
            raise WorkflowError('heldout trials incomplete; no validation evidence')
        if (incumbent != self.plan['incumbent'] or candidate != self.plan['candidate']
                or tuple(cases) != (result['evidence_ref'],)):
            raise ValueError('validation scope differs from frozen evaluation')
        passed = result['decision'] == 'benefit'
        return Validation(passed, passed, result['evidence_ref'])

    def promote(self, learning):
        result = self.result()
        if result is None:
            raise WorkflowError('heldout trials incomplete')
        if result['decision'] != 'benefit':
            return learning.active
        # Existing Learning's trusted validation callback must be bound to this
        # adapter (or a dispatcher resolving this round), not a model's verdict.
        return learning.evaluate(self.plan['candidate'], (result['evidence_ref'],),
                                 self.state['operation_id'] + ':validation')


class TuningController:
    """Wrap CycleController to drain tuning_due after recovery, before new tasks.

    evaluation_factory(event, operation_id, remaining) returns/reopens the SAME
    SkillEvaluation. It selects a frozen candidate and heldout set; user guidance
    is part of its structured event. Manual requests use request(steering).
    No programming loop is implemented here; step runs one paired arm at a time.
    """
    def __init__(self, path, controller, *, evaluation_factory):
        self.path, self.controller, self.factory = Path(path), controller, evaluation_factory
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(pending=None, completed={})

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, self.state)

    def request(self, steering):
        return self.controller.scheduler.enqueue_tuning(uuid.uuid4().hex, steering)

    def bind_pending(self, evaluation):
        """After explicitly observing a lost setup return, bind its retained round."""
        if self.state['pending'] is None or 'path' in self.state['pending']:
            raise WorkflowError('only an unbound pending setup can be reconciled')
        if not isinstance(evaluation, SkillEvaluation):
            raise ValueError('retained canonical evaluation required')
        self.state['pending'].update(path=str(evaluation.path.resolve()), plan=evaluation.state['plan'])
        self._save()

    def step(self):
        scheduler = self.controller.scheduler
        if scheduler.state['stopped']:
            return None
        events = scheduler.extension_events()
        if self.controller.state['pending'] or any(e['kind'] in ('paired_recovery_due', 'candidate_validation_due') for e in events):
            return self.controller.step()
        pending = self.state['pending']
        if pending is None:
            event = next((e for e in events if e['kind'] in ('tuning_due', 'manual_tuning')), None)
            if event is None:
                return self.controller.step()
            pending = self.state['pending'] = dict(event=event, operation_id=uuid.uuid4().hex)
            self._save()  # Lost factory returns require reopen, never create anew.
            evaluation = self.factory(copy.deepcopy(event), pending['operation_id'], scheduler.remaining())
            if not isinstance(evaluation, SkillEvaluation):
                raise ValueError('canonical evaluation required')
            pending.update(path=str(evaluation.path.resolve()), plan=evaluation.state['plan'])
            self._save()
        else:
            if 'path' not in pending:
                raise WorkflowError('tuning setup pending; explicitly bind retained evaluation')
            evaluation = self.factory(copy.deepcopy(pending['event']), pending['operation_id'], scheduler.remaining())
            if (str(evaluation.path.resolve()) != pending['path']
                    or evaluation.state['plan'] != pending['plan']):
                raise WorkflowError('tuning factory changed retained round')
        if evaluation.plan['steering'] != pending['event'].get('steering', {}):
            raise ValueError('evaluation dropped or changed structured steering')
        if scheduler.state['skills'] not in (evaluation.plan['incumbent'], evaluation.plan['candidate']):
            raise WorkflowError('active skills changed during frozen tuning round')
        if evaluation.result() is None:
            evaluation.step()
        result = evaluation.result()
        if result is None:
            return None
        incumbent = evaluation.plan['incumbent']
        active = evaluation.promote(self.controller.learning)
        if active == evaluation.plan['candidate']:
            scheduler.publish_validated_skills(incumbent, active)
        event = pending['event']
        scheduler.complete_extension(event['id'], result['evidence_ref'], actor_seconds=result['actor_seconds'])
        self.state['completed'][event['id']] = result['evidence_ref']
        self.state['pending'] = None
        self._save()
        return result
