"""Matched transfer trials and an independent evidence-reading memory judge.

Execution callbacks must use fresh Actors, disable learning publication, and
observe retained external operations when re-entering the same arm directory.
The evaluator never admits memories or controls rewrite counts.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

from optimizer.jobs import atomic, canonical
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from .subject_execution import SubjectOutcome
from .time_budget import AttemptTimeBudget


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('elapsed allocation must be finite and positive')
    return value


def _inconclusive(reason):
    return dict(decision='inconclusive', reason=reason, evidence=[], instructions=None)


class MemoryVerifierAgent:
    """The judge can inspect actual logs; a numeric tie does not force rejection."""
    def __init__(self, complete, *, clock=time.monotonic):
        self.complete, self.clock, self.last_trace = complete, clock, []

    def review(self, proposal, arms, read_artifact, *, max_seconds):
        deadline = self.clock() + _seconds(max_seconds)
        self.last_trace = []
        known, required = set(), set()
        for arm in arms.values():
            for role, digest in arm.artifacts.items():
                known.add(digest)
                if any(word in role.lower() for word in ('log', 'result', 'execution')):
                    required.add(digest)
        read = set()
        messages = [{'role': 'system', 'content': (
            'You are an independent memory usefulness verifier. Inspect matched transfer '
            'trial metrics and retained Actor/tool logs through read_artifact. Compare '
            'control accepted memory with treatment candidate memory. Both succeeding '
            'does not prove uselessness: logs may establish more direct reasoning, better '
            'transfer, fewer mistakes or unnecessary work. Conversely, apparent numeric '
            'gains may be accidental. Cite actual observed evidence and explain why the '
            'memory caused useful behavior or needs improvement. Host correctness is '
            'authoritative: never accept a failed treatment. Return one JSON action: '
            '{"read_artifact":{"sha256":"...","offset":0,"limit":8192}} or '
            '{"decision":"accept|rewrite|reject|inconclusive","reason":"...",'
            '"evidence":["actually read artifact hashes"],"instructions":null|"concrete rewriting guidance"}. '
            'Inspect supplied result/log artifacts before a substantive judgment. '
            'A rewrite requires concrete instructions; other decisions have null instructions.')},
            {'role': 'user', 'content': json.dumps(dict(proposal=proposal,
                arms={key: value.as_dict() for key, value in arms.items()}), sort_keys=True)}]

        def remaining():
            available = deadline - self.clock()
            if available <= 0:
                raise TimeoutError('memory judge elapsed allocation exhausted')
            return available

        self.last_trace.append(dict(kind='request', messages=messages.copy()))
        while True:
            raw = self.complete(tuple(messages), remaining_seconds=remaining())
            remaining()
            self.last_trace.append(dict(kind='response', text=raw))
            messages.append(dict(role='assistant', content=raw))
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError('judge action must be an object')
                if set(value) == {'read_artifact'}:
                    request = value['read_artifact']
                    if not isinstance(request, dict) or set(request) - {'sha256', 'offset', 'limit'}:
                        raise ValueError('invalid artifact read')
                    digest = request['sha256']
                    if digest not in known:
                        raise ValueError('artifact is not retained by the matched trial')
                    data = read_artifact(digest, remaining_seconds=remaining())
                    if not isinstance(data, bytes) or _hash(data) != digest:
                        raise ValueError('artifact bytes do not match identity')
                    offset, limit = request.get('offset', 0), request.get('limit', 8192)
                    if type(offset) is not int or not 0 <= offset <= len(data) or type(limit) is not int or not 1 <= limit <= 65536:
                        raise ValueError('invalid artifact page')
                    page = data[offset:offset + limit]
                    if page or not data:
                        read.add(digest)
                    observation = dict(sha256=digest, offset=offset, size=len(data),
                        text=page.decode('utf-8', errors='replace'),
                        next_offset=offset + limit if offset + limit < len(data) else None)
                    self.last_trace.append(dict(kind='read', request=request, observation=observation))
                else:
                    if set(value) != {'decision', 'reason', 'evidence', 'instructions'} or value['decision'] not in {'accept', 'rewrite', 'reject', 'inconclusive'}:
                        raise ValueError('invalid memory judgment')
                    if not isinstance(value['reason'], str) or not value['reason'].strip():
                        raise ValueError('judgment requires an evidence-based reason')
                    evidence = value['evidence']
                    if not isinstance(evidence, list) or any(item not in read for item in evidence):
                        raise ValueError('judgment cites evidence not actually read')
                    if value['decision'] != 'inconclusive' and (required - read or not evidence):
                        raise ValueError('inspect both arms result/log artifacts before judgment')
                    if value['decision'] == 'accept' and not arms['treatment'].host_verified:
                        raise ValueError('failed treatment cannot support memory acceptance')
                    if value['decision'] == 'rewrite':
                        if not isinstance(value['instructions'], str) or not value['instructions'].strip():
                            raise ValueError('rewrite requires concrete instructions')
                    elif value['instructions'] is not None:
                        raise ValueError('only rewrites carry instructions')
                    remaining()
                    self.last_trace.append(dict(kind='decision', value=value))
                    return value
                remaining()
            except (ValueError, KeyError, TypeError) as error:
                observation = dict(error=str(error))
                self.last_trace.append(dict(kind='feedback', observation=observation))
            messages.append(dict(role='user', content=json.dumps(observation, sort_keys=True)))


class MemoryVerifier:
    def __init__(self, complete, select_trial, execute_arm, *, model_identity,
                 runtime_identity, max_seconds=1800, arm_seconds=600,
                 clock=time.monotonic, wall_clock=time.time):
        self.select_trial, self.execute_arm = select_trial, execute_arm
        self.judge = MemoryVerifierAgent(complete, clock=clock)
        self.model, self.runtime = model_identity, runtime_identity
        self.seconds, self.arm_seconds = _seconds(max_seconds), _seconds(arm_seconds)
        self.clock, self.wall_clock, self.last_trace = clock, wall_clock, []

    @contextmanager
    def _lock(self, directory):
        with (directory / '.evaluation.lock').open('a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def evaluate(self, proposal, baseline_bytes, treatment_bytes, directory, *,
                 remaining_seconds, task=None, sequence=0):
        parent_deadline = self.clock() + _seconds(remaining_seconds)
        if not isinstance(baseline_bytes, bytes) or not isinstance(treatment_bytes, bytes):
            raise TypeError('paired memories must be immutable bytes')
        if type(sequence) is not int or sequence < 0:
            raise ValueError('sequence must be nonnegative')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        identity = dict(proposal_sha256=_hash(canonical(proposal)), baseline_sha256=_hash(baseline_bytes),
            treatment_sha256=_hash(treatment_bytes), model=self.model, runtime=self.runtime,
            max_seconds=self.seconds, arm_seconds=self.arm_seconds, sequence=sequence)
        identity = json.loads(canonical(identity))
        path = directory / 'evaluation.json'
        with self._lock(directory):
            state = json.loads(path.read_bytes()) if path.exists() else dict(
                identity=identity, allocation=min(self.seconds, remaining_seconds), arms={}, trace=[], task=None)
            if state['identity'] != identity or task is not None and state['task'] is not None and state['task'] != task:
                raise ValueError('matched trial identity changed')
            if state.get('decision', {}).get('decision') in {'accept', 'rewrite', 'reject'}:
                self.last_trace = state['trace']
                return state['decision']
            # Persist allocation/identity before any budget interval or callback.
            atomic(path, canonical(state))
            store = ContentAddressedArtifacts(directory / 'artifacts')
            for role, data in (('control', baseline_bytes), ('treatment', treatment_bytes)):
                store.put_bytes(data)
            budget = AttemptTimeBudget(directory / 'elapsed.json', state['allocation'],
                                       clock=self.clock, wall_clock=self.wall_clock)

            def save():
                self.last_trace = state['trace']
                atomic(path, canonical(state))

            def remaining():
                available = min(parent_deadline, budget.deadline) - self.clock()
                if available <= 0:
                    raise TimeoutError('matched trial elapsed allocation exhausted')
                return available

            try:
                if state['task'] is None:
                    selected = task if task is not None else self.select_trial(proposal, remaining_seconds=remaining())
                    remaining()
                    if not isinstance(selected, dict) or not selected:
                        raise ValueError('select exactly one nonempty transfer task')
                    frozen = json.loads(canonical(selected))
                    allocation = min(self.arm_seconds, remaining() / 3)
                    state['task'], state['arm_allocation'] = frozen, allocation
                    state['order'] = ['control', 'treatment'] if sequence % 2 == 0 else ['treatment', 'control']
                    state['trace'].append(dict(kind='task-frozen', task=state['task'],
                        arm_seconds=state['arm_allocation'], order=state['order']))
                    save()
                for name in state['order']:
                    if state['arms'].get(name, {}).get('outcome') is not None:
                        continue
                    arm_dir = directory / name
                    arm_dir.mkdir(exist_ok=True)
                    arm_budget = AttemptTimeBudget(arm_dir / 'elapsed.json', state['arm_allocation'],
                                                  clock=self.clock, wall_clock=self.wall_clock)
                    try:
                        allowance = arm_budget.deadline - self.clock()
                        if allowance <= 0 or remaining() < allowance:
                            raise TimeoutError('insufficient remaining matched arm allocation')
                        state['trace'].append(dict(kind='arm-started', arm=name,
                            memory_sha256=_hash(treatment_bytes if name == 'treatment' else baseline_bytes),
                            remaining_seconds=allowance))
                        save()
                        outcome = self.execute_arm(json.loads(canonical(state['task'])), treatment_bytes if name == 'treatment' else baseline_bytes,
                                                   arm_dir, remaining_seconds=allowance)
                        if not isinstance(outcome, SubjectOutcome) or type(outcome.host_verified) is not bool:
                            raise TypeError('arm must return a host-owned SubjectOutcome')
                        for digest in outcome.artifacts.values():
                            store.put_bytes(outcome.read_artifact(digest))
                        status = outcome.observations
                        evidence = status.get('evidence', {})
                        statuses = [status.get('classification'), status.get('status'), status.get('actor_status')]
                        if isinstance(evidence, dict):
                            statuses += [evidence.get('classification'), evidence.get('actor_status')]
                        invalid = any(any(word in str(item).lower().replace('_', '-') for word in
                            ('infrastructure', 'provider', 'timeout', 'timed-out', 'time-budget',
                             'budget-exhausted', 'cancelled', 'interrupted', 'context-budget')) for item in statuses)
                        valid = self.clock() <= arm_budget.deadline and not invalid
                        state['arms'][name] = dict(outcome=outcome.as_dict(), valid=valid,
                            error_type=None if valid else 'InvalidArmStatus' if invalid else 'TimeoutError')
                        state['trace'].append(dict(kind='arm-completed', arm=name, **state['arms'][name]))
                    except Exception as error:
                        state['arms'][name] = dict(outcome=None, valid=False, error_type=type(error).__name__)
                        state['trace'].append(dict(kind='arm-inconclusive', arm=name, error_type=type(error).__name__))
                    finally:
                        arm_budget.close()
                        save()
                validity = {key: value.get('valid', False) for key, value in state['arms'].items()}
                if set(validity) != {'control', 'treatment'} or not all(validity.values()):
                    decision = _inconclusive('Incomplete, late or infrastructure-failed arms cannot establish memory benefit.')
                else:
                    arms = {name: SubjectOutcome(**row['outcome'], artifact_root=store.root)
                            for name, row in state['arms'].items()}
                    self.judge.last_trace = []
                    try:
                        decision = self.judge.review(proposal, arms,
                            lambda digest, **kw: store.read_bytes(digest), max_seconds=remaining())
                    finally:
                        state['trace'].extend(self.judge.last_trace)
                decision.update(evaluation_id=_hash(canonical(identity)), trial_task=state['task'], validity=validity,
                    arms={name: row['outcome'] for name, row in state['arms'].items()},
                    artifacts={name + ':' + role: digest for name, row in state['arms'].items()
                               for role, digest in (row['outcome'] or {}).get('artifacts', {}).items()})
                state['decision'] = decision
                save()
                return decision
            except Exception as error:
                decision = _inconclusive('Matched trial or judge did not complete; inspect retained evidence before resuming.')
                decision['error_type'] = type(error).__name__
                decision.update(evaluation_id=_hash(canonical(identity)), trial_task=state['task'],
                                validity={key: row.get('valid', False) for key, row in state['arms'].items()},
                                arms={name: row['outcome'] for name, row in state['arms'].items()},
                                artifacts={name + ':' + role: digest for name, row in state['arms'].items()
                                           for role, digest in (row['outcome'] or {}).get('artifacts', {}).items()})
                state['decision'] = decision
                save()
                return decision
            finally:
                budget.close()
                save()
