"""Ordered, resumable subject progress; curriculum agents supply the decisions."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import re

from optimizer.jobs import atomic, canonical


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be nonempty text')
    return value


def _evidence(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('decision requires evidence references')
    return [_text(item, 'evidence reference') for item in value]


@dataclass(frozen=True)
class Subject:
    subject_id: str
    objective: str
    min_waves: int = 3
    task: dict | None = None

    def __post_init__(self):
        if not isinstance(self.subject_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', self.subject_id):
            raise ValueError('subject_id must be a stable identifier')
        _text(self.objective, 'objective')
        if type(self.min_waves) is not int or self.min_waves < 1:
            raise ValueError('min_waves must be a positive integer')
        if self.task is not None and not isinstance(self.task, dict):
            raise ValueError('task must be an object or null')
        if self.task is not None:
            object.__setattr__(self, 'task', json.loads(canonical(self.task)))

    def as_dict(self):
        return dict(subject_id=self.subject_id, objective=self.objective,
                    min_waves=self.min_waves, task=json.loads(canonical(self.task)))

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) - {'subject_id', 'objective', 'min_waves', 'task'}:
            raise ValueError('invalid subject fields')
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError('subject requires subject_id and objective') from error


def load_subjects(path):
    value = json.loads(Path(path).read_bytes())
    if isinstance(value, dict) and set(value) == {'subjects'}:
        value = value['subjects']
    if not isinstance(value, list):
        raise ValueError('subjects must be a list')
    return _subjects(value)


def _subjects(rows):
    result = tuple(row if isinstance(row, Subject) else Subject.from_mapping(row) for row in rows)
    if not result or len({row.subject_id for row in result}) != len(result):
        raise ValueError('subjects must be nonempty with unique identifiers')
    return result


class SubjectScheduler:
    """One durable state, locked across memory-review and wave writers.

    ``frontier`` tracks first-pass progression independently of ``current`` so a
    revisit returns to the outstanding subject rather than restarting the list.
    Every wave owns immutable memory bytes identified by its snapshot hash.
    """
    def __init__(self, root, subjects):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'subjects.json'
        self.subjects = _subjects(subjects)
        self.identities = [row.as_dict() for row in self.subjects]
        self.index = {row.subject_id: i for i, row in enumerate(self.subjects)}
        with self._transaction() as state:
            if not state:
                state.update(schema='ascend-subjects-v1', subjects=self.identities,
                             frontier=0, current=0, waves={}, repairs={}, reviews={}, decisions=[])
            elif state.get('schema') != 'ascend-subjects-v1' or state.get('subjects') != self.identities:
                raise ValueError('retained subject configuration changed')

    @contextmanager
    def _transaction(self):
        with (self.root / '.subjects.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_bytes()) if self.path.exists() else {}
            yield state
            atomic(self.path, canonical(state))

    def state(self):
        with self._transaction() as state:
            return json.loads(canonical(state))

    @staticmethod
    def _open_repairs(state, subject_id=None):
        return [row for row in state['repairs'].values() if row['status'] == 'open'
                and (subject_id is None or row['subject_id'] == subject_id)]

    def context(self):
        state = self.state()
        current = state['current']
        return dict(current_subject=None if current is None else self.identities[current],
                    subjects=self.identities,
                    completed_waves={row.subject_id: sum(wave['status'] == 'completed' and
                        wave['subject_id'] == row.subject_id for wave in state['waves'].values())
                        for row in self.subjects},
                    repairs=self._open_repairs(state),
                    pending_reviews=[key for key, row in state['reviews'].items() if row['status'] == 'pending'],
                    complete=self.is_complete(state))

    def is_complete(self, state=None):
        state = self.state() if state is None else state
        return (state['frontier'] == len(self.subjects) and state['current'] is None
                and not self._open_repairs(state)
                and all(row['status'] != 'pending' for row in state['reviews'].values())
                and all(row['status'] == 'completed' for row in state['waves'].values()))

    def start_wave(self, wave_id, memory_snapshot_sha256):
        _text(wave_id, 'wave_id')
        if not isinstance(memory_snapshot_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', memory_snapshot_sha256):
            raise ValueError('memory snapshot must be a SHA-256')
        with self._transaction() as state:
            if wave_id in state['waves']:
                retained = state['waves'][wave_id]
                if retained['memory_snapshot_sha256'] != memory_snapshot_sha256:
                    raise ValueError('wave memory snapshot changed')
                return dict(retained)
            if state['current'] is None:
                raise ValueError('no current subject; revisit an earlier subject to schedule more work')
            row = dict(wave_id=wave_id, subject_id=self.subjects[state['current']].subject_id,
                       memory_snapshot_sha256=memory_snapshot_sha256, status='running', review_ids=[])
            state['waves'][wave_id] = row
            return dict(row)

    def complete_wave(self, wave_id, review_ids=()):
        ids = list(review_ids)
        if len(set(ids)) != len(ids):
            raise ValueError('review IDs must be unique')
        for key in ids:
            _text(key, 'review ID')
        with self._transaction() as state:
            row = state['waves'][wave_id]
            if row['status'] == 'completed':
                if row['review_ids'] != ids:
                    raise ValueError('completed wave reviews changed')
                return dict(row)
            for key in ids:
                if key in state['reviews']:
                    raise ValueError('review already belongs to another wave')
            for key in ids:
                state['reviews'][key] = dict(wave_id=wave_id, subject_id=row['subject_id'], status='pending')
            row.update(status='completed', review_ids=ids)
            return dict(row)

    def resolve_review(self, review_id, outcome):
        if outcome not in {'accepted', 'rejected'}:
            raise ValueError('review outcome must be accepted or rejected')
        with self._transaction() as state:
            row = state['reviews'][review_id]
            if row['status'] not in {'pending', outcome}:
                raise ValueError('review already has a different outcome')
            row['status'] = outcome

    def add_repair(self, repair_id, subject_id, project_id, reason, evidence):
        if subject_id not in self.index:
            raise ValueError('unknown repair subject')
        row = dict(repair_id=_text(repair_id, 'repair_id'), subject_id=subject_id,
                   project_id=_text(project_id, 'project_id'), reason=_text(reason, 'reason'),
                   evidence=_evidence(evidence), status='open')
        with self._transaction() as state:
            old = state['repairs'].get(repair_id)
            if old is not None:
                if any(old[key] != value for key, value in row.items() if key != 'status'):
                    raise ValueError('repair identity reused with different evidence')
                return dict(old)
            state['repairs'][repair_id] = row
            return dict(row)

    def resolve_repair(self, repair_id, evidence):
        references = _evidence(evidence)
        with self._transaction() as state:
            row = state['repairs'][repair_id]
            if row['status'] == 'resolved' and row['resolution_evidence'] != references:
                raise ValueError('repair resolution evidence changed')
            row.update(status='resolved', resolution_evidence=references)

    def decide(self, decision):
        """Apply an agent decision; invalid decisions leave retained state intact."""
        action = decision.get('action')
        reason = _text(decision.get('reason'), 'reason')
        evidence = _evidence(decision.get('evidence'))
        if action not in {'continue', 'advance', 'revisit', 'repair', 'complete'}:
            raise ValueError('unknown subject decision')
        with self._transaction() as state:
            target = state['current']
            if action == 'complete':
                if not self.is_complete(state):
                    raise ValueError('completion requires all subjects and reviews settled without open repairs')
            elif action in {'revisit', 'repair'}:
                subject_id = decision.get('subject_id')
                if action == 'repair':
                    repair = state['repairs'].get(decision.get('repair_id'))
                    if repair is None or repair['status'] != 'open':
                        raise ValueError('repair decision requires an open repair')
                    subject_id = repair['subject_id']
                if subject_id not in self.index or self.index[subject_id] > state['frontier']:
                    raise ValueError('can only revisit previously reached subjects')
                target = self.index[subject_id]
            elif action == 'advance':
                if target is None:
                    raise ValueError('no current subject to advance')
                subject = self.subjects[target]
                waves = [row for row in state['waves'].values() if row['subject_id'] == subject.subject_id]
                if sum(row['status'] == 'completed' for row in waves) < subject.min_waves:
                    raise ValueError('subject minimum waves not complete')
                if any(row['status'] != 'completed' for row in waves) or self._open_repairs(state, subject.subject_id):
                    raise ValueError('subject has running waves or open repairs')
                if target == state['frontier']:
                    state['frontier'] += 1
                target = state['frontier'] if state['frontier'] < len(self.subjects) else None
            elif target is None:
                raise ValueError('no current subject to continue')
            state['current'] = target
            state['decisions'].append(dict(decision, reason=reason, evidence=evidence))
            return json.loads(canonical(state))


class SubjectCurriculumAgent:
    """Small provider-neutral decision boundary; execution remains caller owned."""
    def __init__(self, complete):
        self.complete = complete

    def decide(self, scheduler, *, memory_context, remaining_seconds):
        if remaining_seconds <= 0:
            raise TimeoutError('subject curriculum elapsed allowance exhausted')
        context = dict(scheduler.context(), memory=memory_context, remaining_seconds=remaining_seconds)
        raw = self.complete(({'role': 'system', 'content': (
            'Choose the next curriculum action using accepted memory and retained evidence. '
            'Return JSON {"action":"continue|advance|revisit|repair|complete","reason":"...",'
            '"evidence":["artifact or memory reference"],"subject_id":"optional",'
            '"repair_id":"optional"}. Run the subject minimum waves before advancing; '
            'repair incomplete or accidentally successful projects; revisit a subject when '
            'memory lacks useful coverage. Pending reviews permit provisional advancement. '
            'For a repair, name its existing repair_id; for a revisit name subject_id. '
            'After late memory rejection, reassess the gap: revisit a subject or choose complete '
            'when all subject minima/reviews/repairs are settled and more work is unnecessary.')} ,
            {'role': 'user', 'content': json.dumps(context, sort_keys=True)}),
            remaining_seconds=remaining_seconds)
        decision = json.loads(raw)
        if not isinstance(decision, dict):
            raise ValueError('subject decision must be an object')
        return scheduler.decide(decision)
