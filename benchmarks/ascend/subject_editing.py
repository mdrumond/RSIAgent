"""Durable curriculum previews; only explicit approval changes future work."""
from contextlib import contextmanager
import difflib
import fcntl
import json
import math
import re
from pathlib import Path
import time
import uuid

from optimizer.jobs import atomic, canonical
from .subject_memory import digest
from .subjects import Subject


def validate_project(item, *, runtime=None, policy=None):
    """Validate existing executable contracts without executing model-authored code."""
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    if not isinstance(item, dict) or set(item) - {'project_id', 'spec', 'task', 'repairs'}:
        raise ValueError('invalid project fields')
    identifier = item.get('project_id')
    if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', identifier):
        raise ValueError('project requires a stable identifier')
    repairs = item.get('repairs', [])
    if not isinstance(repairs, list) or any(not isinstance(key, str) for key in repairs):
        raise ValueError('repairs must be a list of repair IDs')
    task = item.get('task') or {}
    if not isinstance(task, dict):
        raise ValueError('project task must be an object')
    if task.get('kind') == 'optimizer':
        from optimizer.cases import validate_submission
        if not isinstance(task.get('task'), dict) or not isinstance(task['task'].get('task_id'), str):
            raise ValueError('optimizer project requires a task_id and submission')
        validate_submission(task['task']['submission'])
    else:
        spec = ProjectSpec.from_mapping(item['spec'])
        if spec.project_id != identifier:
            raise ValueError('project and spec identifiers differ')
        if task:
            from .phase2_targets import TargetContract
            if task.get('kind') != 'target' or TargetContract.from_mapping(task['contract']).spec != spec:
                raise ValueError('target contract differs from project spec')
            if not isinstance(task.get('artifact_root'), str) or not task['artifact_root'].strip():
                raise ValueError('target requires a retained qualification artifact_root')
            from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
            contract = TargetContract.from_mapping(task['contract'])
            try:
                contract.verify(ContentAddressedArtifacts(Path(task['artifact_root'])),
                                contract.runtime if runtime is None else runtime, policy)
            except (OSError, KeyError, TypeError) as error:
                raise ValueError('target qualification evidence is unavailable or invalid') from error
    return item


class SubjectEdits:
    """The provider formats requests; approval compares the exact retained base.

    The caller supplies its curriculum role completion and target policy validator.
    Restart approval returns a plan; the actor-control owner starts its new attempt.
    """
    def __init__(self, root, scheduler, complete, *, validate_item=validate_project, target_policy=None):
        self.root, self.scheduler, self.complete = Path(root), scheduler, complete
        self.directory = self.root / 'curriculum-edits'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.validate_item, self.target_policy = validate_item, target_policy

    @contextmanager
    def _lock(self):
        with (self.directory / '.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _path(self, identifier):
        if not isinstance(identifier, str) or len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('invalid curriculum proposal ID')
        return self.directory / (identifier + '.json')

    def get(self, identifier):
        return json.loads(self._path(identifier).read_bytes())

    def _campaign(self):
        path = self.root / 'subject-campaign.json'
        return json.loads(path.read_bytes()) if path.exists() else {'projects': {}, 'pending': None}

    def _context(self):
        subjects, campaign = self.scheduler.state(), self._campaign()
        return subjects, campaign, digest(dict(subjects=subjects, campaign=campaign))

    def _validate(self, kind, value, state, campaign):
        if kind == 'subjects':
            if not isinstance(value, list) or not value:
                raise ValueError('subjects must be a nonempty list')
            rows = [Subject.from_mapping(row).as_dict() for row in value]
            if len({row['subject_id'] for row in rows}) != len(rows):
                raise ValueError('subject IDs must be unique')
            protected = min(len(state['subjects']), state['frontier'] + 1)
            if rows[:protected] != state['subjects'][:protected]:
                raise ValueError('preserve active and reached subjects exactly')
            for row in rows[protected:]:
                task = row['task']
                if task is not None:
                    if task.get('kind') == 'optimizer':
                        self.validate_item(dict(project_id=row['subject_id'], task=task, repairs=[]))
                    elif task.get('kind') == 'target':
                        spec = dict(task['contract']['spec'])
                        self.validate_item(dict(project_id=spec['project_id'], spec=spec, task=task, repairs=[]))
                    else:
                        raise ValueError('subject task must be target or optimizer')
            return rows
        if kind == 'restart':
            if not isinstance(value, dict) or set(value) != {'item', 'instruction'} or not isinstance(value['instruction'], str) or not value['instruction'].strip():
                raise ValueError('restart requires item and nonempty instruction')
            rows = [dict(subject_id='', item=value['item'])]
        else:
            if not isinstance(value, list) or any(not isinstance(row, dict) or set(row) != {'subject_id', 'item'} for row in value):
                raise ValueError('project queue requires subject_id/item rows')
            rows = value
        ids, known_subjects = set(), {row['subject_id'] for row in state['subjects']}
        reserved = {row['item']['project_id'] for group in state.get('claimed_projects', {}).values() for row in group}
        reserved.update((campaign.get('pending') or {}).get('projects', []))
        for row in rows:
            item = row['item']
            self.validate_item(item)
            identifier = item['project_id']
            if identifier in ids or identifier in campaign['projects'] or identifier in reserved:
                raise ValueError('project ID is duplicate or already started; use a fresh ID')
            ids.add(identifier)
            if kind != 'restart' and row['subject_id'] not in known_subjects:
                raise ValueError('unknown project subject')
            if any(key not in state['repairs'] or state['repairs'][key]['status'] != 'open'
                   for key in item.get('repairs', [])):
                raise ValueError('projects may only reference existing open repairs')
        return value

    def preview(self, kind, instructions, *, remaining_seconds, project=None):
        if kind not in {'subjects', 'projects', 'restart'}:
            raise ValueError('unknown curriculum edit')
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError('describe the requested change')
        if type(remaining_seconds) not in (int, float) or not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
            raise ValueError('elapsed allowance must be finite and positive')
        deadline = time.monotonic() + remaining_seconds
        state, campaign, base = self._context()
        before = state['subjects'] if kind == 'subjects' else state.get('project_queue', [])
        if kind == 'restart':
            before = campaign['projects'].get(project)
            if before is None:
                pending = campaign.get('pending') or {}
                plan = next((row for row in pending.get('items', []) if row['project_id'] == project), None)
                if plan is not None:
                    before = dict(plan=plan, objective=pending.get('subject', {}).get('objective', ''))
            if before is None:
                raise ValueError('unknown restart project')
        schema = {'subjects': 'a complete subject list using subject_id, objective, min_waves, task',
                  'projects': 'the complete pending queue [{"subject_id":"...","item":{project_id,spec or task,repairs}}]',
                  'restart': '{"item":{project_id,spec or task,repairs},"instruction":"reviewed actor prompt"}'}[kind]
        from benchmarks.a5kernels.phase1_curriculum_selector import _grammar
        grammar = _grammar() if self.target_policy is None else self.target_policy.grammar(_grammar())
        messages = [dict(role='system', content=(
            'You are the curriculum editor. Format the user request as existing executable subject/project contracts. '
            'Return JSON only: ' + schema + '. Preserve reached/active subjects and existing repair obligations. '
            'Use fresh project IDs. Do not claim execution or verification. Do not apply changes.')),
            dict(role='user', content=json.dumps(dict(request=instructions, before=before,
                subjects=state['subjects'], repairs=state['repairs'], project=project,
                known_projects=list(campaign['projects']), project_grammar=grammar), sort_keys=True))]
        while True:
            available = deadline - time.monotonic()
            if available <= 0:
                raise TimeoutError('curriculum editing elapsed allowance exhausted')
            raw = self.complete(messages, remaining_seconds=available)
            try:
                after = self._validate(kind, json.loads(raw), state, campaign)
                break
            except (ValueError, KeyError, TypeError) as error:
                messages.extend((dict(role='assistant', content=raw), dict(role='user', content=str(error))))
        if time.monotonic() >= deadline:
            raise TimeoutError('curriculum editing elapsed allowance exhausted')
        identifier = uuid.uuid4().hex
        pretty = lambda value: json.dumps(value, indent=2, sort_keys=True).splitlines(keepends=True)
        result = dict(id=identifier, kind=kind, project=project, base=base, scheduler_base=digest(state),
            instructions=instructions, before=before, after=after, status='proposed',
            diff=''.join(difflib.unified_diff(pretty(before), pretty(after), fromfile='before', tofile='after')))
        atomic(self._path(identifier), canonical(result))
        return result

    def approve(self, identifier):
        with self._lock():
            proposal = self.get(identifier)
            if proposal['status'] == 'approved':
                return proposal
            state, campaign, base = self._context()
            # A scheduler receipt proves apply completed before an interrupted acknowledgement.
            receipt = state.get('edits', {}).get(identifier)
            if receipt is None:
                if base != proposal['base']:
                    raise ValueError('curriculum proposal is stale; preview again')
                self._validate(proposal['kind'], proposal['after'], state, campaign)
                if proposal['kind'] != 'restart':
                    receipt = self.scheduler.edit(proposal['kind'], proposal['after'],
                        expected=proposal['scheduler_base'], proposal_id=identifier)
            proposal.update(status='approved', receipt=receipt)
            atomic(self._path(identifier), canonical(proposal))
            return proposal
