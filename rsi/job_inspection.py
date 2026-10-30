"""Read authoritative canonical accounting without resuming or changing a task."""
import json
import os
from pathlib import Path
import time

from core.artifacts import ContentAddressedArtifacts, canonical_bytes
from hashlib import sha256
from .runtime import ActorClock, WorkflowError
from .skills import SkillBundle


def inspect_task(path, *, clock=time.time):
    path = Path(path)
    state = json.loads((path / 'workflow.json').read_text())
    now = clock()
    intervals = state['intervals']
    accounting = ActorClock(intervals, lambda: None, lambda: now,
                            state.get('timing_uncertainties', {}), state.get('timing_receipts', {}))
    union, remaining = None, {}
    condition = 'uncertain' if accounting.uncertainties else 'active' if accounting.unresolved else 'complete'
    try:
        union = accounting.seconds()
    except WorkflowError:
        condition = 'uncertain' if not accounting.unresolved else condition
    for kind in ('actor_seconds', 'wall_seconds'):
        cap = state['limits'][kind]
        if cap is None:
            remaining[kind] = None  # Only disabled caps are explicitly unlimited.
        elif kind == 'actor_seconds' and union is not None:
            remaining[kind] = max(0, cap - union)
        elif kind == 'wall_seconds' and now >= state['created']:
            remaining[kind] = max(0, cap - (now - state['created']))
    def category(kind):
        if accounting.uncertainties and kind in ActorClock.BILLABLE:
            return None
        records = {k: v for k, v in intervals.items() if v[0] == kind}
        try:
            return ActorClock(records, lambda: None, lambda: now).seconds(kind)
        except WorkflowError:
            return None
    result = dict(task_id=path.name, charged_seconds={k: category(k) for k in sorted(ActorClock.BILLABLE)},
        excluded_seconds={k: category(k) for k in sorted(ActorClock.EXCLUDED)},
        actor_union_seconds=union, remaining=remaining, accounting_state=condition,
        intervals=intervals, clock_domain='controller', timing_uncertainties=state.get('timing_uncertainties', {}),
        timing_receipts=state.get('timing_receipts', {}),
        source_hash=state['source'], stage=state['stage'])
    manifest = path / 'runner.json'
    if manifest.exists():
        store = ContentAddressedArtifacts(path / 'artifacts')
        contract = store.read_json(json.loads(manifest.read_text())['contract'])
        bundle = SkillBundle.from_dict(contract['skills'])
        result['skills_ref'] = sha256(canonical_bytes(bundle.to_dict())).hexdigest()
    return result


def inspect_cycle(instance, request):
    if not request.get('cycle_id'):
        return {'tasks': []}
    root = Path(instance).resolve() / 'runs'
    cycle = (root / request['cycle_id']).resolve()
    if cycle.parent != root:
        raise ValueError('cycle identity escapes instance runs')
    tasks = []
    for folder, directories, filenames in os.walk(cycle):
        directories[:] = sorted(d for d in directories if d not in {
            'project', 'inputs', 'artifacts', 'cache', 'frozen', 'node_modules', '.git', 'memory'})
        if not {'workflow.json', 'runner.json'} <= set(filenames):
            continue
        task = inspect_task(Path(folder))
        task['task_id'] = str(Path(folder).relative_to(cycle))
        tasks.append(task)
    result = {'tasks': tasks}
    for name in ('cycle.json', 'learning.json'):
        path = cycle / name
        if path.exists():
            state = json.loads(path.read_text())
            if name == 'cycle.json':
                result['skills_ref'] = state.get('skills')
                result['candidate_refs'] = [row['reference'] for row in state.get('candidates', [])]
            else:
                result['active_skills_ref'] = state.get('active')
    for task in tasks:
        if task.get('skills_ref') in result.get('candidate_refs', []):
            task['candidate_ref'] = task['skills_ref']
    return result
