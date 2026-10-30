"""Write-ahead configuration adoption between canonical cycle operations."""
import copy
import fcntl
import json
import os

from rsi.configuration import _snapshot, validate_configuration
from rsi.contracts import BudgetLimits, CycleSpec, Mode, ProjectSpec
from rsi.python_execution import atomic_json


EDITABLE = {'prompts', 'phase1_projects', 'phase2_projects', 'project_configs', 'budgets'}


def checked(snapshot):
    validate_configuration(snapshot['payload'])
    if _snapshot(snapshot['payload']) != snapshot:
        raise ValueError('configuration revision integrity failure')
    return copy.deepcopy(snapshot)


def projects(config, mode, rows=None):
    rows = config['phase1_projects' if mode == Mode.PHASE1 else 'phase2_projects'] if rows is None else rows
    limits = BudgetLimits.from_dict(config['budgets']).to_dict()
    result = []
    from dataclasses import replace
    for value in rows:
        project = value if isinstance(value, ProjectSpec) else ProjectSpec.from_dict(value)
        caps = {key: min(v for v in (cap, limits[key]) if v is not None)
                if cap is not None or limits[key] is not None else None
                for key, cap in project.budgets.to_dict().items()}
        result.append(replace(project, budgets=BudgetLimits.from_dict(caps)))
    return tuple(result)


def recover(root, store, *, locked=False):
    """Complete only an authenticated, already-admitted local file transaction.

    No actors/hooks run. Unknown/manual state is refused rather than overwritten.
    Scheduler's existing cycle.lock serializes adoption and normal dispatch.
    """
    journal = root / 'revision.pending.json'
    if not journal.exists():
        return
    descriptor = None
    if not locked:
        descriptor = os.open(root / 'cycle.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(descriptor)
            raise
    try:
        transaction = json.loads(journal.read_text())
        record = store.read_json(transaction['reference'])
        for name, old, new in (('cycle.json', 'old_state', 'state'),
                               ('service.json', 'old_binding', 'binding')):
            current = json.loads((root / name).read_text())
            if current not in (store.read_json(record[old]), store.read_json(record[new])):
                raise ValueError('revision transaction conflicts with retained state')
        atomic_json(root / 'cycle.json', store.read_json(record['state']))
        atomic_json(root / 'service.json', store.read_json(record['binding']))
        history = root / 'revisions'
        history.mkdir(exist_ok=True)
        os.replace(journal, history / (transaction['reference'] + '.json'))
        descriptor_sync = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor_sync)
        finally:
            os.close(descriptor_sync)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def adopt(cycle, snapshot):
    snapshot = checked(snapshot)
    scheduler, state = cycle.scheduler, cycle.scheduler.state
    if state['pending'] or cycle.controller.state['pending'] or state['phase'] not in ('original', 'complete'):
        raise ValueError('configuration adoption requires an idle task/wave boundary')
    if any('result' not in op for op in cycle.learning.state['operations'].values()):
        raise ValueError('reconcile learning before configuration adoption')
    if any(e['kind'] in ('paired_recovery_due', 'candidate_validation_due', 'curriculum_requested')
           for e in scheduler.extension_events()):
        raise ValueError('drain validation/recovery before configuration adoption')
    old, new = cycle.config, snapshot['payload']
    if {k: v for k, v in old.items() if k not in EDITABLE} != {
            k: v for k, v in new.items() if k not in EDITABLE}:
        raise ValueError('baseline/provider/loop policy changes require a new cycle')
    selected = 'phase1_projects' if cycle.spec.mode == Mode.PHASE1 else 'phase2_projects'
    cursor = state['cursor']
    prior = tuple(cycle.spec.projects[:cursor])
    incoming = projects(new, cycle.spec.mode)
    old_rows = old[selected]
    if len(incoming) < cursor or [p.project_id for p in incoming[:cursor]] != [p.project_id for p in prior]:
        raise ValueError('admitted project identities/order cannot change')
    if new[selected][:cursor] != old_rows[:cursor]:
        raise ValueError('admitted project contracts cannot change')
    # Preserve actual intersected limits of historical tasks, including old defaults.
    incoming = (*prior, *incoming[cursor:])
    for spec in incoming[cursor:]:
        cycle.services.project_config(new, spec)
    if len(incoming) == cursor and state['phase'] != 'complete':
        raise ValueError('cannot remove the active wave remainder before its memory boundary')
    config_ref = cycle.store.put_json(snapshot)
    spec = CycleSpec(cycle.spec.cycle_id, cycle.spec.mode, tuple(incoming),
        cycle.spec.baseline_kdb, cycle.spec.baseline_skills, cycle.spec.skill_map_ref, config_ref)
    binding = dict(cycle.binding, spec=spec.to_dict(), config=snapshot)
    revised = copy.deepcopy(state)
    revised['config']['spec'] = spec.to_dict()
    revised['events'].append(dict(kind='configuration_adopted',
        previous=cycle.spec.configuration_ref, configuration=config_ref, cursor=cursor))
    if state['phase'] == 'complete' and len(incoming) > cursor:
        revised['phase'], revised['wave_results'] = 'original', []
        revised['wave'] += 1
    record = dict(old_state=cycle.store.put_json(state), old_binding=cycle.store.put_json(cycle.binding),
                  state=cycle.store.put_json(revised), binding=cycle.store.put_json(binding))
    reference = cycle.store.put_json(record)
    atomic_json(cycle.root / 'revision.pending.json', dict(reference=reference))
    recover(cycle.root, cycle.store, locked=True)
    cycle.binding, cycle.config, cycle.spec = binding, copy.deepcopy(new), spec
    scheduler.state, scheduler.config, scheduler.spec = revised, revised['config'], spec
    return dict(reference=reference, configuration=config_ref, revision=snapshot['revision'], cursor=cursor)
