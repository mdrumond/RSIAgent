"""Read-only timing projection; never infer time from provider or remote epochs."""
import math
from rsi.runtime import ActorClock


def _union(intervals):
    total, boundary = 0.0, float('-inf')
    for start, end in sorted(intervals):
        total += max(0, end - max(start, boundary))
        boundary = max(boundary, end)
    return total


def task_inspection(task):
    result = {key: task[key] for key in ('task_id', 'remaining', 'actor_allocation', 'skills_ref', 'candidate_ref', 'supervision') if key in task}
    if 'actor_union_seconds' in task or 'intervals' not in task:
        return {**result, **{key: task[key] for key in ('charged_seconds', 'excluded_seconds',
                'actor_union_seconds', 'accounting_state') if key in task}}
    if task.get('clock_domain') != 'controller' or not isinstance(task['intervals'], dict):
        return dict(result, accounting_state='invalid-clock-domain')
    groups = {}
    unresolved = bool(task.get('timing_uncertainties'))
    for interval in task['intervals'].values():
        if not isinstance(interval, list) or len(interval) != 3:
            return dict(result, accounting_state='invalid-interval')
        category, start, end = interval
        if (not isinstance(category, str) or category not in ActorClock.BILLABLE | ActorClock.EXCLUDED
                or type(start) not in (int, float) or not math.isfinite(start)
                or end is not None and (type(end) not in (int, float)
                    or not math.isfinite(end) or end < start)):
            return dict(result, accounting_state='invalid-interval')
        if end is None:
            unresolved = True
        else:
            groups.setdefault(category, []).append((start, end))
    return dict(result, accounting_state='unresolved' if unresolved else 'reconciled',
        charged_seconds={k: _union(v) for k, v in groups.items() if k in ActorClock.BILLABLE},
        excluded_seconds={k: _union(v) for k, v in groups.items() if k in ActorClock.EXCLUDED},
        actor_union_seconds=None if unresolved else _union(
            [pair for k, v in groups.items() if k in ActorClock.BILLABLE for pair in v]))


def inspection(value, *, instance=None):
    supplied = value.get('inspection')
    tasks = supplied.get('tasks', []) if isinstance(supplied, dict) else []
    projects = [project for wave in supplied.get('waves', []) for project in wave['projects']] if isinstance(supplied, dict) else []
    legacy = (not supplied or any('actor_allocation' not in task for task in tasks)
              or any('budget_source' not in project for project in projects))
    if legacy and instance is not None:
        from pathlib import Path
        from rsi.job_inspection import inspect_cycle
        cycle_id = value.get('job', {}).get('request', {}).get('cycle_id')
        if cycle_id:
            root = (Path(instance).resolve() / 'runs').resolve()
            cycle = (root / cycle_id).resolve()
            if cycle.parent != root:
                raise ValueError('cycle identity escapes instance runs')
            if (cycle / 'cycle.json').is_file():
                # Old pinned CLI remains the command authority. Only retained
                # read-only accounting is projected with the current inspector.
                supplied = inspect_cycle(instance, {'cycle_id': cycle_id})
                value = dict(value, inspection=supplied)
    # Old evidence without local workflow access cannot prove an allocation.
    # Keep its scheduler ceiling distinct instead of presenting it as effective.
    if isinstance(supplied, dict):
        waves = []
        for wave in supplied.get('waves', []):
            rows = []
            for project in wave['projects']:
                row = dict(project)
                if 'budget_source' not in row and (
                        'actor_allocation' in row or 'remaining_actor_seconds' in row):
                    if 'actor_allocation' in row:
                        row['wave_actor_ceiling'] = row.pop('actor_allocation')
                    row.pop('remaining_actor_seconds', None)
                    row['budget_source'] = 'unavailable'
                rows.append(row)
            waves.append(dict(wave, projects=rows))
        value = dict(value, inspection=dict(supplied, waves=waves))

    supplied = value.get('inspection')
    waves = supplied.get('waves', []) if isinstance(supplied, dict) else []
    roles = supplied.get('role_supervision', []) if isinstance(supplied, dict) else []
    tasks = supplied.get('tasks', []) if isinstance(supplied, dict) else []
    if not tasks:
        latest = {}
        for event in value.get('events', []):
            if (isinstance(event, dict) and event.get('type') == 'workflow-accounting'
                    and isinstance(event.get('task_id'), str)):
                latest[event['task_id']] = event
        tasks = list(latest.values())
    if not tasks:
        if waves or roles:
            return {'available': True, 'tasks': [], 'waves': waves, 'role_supervision': roles}
        return {'available': False, 'reason': 'No authoritative task time/skill snapshots in CLI evidence.'}
    return {'available': True, 'tasks': [task_inspection(task) for task in tasks if isinstance(task, dict)],
            'waves': waves, 'role_supervision': roles}
