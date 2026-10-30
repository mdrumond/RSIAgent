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
    result = {key: task[key] for key in ('task_id', 'remaining', 'skills_ref', 'candidate_ref') if key in task}
    if 'intervals' not in task:
        return {**result, **{key: task[key] for key in ('charged_seconds', 'excluded_seconds',
                'actor_union_seconds', 'accounting_state') if key in task}}
    if task.get('clock_domain') != 'controller' or not isinstance(task['intervals'], dict):
        return dict(result, accounting_state='invalid-clock-domain')
    groups = {}
    unresolved = False
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


def inspection(value):
    supplied = value.get('inspection')
    tasks = supplied.get('tasks', []) if isinstance(supplied, dict) else []
    if not tasks:
        latest = {}
        for event in value.get('events', []):
            if (isinstance(event, dict) and event.get('type') == 'workflow-accounting'
                    and isinstance(event.get('task_id'), str)):
                latest[event['task_id']] = event
        tasks = list(latest.values())
    if not tasks:
        return {'available': False, 'reason': 'No authoritative task time/skill snapshots in CLI evidence.'}
    return {'available': True, 'tasks': [task_inspection(task) for task in tasks if isinstance(task, dict)]}
