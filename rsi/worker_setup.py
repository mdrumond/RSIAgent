"""Controller-owned preRunner boundaries; never workflow results or actor grants."""
import json
import time

from rsi.jobs import _alive
from rsi.python_execution import atomic_json


def checkpoint(request, phase, binding_interval=None):
    worker = json.loads((request.root.parent / 'worker.json').read_text())
    atomic_json(request.root.parent / ('worker-setup-' + request.dispatch_id + '.json'),
        dict(dispatch_id=request.dispatch_id, process=worker['process'], phase=phase,
             observed_at=time.time(), binding_interval=binding_interval))


def binding_intervals(branch):
    return [row['binding_interval'] for path in branch.glob('worker-setup-*.json')
            if (row := json.loads(path.read_text())).get('phase') == 'local-setup'
            and row.get('binding_interval') is not None]


def close_pre_runner(branch, dispatch, admission, *, setup_only=False, binding_required=False):
    """Derive closure from factory ordering, not manual no-effects assertions.

    The shipped local binder only authenticates/copies inputs. Provider creation
    and commands are unreachable until Runner.create has created its root.
    A partial runner or an unknown binder keeps its original reservation even
    after process loss. New local-only setup checkpoints also cover CPU tests.
    """
    if binding_required and not setup_only:
        return None
    worker = json.loads((branch / 'worker.json').read_text())
    process = worker.get('process')
    if (worker['dispatch_id'] != dispatch or not process or _alive(process)
            or admission['pid'] != worker['pid'] or admission['dispatch_id'] != dispatch
            or (branch / 'runner').exists() or (branch / 'wave-result.json').exists()):
        return None
    path = branch / ('worker-setup-' + dispatch + '.json')
    row = json.loads(path.read_text()) if path.exists() else None
    if row and (row['dispatch_id'] != dispatch or row['process'] != process):
        return None
    if not row or row['phase'] != 'local-setup':
        raw = admission['receipt'].get('raw_measurements', {})
        submitted, terminal = raw.get('submission', {}), raw.get('terminal', {})
        if (not setup_only or not submitted.get('handle')
                or submitted['handle'] != terminal.get('handle')
                or terminal.get('state') != 'completed' or terminal.get('exit') != 0):
            return None
    closed = dict(dispatch_id=dispatch, process=process, closed=True,
        boundary=row, setup_only_binding=setup_only,
        admission_ended=admission['ended'], setup_closed_observed_at=time.time(),
        setup_elapsed=None, setup_actor_billable=False,
        uncertainty='setup end/elapsed unknown after process loss',
        observation='closed preRunner setup; canonical actor/provider and commands unreachable')
    atomic_json(branch / ('pre-runner-closure-' + dispatch + '.json'), closed)
    return closed
