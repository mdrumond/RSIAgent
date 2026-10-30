"""CLI job adapter over the production service's single canonical controller."""
from dataclasses import asdict, is_dataclass
import json

from .runtime import WorkflowError


class JobStop(KeyboardInterrupt):
    """Cooperative local stop: unwind owned child cleanup before persisting stop."""


def run_job(instance, request, configuration, *, should_stop, emit):
    # Lazy import leaves read-only CLI operations free of provider dependencies.
    from .services import InstanceServices

    services = InstanceServices(instance)
    cycle = None
    cycle_id = request['cycle_id']
    try:
        if should_stop():
            return {'state': 'stopped', 'cycle_id': cycle_id, 'admitted': False}
        if request['action'] in ('start', 'revisit'):
            cycle = services.create_cycle(cycle_id, request['mode'],
                projects=request.get('projects'), configuration=configuration,
                recover=request.get('recover'), lineage=request.get('previous_cycle'))
        elif request['action'] == 'resume':
            cycle = services.open_cycle(cycle_id)
            if cycle.spec.mode.value != request['mode']:
                raise ValueError('retained cycle belongs to another phase')
            # Reconciliation observes retained effects. It cannot dispatch unknown work.
            if cycle.scheduler.state['pending'] or cycle.controller.state['pending']:
                try:
                    cycle.reconcile()
                except (WorkflowError, ValueError) as exc:
                    return dict(state='suspended', cycle_id=cycle_id, error=str(exc),
                                evidence=services.status(cycle_id))
            cycle.scheduler.resume()
        else:
            raise ValueError('unsupported job action')
        emit({'kind': 'cycle-admitted', 'cycle_id': cycle_id,
              'configuration_ref': cycle.spec.configuration_ref})
        while True:
            if should_stop():
                cycle.request_stop()
                return dict(state='stopped', cycle_id=cycle_id, evidence=services.status(cycle_id))
            before = services.status(cycle_id)
            if cycle.scheduler.state['stopped']:
                return dict(state='stopped', cycle_id=cycle_id, evidence=before)
            terminal = cycle.scheduler.state['phase'] in ('complete', 'wave_limit')
            extensions = cycle.scheduler.extension_events()
            if terminal and not extensions:
                return dict(state=cycle.scheduler.state['phase'], cycle_id=cycle_id, evidence=before)
            try:
                result = cycle.step()
            except WorkflowError as exc:
                return dict(state='suspended', cycle_id=cycle_id, error=str(exc),
                            evidence=services.status(cycle_id))
            after = services.status(cycle_id)
            emit({'kind': 'controller-step', 'cycle_id': cycle_id,
                  'result': asdict(result) if is_dataclass(result) else result, 'state': after})
            if json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True):
                return dict(state='blocked', cycle_id=cycle_id, evidence=after)
    except JobStop:
        if cycle is not None:
            cycle.request_stop()
        return dict(state='stopped', cycle_id=cycle_id, evidence=services.status(cycle_id))
    finally:
        if cycle is not None:
            cycle.close()
