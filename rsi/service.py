"""Pinned job adapter. All task and recovery policy stays in existing controllers.

run_job(instance, request, configuration=None, *, should_stop, emit) -> JSON dict.
The optional third positional argument is the dispatcher's frozen snapshot;
otherwise request.configuration supplies it. No credentials belong in request.
"""
import os
import signal
import threading

from rsi.services import InstanceServices


def run_job(instance, request, configuration=None, *, should_stop, emit):
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('owned job service must run on the worker main thread')
    action = request['action']
    if action not in ('start', 'resume', 'revisit', 'adopt'):
        raise ValueError('unknown job action')
    snapshot = configuration if configuration is not None else request.get('configuration')
    services = InstanceServices(instance)
    cycle_id = request['cycle_id']
    if should_stop():
        return dict(status='stopped', cycle_id=cycle_id, dispatched=False)
    if action in ('start', 'revisit'):
        if snapshot is None:
            raise ValueError('dispatcher frozen configuration required')
        cycle = services.create_cycle(cycle_id, request['mode'],
            projects=request.get('projects'), configuration=snapshot, recover=request.get('recover'))
    else:
        cycle = services.open_cycle(cycle_id)
    done = threading.Event()
    monitor = None
    previous = None
    dispatching = False
    try:
        if action == 'adopt':
            if snapshot is None:
                raise ValueError('explicit frozen adoption revision required')
            result = cycle.adopt_revision(snapshot)
            emit(dict(event='configuration_adopted', cycle_id=cycle_id, **result))
            return dict(status='adopted', cycle_id=cycle_id, **result)
        if action == 'resume':
            # Retained work is observed first. No pending provider/host work is replayed.
            if cycle.scheduler.state['pending'] or cycle.controller.state['pending']:
                try:
                    cycle.reconcile()
                except (ValueError, RuntimeError) as exc:
                    return dict(status='suspended', cycle_id=cycle_id, error=str(exc),
                                pending=cycle.scheduler.state['pending'])
            cycle.scheduler.resume()
        def interrupt(*_args):
            if dispatching:
                raise KeyboardInterrupt('owned job stop requested')
        previous = signal.signal(signal.SIGINT, interrupt)
        def watch():
            while not done.wait(.05):
                if should_stop():
                    # Signal only this owned worker; canonical finally blocks own children.
                    os.kill(os.getpid(), signal.SIGINT)
                    return
        monitor = threading.Thread(target=watch, name='rsi-owned-stop', daemon=True)
        monitor.start()
        while not should_stop():
            dispatching = True
            try:
                cycle.step()  # Existing controllers are the sole policy drivers.
            finally:
                dispatching = False
            state = cycle.scheduler.state
            emit(dict(event='cycle_observed', cycle_id=cycle_id, phase=state['phase'],
                      cursor=state['cursor'], wave=state['wave'], spent=state['spent']))
            if state['pending'] or cycle.controller.state['pending']:
                return dict(status='suspended', cycle_id=cycle_id, pending=state['pending'])
            if state['phase'] in ('complete', 'wave_limit'):
                extensions = cycle.scheduler.extension_events()
                actionable = [e for e in extensions if e['kind'] in (
                    'paired_recovery_due', 'candidate_validation_due')]
                if actionable:
                    continue
                return dict(status='awaiting_tuning' if any(e['kind'] == 'tuning_due'
                            for e in extensions) else state['phase'], cycle_id=cycle_id,
                            remaining=cycle.scheduler.remaining(), events=extensions)
        cycle.scheduler.stop()
        return dict(status='stopped', cycle_id=cycle_id, pending=cycle.scheduler.state['pending'])
    except (ValueError, RuntimeError) as exc:
        if cycle.scheduler.state['pending'] or cycle.controller.state['pending']:
            emit(dict(event='cycle_suspended', cycle_id=cycle_id, error=str(exc)))
            return dict(status='suspended', cycle_id=cycle_id, error=str(exc),
                        pending=cycle.scheduler.state['pending'],
                        controller_pending=cycle.controller.state['pending'])
        raise
    except KeyboardInterrupt:
        # ProjectRunner/PythonExecution/IsolatedRoles have unwound and reaped first.
        cycle.scheduler.stop()
        emit(dict(event='cycle_stopped', cycle_id=cycle_id,
                  pending=cycle.scheduler.state['pending'], controller=cycle.controller.state['pending']))
        return dict(status='stopped', cycle_id=cycle_id, pending=cycle.scheduler.state['pending'],
                    controller_pending=cycle.controller.state['pending'])
    finally:
        done.set()
        if monitor is not None:
            monitor.join()
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
        cycle.close()
