"""Current subject composition with retained CPL execution and environment admission."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import time

from coding.pi.rpc import PiCancelled

from .cpl_native import CplNativeCandidateBackend, CplNativeMeasurements
from .execution import _deadline, process_deadline
from .actor_time import current_budget, preparation, preparation_active, prepare_until_ready, current_worker
from .phase1_cli import load_native_config
from .subject_runtime import compose_production
from .wave_environment import NativeWaveEnvironmentCheck
from .environment_admission import EnvironmentAdmission, native_reservation


def _allocation_error(message, remaining):
    error = TimeoutError(message)
    error.interruption_reason = 'operator-stop' if getattr(remaining, 'stop_requested', lambda: False)() else 'deadline-exhausted'
    return error


class _CheckedExecutor:
    def __init__(self, executor, check):
        self._executor, self._check = executor, check

    def __getattr__(self, name):
        return getattr(self._executor, name)

    def bind_case(self, *args, **kwargs):
        return _CheckedExecutor(self._executor.bind_case(*args, **kwargs), self._check)

    def _call(self, name, *args, **kwargs):
        with self._check.reserve():
            self._check()
            # A candidate failure does not establish a broken environment.
            # Fresh health remains mandatory at the next dispatch.
            return getattr(self._executor, name)(*args, **kwargs)

    def execute(self, *args, **kwargs):
        return self._call('execute', *args, **kwargs)

    def compile(self, *args, **kwargs):
        return self._call('compile', *args, **kwargs)


class _CheckedActor:
    def __init__(self, actor, check, remaining):
        object.__setattr__(self, '_wrapped', actor)
        object.__setattr__(self, '_check', check)
        object.__setattr__(self, '_remaining', remaining)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        setattr(self._wrapped, name, value)

    def __call__(self, request):
        seconds = self._remaining()
        if request.remaining_seconds is not None:
            seconds = min(seconds, request.remaining_seconds)
        if seconds <= 0:
            raise _allocation_error('native project elapsed allocation exhausted before dispatch', self._remaining)
        self._check()
        remaining = min(seconds, self._remaining())
        try:
            return self._wrapped(replace(request, remaining_seconds=remaining))
        except Exception as error:
            if (isinstance(error, (PiCancelled, TimeoutError)) and self._remaining.stop_requested()
                    and not getattr(error, 'interruption_reason', None)):
                error.interruption_reason = 'operator-stop'
            raise


def compose_cpl_production(settings, *, target, root, initial_memory, max_seconds):
    """Supply to ``build_subject_campaign(compose=...)`` for native A3 runs.

    The ignored native config supplies transport_target, runtime and optional
    registry_workspace. The usual phase1_arguments still select model, KDB,
    guide, environment file and per-arm workspace/device overrides. stop_file
    belongs to this campaign only, never to a historical producer directory.
    """
    if target != 'a3':
        raise ValueError('CPL retrospective composition requires the native A3 binding')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker-config')
    options, _ = parser.parse_known_args(settings['phase1_arguments'])
    native = load_native_config(options.config, options.worker_config)
    if native.get('execution') == 'local':
        raise ValueError('CPL composition requires a registered remote runtime')
    for field in ('transport_target', 'runtime'):
        if not native.get(field):
            raise ValueError('native CPL configuration requires ' + field)
    stop = Path(settings['stop_file']) if settings.get('stop_file') else None
    transports = []
    def remaining():
        if stop is not None and stop.exists():
            return 0
        budget, worker = current_budget(), current_worker()
        if budget is not None and (worker.remaining() if worker is not None else budget.remaining()) <= 0:
            return 0
        if preparation_active():
            return float('inf')
        return max(0, min(budget.remaining() if budget else max_seconds,
                          _deadline.get() - time.monotonic()))
    remaining.stop_requested = lambda: stop is not None and stop.exists()
    def transport_factory(**kwargs):
        kwargs.pop('validation_wrapper', None)
        allocation = {key: native.get(key) for key in ('transport_target', 'runtime', 'registry_workspace')}
        allocation.update({key: str(kwargs.get(key)) for key in ('remote_workspace', 'physical_device', 'profile')})
        namespace = hashlib.sha256(json.dumps(allocation, sort_keys=True).encode()).hexdigest()
        recovery_root = Path(kwargs['state_directory'])
        kwargs['state_directory'] = recovery_root / 'allocations' / namespace
        transport = CplNativeCandidateBackend(**kwargs, target=native['transport_target'],
            runtime=native['runtime'], registry_workspace=native.get('registry_workspace'),
            remaining_seconds=remaining)
        transport.recovery_root = recovery_root
        transports.append(transport)
        return transport
    def measurement_factory(**kwargs):
        for key in ('validation_wrapper', 'cpl_remote', 'profile', 'remote_candidate_directory', 'physical_device'):
            kwargs.pop(key, None)
        return CplNativeMeasurements(**kwargs)
    bound = compose_production(settings, target=target, root=root,
        initial_memory=initial_memory, max_seconds=max_seconds,
        native_options={'transport_factory': transport_factory, 'measurement_factory': measurement_factory})
    check = NativeWaveEnvironmentCheck(bound['paths'], transports)
    cache = settings.get('environment_check_cache') or ((stop.parent if stop else Path(root)) / 'environment-check-cache')
    operation_root = Path(settings.get('native_operation_root') or root).resolve()
    admission = EnvironmentAdmission(check, transports, bound['paths'].root, cache=cache, pending_root=operation_root,
        dependencies={key: native.get(key) for key in ('target','arch','compiler_version','environment_identity','allowed_devices','health_workspace')})
    def ensure_environment(*, force=False):
        if remaining() <= 0:
            raise _allocation_error('native environment allocation exhausted or campaign stopped', remaining)
        with preparation('device health, runtime admission and known-good probe'):
            with native_reservation(operation_root, remaining):
                prepare_until_ready(lambda: admission(remaining, force=force),
                    stopped=remaining.stop_requested)
    ensure_environment.invalidate = admission.invalidate
    ensure_environment.reserve = lambda: native_reservation(operation_root, remaining)
    actor = bound['actor']
    if hasattr(actor, 'stopped'):
        actor.stopped = remaining.stop_requested
    actor.executor = _CheckedExecutor(actor.executor, ensure_environment)
    def checked_measurement(callback):
        def measure(*args, **kwargs):
            with ensure_environment.reserve():
                ensure_environment()
                return callback(*args, **kwargs)
        return measure
    for name in ('request_timing', 'request_profile'):
        callback = getattr(actor, name, None)
        if callback is not None:
            setattr(actor, name, checked_measurement(callback))
    bound['actor'] = _CheckedActor(actor, ensure_environment, remaining)
    bound.update(check_environment=ensure_environment, transports=tuple(transports),
                 environment_check_cache=str(Path(cache).resolve()),
                 native_operation_root=str(operation_root))
    return bound
