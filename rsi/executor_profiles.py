"""Host-only worker environment and guarded executor profiles, never credentials in inputs."""
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path


def _callable(name):
    module, attribute = name.split(':', 1)
    return getattr(importlib.import_module(module), attribute)


def public_identity(executor):
    return _public(executor.metadata)


def _public(metadata):
    result = {key: metadata[key] for key in ('target', 'runtime', 'physical_device', 'executor_identity')}
    result.update({key: metadata[key] for key in ('outer_handle', 'runtime_identity',
        'controller_placement', 'logical_device') if key in metadata})
    return result


class GuardDenied(ValueError):
    def __init__(self, receipt):
        super().__init__('trusted physical-device admission was denied or unavailable')
        self.admission_receipt = receipt


@dataclass(frozen=True)
class ExecutorProfiles:
    path: str

    @classmethod
    def from_environment(cls, environment=None):
        environment = os.environ if environment is None else environment
        path = environment.get('RSI_EXECUTOR_PROFILES')
        return cls(path) if path else None

    def _record(self, identity):
        return dict(json.loads(Path(self.path).expanduser().read_text())['profiles'][identity])

    def _executor(self, identity, value):
        if 'factory' in value:
            return _callable(value['factory'])(identity, value.get('configuration', {}))
        from rsi.cpl_execution import CplConfig, CplExecutor
        options = {k: v for k, v in value.items() if k not in {
            'metadata', 'guard', 'guard_configuration', 'worker_environment', 'worker_configuration'}}
        options['installed_client'] = Path(options['installed_client']).expanduser()
        options['trusted_resources'] = tuple(Path(p).expanduser() for p in options.get('trusted_resources', ()))
        return CplExecutor(identity, CplConfig(**options))

    def metadata(self, identity):
        value = self._record(identity)
        if not value.get('guard'):
            raise ValueError('NPU profile requires a trusted admission/command health guard')
        return _public(value['metadata']) if 'metadata' in value else public_identity(self._executor(identity, value))

    def __call__(self, identity):
        value = self._record(identity)
        if not value.get('guard'):
            raise ValueError('NPU executor profile requires a trusted per-command health/policy guard')
        return GuardedExecutor(self._executor(identity, value), self, identity)

    def guard(self, identity, request, *, phase, remaining):
        value = self._record(identity)
        if not value.get('guard') or remaining() is not None and remaining() <= 0:
            raise ValueError('trusted bounded NPU admission guard and elapsed allocation required')
        metadata = self.metadata(identity)
        configuration = dict(value.get('guard_configuration', {}))
        if metadata.get('controller_placement') != 'local':
            configuration['executor'] = self._executor(identity, value)
        scope = request.root.parent if hasattr(request, 'root') else request.project.parent.parent
        evidence = Path(configuration.setdefault('evidence_root', str(scope / 'device-admission')))
        evidence.mkdir(parents=True, exist_ok=True)
        try:
            result = _callable(value['guard'])(request, phase=phase, metadata=metadata,
                configuration=configuration, remaining=remaining)
        except Exception as exc:
            raise GuardDenied(dict(admitted=False, physical_device=metadata['physical_device'],
                observation='trusted guard did not complete', error_type=type(exc).__name__)) from exc
        if (not isinstance(result, dict) or result.get('admitted') is not True
                or not result.get('observation', '').strip()
                or str(result.get('physical_device')) != str(metadata['physical_device'])):
            raise GuardDenied(result if isinstance(result, dict) else dict(admitted=False,
                physical_device=metadata['physical_device'], observation='invalid guard response'))
        return result

    def worker_environment(self, identity, request, *, inherited):
        value = self._record(identity)
        metadata = self.metadata(identity)
        if str(metadata['physical_device']) != request.device:
            raise ValueError('worker profile does not bind the leased physical device')
        if metadata.get('controller_placement') != 'local':
            return dict(inherited)  # GZ runtime activation and mask belong to target executor.
        if not value.get('worker_environment'):
            raise ValueError('local NPU worker requires trusted environment admission')
        options = dict(value.get('worker_configuration', {}))
        if (options.get('physical_device') != metadata['physical_device']
                or options.get('outer_cpl_handle') != metadata.get('outer_handle')):
            raise ValueError('local worker outer admission/device differs from frozen profile')
        options['cache_directory'] = str(request.root.parent / 'target-cache')
        result = _callable(value['worker_environment'])(inherited=dict(inherited), **options)
        if (result.get('ASCEND_RT_VISIBLE_DEVICES') != request.device
                or result.get('CATLASS_DSL_CACHE_DIR') != options['cache_directory']
                or result.get('RSI_OUTER_CPL_HANDLE') != metadata['outer_handle']
                or result.get('RSI_CPL_RUNTIME') != metadata['runtime']):
            raise ValueError('local worker environment does not match frozen lease/cache/runtime')
        return result


@dataclass(frozen=True)
class GuardedExecutor:
    executor: object
    profiles: ExecutorProfiles
    identity: str

    @property
    def metadata(self):
        return self.executor.metadata

    def execute(self, request, *, observe, remaining):
        if self.profiles.metadata(self.identity) != public_identity(self.executor):
            raise ValueError('target profile changed since the frozen executor was constructed')
        try:
            receipt = self.profiles.guard(self.identity, request, phase='command', remaining=remaining)
        except GuardDenied as exc:
            observe('command', guard_receipt=exc.admission_receipt)
            raise
        observe('command', guard_receipt=receipt)
        return self.executor.execute(request, observe=observe, remaining=remaining)

    def reconcile(self, request, **kwargs):
        return self.executor.reconcile(request, **kwargs)


@dataclass(frozen=True)
class ProfileCommandGuard:
    profiles: object
    identity: str

    def __call__(self, request, *, remaining):
        return self.profiles.guard(self.identity, request, phase='command', remaining=remaining)


@dataclass(frozen=True)
class FrozenExecutorFactory:
    factory: object
    identities_json: str

    def metadata(self, identity):
        metadata = self.factory.metadata(identity)
        if metadata != json.loads(self.identities_json)[identity]:
            raise ValueError('host worker profile differs from frozen device/runtime identity')
        return metadata

    def __call__(self, identity):
        executor = self.factory(identity)
        if public_identity(executor) != json.loads(self.identities_json)[identity]:
            raise ValueError('host executor profile differs from frozen device/runtime identity')
        return executor

    def worker_environment(self, identity, request, **kwargs):
        self.metadata(identity)
        return self.factory.worker_environment(identity, request, **kwargs)

    def guard(self, identity, request, **kwargs):
        self.metadata(identity)
        return self.factory.guard(identity, request, **kwargs)
