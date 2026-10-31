"""Lazy host-only executor resolution; profile contents never enter instance inputs."""
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path


def public_identity(executor):
    metadata = executor.metadata
    result = {key: metadata[key] for key in ('target', 'runtime', 'physical_device', 'executor_identity')}
    result.update({key: metadata[key] for key in ('outer_handle', 'runtime_identity',
        'controller_placement', 'logical_device') if key in metadata})
    return result


@dataclass(frozen=True)
class ExecutorProfiles:
    path: str

    @classmethod
    def from_environment(cls, environment=None):
        environment = os.environ if environment is None else environment
        path = environment.get('RSI_EXECUTOR_PROFILES')
        return cls(path) if path else None

    def __call__(self, identity):
        value = dict(json.loads(Path(self.path).expanduser().read_text())['profiles'][identity])
        if 'factory' in value:
            module, attribute = value['factory'].split(':', 1)
            factory = getattr(importlib.import_module(module), attribute)
            return factory(identity, value.get('configuration', {}))
        from rsi.cpl_execution import CplConfig, CplExecutor
        value['installed_client'] = Path(value['installed_client']).expanduser()
        value['trusted_resources'] = tuple(Path(p).expanduser() for p in value.get('trusted_resources', ()))
        return CplExecutor(identity, CplConfig(**value))


@dataclass(frozen=True)
class FrozenExecutorFactory:
    factory: object
    identities_json: str

    def __call__(self, identity):
        executor = self.factory(identity)
        if public_identity(executor) != json.loads(self.identities_json)[identity]:
            raise ValueError('host executor profile differs from frozen device/runtime identity')
        return executor
