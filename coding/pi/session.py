"""Compatibility names and an unstarted registered native session binding.

Programming entrypoints transfer this binding to the canonical host controller.
It owns no provider gateway, conversation loop, schema engine or CLI child.
"""
import os
from pathlib import Path

from .native import NativeError, NativeSession, runtime_preflight

ROOT = Path(__file__).resolve().parents[2]
ENGINE_IDENTITY = dict(coding_engine='pi', pi_version='1.3.0', adapter_version=6,
                       context_policy='native-sdk-session-v1', budget_policy='elapsed-time-v1')
RuntimeUnavailable = NativeError


class ActionError(NativeError):
    """Native settlement lacked the required host stage commit."""


class PiCodingSession:
    """Retain a registered model/environment for canonical session adoption."""
    def __init__(self, *, model, schemas=None, environ=None, deadline=None, **options):
        from benchmarks.a3_experiments import BackendModel
        from benchmarks.a3_model_profiles import load_a3_model_profile
        self.model = getattr(model, 'value', model)
        self.profile = load_a3_model_profile(BackendModel(self.model))
        self.environ = os.environ if environ is None else environ
        self.schemas, self.options, self.deadline = dict(schemas or {}), options, deadline
        self.closed, self.rpc, self.native_session = False, None, None
        self.usage_source = None

    def open_native(self, root, **options):
        if self.closed or self.native_session is not None:
            raise ValueError('registered session binding already consumed')
        self.native_session = NativeSession(root, profile_id=self.profile.profile_id,
                                            environ=self.environ, **options)
        return self.native_session

    @property
    def completion_tokens(self):
        source = self.native_session or self.usage_source
        return source.completion_tokens if source is not None else 0

    def close(self):
        if not self.closed and self.native_session:
            self.native_session.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
