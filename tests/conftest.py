from test_native_pi_filesystem import scripted_provider
import pytest


@pytest.fixture
def scripted_pi(monkeypatch):
    """Inject one typed role decision; canonical host effects remain real.

    These composition fixtures explicitly supply a completion transport. They
    do not exercise the SDK; native HTTP/IPC behavior has separate fixtures.
    """
    import json
    import time
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, openai_compatible_transport
    from benchmarks.ascend.native_curriculum import NativeSubjectRole
    from core.artifacts import ContentAddressedArtifacts
    providers = []
    original = DirectCompletionProvider.__init__
    def configured(provider, *args, **kwargs):
        original(provider, *args, **kwargs)
        if provider._transport is not openai_compatible_transport:
            providers.append(provider)
    def decision(role, messages, seconds, *, wave, query=None, artifacts=None, validate=None):
        if seconds <= 0:
            raise TimeoutError('fixture role elapsed allowance exhausted')
        provider = next(provider for provider in reversed(providers)
                        if provider.coding_environment() == role.environ)
        started = time.monotonic()
        response = provider.complete(role.profile, messages, max_seconds=seconds)
        if validate:
            validate(response.text)
        if role.validator:
            role.validator(json.loads(response.text))
        role.last_store = ContentAddressedArtifacts(role.root/'fixture-artifacts')
        role.last_receipt = dict(scope='injected-composition-fixture', response=response.text,
            actor_seconds=time.monotonic()-started, usage=[dict(output=response.completion_tokens)],
            input_ref=role.last_store.put_json(messages))
        role.last_store.put_json(role.last_receipt)
        return response.text
    monkeypatch.setattr(DirectCompletionProvider, '__init__', configured)
    monkeypatch.setattr(NativeSubjectRole, '_call', decision)
    monkeypatch.setattr('coding.pi.actor.runtime_preflight', lambda: None)
    monkeypatch.setattr('coding.pi.runtime_preflight', lambda: None)


@pytest.fixture(autouse=True)
def native_provider_is_loopback_only(monkeypatch):
    """Harness tests must bind native requests to an actual local fixture."""
    from urllib.parse import urlparse
    from coding.pi.native import NativeSession
    original = NativeSession.prompt
    def prompt(session, *args, **kwargs):
        host = urlparse(session.profile['base_url']).hostname
        if host not in {'127.0.0.1', 'localhost', '::1'}:
            raise AssertionError('native test provider must use a loopback HTTP fixture before dispatch')
        return original(session, *args, **kwargs)
    monkeypatch.setattr(NativeSession, 'prompt', prompt)
