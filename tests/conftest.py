from test_native_pi_filesystem import scripted_provider
import pytest


@pytest.fixture
def scripted_pi(monkeypatch):
    """Keep existing host-composition scripts offline at the typed session seam."""
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
