import pytest


@pytest.fixture
def scripted_pi(monkeypatch):
    """Keep existing host-composition scripts offline at the typed session seam."""
    from tests.pi_fakes import ScriptedPiSession
    from coding.pi.actor import PiActorDriver
    monkeypatch.setattr('coding.pi.actor.runtime_preflight', lambda: None)
    monkeypatch.setattr('coding.pi.runtime_preflight', lambda: None)
    monkeypatch.setattr(PiActorDriver, '_new_session',
                        lambda self, profile: ScriptedPiSession(self.provider, profile))
