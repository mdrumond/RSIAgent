import pytest


@pytest.fixture
def scripted_pi(monkeypatch):
    """Keep existing host-composition scripts offline at the typed session seam."""
    from tests.pi_fakes import ScriptedPiSession
    from coding.pi.actor import PiActorDriver
    from benchmarks.a5kernels.phase2_source_provider import PiPhase2SourceProvider
    monkeypatch.setattr('coding.pi.actor.runtime_preflight', lambda: None)
    monkeypatch.setattr('coding.pi.runtime_preflight', lambda: None)
    monkeypatch.setattr(PiActorDriver, '_new_session',
                        lambda self, profile: ScriptedPiSession(self.provider, profile))
    monkeypatch.setattr(PiPhase2SourceProvider, '_new_coding_session',
                        lambda self, profile, consumed: ScriptedPiSession(self.provider, profile))
