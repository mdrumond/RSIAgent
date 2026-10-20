import json

import pytest

from benchmarks.pi_coding import smoke


def test_callable_preflight_is_offline_and_preserves_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, 'runtime_preflight', lambda: None)
    ledger = tmp_path / 'budget.json'
    ledger.write_text('{"limit_usd":5,"reservations":[]}')
    original = ledger.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('check-only must not instantiate transport or run worker')
    monkeypatch.setattr(smoke, 'run_worker', forbidden)
    result = smoke.run_smoke(config={'lock_directory': str(tmp_path / 'locks'),
                                    'transport_identity': 'test-v1'},
                             budget_ledger=ledger, root=tmp_path / 'jobs',
                             window='test', check_only=True, transport_factory=forbidden)
    assert result['remote_validated'] is False
    assert result['max_requests'] == 20
    assert ledger.read_bytes() == original
    assert not (tmp_path / 'jobs').exists()


@pytest.mark.parametrize('budget', [0, -1, 2.01, float('nan'), float('inf')])
def test_callable_rejects_budget_before_runtime_or_provider(budget):
    with pytest.raises(ValueError, match='incremental budget'):
        smoke.run_smoke(config={}, budget_ledger='unused', root='unused',
                        window='test', budget_usd=budget)


def test_callable_preserves_failed_run_report_and_forwards_transport(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, 'runtime_preflight', lambda: None)
    monkeypatch.setattr(smoke, 'OptimizerAgent', lambda **kwargs: object())
    guide = tmp_path / 'guide'
    guide.write_text('guide')
    ledger = tmp_path / 'budget.json'
    ledger.write_text('{"limit_usd":5,"reservations":[]}')
    factory = object()
    def worker(root, job_id, config, **kwargs):
        assert kwargs['transport_factory'] is factory
        assert config['max_cases'] == 3
        raise RuntimeError('private diagnostics must stay private')
    monkeypatch.setattr(smoke, 'run_worker', worker)
    report = smoke.run_smoke(config={'guide': str(guide), 'lock_directory': str(tmp_path),
                                    'transport_identity': 'test-v1'},
                            budget_ledger=ledger, root=tmp_path / 'jobs', window='test',
                            transport_factory=factory)
    assert report['failure_type'] == 'RuntimeError'
    assert not report['host_verified']
    assert json.loads((tmp_path / 'jobs/smoke-summary.json').read_text()) == report
