"""CLI lifecycle carries the original operation and structured host observation."""
import json
from types import SimpleNamespace

from rsi import job_cli


def test_recovery_request_keeps_cycle_and_original_operation(tmp_path, monkeypatch):
    proof = tmp_path / 'observation.json'
    observation = {'controller_proof': {'handle': 'pi:original', 'actor_closed': True,
                                      'observation': 'actual closure'}}
    proof.write_text(json.dumps(observation))
    monkeypatch.setattr(job_cli.jobs, 'status', lambda *args: dict(request={
        'mode': 'phase1', 'cycle_id': 'same-cycle', 'action': 'start'}))
    requests = []
    monkeypatch.setattr(job_cli.jobs, 'start', lambda root, request: requests.append(request) or request)
    args = SimpleNamespace(command='phase1', action='recover', instance=tmp_path,
                           job='retained-job', operation='original-operation', observation=proof)
    result = job_cli.execute(args)
    assert result['cycle_id'] == 'same-cycle'
    assert result['previous_job'] == 'retained-job'
    assert result['operation_id'] == 'original-operation'
    assert result['observation'] == observation
    assert len(requests) == 1
