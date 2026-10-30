"""Owned worker plus real canonical CPU controllers; only decisions are scripted."""
import json
import os

from test_rsi_services import instance as bound_instance
from rsi.configuration import _bytes, _publish, show_configuration
from rsi.job_worker import run_owned
from rsi import jobs
import rsi.service


def test_worker_runs_real_cpu_canonical_project_and_retains_evidence(bound_instance, monkeypatch):
    services = bound_instance
    monkeypatch.setattr(rsi.service, 'InstanceServices', lambda root: services)
    root, directory = jobs._directory(services.root)
    import uuid
    job_id = str(uuid.uuid4())
    request = {'action': 'start', 'mode': 'phase1', 'cycle_id': 'cli-wave', 'recover': False}
    record = dict(job_id=job_id, state='launching', process=None, request=request,
                  configuration=show_configuration(root))
    _publish(jobs._path(directory, job_id), _bytes(record))
    lock = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    run_owned(root, job_id, lock, rsi.service.run_job)
    retained = jobs.evidence(root, job_id)
    assert retained['job']['state'] == 'completed'
    assert retained['job']['result']['status'] in ('complete', 'awaiting_tuning')
    state = services.status('cli-wave')['cycle.json']
    assert len(state['originals']) == 1 and state['originals'][0]['outcome'] == 'accepted'
    operation = state['originals'][0]['operation_id']
    workflow = json.loads((root / 'runs/cli-wave/projects' / operation / 'workflow.json').read_text())
    assert workflow['outcome'] == 'accepted'
    stages = [row['evidence']['stage'] for row in workflow['history']]
    assert stages == ['preflight', 'task_construction', 'design', 'implementation',
                      'test', 'debug', 'optimisation', 'validation', 'acceptance']
    assert retained['events']
    assert all(row[2] is not None for row in workflow['intervals'].values())
