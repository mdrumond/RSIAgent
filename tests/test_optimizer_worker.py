import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer.transport import CheckedTransport, ObservationUnavailable, WorkerStopped, exclusive


def config(tmp_path):
    return dict(profile='bz-a3-1', target='bz-a3-1', runtime='py311-torch',
                physical_device=10, arch='dav-2201', cpl_remote='/remote',
                validation_wrapper='/profiles/catlass-validation.sh', remote_workspace='/work')


def test_uncertain_dispatch_is_never_submitted_twice(tmp_path):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=75, stdout='connection lost', stderr='')
    transport = CheckedTransport(config(tmp_path), run=run)
    directory = tmp_path / 'operation'
    with pytest.raises(ObservationUnavailable):
        transport.execute(directory, ['true'])
    with pytest.raises(ObservationUnavailable):
        transport.execute(directory, ['true'])
    assert len(calls) == 1


def test_retained_handle_is_observed_and_persisted(tmp_path):
    calls, handles = [], []
    def run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            return SimpleNamespace(returncode=75, stdout='BZ_A3_REMOTE_HANDLE=remote:bz-a3-1:job:abc\nCATLASS_VALIDATION_HANDLE=bz-a3-1:abc\nCATLASS_VALIDATION_STATE=observation-unavailable', stderr='')
        return SimpleNamespace(returncode=0, stdout='CATLASS_VALIDATION_STATE=completed', stderr='')
    transport = CheckedTransport(config(tmp_path), run=run, retain=lambda h: handles.append(h))
    transport.execute(tmp_path / 'op', ['true'])
    assert '--resume' in calls[1] and 'abc' in calls[1]
    assert handles == ['remote:bz-a3-1:job:abc', 'bz-a3-1:abc']


def test_stop_cancels_retained_native_handle(tmp_path):
    calls = []
    directory = tmp_path / 'op'
    directory.mkdir()
    (directory / 'transport.json').write_text(json.dumps({'state': 'running', 'native_handle': 'remote:bz-a3-1:job:abc', 'handle': 'bz-a3-1:abc'}))
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='{}', stderr='')
    transport = CheckedTransport(config(tmp_path), run=run, stopped=lambda: True)
    with pytest.raises(WorkerStopped):
        transport.execute(directory, ['true'])
    assert calls[0][-2:] == ['cancel', 'remote:bz-a3-1:job:abc']


def test_failed_profile_observer_is_terminal(tmp_path):
    directory = tmp_path / 'op'
    directory.mkdir()
    (directory / 'transport.json').write_text(json.dumps({'state': 'running', 'native_handle': 'remote:bz-a3-1:job:abc', 'handle': 'bz-a3-1:abc'}))
    def run(argv, **kwargs):
        return SimpleNamespace(returncode=1, stdout='BZ_A3_JOB_STATE=failed exit=1', stderr='compile failed')
    with pytest.raises(RuntimeError, match='remote operation failed'):
        CheckedTransport(config(tmp_path), run=run).execute(directory, ['true'])


def test_exclusive_lock_rejects_second_worker(tmp_path):
    with exclusive(tmp_path / 'lock'):
        with pytest.raises(BlockingIOError):
            with exclusive(tmp_path / 'lock'):
                pass


def test_compact_publish_excludes_tensor_binary_and_timeline(tmp_path):
    import tarfile
    from optimizer.remote_worker import publish
    for name in ['baseline.pt', 'kernel.so', 'trace.json', 'kernel.cpp', 'compile.log', 'timing.json']:
        (tmp_path / name).write_text('data')
    compact = tmp_path / 'Memory' / 'compact'
    compact.mkdir(parents=True)
    (compact / 'metrics.csv').write_text('name,value\nbw,12')
    publish(tmp_path, {'status': 'verified'})
    with tarfile.open(tmp_path / 'compact.tar.gz') as archive:
        names = archive.getnames()
        assert 'baseline.pt' not in names and 'trace.json' not in names and 'kernel.so' not in names
        assert {'kernel.cpp', 'compile.log', 'timing.json', 'result.json', 'Memory/compact/metrics.csv'} <= set(names)


def test_remote_backend_hydrates_authenticated_evidence_without_resubmission(tmp_path):
    import shutil
    from optimizer.jobs import JobStore
    from optimizer.remote_worker import publish
    from optimizer.worker import RemoteBackend
    store = JobStore(tmp_path / 'jobs')
    job = store.create('def forward(x): return x', 'forward', {}, {})['job_id']
    remote_result = tmp_path / 'remote-result'
    remote_result.mkdir()
    publish(remote_result, {'status': 'verified', 'suite_sha256': 'abc'})
    class Transport:
        executions = 0
        def transfer(self, action, source, destination):
            if action == 'download':
                shutil.copyfile(remote_result / 'compact.tar.gz', destination)
        def execute(self, directory, argv):
            self.executions += 1
    transport = Transport()
    backend = RemoteBackend(store, job, config(tmp_path), transport=transport)
    directory = store.root / job / 'execution' / 'candidate'
    first = backend.evaluate('source', directory, block_count=1)
    assert first == backend.evaluate('source', directory, block_count=1)
    assert transport.executions == 1
    assert store.artifacts(job)[0]['name'] == 'candidate/result.json'


def test_pending_download_observes_original_transfer_then_fetches(tmp_path):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if 'download' in argv:
            return SimpleNamespace(returncode=75, stdout=json.dumps({'state': 'running', 'handle': 'remote:bz-a3-1:transfer:one'}), stderr='')
        if 'fetch' in argv:
            Path(argv[-1]).write_text('payload')
        return SimpleNamespace(returncode=0, stdout=json.dumps({'state': 'completed', 'handle': 'remote:bz-a3-1:transfer:one'}), stderr='')
    handles = []
    transport = CheckedTransport(config(tmp_path), run=run, retain=handles.append)
    destination = tmp_path / 'result.tar.gz'
    transport.transfer('download', '/remote/result.tar.gz', destination)
    transport.transfer('download', '/remote/result.tar.gz', destination)
    assert len(calls) == 3
    assert calls[1][2:4] == ['observe', 'remote:bz-a3-1:transfer:one']
    assert handles == ['remote:bz-a3-1:transfer:one']


@pytest.mark.parametrize('error,status', [(ObservationUnavailable('lost'), 'running'), (WorkerStopped('deadline'), 'budget-exhausted')])
def test_loop_preserves_recoverable_transport_state(tmp_path, error, status):
    from optimizer.jobs import JobStore, canonical, sha
    from optimizer.loop import OptimizationLoop
    body = {'files': {}, 'memory_entries': []}
    store = JobStore(tmp_path)
    job = store.create('def forward(x): return x', 'forward',
                       {'args': [{'shape': [5], 'dtype': 'float32'}]},
                       {**body, 'snapshot_id': sha(canonical(body))})['job_id']
    class Agent:
        identity = {'model': 'test'}
        def complete(self, stage, context, record):
            return {'action': 'cases', 'signatures': [], 'summary': 'test'}
    class Backend:
        def prepare(self, *args):
            raise error
    with pytest.raises(type(error)):
        OptimizationLoop(store, job, Agent(), Backend()).run()
    assert store.get(job)['status'] == status
