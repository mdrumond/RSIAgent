import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer.transport import CheckedTransport, ObservationUnavailable, WorkerStopped, exclusive


def config(tmp_path):
    return dict(profile='bz-a3-1', target='bz-a3-1', runtime='py311-torch',
                physical_device=10, arch='dav-2201', cpl_remote='/remote',
                validation_wrapper='/profiles/catlass-validation.sh', remote_workspace='/work')


@pytest.mark.parametrize('failure', ['config', 'guide'])
def test_cli_bootstrap_failure_is_durable(tmp_path, failure):
    import subprocess
    import sys
    from optimizer.jobs import JobStore
    store = JobStore(tmp_path / 'jobs')
    key = store.create('def f(x): return x', 'f', {}, {})['job_id']
    invalid = tmp_path / 'invalid.json'
    settings = config(tmp_path)
    settings['guide'] = str(tmp_path / 'missing-guide')
    settings['lock_directory'] = str(tmp_path / 'locks')
    invalid.write_text(json.dumps({} if failure == 'config' else settings))
    result = subprocess.run([sys.executable, '-m', 'optimizer.worker', '--root', str(store.root),
                             '--job-id', key, '--config', str(invalid)], capture_output=True, text=True)
    assert result.returncode != 0 and 'Traceback' in result.stderr
    assert store.get(key)['status'] == 'failed'
    error = next(row['payload'] for row in store.events(key) if row['kind'] == 'worker-startup-error')
    assert error['type'] == ('ValueError' if failure == 'config' else 'FileNotFoundError')
    assert ('checked A3 profile' if failure == 'config' else 'missing-guide') in error['message']


def test_duplicate_cli_startup_cannot_fail_active_job(tmp_path):
    import subprocess
    import sys
    from optimizer.jobs import JobStore
    store = JobStore(tmp_path / 'jobs')
    key = store.create('def f(x): return x', 'f', {}, {})['job_id']
    settings = config(tmp_path)
    settings['arch'] = 'dav2201'
    settings['guide'] = str(tmp_path / 'guide')
    settings['lock_directory'] = str(tmp_path / 'locks')
    Path(settings['guide']).write_text('guide')
    path = tmp_path / 'worker.json'
    path.write_text(json.dumps(settings))
    with exclusive(store.root / key / 'worker.lock'):
        result = subprocess.run([sys.executable, '-m', 'optimizer.worker', '--root', str(store.root),
                                 '--job-id', key, '--config', str(path)], capture_output=True, text=True)
    assert result.returncode != 0 and 'BlockingIOError' in result.stderr
    assert store.get(key)['status'] == 'queued'
    assert not any(row['kind'] == 'worker-startup-error' for row in store.events(key))


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


def test_transfer_retains_diagnostics_before_json_parse(tmp_path):
    source = tmp_path / 'package.tar.gz'
    source.write_bytes(b'package')
    def run(argv, **kwargs):
        return SimpleNamespace(returncode=2, stdout='', stderr='SSH transfer capability probe failed')
    transport = CheckedTransport(config(tmp_path), run=run)
    with pytest.raises(ObservationUnavailable):
        transport.transfer('upload', source, '/work/package.tar.gz')
    log = source.with_name(source.name + '.upload.log').read_text()
    assert 'returncode=2' in log and 'SSH transfer capability probe failed' in log
    with pytest.raises(ObservationUnavailable, match='uncertain transfer'):
        transport.transfer('upload', source, '/work/package.tar.gz')


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
    assert json.loads((directory / 'transport.json').read_text())['state'] == 'cancelled'


def test_reconnect_retries_failed_cancellation_before_marking_stopped(tmp_path, monkeypatch):
    from optimizer.jobs import JobStore, canonical, sha
    from optimizer.worker import run_worker
    import optimizer.worker as module
    store = JobStore(tmp_path / 'jobs')
    body = {'files': {}, 'memory_entries': []}
    key = store.create('def f(x): return x', 'f', {},
                       {**body, 'snapshot_id': sha(canonical(body))})['job_id']
    store.transition(key, 'running', stage='observation-unavailable')
    store.stop(key)
    operation = store.root / key / 'execution' / 'candidate'
    operation.mkdir(parents=True)
    marker = operation / 'transport.json'
    marker.write_text(json.dumps({'state': 'running', 'handle': 'bz-a3-1:original',
                                  'native_handle': 'remote:bz-a3-1:job:original'}))
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0, stdout='{}', stderr='')
    monkeypatch.setattr(module, 'CheckedTransport', lambda config, **kwargs:
                        CheckedTransport(config, run=run, **kwargs))
    settings = config(tmp_path)
    settings['guide'] = str(tmp_path / 'guide')
    Path(settings['guide']).write_text('guide')
    settings['lock_directory'] = str(tmp_path / 'locks')
    agent = SimpleNamespace(identity={'model': 'unused'})
    with pytest.raises(ObservationUnavailable):
        run_worker(store.root, key, settings, agent=agent)
    assert store.get(key)['status'] == 'running'
    assert json.loads(marker.read_text())['state'] == 'running'
    assert run_worker(store.root, key, settings, agent=agent)['status'] == 'stopped'
    assert len(calls) == 2
    assert all(argv[-2:] == ['cancel', 'remote:bz-a3-1:job:original'] for argv in calls)
    assert json.loads(marker.read_text())['state'] == 'cancelled'


def test_stop_recovery_does_not_claim_unknown_dispatch_cancelled(tmp_path):
    directory = tmp_path / 'operation'
    directory.mkdir()
    marker = directory / 'transport.json'
    marker.write_text('{"state":"dispatching"}')
    with pytest.raises(ObservationUnavailable):
        CheckedTransport(config(tmp_path)).cancel_pending(directory)
    assert json.loads(marker.read_text())['state'] == 'dispatching'


def test_failed_profile_observer_is_terminal(tmp_path):
    directory = tmp_path / 'op'
    directory.mkdir()
    (directory / 'transport.json').write_text(json.dumps({'state': 'running', 'native_handle': 'remote:bz-a3-1:job:abc', 'handle': 'bz-a3-1:abc'}))
    def run(argv, **kwargs):
        return SimpleNamespace(returncode=1, stdout='BZ_A3_JOB_STATE=failed exit=1', stderr='compile failed')
    with pytest.raises(RuntimeError, match='remote operation failed'):
        CheckedTransport(config(tmp_path), run=run).execute(directory, ['true'])


def test_remote_timeout_is_budget_boundary_without_compact_download(tmp_path):
    directory = tmp_path / 'op'
    directory.mkdir()
    (directory / 'transport.json').write_text(json.dumps({'state': 'running', 'handle': 'bz-a3-1:abc'}))
    def run(argv, **kwargs):
        return SimpleNamespace(returncode=124, stdout='BZ_A3_JOB_STATE=failed exit=124', stderr='')
    with pytest.raises(WorkerStopped, match='budget exhausted'):
        CheckedTransport(config(tmp_path), run=run).execute(directory, ['true'])
    assert json.loads((directory / 'transport.json').read_text())['state'] == 'cancelled'


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


def test_cancelled_backend_restarts_only_in_fresh_generation(tmp_path):
    import shutil
    from optimizer.jobs import JobStore
    from optimizer.remote_worker import publish
    from optimizer.worker import RemoteBackend
    store = JobStore(tmp_path / 'jobs')
    job = store.create('def forward(x): return x', 'forward', {}, {})['job_id']
    completed = tmp_path / 'remote'
    completed.mkdir()
    publish(completed, {'status': 'verified'})
    class Transport:
        calls = []
        def transfer(self, action, source, destination):
            if action == 'download':
                shutil.copyfile(completed / 'compact.tar.gz', destination)
        def execute(self, directory, argv):
            self.calls.append(directory)
    transport = Transport()
    backend = RemoteBackend(store, job, config(tmp_path), transport=transport)
    directory = store.root / job / 'execution' / 'candidate'
    directory.mkdir(parents=True)
    (directory / 'transport.json').write_text(json.dumps({'state': 'cancelled', 'handle': 'bz-a3-1:old'}))
    assert backend.evaluate('source', directory, block_count=1)['status'] == 'verified'
    assert transport.calls == [directory / 'retry']


def test_terminal_candidate_crash_without_archive_is_feedback_not_job_crash(tmp_path):
    from optimizer.jobs import JobStore
    from optimizer.worker import RemoteBackend
    store = JobStore(tmp_path / 'jobs')
    job = store.create('def forward(x): return x', 'forward', {}, {})['job_id']
    class Transport:
        def transfer(self, action, source, destination):
            if action == 'download':
                raise RuntimeError('missing compact archive')
        def execute(self, directory, argv):
            (directory / 'transport.log').write_text('candidate exited 139')
            raise RuntimeError('remote operation failed')
    backend = RemoteBackend(store, job, config(tmp_path), transport=Transport())
    result = backend.evaluate('source', store.root / job / 'execution' / 'candidate', block_count=1)
    assert result['status'] == 'error'
    assert 'no compact result' in result['error']
    assert {row['name'] for row in store.artifacts(job)} == {'candidate/transport.log', 'candidate/result.json'}


def test_profile_observation_loss_preserves_pending_and_history_on_retry(tmp_path):
    from test_optimizer_loop import fixture, Agent, Backend, candidate
    from optimizer.loop import OptimizationLoop
    store, job = fixture(tmp_path, attempts=3)
    class LostProfile(Backend):
        def profile(self, *args, **kwargs):
            if not self.profiles:
                self.profiles.append('lost')
                raise ObservationUnavailable('pending exact handle')
            return {'status': 'available'}
    backend = LostProfile()
    agent = Agent([{'action': 'cases', 'signatures': [], 'summary': ''}, candidate('fast'), {'action': 'finish', 'summary': ''}])
    with pytest.raises(ObservationUnavailable):
        OptimizationLoop(store, job, agent, backend).run()
    state = [e['payload'] for e in store.events(job) if e['kind'] == 'optimizer-state'][-1]
    assert state['pending']['history_recorded']
    assert len(state['history']) == 1
    assert OptimizationLoop(store, job, agent, backend).run()['status'] == 'completed'
    state = [e['payload'] for e in store.events(job) if e['kind'] == 'optimizer-state'][-1]
    assert len(state['history']) == 1
    assert len(backend.evaluations) == 1


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


@pytest.mark.parametrize('boundary', ['stop', 'deadline'])
def test_final_reserved_profile_respects_stop_and_elapsed_deadline(tmp_path, boundary):
    from test_optimizer_loop import fixture, Agent, Backend, candidate
    from optimizer.loop import OptimizationLoop
    store, job = fixture(tmp_path, attempts=1, seconds=10)
    class BoundBackend(Backend):
        def evaluate(self, *args, **kwargs):
            result = super().evaluate(*args, **kwargs)
            if boundary == 'stop':
                store.stop(job)
            else:
                store.account_time(job, 10)
            return result
    backend = BoundBackend()
    OptimizationLoop(store, job, Agent([{'action': 'cases', 'signatures': [], 'summary': ''}, candidate('fast')]), backend).run()
    assert backend.profiles == []
