from canonical_optimizer_fixtures import canonical_optimizer_backend

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer.transport import CheckedTransport, ObservationUnavailable, WorkerStopped, exclusive


@pytest.mark.parametrize('boundary', ['stop', 'deadline'])
def test_stalled_client_is_bounded_and_preserves_unknown_dispatch(tmp_path, boundary):
    import sys
    import time
    wrapper = tmp_path / 'wrapper'
    wrapper.write_text('#!' + sys.executable + '\nimport time\nprint("client started", flush=True)\ntime.sleep(30)\n')
    wrapper.chmod(0o755)
    settings = config(tmp_path) | {'validation_wrapper': str(wrapper)}
    start = time.monotonic()
    transport = CheckedTransport(settings, stopped=lambda: boundary == 'stop' and time.monotonic() - start > .2,
                                 deadline=start + (.2 if boundary == 'deadline' else 30))
    with pytest.raises(ObservationUnavailable, match='interrupted'):
        with exclusive(tmp_path / 'worker.lock'):
            transport.execute(tmp_path / 'op', ['true'])
    assert time.monotonic() - start < 3
    assert json.loads((tmp_path / 'op/transport.json').read_text())['state'] == 'dispatching'
    assert 'client started' in (tmp_path / 'op/transport.log').read_text()
    with exclusive(tmp_path / 'worker.lock'):
        pass


def test_interrupted_client_retains_partial_handles(tmp_path):
    import sys
    import time
    wrapper = tmp_path / 'wrapper'
    wrapper.write_text('#!' + sys.executable + '\nimport time\nprint("BZ_A3_REMOTE_HANDLE=remote:bz-a3-1:job:abc", flush=True)\nprint("CATLASS_VALIDATION_HANDLE=bz-a3-1:abc", flush=True)\ntime.sleep(30)\n')
    wrapper.chmod(0o755)
    handles = []
    with pytest.raises(ObservationUnavailable, match='interrupted'):
        CheckedTransport(config(tmp_path) | {'validation_wrapper': str(wrapper)},
                         deadline=time.monotonic() + .2, retain=handles.append).execute(tmp_path / 'op', ['true'])
    assert handles == ['remote:bz-a3-1:job:abc', 'bz-a3-1:abc']
    assert json.loads((tmp_path / 'op/transport.json').read_text())['native_handle'] == handles[0]


def test_stalled_upload_preserves_sentinel_and_diagnostics(tmp_path):
    import sys
    import time
    client = tmp_path / 'client'
    client.write_text('#!' + sys.executable + '\nimport time\nprint("upload started", flush=True)\ntime.sleep(30)\n')
    client.chmod(0o755)
    source = tmp_path / 'package'
    source.write_bytes(b'bytes')
    transport = CheckedTransport(config(tmp_path) | {'cpl_remote': str(client)}, deadline=time.monotonic() + .2)
    with pytest.raises(ObservationUnavailable):
        transport.transfer('upload', source, '/work/package')
    assert json.loads((tmp_path / 'package.upload.json').read_text()) == {'state': 'dispatching'}
    assert 'upload started' in (tmp_path / 'package.upload.log').read_text()
    with pytest.raises(ObservationUnavailable, match='uncertain transfer'):
        transport.transfer('upload', source, '/work/package')


def test_cancel_cleanup_ignores_stop_but_has_its_own_bound(tmp_path, monkeypatch):
    import sys
    import time
    import optimizer.transport as module
    monkeypatch.setattr(module, 'CANCEL_TIMEOUT', .2)
    client = tmp_path / 'client'
    client.write_text('#!' + sys.executable + '\nimport time\nprint("cancel attempted", flush=True)\ntime.sleep(30)\n')
    client.chmod(0o755)
    directory = tmp_path / 'op'
    directory.mkdir()
    original = {'state': 'running', 'handle': 'bz-a3-1:abc', 'native_handle': 'remote:bz-a3-1:job:abc'}
    (directory / 'transport.json').write_text(json.dumps(original))
    start = time.monotonic()
    with pytest.raises(ObservationUnavailable, match='cancellation unavailable'):
        CheckedTransport(config(tmp_path) | {'cpl_remote': str(client)},
                         stopped=lambda: True, deadline=start - 1).cancel_pending(directory)
    assert time.monotonic() - start < 3
    assert 'cancel attempted' in (directory / 'cancel.log').read_text()
    assert json.loads((directory / 'transport.json').read_text()) == original


def test_interrupted_fetch_never_publishes_partial_destination(tmp_path):
    import sys
    import time
    client = tmp_path / 'client'
    client.write_text('#!' + sys.executable + '\nimport sys,time\nfrom pathlib import Path\nPath(sys.argv[-1]).write_bytes(b"partial")\ntime.sleep(30)\n')
    client.chmod(0o755)
    destination = tmp_path / 'compact.tar.gz'
    marker = tmp_path / 'compact.tar.gz.download.json'
    marker.write_text(json.dumps({'state': 'completed', 'handle': 'remote:bz-a3-1:transfer:abc'}))
    with pytest.raises(ObservationUnavailable, match='fetch unavailable'):
        CheckedTransport(config(tmp_path) | {'cpl_remote': str(client)},
                         deadline=time.monotonic() + .2).transfer('download', '/work/compact.tar.gz', destination)
    assert not destination.exists()
    assert (tmp_path / 'compact.tar.gz.fetching').read_bytes() == b'partial'
    assert json.loads(marker.read_text())['handle'] == 'remote:bz-a3-1:transfer:abc'


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


def test_legacy_remote_timeout_without_native_handle_remains_unknown(tmp_path):
    directory = tmp_path / 'op'
    directory.mkdir()
    legacy = 'bz-a3-1:abc'
    (directory / 'transport.json').write_text(json.dumps({'state': 'running', 'handle': legacy}))
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=124, stdout='BZ_A3_JOB_STATE=failed exit=124', stderr='')
    with pytest.raises(ObservationUnavailable):
        CheckedTransport(config(tmp_path), run=run).execute(directory, ['true'])
    retained = json.loads((directory / 'transport.json').read_text())
    assert retained['handle'] == legacy and not retained.get('native_handle')
    assert retained['state'] != 'cancelled'
    assert 'BZ_A3_JOB_STATE=failed exit=124' in (directory / 'transport.log').read_text()
    assert len(calls) == 1 and '--resume' in calls[0]
    assert not any('cancel' in argv or 'download' in argv for argv in calls)


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
            # Keep the workload in its retained command's process group so
            # exact-handle cancellation also reaches Python/compiler children.
            assert 'timeout --foreground --signal=TERM' in argv[-1]
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


@pytest.mark.parametrize('failure', ['bad-digest', 'missing-result', 'commit', 'result-marker'])
def test_compact_import_failure_does_not_publish_result_and_can_retry(tmp_path, monkeypatch, failure):
    import io
    import shutil
    import tarfile
    from optimizer.jobs import JobStore, canonical, sha
    from optimizer.worker import RemoteBackend
    store = JobStore(tmp_path / 'jobs')
    job = store.create('def forward(x): return x', 'forward', {}, {})['job_id']
    compact = tmp_path / 'remote.tar.gz'
    def publish(valid):
        files = {'result.json': canonical({'status': 'verified'}), 'metrics.csv': b'metric,1'}
        if not valid and failure == 'missing-result':
            files.pop('result.json')
        rows = [dict(path=name, sha256=sha(data), size=len(data)) for name, data in files.items()]
        if not valid and failure == 'bad-digest':
            rows[-1]['sha256'] = '0' * 64
        with tarfile.open(compact, 'w:gz') as archive:
            for name, data in [*files.items(), ('files.json', canonical(rows))]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    class Transport:
        def transfer(self, action, source, destination):
            if action == 'download':
                shutil.copyfile(compact, destination)
        def execute(self, directory, argv):
            pass
    backend = RemoteBackend(store, job, config(tmp_path), transport=Transport())
    directory = store.root / job / 'execution' / 'candidate'
    publish(False)
    batch = store._event_batch
    from optimizer import worker
    original_atomic = worker.atomic
    if failure == 'result-marker':
        def fail_marker(path, data):
            if path == directory / 'result.json':
                raise OSError('interrupted after inventory commit')
            return original_atomic(path, data)
        monkeypatch.setattr(worker, 'atomic', fail_marker)
    if failure == 'commit':
        def fail(*args):
            raise OSError('interrupted artifact commit')
        monkeypatch.setattr(store, '_event_batch', fail)
    with pytest.raises((ValueError, OSError)):
        backend.evaluate('source', directory, block_count=1)
    assert not (directory / 'result.json').exists()
    assert len(store.artifacts(job)) == (2 if failure == 'result-marker' else 0)
    monkeypatch.setattr(store, '_event_batch', batch)
    monkeypatch.setattr(worker, 'atomic', original_atomic)
    publish(True)
    assert backend.evaluate('source', directory, block_count=1) == {'status': 'verified'}
    assert len(store.artifacts(job)) == 2


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
    agent.profile = True
    loop = OptimizationLoop(store, job, agent, backend)
    with pytest.raises(ObservationUnavailable, match='pending exact handle'):
        loop.run()
    assert loop.fixture_runner.workflow.pending is not None
    assert loop.fixture_runner.workflow.stage.value == 'optimisation'
    interrupted = [row for row in loop.fixture_runner.events if row['type'] == 'domain-interrupted']
    assert len(interrupted) == 1 and interrupted[0]['name'] == 'profile'
    state = loop.state
    assert len(state['history']) == 1
    from canonical_optimizer_fixtures import retained_native, unresolved_authority
    from rsi.runtime import WorkflowError
    before, budget = retained_native(loop), store.get(job)
    assert state['best']['speedup'] == 2
    assert backend.profiles == ['lost']
    resumed = OptimizationLoop(store, job, agent, backend)
    with pytest.raises(WorkflowError, match='reconciliation'):
        resumed.run()
    assert unresolved_authority(retained_native(resumed)) == unresolved_authority(before)
    after = store.get(job)
    assert all(after[key] == budget[key] for key in ('max_seconds', 'max_attempts', 'attempts'))
    assert after['elapsed_seconds'] >= budget['elapsed_seconds']
    assert len(resumed.state['history']) == len(backend.evaluations) == 1
    assert resumed.state['best']['speedup'] == 2 and backend.profiles == ['lost']


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
        canonical_fixture = True
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


def test_injected_transport_receives_host_controls_and_retains_handle(tmp_path):
    import time
    from optimizer.jobs import JobStore
    from optimizer.worker import RemoteBackend
    store = JobStore(tmp_path / 'jobs')
    key = store.create('def f(x): return x', 'f', {}, {})['job_id']
    captured = {}
    sentinel = object()
    def factory(settings, **controls):
        captured.update(controls)
        return sentinel
    backend = RemoteBackend(store, key, config(tmp_path), transport_factory=factory)
    assert backend.transport is sentinel
    assert captured['deadline'] > time.monotonic()
    assert not captured['stopped']()
    captured['retain']('custom:retained')
    assert store.get(key)['handles'][0]['handle'] == 'custom:retained'
    store.stop(key)
    assert captured['stopped']()
    with pytest.raises(ValueError, match='not both'):
        RemoteBackend(store, key, config(tmp_path), transport=sentinel, transport_factory=factory)


def test_injected_worker_requires_explicit_lock_directory(tmp_path):
    from optimizer.worker import run_worker
    with pytest.raises(ValueError, match='lock_directory'):
        run_worker(tmp_path / 'jobs', 'unused', {}, transport_factory=lambda **kw: None)


def test_injected_worker_routes_stop_and_pins_transport_identity(tmp_path):
    from optimizer.jobs import JobStore, canonical, sha
    from optimizer.worker import run_worker
    store = JobStore(tmp_path / 'jobs')
    body = {'files': {}, 'memory_entries': []}
    key = store.create('def f(x): return x', 'f', {},
                       {**body, 'snapshot_id': sha(canonical(body))})['job_id']
    store.transition(key, 'running')
    store.stop(key)
    directory = store.root / key / 'execution' / 'candidate'
    directory.mkdir(parents=True)
    (directory / 'transport.json').write_text('{"state":"running","handle":"custom:one"}')
    guide = tmp_path / 'guide'
    guide.write_text('guide')
    settings = {k: v for k, v in config(tmp_path).items()
                if k not in {'validation_wrapper', 'cpl_remote'}}
    settings.update(guide=str(guide), lock_directory=str(tmp_path / 'locks'),
                    transport_identity='test-transport-v1')
    calls = []
    def factory(settings, **controls):
        return SimpleNamespace(cancel_pending=lambda path: calls.append(path))
    agent = SimpleNamespace(identity={'model': 'unused'})
    assert run_worker(store.root, key, settings, agent=agent,
                      transport_factory=factory)['status'] == 'stopped'
    assert calls == [directory]
    settings['transport_identity'] = 'test-transport-v2'
    with pytest.raises(ValueError, match='changed on resume'):
        run_worker(store.root, key, settings, agent=agent, transport_factory=factory)


def test_completed_remote_result_survives_interrupted_client_at_deadline(tmp_path):
    expired = False
    def run(argv, **kwargs):
        nonlocal expired
        expired = True
        return SimpleNamespace(returncode=-15, interrupted=True, stderr='', stdout=(
            'BZ_A3_REMOTE_HANDLE=remote:bz-a3-1:job:completed\n'
            'CATLASS_VALIDATION_HANDLE=bz-a3-1:completed\n'
            'CATLASS_VALIDATION_STATE=completed\n'))
    transport = CheckedTransport(config(tmp_path), run=run, stopped=lambda: expired)
    directory = tmp_path / 'op'
    transport.execute(directory, ['true'])
    assert json.loads((directory / 'transport.json').read_text())['state'] == 'completed'


@pytest.mark.parametrize('boundary', ['deadline', 'stop', 'client-timeout', 'failed-cancel', 'unknown-dispatch'])
def test_interrupted_observation_cancels_only_known_operation_at_real_stop(tmp_path, boundary):
    import time
    stopped = False
    calls = []
    transport = None
    def run(argv, **kwargs):
        nonlocal stopped
        calls.append(argv)
        if 'cancel' in argv:
            return SimpleNamespace(returncode=1 if boundary == 'failed-cancel' else 0, stdout='', stderr='')
        stopped = boundary != 'client-timeout'
        if boundary == 'deadline':
            transport.deadline = time.monotonic() - 1
        output = '' if boundary == 'unknown-dispatch' else (
            'BZ_A3_REMOTE_HANDLE=remote:bz-a3-1:job:deadline-test\n'
            'CATLASS_VALIDATION_HANDLE=bz-a3-1:deadline-test\n')
        return SimpleNamespace(returncode=-15, stdout=output, stderr='', interrupted=True)
    transport = CheckedTransport(config(tmp_path), run=run,
        stopped=lambda: stopped or time.monotonic() >= transport.deadline)
    error = WorkerStopped if boundary in {'deadline', 'stop'} else ObservationUnavailable
    with pytest.raises(error):
        transport.execute(tmp_path / 'profile', ['true'])
    cancelled = [argv for argv in calls if 'cancel' in argv]
    assert bool(cancelled) == (boundary in {'deadline', 'stop', 'failed-cancel'})
    if cancelled:
        assert cancelled[0][-1] == 'remote:bz-a3-1:job:deadline-test'
    state = json.loads((tmp_path / 'profile/transport.json').read_text())
    assert (state['state'] == 'cancelled') == (boundary in {'deadline', 'stop'})
