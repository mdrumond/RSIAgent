"""Detached optimizer worker: python -m optimizer.worker --root ... --job-id ... --config ..."""
import argparse
import dataclasses
import io
import json
import os
from pathlib import Path
import shlex
import tarfile
import time

from .jobs import JobStore, atomic, canonical, sha
from .transport import CheckedTransport, ObservationUnavailable, WorkerStopped, exclusive


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {'profile', 'target', 'runtime', 'physical_device', 'arch', 'cpl_remote',
                'validation_wrapper', 'remote_workspace', 'guide'}
    if not required <= config.keys() or config['profile'] not in {'bz-a3-1', 'bz-a3-2'}:
        raise ValueError('worker requires configured checked A3 profile and paths')
    if config['target'] != config['profile'] or config['runtime'] != 'py311-torch':
        raise ValueError('target/runtime must match checked native A3 route')
    if type(config['physical_device']) is not int or config['physical_device'] < 0:
        raise ValueError('physical_device must be a nonnegative integer')
    if type(config.get('max_cases', 24)) is not int or not 1 <= config.get('max_cases', 24) <= 24:
        raise ValueError('max_cases must be between one and 24')
    if not all(Path(config[key]).is_absolute() for key in ('cpl_remote', 'validation_wrapper', 'remote_workspace', 'guide')):
        raise ValueError('worker paths must be absolute')
    return config


class RemoteBackend:
    def __init__(self, store, job_id, config, *, transport=None):
        self.store, self.job_id, self.config = store, job_id, config
        self.remote = Path(config['remote_workspace']) / ('optimizer-' + job_id)
        self.started = time.monotonic()
        remaining = store.get(job_id)['max_seconds'] - store.get(job_id)['elapsed_seconds']
        self.deadline = self.started + remaining
        self.transport = transport or CheckedTransport(config, retain=self._retain, stopped=self._stopped, deadline=self.deadline)

    def _retain(self, handle):
        self.store.retain_handle(self.job_id, handle, operation='optimizer')

    def _stopped(self):
        return self.store.get(self.job_id)['stop_requested'] or time.monotonic() >= self.deadline

    def _operation(self, operation, directory, **payload):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        transport_state = directory / 'transport.json'
        if transport_state.exists() and json.loads(transport_state.read_text()).get('state') == 'cancelled':
            # A cancelled process may leave partial compiler/frozen files. Resume
            # uses a fresh generation only after exact-handle cancellation succeeds.
            return self._operation(operation, directory / 'retry', **payload)
        result_path = directory / 'result.json'
        if result_path.exists():
            return json.loads(result_path.read_text())
        relative = directory.relative_to(self.store.root / self.job_id / 'execution')
        remote_directory = self.remote / 'execution' / relative
        baseline = self.store.root / self.job_id / 'execution' / 'baseline'
        while (baseline / 'transport.json').exists() and json.loads((baseline / 'transport.json').read_text()).get('state') == 'cancelled':
            baseline = baseline / 'retry'
        reference = (json.loads((baseline / 'result.json').read_text()).get('remote_reference')
                     if (baseline / 'result.json').exists() else None)
        request = {'operation': operation, 'directory': str(remote_directory),
                   'baseline': str(remote_directory / 'frozen') if operation == 'prepare' else reference or str(self.remote / 'execution' / 'baseline' / 'frozen'),
                   'arch': self.config['arch'], 'max_cases': self.config.get('max_cases', 24), **payload}
        archive_path = directory / 'package.tar.gz'
        source_root = Path(__file__).resolve().parents[1]
        with tarfile.open(archive_path, 'w:gz') as archive:
            for family in ('optimizer', 'benchmarks'):
                for path in (source_root / family).rglob('*.py'):
                    archive.add(path, arcname=str(Path('source') / path.relative_to(source_root)))
            data = canonical(request)
            item = tarfile.TarInfo('request.json')
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
        # Parent workspace is registry-configured and must already exist.
        upload = Path(self.config['remote_workspace']) / f'{self.job_id}-{sha(str(relative).encode())[:12]}.tar.gz'
        state_path = directory / 'transport.json'
        if not state_path.exists():
            self.transport.transfer('upload', archive_path, upload)
        quoted = shlex.quote
        command = (f'test -d {quoted(self.config["remote_workspace"])} && mkdir -p {quoted(str(remote_directory))} '
                   f'&& tar -xzf {quoted(str(upload))} -C {quoted(str(remote_directory))} '
                   f'&& cd {quoted(str(remote_directory / "source"))} '
                   f'&& timeout --signal=TERM {max(1, int(self.deadline - time.monotonic()))} '
                   f'python -m optimizer.remote_worker {quoted(str(remote_directory / "request.json"))}')
        failure = None
        try:
            self.transport.execute(directory, ['bash', '-c', command])
        except RuntimeError as exc:
            if isinstance(exc, (ObservationUnavailable, WorkerStopped)):
                raise
            # Terminal compiler/reference failures still have compact evidence.
            failure = str(exc)
            self.store.add_event(self.job_id, 'remote-operation-error', {'operation': operation, 'error': str(exc)})
        if (directory / 'transport.log').exists():
            self.store.put_artifact(self.job_id, str(relative / 'transport.log'), (directory / 'transport.log').read_bytes())
        compact = directory / 'compact.tar.gz'
        try:
            self.transport.transfer('download', remote_directory / 'compact.tar.gz', compact)
        except RuntimeError as exc:
            if failure is None or isinstance(exc, (ObservationUnavailable, WorkerStopped)):
                raise
            result = {'status': 'error', 'error': failure + '; no compact result was published',
                      'operation': operation, 'cases': [], 'remote_directory': str(remote_directory)}
            if (directory / 'transport.log').exists():
                result['runtime_log_tail'] = (directory / 'transport.log').read_text(errors='replace')[-12000:]
            atomic(result_path, canonical(result))
            self.store.put_artifact(self.job_id, str(relative / 'result.json'), canonical(result))
            return result
        with tarfile.open(compact) as archive:
            rows = json.load(archive.extractfile('files.json'))
            for row in rows:
                name = Path(row['path'])
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError('invalid compact artifact path')
                data = archive.extractfile(row['path']).read()
                if sha(data) != row['sha256'] or len(data) != row['size']:
                    raise ValueError('compact artifact digest differs')
                self.store.put_artifact(self.job_id, str(relative / name), data)
                if name == Path('result.json'):
                    atomic(result_path, data)
        if not result_path.exists():
            raise ValueError('remote compact archive has no result')
        return json.loads(result_path.read_text())

    def prepare(self, spec, directory):
        result = self._operation('prepare', directory, spec=spec)
        if result.get('status') == 'error':
            raise ValueError(result['error'])
        # Baseline diagnostics are independent of canonical timings.
        try:
            result['profiling'] = self.profile(None, Path(directory) / 'profile')
        except (ObservationUnavailable, WorkerStopped):
            raise
        except Exception as exc:
            result['profiling'] = {'status': 'unavailable', 'error': str(exc)}
        return result

    def evaluate(self, source, directory, *, block_count):
        return self._operation('evaluate', directory, source=source, block_count=block_count)

    def compile(self, source, directory, *, block_count):
        return self._operation('compile', directory, source=source, block_count=block_count)

    def profile(self, source, directory, *, block_count=1):
        return self._operation('profile', directory, source=source, block_count=block_count)


def knowledge_query(config, directory):
    if not config.get('knowledge'):
        return None
    settings = config['knowledge']
    if 'kdb' in settings:
        from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
        from benchmarks.a5kernels.knowledge import KnowledgeDB
        from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, KnowledgeQuery, ProgressiveMemoryJournal
        from benchmarks.a5kernels.production import _transformers_cache
        embeddings = PinnedBGEEmbeddings(cache_dir=_transformers_cache(Path(settings['bge_cache'])))
        database = KnowledgeDB.open_read_only(Path(settings['kdb']), embeddings)
        try:
            manifests = database.selected_manifests(settings['collection'])
            actual = (manifests[0].fingerprint if isinstance(settings['collection'], str)
                      else {manifest.collection: manifest.fingerprint for manifest in manifests})
            if actual != settings['manifest']:
                raise ValueError('optimizer selected KDB manifests changed')
            gate = KnowledgeAgent(enabled=True, database=database, collection=settings['collection'],
                                  journal=ProgressiveMemoryJournal(directory / 'knowledge.jsonl'))
        finally:
            database.close()
        return lambda text: [dataclasses.asdict(row) for row in gate.query(KnowledgeQuery(query=text))]
    from benchmarks.a3kernels.live_composition import local_knowledge_factory
    from benchmarks.a3_experiments import KnowledgeMode
    from benchmarks.a3kernels.knowledge_agent import KnowledgeQuery
    from types import SimpleNamespace
    gate = local_knowledge_factory(SimpleNamespace(**{key: Path(value) for key, value in settings.items()}))(
        SimpleNamespace(knowledge=KnowledgeMode.WITH_KDB), SimpleNamespace(root=directory))
    return lambda text: [dataclasses.asdict(row) for row in gate.query(KnowledgeQuery(query=text))]


def run_worker(root, job_id, config, *, agent=None):
    from .agent import OptimizerAgent
    from .loop import OptimizationLoop
    from .provider import BoundedAgent
    from benchmarks.a3_experiments import BackendModel
    store = JobStore(root)
    directory = store._directory(job_id)
    device_lock = Path(config.get('lock_directory', Path(config['validation_wrapper']).parent / '.optimizer-locks')) / f'{config["target"]}-{config["physical_device"]}.lock'
    with exclusive(directory / 'worker.lock'), exclusive(device_lock, wait=True,
                              stopped=lambda: store.get(job_id)['stop_requested']):
        if config.get('env_file'):
            from dotenv import load_dotenv
            load_dotenv(config['env_file'], override=False)
        guide = Path(config['guide']).read_text()
        identity = {key: config[key] for key in ('target', 'profile', 'runtime', 'arch', 'physical_device', 'remote_workspace')}
        identity['guide_sha256'] = sha(guide.encode())
        identity['max_cases'] = config.get('max_cases', 24)
        identity['knowledge'] = config.get('knowledge')
        previous = next((event['payload'] for event in store.events(job_id)
                         if event['kind'] == 'worker-configuration'), None)
        if previous is not None and previous != identity:
            raise ValueError('worker runtime, device, workspace or guide changed on resume')
        if previous is None:
            store.add_event(job_id, 'worker-configuration', identity)
        agent = agent or OptimizerAgent(guide=guide,
                                        model=BackendModel(config.get('model_profile', BackendModel.DEEPSEEK_FLASH.value)))
        backend = RemoteBackend(store, job_id, config)
        agent = BoundedAgent(agent, deadline=backend.deadline,
                             stopped=lambda: store.get(job_id)['stop_requested'])
        store.add_event(job_id, 'worker-started', {'pid': os.getpid(), 'profile': config['profile'], 'device': config['physical_device']})
        try:
            if store.get(job_id)['stop_requested']:
                # A previous Stop may have lost its cancellation response. Drain
                # retained operations before the loop can mark the job stopped.
                for marker in sorted((directory / 'execution').rglob('transport.json')):
                    backend.transport.cancel_pending(marker.parent)
            return OptimizationLoop(store, job_id, agent, backend,
                                    knowledge=knowledge_query(config, directory),
                                    max_cases=config.get('max_cases', 24)).run()
        except WorkerStopped:
            row = store.get(job_id)
            if row['status'] in {'running', 'failed'}:
                if row['status'] == 'failed':
                    # Loop may classify backend cancellation before worker observes it.
                    store.add_event(job_id, 'worker-stopped', {'stop_requested': row['stop_requested']})
                else:
                    store.transition(job_id, 'stopped')
            return store.get(job_id)
        except ObservationUnavailable as exc:
            if store.get(job_id)['status'] == 'running':
                store.transition(job_id, 'running', stage='observation-unavailable')
            store.add_event(job_id, 'observation-unavailable', {'error': str(exc)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_worker(args.root, args.job_id, load_config(args.config))))
    except (ObservationUnavailable, WorkerStopped, BlockingIOError):
        # Recovery boundaries and a duplicate worker losing its lock are not
        # startup failures and must not overwrite the active job's status.
        raise
    except Exception as exc:
        store = JobStore(args.root)
        try:
            with exclusive(store._directory(args.job_id) / 'worker.lock'):
                store.add_event(args.job_id, 'worker-startup-error',
                                {'type': type(exc).__name__, 'message': str(exc)})
                if store.get(args.job_id)['status'] in {'queued', 'running'}:
                    store.transition(args.job_id, 'failed', stage='worker-startup-error')
        except (BlockingIOError, OSError, ValueError):
            # Another worker may already have claimed the job, or the supplied
            # identity may not exist. Preserve the original stderr traceback.
            pass
        raise


if __name__ == '__main__':
    main()
