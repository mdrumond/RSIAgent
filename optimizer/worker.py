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
        self.transport = transport or CheckedTransport(config, retain=self._retain, stopped=self._stopped)

    def _retain(self, handle):
        self.store.retain_handle(self.job_id, handle, operation='optimizer')

    def _stopped(self):
        return self.store.get(self.job_id)['stop_requested'] or time.monotonic() >= self.deadline

    def _operation(self, operation, directory, **payload):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / 'result.json'
        if result_path.exists():
            return json.loads(result_path.read_text())
        relative = directory.relative_to(self.store.root / self.job_id / 'execution')
        remote_directory = self.remote / 'execution' / relative
        request = {'operation': operation, 'directory': str(remote_directory),
                   'baseline': str(self.remote / 'execution' / 'baseline' / 'frozen'),
                   'arch': self.config['arch'], **payload}
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
        try:
            self.transport.execute(directory, ['bash', '-c', command])
        except RuntimeError as exc:
            if isinstance(exc, (ObservationUnavailable, WorkerStopped)):
                raise
            # Terminal compiler/reference failures still have compact evidence.
            self.store.add_event(self.job_id, 'remote-operation-error', {'operation': operation, 'error': str(exc)})
        if (directory / 'transport.log').exists():
            self.store.put_artifact(self.job_id, str(relative / 'transport.log'), (directory / 'transport.log').read_bytes())
        compact = directory / 'compact.tar.gz'
        self.transport.transfer('download', remote_directory / 'compact.tar.gz', compact)
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

    def profile(self, source, directory, *, block_count=1):
        return self._operation('profile', directory, source=source, block_count=block_count)


def knowledge_query(config, directory):
    if not config.get('knowledge'):
        return None
    from benchmarks.a3kernels.live_composition import local_knowledge_factory
    from benchmarks.a3_experiments import KnowledgeMode
    from benchmarks.a3kernels.knowledge_agent import KnowledgeQuery
    from types import SimpleNamespace
    settings = config['knowledge']
    gate = local_knowledge_factory(SimpleNamespace(**{key: Path(value) for key, value in settings.items()}))(
        SimpleNamespace(knowledge=KnowledgeMode.WITH_KDB), SimpleNamespace(root=directory))
    return lambda text: [dataclasses.asdict(row) for row in gate.query(KnowledgeQuery(query=text))]


def run_worker(root, job_id, config, *, agent=None):
    from .agent import OptimizerAgent
    from .loop import OptimizationLoop
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
        previous = next((event['payload'] for event in store.events(job_id)
                         if event['kind'] == 'worker-configuration'), None)
        if previous is not None and previous != identity:
            raise ValueError('worker runtime, device, workspace or guide changed on resume')
        if previous is None:
            store.add_event(job_id, 'worker-configuration', identity)
        agent = agent or OptimizerAgent(guide=guide,
                                        model=BackendModel(config.get('model_profile', BackendModel.GPT_5_6_SOL.value)))
        backend = RemoteBackend(store, job_id, config)
        store.add_event(job_id, 'worker-started', {'pid': os.getpid(), 'profile': config['profile'], 'device': config['physical_device']})
        try:
            return OptimizationLoop(store, job_id, agent, backend,
                                    knowledge=knowledge_query(config, directory)).run()
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
            store.add_event(job_id, 'observation-unavailable', {'error': str(exc)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_worker(args.root, args.job_id, load_config(args.config))))


if __name__ == '__main__':
    main()
