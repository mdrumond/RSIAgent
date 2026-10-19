"""Optimizer HTTP adapter. Python submissions are only executed by detached workers."""
import hashlib
import fcntl
import json
from pathlib import Path
import secrets
import subprocess
import sys

from flask import abort, jsonify, render_template, request

from optimizer.jobs import JobStore, atomic, canonical
from optimizer.memory import discover_sources, freeze_source


class OptimizerDashboard:
    def __init__(self, config_path, launcher=None):
        self.settings = json.loads(Path(config_path).read_text()).get('optimizer', {})
        self.enabled = self.settings.get('enabled') is True
        self.reason = 'Configure optimizer.enabled, root, memory_roots and worker_config to start jobs.'
        self.launcher = launcher or subprocess.Popen
        self.store = None
        if self.enabled:
            root = Path(self.settings.get('root', ''))
            worker = Path(self.settings.get('worker_config', ''))
            if not root.is_absolute() or not worker.is_absolute() or not worker.is_file():
                self.enabled = False
                self.reason = 'Configure absolute optimizer root and an existing absolute worker_config.'
            else:
                self.store = JobStore(root)

    def sources(self):
        rows = discover_sources(self.settings.get('memory_roots', []),
                                snapshot_roots=self.settings.get('snapshot_roots', []))
        return sorted(rows, key=lambda row: Path(row['root']).stat().st_mtime
                      if Path(row['root']).exists() else 0, reverse=True)

    def launch(self, job_id):
        try:
            with (self.store._directory(job_id) / 'worker.log').open('ab') as log:
                self.launcher([sys.executable, '-m', 'optimizer.worker', '--root',
                               str(self.store.root), '--job-id', job_id, '--config',
                               self.settings['worker_config']], stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              cwd=str(Path(__file__).resolve().parents[1]))
        except OSError as error:
            self.store.add_event(job_id, 'worker-launch-failed', {'error': str(error)})
            self.store.transition(job_id, 'failed', stage='worker-launch-failed')

    def worker_active(self, job_id):
        """The detached worker owns this lock even while awaiting its device."""
        path = self.store._directory(job_id) / 'worker.lock'
        try:
            with path.open('rb') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                return False
        except FileNotFoundError:
            return False

    def action(self, action, job_id, data, key):
        if not key or len(key) > 200:
            raise ValueError('Idempotency-Key is required (maximum 200 characters)')
        identity = hashlib.sha256(key.encode()).hexdigest()
        receipt = self.store.root / ('action-' + identity + '.json')
        fingerprint = hashlib.sha256(canonical([action, job_id, data])).hexdigest()
        with self.store._lock(self.store.root):
            if receipt.exists():
                previous = json.loads(receipt.read_bytes())
                if previous['fingerprint'] != fingerprint:
                    raise ValueError('idempotency key reused with different action')
                return self.store.get(previous['job_id'])
            # Creation takes its own store lock, so release it before creating.
            if action != 'create':
                if action == 'stop':
                    job = self.store.stop(job_id)
                elif action == 'recover':
                    job = self.store.get(job_id)
                    if (job['status'] != 'running' or job['stage'] != 'observation-unavailable'
                            or self.worker_active(job_id)):
                        raise ValueError('recover requires an inactive worker awaiting observation')
                    self.launch(job_id)
                else:
                    job = self.store.resume(job_id,
                        additional_attempts=data.get('additional_attempts'),
                        additional_seconds=data.get('additional_seconds'))
                    self.launch(job_id)
                atomic(receipt, canonical({'fingerprint': fingerprint, 'job_id': job_id}))
                return self.store.get(job_id)
        source = data.get('source', '')
        root = data.get('memory_root')
        if root not in self.settings.get('memory_roots', []):
            raise ValueError('select a configured completed memory source')
        if not isinstance(data.get('signature'), list) or not data['signature']:
            raise ValueError('provide at least one representative input signature')
        if not isinstance(data.get('init_args', []), list) or not isinstance(data.get('init_kwargs', {}), dict):
            raise ValueError('constructor arguments must be a JSON array and object')
        snapshot = freeze_source(root, snapshot_roots=self.settings.get('snapshot_roots', []))
        job = self.store.create(source, data.get('entrypoint'),
            {'signatures': data['signature']}, snapshot,
            name=data.get('name', ''), init_args=data.get('init_args'),
            init_kwargs=data.get('init_kwargs'), idempotency_key=key,
            max_attempts=data.get('max_attempts', 25), max_seconds=data.get('max_seconds', 7200))
        with self.store._lock(self.store.root):
            if not receipt.exists():
                self.launch(job['job_id'])
                atomic(receipt, canonical({'fingerprint': fingerprint, 'job_id': job['job_id']}))
        return self.store.get(job['job_id'])


def register_optimizer(app, config_path, csrf_token, launcher=None):
    manager = OptimizerDashboard(config_path, launcher)
    app.extensions['optimizer'] = manager
    from .optimizer_results import register_results
    register_results(app, manager)

    @app.get('/optimizer')
    def optimizer_page():
        return render_template('optimizer.html', optimizer=manager, csrf_token=csrf_token,
                               jobs=manager.store.list() if manager.enabled else [])

    @app.get('/optimizer/<job_id>')
    def optimizer_job(job_id):
        if not manager.enabled:
            abort(404)
        try:
            job = manager.store.get(job_id)
        except (ValueError, OSError):
            abort(404)
        return render_template('optimizer_job.html', job=job, csrf_token=csrf_token)

    @app.get('/api/optimizer/sources')
    def optimizer_sources():
        return jsonify(manager.sources() if manager.enabled else [])

    @app.get('/api/optimizer/jobs')
    @app.get('/api/optimizer/jobs/<job_id>')
    @app.get('/api/optimizer/jobs/<job_id>/events')
    @app.get('/api/optimizer/jobs/<job_id>/events/<int:sequence>')
    @app.get('/api/optimizer/jobs/<job_id>/submission')
    def optimizer_read(job_id=None, sequence=None):
        if not manager.enabled:
            abort(403)
        try:
            if job_id is None:
                result = manager.store.list()
            elif request.path.endswith('/submission'):
                result = manager.store.submission(job_id)
                snapshot = result['memory_snapshot']
                result['memory_snapshot'] = {key: value for key, value in snapshot.items()
                                             if key not in {'files', 'memory_entries'}}
                result['memory_snapshot']['memory_count'] = len(snapshot.get('memory_entries', []))
            elif '/events' not in request.path:
                result = manager.store.get(job_id)
                result['worker_active'] = manager.worker_active(job_id)
            else:
                manager.store.get(job_id)
                events = manager.store.events(job_id)
                result = events[sequence] if sequence is not None else [
                    {k: row[k] for k in ('sequence', 'timestamp', 'kind', 'sha256')}
                    for row in events]
            return jsonify(result)
        except (OSError, ValueError, IndexError):
            abort(404)

    @app.post('/api/optimizer/jobs')
    @app.post('/api/optimizer/jobs/<job_id>/<action>')
    def optimizer_action(job_id=None, action='create'):
        if not manager.enabled or not secrets.compare_digest(
                request.headers.get('X-CSRF-Token', ''), csrf_token):
            abort(403)
        if action not in {'create', 'stop', 'resume', 'recover'}:
            abort(404)
        try:
            data = request.get_json(silent=True) or request.form.to_dict()
            if not isinstance(data, dict):
                raise ValueError('request must be an object')
            if request.files.get('source_file'):
                data['source'] = request.files['source_file'].read().decode('utf-8')
            if request.form:
                for name in ('signature', 'init_args', 'init_kwargs', 'max_attempts', 'max_seconds'):
                    if name in data:
                        data[name] = json.loads(data[name])
            job = manager.action(action, job_id, data, request.headers.get('Idempotency-Key'))
            return jsonify(job), 202
        except (ValueError, TypeError, KeyError, OSError) as error:
            return jsonify(error=str(error)), 409
