"""Read-only subject campaign projections and incremental actor logs."""
import base64
import hashlib
import fcntl
import json
from pathlib import Path

from flask import abort, jsonify, render_template, request, Response

from .store import _json, _source


def _inside(root, path):
    return path.resolve().is_relative_to(root.resolve())


class SubjectViews:
    def __init__(self, config_path):
        config = _json(Path(config_path), {})
        self.campaigns = {row['id']: row for row in config.get('subject_campaigns', [])}

    def campaign(self, key):
        config = self.campaigns[key]
        root = Path(config['root'])
        state = _json(root / 'subject-campaign.json')
        subjects = _json(root / 'subjects.json')
        if not isinstance(state, dict) or not isinstance(subjects, dict):
            raise ValueError('Subject campaign artifacts are unavailable')
        pending = state.get('pending') or {}
        rows = dict(state.get('projects', {}))
        for plan in pending.get('items', []):
            rows.setdefault(plan['project_id'], dict(plan=plan))
        active = _json(root / 'memory/active.json', {})
        published_projects = {memory.get('provenance', {}).get('project_id')
            for version in active.get('active', {}).values()
            if (memory := _json(root / 'memory/versions' / (version + '.json'), {}))}
        jobs = {_json(path, {}).get('review_id'): _json(path, {})
                for path in (root / 'memory-reviews').glob('*/job.json')}
        waves = subjects.get('waves', {})
        running = False
        lock_path = root / '.campaign.lock'
        if lock_path.exists():
            with lock_path.open('rb') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    running = True
        projects = []
        for pid, row in rows.items():
            plan = row.get('plan', {})
            ordinal = row.get('wave_index', 0)
            completed = state.get('waves', [])
            wave = completed[ordinal - 1] if 0 < ordinal <= len(completed) else None
            if pid in pending.get('projects', []):
                wave = pending['wave_id']
            outcome = row.get('outcome') or {}
            review = row.get('review') or {}
            review_id = row.get('manual_review_id', pid)
            published = pid in active.get('admissions', {}) or review_id in active.get('admissions', {}) or pid in published_projects
            directory = root / row['directory'] if row.get('directory') else root / 'waves' / str(wave) / pid
            started = directory.is_dir() and _inside(root, directory)
            execution = 'superseded' if row.get('superseded_by') else ('verified' if outcome.get('host_verified') else 'failed') if outcome else (
                'running' if started and running else 'stopped' if started else 'pending')
            projects.append(dict(id=pid, project_id=pid, attempt_id=_json(directory / "manual-execution.json", {}).get("attempt_id", pid), wave_id=wave,
                subject_id=row.get('subject_id') or waves.get(wave, {}).get('subject_id', pending.get('subject', {}).get('subject_id')),
                kind=plan.get('task', {}).get('kind', plan.get('kind', 'graph')),
                objective=row.get('objective', plan.get('spec', {}).get('hypothesis', '')),
                execution=execution, project_review=review.get('decision', 'pending'),
                memory_admission=('manually-accepted' if review_id in active.get('manual_overrides', {})
                                  else jobs.get(review_id, {}).get('status', 'not-submitted')),
                publication='published' if published else 'unpublished', merged=published,
                directory=row.get('directory'), restart_of=row.get('restart_of'), superseded_by=row.get('superseded_by'),
                plan=plan, review=review, outcome=outcome,
                memory=review.get('memory'), source=_source(root, outcome.get('source_sha256')) or ''))
        return dict(id=key, name=config.get('name', key), target=config.get('target'),
                    terminal=state.get('terminal'), subjects=subjects.get('subjects', []),
                    projects=projects, pending=pending, project_queue=subjects.get('project_queue', []),
                    stats=dict(projects=len(projects),
                    verified=sum(p['execution'] == 'verified' for p in projects),
                    published=sum(p['merged'] for p in projects)))

    def summaries(self):
        result = []
        for key in self.campaigns:
            try:
                result.append(self.campaign(key))
            except (ValueError, OSError) as error:
                result.append(dict(id=key, name=self.campaigns[key].get('name', key), error=str(error)))
        return result

    def project(self, key, pid):
        campaign = self.campaign(key)
        return campaign, next(row for row in campaign['projects'] if row['id'] == pid)

    def streams(self, key, pid):
        _, project = self.project(key, pid)
        root = Path(self.campaigns[key]['root'])
        directory = root / project['directory'] if project.get('directory') else root / 'waves' / str(project['wave_id']) / pid
        paths = [*directory.glob('**/sft-traces/*.jsonl'), *directory.glob('**/evidence.jsonl'),
                 *directory.glob('**/jobs/*/events.jsonl')]
        return root, {p.relative_to(root).as_posix(): p for p in sorted(paths)
                      if p.is_file() and _inside(root, p)}

    @staticmethod
    def _payload(root, stream, row):
        # Provider/optimizer ledgers refer to CAS bytes rather than inline messages.
        payload = row.get('payload', {})
        digest = payload.get('sha256') if isinstance(payload, dict) else None
        if isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest):
            branch = root / stream
            for directory in (branch.parent, branch.parent.parent, root):
                content = _source(directory, digest)
                optimizer_blob = directory / 'artifacts' / digest
                if content is None and optimizer_blob.is_file() and _inside(root, optimizer_blob):
                    data = optimizer_blob.read_bytes()
                    if hashlib.sha256(data).hexdigest() == digest:
                        content = data.decode('utf-8', errors='replace')
                if content is not None:
                    try:
                        row = dict(row, retained_payload=json.loads(content))
                    except ValueError:
                        row = dict(row, retained_payload=content)
                    break
        # Traces already redact credentials; apply the same rule to other ledgers.
        from benchmarks.a5kernels.sft_dataset import _redact
        return _redact(row)

    def logs(self, key, pid, cursor=None, limit=100):
        if not 1 <= limit <= 500:
            raise ValueError('limit must be between 1 and 500')
        positions = {} if cursor in (None, 'start') else json.loads(base64.urlsafe_b64decode(cursor))
        if not isinstance(positions, dict):
            raise ValueError('invalid log cursor')
        root, streams = self.streams(key, pid)
        events, reset, earlier = [], False, False
        for name, path in streams.items():
            with path.open('rb') as handle:
                size = path.stat().st_size
                old = positions.get(name)
                if old is not None:
                    offset, anchor = old
                    if type(offset) is not int or offset < 0:
                        raise ValueError('invalid log offset')
                    handle.seek(max(0, offset - 64))
                    actual = hashlib.sha256(handle.read(min(offset, 64))).hexdigest()
                    if offset > size or actual != anchor:
                        offset, reset = 0, True
                elif cursor is None and size > 65536:
                    handle.seek(size - 65536)
                    handle.readline()
                    offset, earlier = handle.tell(), True
                else:
                    offset = 0
                handle.seek(offset)
                while len(events) < limit:
                    start = handle.tell()
                    line = handle.readline(1048577)
                    if not line.endswith(b'\n'):
                        handle.seek(start)
                        break
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        row = dict(kind='unreadable', payload='Invalid log record')
                    events.append(dict(id=f'{name}:{start}', stream=name, offset=start,
                                       event=self._payload(root, name, row)))
                offset = handle.tell()
                handle.seek(max(0, offset - 64))
                positions[name] = [offset, hashlib.sha256(handle.read(min(offset, 64))).hexdigest()]
        token = base64.urlsafe_b64encode(json.dumps(positions, separators=(',', ':')).encode()).decode()
        return dict(events=events, cursor=token, reset=reset, earlier=earlier)


def register_subjects(app, config_path):
    views = SubjectViews(config_path)
    app.extensions['dashboard_subjects'] = views

    def lookup(key, pid=None):
        try:
            return views.campaign(key) if pid is None else views.project(key, pid)
        except (KeyError, StopIteration):
            abort(404)
        except (ValueError, OSError):
            abort(503)

    @app.context_processor
    def subject_context():
        return dict(subject_campaigns=views.summaries())

    @app.get('/campaign/<campaign_id>')
    def subject_campaign_page(campaign_id):
        return render_template('subject_campaign.html', campaign=lookup(campaign_id))

    @app.get('/campaign/<campaign_id>/project/<project_id>')
    def subject_project_page(campaign_id, project_id):
        campaign, project = lookup(campaign_id, project_id)
        return render_template('subject_project.html', campaign=campaign, project=project)

    @app.get('/api/campaigns/<campaign_id>')
    def subject_campaign_api(campaign_id):
        return jsonify(lookup(campaign_id))

    @app.get('/api/campaigns/<campaign_id>/projects/<project_id>/logs')
    def subject_logs(campaign_id, project_id):
        lookup(campaign_id, project_id)
        try:
            return jsonify(views.logs(campaign_id, project_id, request.args.get('cursor'),
                                      int(request.args.get('limit', 100))))
        except (ValueError, TypeError, OSError):
            return jsonify(error='Invalid or unavailable log cursor'), 400

    @app.get('/api/campaigns/<campaign_id>/projects/<project_id>/logs/download')
    def subject_log_download(campaign_id, project_id):
        lookup(campaign_id, project_id)
        def records():
            root, streams = views.streams(campaign_id, project_id)
            for name, path in streams.items():
                size = path.stat().st_size
                with path.open('rb') as handle:
                    while handle.tell() < size:
                        offset = handle.tell()
                        line = handle.readline(size - offset)
                        if not line.endswith(b'\n'):
                            break
                        try:
                            row = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            row = dict(kind='unreadable', payload='Invalid log record')
                        yield json.dumps(dict(stream=name, offset=offset,
                            event=views._payload(root, name, row))) + '\n'

        return Response(records(), mimetype='application/x-ndjson', headers={
            'Content-Disposition': 'attachment; filename="actor-logs.jsonl"'})
