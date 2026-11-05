"""Portable, inert campaign/project exports from ledger and explicit manifests.

Optional package-inputs.json: files [{path, kind, project?, revision?}], git [{path,
kind, project?}]. Kinds: source-history, skills, memory,
cited-entry, knowledge-history, machinery, dependencies. Campaigns additionally
snapshot knowledge.sqlite. No directory discovery outside project sources.
Writers must be stopped before packing; runtime bindings must be supplied anew.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile

from .storage_archive import compress_file, restore_file, iter_artifact_bytes

PORTABLE_TABLES = ('storage_campaign', 'projects', 'attempts', 'events',
    'storage_attempts', 'storage_iterations', 'storage_artifacts', 'storage_revisions',
    'storage_deployments', 'storage_jobs', 'recovery_queue', 'recovery_attempts',
    'recovery_lessons', 'generation_drafts', 'generation_events', 'requests', 'control')


def _json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _safe(path):
    path = Path(path)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Expected portable relative path')
    return path


def _excluded(path):
    return any(p in {'.git', 'node_modules', '.venv', '__pycache__', 'services',
                     'credentials', 'runtimes', '.ssh'} or p.startswith('.env')
               or p.endswith(('.pem', '.key')) for p in Path(path).parts)


def _local(root, name):
    path = root / _safe(name)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Input escapes campaign')
    return path


def _files(directory):
    # scandir iterates one directory at a time; never follows directory symlinks.
    with os.scandir(directory) as entries:
        for entry in entries:
            if _excluded(entry.name):
                continue
            if entry.is_symlink():
                raise ValueError('Source symlinks require explicit collected files')
            if entry.is_dir(follow_symlinks=False):
                yield from _files(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path)


def _repository_files(directory):
    for path in _files(directory):
        # Local transport/credential config and executable hooks are not history.
        if path.relative_to(directory).parts[0] not in ('config', 'hooks') and not path.name.endswith('.lock'):
            yield path


def _connect(path):
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def _tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _sanitize(value):
    # Runtime routes and process ownership are intentionally not portable.
    blocked = {'owner', 'session', 'environment', 'runtime', 'route', 'services',
               'token', 'api_key', 'password', 'credentials', 'baselineKDB'}
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items() if k not in blocked}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, str) and value.startswith(('{', '[')):
        try:
            return json.dumps(_sanitize(json.loads(value)))
        except ValueError:
            pass
    return value


def pack(root, dest, project=None, require_complete=True):
    """Publish package.json only after all collected chunks verify.

    Reusing an unfinished destination resumes completed compression chunks.
    Missing inputs are recorded; require_complete refuses publication.
    """
    root, dest = Path(root).resolve(), Path(dest).resolve()
    if dest == root or root.is_relative_to(dest) or dest.is_relative_to(root):
        raise ValueError('Package must be separate from campaign')
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'package.json').exists():
        raise FileExistsError('Package already published')
    identity = {'root': str(root), 'project': project}
    progress = dest / 'packing.json'
    if progress.exists() and json.loads(progress.read_text()) != identity:
        raise ValueError('Destination belongs to another export')
    _json(progress, identity)
    chunks = dest / 'chunks'
    chunks.mkdir(exist_ok=True)
    metadata = {'format': 1, 'scope': 'project' if project else 'campaign',
                'project': project, 'files': [], 'missing': [], 'paths': {}, 'repositories': [],
                'paused': True, 'binding_required': True}
    config = root / 'instance.json'
    metadata['config'] = _sanitize(json.loads(config.read_text())) if config.exists() else {'format': 1}
    included = set()

    def missing(kind, path):
        metadata['missing'].append({'kind': kind, 'path': str(path)})

    def collect(path, target, kind, expected=None):
        target = str(_safe(target))
        if _excluded(target):
            missing(kind, target)
            return
        if target in included:
            prior = next(f for f in metadata['files'] if f['path'] == target)
            if expected and prior['archive']['sha256'] != expected:
                raise ValueError('Ledger checksum mismatch: ' + target)
            return
        if not path.is_file() or path.is_symlink():
            missing(kind, target)
            return
        archive = compress_file(path, chunks)
        if expected and archive['sha256'] != expected:
            raise ValueError('Ledger checksum mismatch: ' + target)
        for _ in iter_artifact_bytes(archive):
            pass
        portable = {k: archive[k] for k in ('version', 'state', 'chunks', 'sha256')}
        portable['source_identity'] = {'size': archive['source_identity']['size']}
        metadata['files'].append({'path': target, 'kind': kind, 'mode': path.stat().st_mode & 0o777, 'archive': portable})
        metadata['paths'][str(path)] = target
        included.add(target)

    with closing(_connect(root / 'state.sqlite')) as db:
        db.execute('BEGIN')
        tables = _tables(db)
        if 'storage_campaign' not in tables:
            raise ValueError('Campaign storage ledger required')
        projects = [dict(r) for r in db.execute('SELECT * FROM projects WHERE (? IS NULL OR id=?)', (project, project))]
        if project and not projects:
            raise ValueError('Unknown project')
        ids = {p['id'] for p in projects}
        attempts = [dict(r) for r in db.execute('SELECT * FROM attempts') if r['project'] in ids]
        aids = {a['id'] for a in attempts}
        metadata['ledger'] = {}
        metadata['schema'] = {}
        for table in PORTABLE_TABLES:
            if table not in tables or (project and table in ('control', 'requests', 'generation_drafts', 'generation_events', 'storage_jobs')):
                continue
            metadata['schema'][table] = [dict(r) for r in db.execute('PRAGMA table_info(' + table + ')')]
            rows = []
            for row in db.execute('SELECT * FROM ' + table):
                r = dict(row)
                if table == 'control' and r['key'] not in ('campaign-budget', 'wave', 'fresh-since-recovery'):
                    continue
                if 'project' in r and r['project'] is not None and r['project'] not in ids:
                    continue
                if table == 'projects' and r['id'] not in ids:
                    continue
                if table == 'attempts' and r['id'] not in aids:
                    continue
                if table != 'storage_artifacts' and 'attempt' in r and r['attempt'] not in aids:
                    if r['attempt'] is not None or project:
                        continue
                if table == 'storage_artifacts' and r['project'] not in ids and r['attempt'] not in aids and project:
                    continue
                if table == 'events':
                    r['data'] = json.dumps(_sanitize(json.loads(r['data'])))
                rows.append(_sanitize(r))
            metadata['ledger'][table] = rows
        for p in metadata['ledger']['projects']:
            spec = json.loads(p['spec'])
            source = Path(spec['source'])
            if not source.is_absolute():
                source = _local(root, source)
            if not source.resolve().is_relative_to(root):
                missing('source', p['id'])
            elif source.is_dir():
                metadata['paths'][str(source)] = str(source.relative_to(root))
                for path in _files(source):
                    collect(path, path.relative_to(root), 'source')
            else:
                missing('source', p['id'])
            for name in spec.get('verifier', {}).get('files', []):
                path = Path(name) if Path(name).is_absolute() else _local(root, name)
                if path.resolve().is_relative_to(root):
                    collect(path, path.relative_to(root), 'verifier', spec['verifier'].get('hashes', {}).get(str(path)))
                else:
                    missing('verifier', name)
            p['spec'] = json.dumps(_sanitize(spec))
        for row in metadata['ledger'].get('storage_artifacts', []):
            if row['kind'] == 'source-git' and not row['external']:
                repository = _local(root, row['path'])
                if not repository.is_dir() or not (repository / 'objects').is_dir():
                    missing('source-git', row['path'])
                    continue
                metadata['repositories'].append({'path': row['path'], 'kind': 'source-git'})
                for path in _repository_files(repository):
                    collect(path, path.relative_to(root), 'source-git')
                continue
            if row['external'] or row['status'] not in ('closed', 'archived'):
                missing(row['kind'], row['id'])
                continue
            path = _local(root, row['path'])
            if row.get('encoding', 'plain') != 'plain':
                # Restore through the archive reader, never recompress encoded bytes as plain.
                descriptor = json.loads(row['descriptor'])
                if isinstance(descriptor, str) or 'manifest' in descriptor:
                    name = descriptor if isinstance(descriptor, str) else descriptor['manifest']
                    name = Path(name)
                    if name.is_absolute():
                        name = name.relative_to(root)
                    descriptor = json.loads(_local(root, name).read_text())
                staged = dest / 'collected' / row['id']
                staged.parent.mkdir(exist_ok=True)
                if not staged.exists():
                    restore_file(descriptor, staged)
                path = staged
            collect(path, row['path'], row['kind'], row.get('sha256'))
            metadata['paths'][str(root / row['path'])] = row['path']

    manifest = root / 'package-inputs.json'
    inputs = json.loads(manifest.read_text()) if manifest.exists() else {}
    kinds = {'source-history' if f['kind'] == 'source-git' else f['kind'] for f in metadata['files']}
    covered = {'skills': set(), 'memory': set()}
    used = {kind: {r[field] for r in metadata['ledger'].get('storage_attempts', []) if r.get(field)}
            for kind, field in (('skills', 'skill_revision'), ('memory', 'memory_revision'))}
    for item in inputs.get('files', []):
        if project and (item.get('project') not in (None, project) or
                        item['kind'] in ('knowledge-history', 'machinery', 'dependencies', 'active-kdb')):
            continue
        path = _local(root, item['path'])
        collect(path, item['path'], item['kind'], item.get('sha256'))
        if str(item['path']) in included:
            kinds.add(item['kind'])
            if item['kind'] in covered and item.get('revision'):
                covered[item['kind']].add(item['revision'])
    for item in inputs.get('git', []):
        if project and (item.get('project') not in (None, project) or
                        item['kind'] in ('knowledge-history', 'machinery', 'dependencies', 'active-kdb')):
            continue
        repo = _local(root, item['path'])
        if not repo.is_dir():
            missing(item['kind'], item['path'])
            continue
        git_dir = Path(subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', '--absolute-git-dir'], text=True).strip()).resolve()
        if not git_dir.is_relative_to(root):
            missing(item['kind'], item['path'])
            continue
        name = 'history/' + hashlib.sha256(str(item).encode()).hexdigest()
        metadata['repositories'].append({'path': name, 'kind': item['kind']})
        for path in _repository_files(git_dir):
            collect(path, str(Path(name) / path.relative_to(git_dir)), item['kind'])
        kinds.add(item['kind'])
        if item['kind'] in covered and item.get('revision'):
            covered[item['kind']].add(item['revision'])
    for kind in covered:
        for revision in sorted(used[kind] - covered[kind]):
            missing(kind + '-revision', revision)
    required = {'source-history', 'skills', 'memory', 'cited-entry'}
    if not project:
        required |= {'knowledge-history', 'machinery', 'dependencies'}
        snapshot = dest / 'knowledge.sqlite'
        if not snapshot.exists() and (root / 'knowledge.sqlite').is_file():
            with closing(_connect(root / 'knowledge.sqlite')) as source, closing(sqlite3.connect(snapshot)) as out:
                source.backup(out)
        collect(snapshot, 'knowledge.sqlite', 'active-kdb')
    for kind in sorted(required - kinds):
        missing(kind, 'package-inputs.json')
    metadata['complete'] = not metadata['missing']
    _json(dest / 'inspection.json', metadata)
    if require_complete and not metadata['complete']:
        raise ValueError('Package incomplete; inspect inspection.json')
    _json(dest / 'package.json', metadata)
    # Sidecars are local resume machinery, not portable package content.
    for sidecar in chunks.iterdir():
        if sidecar.suffix != '.gz':
            sidecar.unlink()
    for scratch in ('collected',):
        shutil.rmtree(dest / scratch, ignore_errors=True)
    (dest / 'knowledge.sqlite').unlink(missing_ok=True)
    progress.unlink()
    return metadata


def _rebase(value, paths, dest):
    if isinstance(value, str):
        for old in sorted(paths, key=len, reverse=True):
            value = value.replace(old, str(dest / paths[old]))
        return value
    if isinstance(value, list):
        return [_rebase(v, paths, dest) for v in value]
    if isinstance(value, dict):
        return {_rebase(k, paths, dest): _rebase(v, paths, dest) for k, v in value.items()}
    return value


def unpack(package, dest):
    """Verify into a fresh directory, restore paused records, never execute code."""
    package, dest = Path(package).resolve(), Path(dest).absolute()
    if dest.exists():
        raise FileExistsError(dest)
    metadata = json.loads((package / 'package.json').read_text())
    if metadata.get('format') != 1:
        raise ValueError('Unsupported package')
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.unpack-', dir=dest.parent))
    try:
        for item in metadata['files']:
            path = stage / _safe(item['path'])
            if _excluded(item['path']):
                raise ValueError('Excluded package input')
            path.parent.mkdir(parents=True, exist_ok=True)
            archive = {**item['archive'], 'directory': str(package / 'chunks')}
            for chunk in archive['chunks']:
                _safe(chunk['file'])
            restore_file(archive, path)
            path.chmod(item.get('mode', 0o600) & 0o777)
            if item['kind'] == 'source-git' and path.name == 'rsi-source.json':
                _json(path, _rebase(json.loads(path.read_text()), metadata['paths'], dest))
        for repo in metadata.get('repositories', []):
            directory = stage / _safe(repo['path'])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'objects').mkdir(exist_ok=True)
            (directory / 'refs').mkdir(exist_ok=True)
            bare = 'false' if repo['kind'] == 'source-git' else 'true'
            (directory / 'config').write_text('[core]\n\tbare = ' + bare + '\n\tlogAllRefUpdates = true\n')
        # Construct only portable tables; execution leases/control/services never migrate.
        with closing(sqlite3.connect(stage / 'state.sqlite')) as db, db:
            allowed = set(PORTABLE_TABLES)
            for table, rows in metadata['ledger'].items():
                if table not in allowed:
                    raise ValueError('Unsupported ledger table')
                columns = metadata['schema'][table]
                def identifier(name):
                    if not name.replace('_', '').isalnum():
                        raise ValueError('Invalid column')
                    return '"' + name + '"'
                definitions = []
                for c in columns:
                    kind = c['type'].upper()
                    if kind not in ('TEXT', 'INTEGER', 'REAL', ''):
                        raise ValueError('Unsupported column type')
                    definitions.append(identifier(c['name']) + ' ' + kind +
                                       (' PRIMARY KEY' if c['pk'] else '') +
                                       (' NOT NULL' if c['notnull'] else '') +
                                       (' DEFAULT ' + str(c['dflt_value']) if c['dflt_value'] is not None else ''))
                db.execute('CREATE TABLE "' + table + '" (' + ','.join(definitions) + ')')
                for original in rows:
                    row = _rebase(dict(original), metadata['paths'], dest)
                    if table == 'attempts':
                        row.update(owner=None, session=None)
                    if table == 'requests':
                        row['owner'] = None
                        if row['status'] in ('queued', 'running'):
                            row['status'] = 'paused'
                    if table == 'storage_jobs' and row['status'] != 'complete':
                        row['status'] = 'paused'
                    if table == 'projects':
                        row['spec'] = json.dumps(_rebase(json.loads(row['spec']), metadata['paths'], dest))
                        row['frozen'] = 1
                    if table == 'storage_artifacts':
                        row['encoding'], row['descriptor'] = 'plain', None
                    columns = list(row)
                    db.execute('INSERT INTO "' + table + '" (' + ','.join('"' + k + '"' for k in columns) + ') VALUES (' + ','.join('?' for _ in columns) + ')', list(row.values()))
            db.execute('CREATE UNIQUE INDEX active_attempt ON attempts(project) WHERE outcome IS NULL')
            db.execute('PRAGMA user_version=1')
        result = {'format': 1, 'paused': True, 'binding_required': True,
                  'storage': metadata['ledger'].get('storage_campaign', []),
                  'complete': metadata['complete'], 'missing': metadata['missing']}
        _json(stage / 'restore.json', result)
        config = _rebase(metadata['config'], metadata['paths'], dest)
        config.update(format=1, paused=True, binding_required=True, environment=None,
                      projects=[_rebase(json.loads(p['spec']), metadata['paths'], dest)
                                for p in metadata['ledger']['projects']])
        _json(stage / 'instance.json', config)
        # Exclusive reservation refuses even an empty destination created during restore.
        dest.mkdir()
        try:
            os.rename(stage, dest)
        except BaseException:
            dest.rmdir()
            raise
        return result
    except BaseException:
        shutil.rmtree(stage)
        raise
