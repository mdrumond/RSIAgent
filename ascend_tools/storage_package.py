"""Portable, inert campaign/project exports from ledger and explicit manifests.

Optional package-inputs.json: files [{path, kind, project?, revision?}], git [{path,
kind, project?}]. Kinds: source-history, skills, memory,
cited-entry, knowledge-history, machinery, dependencies. Campaigns additionally
snapshot knowledge.sqlite. No directory discovery outside project sources.
Writers must be stopped before packing; runtime bindings must be supplied anew.
"""
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile

from .storage_archive import archive_files, restore_files
from .storage_cli import Storage

MAX_JSON_BYTES = 8 * 1024 * 1024

PORTABLE_TABLES = ('storage_campaign', 'projects', 'attempts', 'events',
    'storage_attempts', 'storage_iterations', 'storage_artifacts', 'storage_revisions',
    'storage_deployments', 'storage_jobs', 'recovery_queue', 'recovery_attempts',
    'recovery_lessons', 'generation_drafts', 'generation_events', 'requests', 'control',
    'storage_sessions', 'storage_retrievals', 'storage_skill_versions')


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
    blocked = {'owner', 'environment', 'runtime', 'route', 'services',
               'token', 'api_key', 'password', 'credentials', 'credentials_file', 'credential_env', 'baselineKDB'}
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


def _load_json(path):
    # Extension/config manifests are small control data, never a ledger transport.
    with path.open('rb') as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError('JSON control manifest exceeds portable size limit')
    return json.loads(raw)


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _snapshot(root, path, project):
    if path.exists():
        return  # Reuse only a completed, immutable SQLite business snapshot.
    temporary = path.with_suffix('.tmp')
    temporary.unlink(missing_ok=True)
    with closing(_connect(root / 'state.sqlite')) as source, closing(sqlite3.connect(temporary)) as db:
        source.backup(db, pages=256)
        db.execute('PRAGMA cache_size=-2048')
        db.execute('PRAGMA temp_store=FILE')
        # Keep table constraints/indexes; discard executable triggers and live services.
        for kind, name in db.execute("SELECT type,name FROM sqlite_master WHERE type IN ('trigger','view')").fetchall():
            db.execute('DROP ' + kind + ' ' + _quote(name))
        for table in _tables(db) - set(PORTABLE_TABLES):
            if not table.startswith('sqlite_'):
                db.execute('DROP TABLE ' + _quote(table))
        tables = _tables(db)
        if 'storage_campaign' not in tables:
            raise ValueError('Campaign storage ledger required')
        if project:
            if not db.execute('SELECT 1 FROM projects WHERE id=?', (project,)).fetchone():
                raise ValueError('Unknown project')
            db.execute('DELETE FROM projects WHERE id!=?', (project,))
            db.execute('DELETE FROM attempts WHERE project NOT IN (SELECT id FROM projects)')
            for table in tables & set(PORTABLE_TABLES) - {'projects', 'attempts', 'storage_artifacts'}:
                columns = {r[1] for r in db.execute('PRAGMA table_info(' + _quote(table) + ')')}
                if table in ('control', 'requests', 'generation_drafts', 'generation_events', 'storage_jobs'):
                    db.execute('DELETE FROM ' + _quote(table))
                elif 'project' in columns:
                    db.execute('DELETE FROM ' + _quote(table) + ' WHERE project NOT IN (SELECT id FROM projects)')
                elif 'attempt' in columns:
                    db.execute('DELETE FROM ' + _quote(table) + ' WHERE attempt IS NULL OR attempt NOT IN (SELECT id FROM attempts)')
            clause = ''
            if 'storage_retrievals' in tables:
                clause = " AND NOT EXISTS (SELECT 1 FROM storage_retrievals,json_each(citations) j WHERE json_extract(j.value,'$.artifact')=storage_artifacts.id)"
            db.execute("DELETE FROM storage_artifacts WHERE COALESCE(project IN (SELECT id FROM projects),0)=0 AND COALESCE(attempt IN (SELECT id FROM attempts),0)=0" + clause)
        if 'control' in tables:
            db.execute("DELETE FROM control WHERE key NOT IN ('campaign-budget','campaign-plan','wave','fresh-since-recovery')")
        db.create_function('portable_text', 1, _sanitize)
        for table in tables & set(PORTABLE_TABLES):
            columns = [r[1] for r in db.execute('PRAGMA table_info(' + _quote(table) + ')') if r[2].upper() == 'TEXT']
            if columns:
                db.execute('UPDATE ' + _quote(table) + ' SET ' + ','.join(_quote(c) + '=portable_text(' + _quote(c) + ')' for c in columns))
        db.execute('UPDATE attempts SET owner=NULL')
        db.execute('UPDATE projects SET frozen=1')
        if 'requests' in tables:
            db.execute("UPDATE requests SET owner=NULL,status=CASE WHEN status IN ('queued','running') THEN 'failed' ELSE status END")
        if 'control' in tables:
            db.execute("DELETE FROM control WHERE key='pack-capture'")
        db.commit()
        db.execute('VACUUM')  # Remove deleted projects and credentials from free pages.
    os.replace(temporary, path)


def _git_directory(repository):
    return Path(subprocess.check_output(
        ['git', '-C', str(repository), 'rev-parse', '--absolute-git-dir'], text=True).strip()).resolve()


def _revision_files(git_dir, revision, kind, work, collect):
    if len(revision) not in (40, 64) or any(c not in '0123456789abcdef' for c in revision):
        return False
    # NUL records streamed from Git: memory scales with one tree entry/blob block.
    with subprocess.Popen(['git', '--git-dir', str(git_dir), 'ls-tree', '-rz', revision, '--', kind + '/'],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        pending = b''
        while block := process.stdout.read(65536):
            pending += block
            while b'\0' in pending:
                record, pending = pending.split(b'\0', 1)
                header, name = record.split(b'\t', 1)
                mode, object_type, oid = header.decode('ascii').split()
                if object_type != 'blob' or mode == '120000':
                    raise ValueError('Knowledge revision requires regular collected blobs')
                relative = Path('exported') / revision / _safe(os.fsdecode(name))
                path = work / relative
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(path.suffix + '.tmp')
                    with temporary.open('wb') as out:
                        subprocess.run(['git', '--git-dir', str(git_dir), 'cat-file', 'blob', oid],
                                       stdout=out, stderr=subprocess.DEVNULL, check=True)
                    os.replace(temporary, path)
                collect(path, relative, kind)
            if len(pending) > 1024 * 1024:
                process.kill()
                raise ValueError('Oversized Git tree entry')
        return process.wait() == 0 and not pending


def _identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text()
    return dict(pid=pid, start=stat[stat.rfind(')') + 2:].split()[19],
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


@contextmanager
def _capture(root):
    """Admission contract: writers check pack-capture while claiming ownership.

    All checks and the claim share one SQLite write transaction. The claim is
    committed before copying, so snapshots can use independent connections.
    A CLI parent can delegate its claim to its direct Python child.
    """
    me = _identity(os.getpid())
    try:
        parent = _identity(os.getppid())
    except FileNotFoundError:
        parent = None
    token = json.dumps(me)
    owned = False
    with closing(sqlite3.connect(f'file:{root / "state.sqlite"}?mode=rw', uri=True)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        try:
            row = db.execute("SELECT value FROM control WHERE key='pack-capture'").fetchone()
            maintenance = db.execute("SELECT value FROM control WHERE key='storage-maintenance'").fetchone()
            delegated = (row and maintenance and json.loads(row[0]) == parent
                         and json.loads(maintenance[0]) == parent)
            if row and not delegated and not Storage.stopped(row[0]):
                raise ValueError('Pack capture already active or unresolved')
            allowed = [me] + ([parent] if delegated else [])
            for key, value in db.execute("SELECT key,value FROM control WHERE key IN ('owner','skill-writer','memory-writer','storage-maintenance','knowledge-startup')"):
                owner = json.loads(value)
                if owner is None:
                    continue
                if key == 'storage-maintenance' and owner in allowed:
                    continue
                if not Storage.stopped(owner):
                    raise ValueError('Pack requires stopped writers: ' + key)
            for owner, outcome in db.execute('SELECT owner,outcome FROM attempts'):
                if (not owner and outcome is None) or (owner and not Storage.stopped(owner)):
                    raise ValueError('Pack requires stopped attempt writers')
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='requests'").fetchone():
                for kind, owner in db.execute("SELECT kind,owner FROM requests WHERE status IN ('queued','running')"):
                    if kind == 'storage' and owner and json.loads(owner) in allowed:
                        continue
                    if not owner or not Storage.stopped(owner):
                        raise ValueError('Pack requires stopped background writers')
            # Transfer a delegated claim to the actual copying process. If the
            # CLI parent dies, its surviving child must still exclude writers.
            db.execute("INSERT OR REPLACE INTO control VALUES('pack-capture',?)", [token])
            owned = True
            db.commit()
        except BaseException:
            db.rollback()
            raise
        try:
            yield
        finally:
            if owned:
                with db:
                    if delegated and not Storage.stopped(parent):
                        db.execute("UPDATE control SET value=? WHERE key='pack-capture' AND value=?", [row[0], token])
                    else:
                        db.execute("DELETE FROM control WHERE key='pack-capture' AND value=?", [token])


def pack(root, dest, project=None, require_complete=True):
    """Collect an inventory and publish one standard Restic snapshot plus metadata.

    Restic owns repository retries/deduplication. Missing inputs remain explicit.
    """
    root, dest = Path(root).resolve(), Path(dest).resolve()
    if dest == root or root.is_relative_to(dest) or dest.is_relative_to(root):
        raise ValueError('Package must be separate from campaign')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _capture(root), tempfile.TemporaryDirectory(prefix='.package-inputs-', dir=dest.parent) as temporary:
        return _pack(root, dest, project, require_complete, Path(temporary))


def _pack(root, dest, project, require_complete, work):
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (dest / 'package.json').exists():
        raise FileExistsError('Package already published')
    metadata = {'format': 2, 'engine': 'restic', 'repository': '.', 'scope': 'project' if project else 'campaign',
                'project': project, 'files': [], 'missing': [], 'paths': {}, 'repositories': [],
                'paused': True, 'binding_required': True}
    config = root / 'instance.json'
    config_path = work / 'config.json'
    if not config_path.exists():
        _json(config_path, _sanitize(_load_json(config)) if config.exists() else {'format': 1})
    included = {}
    archived = []

    def missing(kind, path):
        metadata['missing'].append({'kind': kind, 'path': str(path)})

    def collect(path, target, kind, expected=None):
        target = str(_safe(target))
        if _excluded(target):
            missing(kind, target)
            return
        if target in included:
            prior = included[target]
            if expected and prior.get('expected_sha256') not in (None, expected):
                raise ValueError('Ledger checksum mismatch: ' + target)
            if expected:
                prior['expected_sha256'] = expected
            return
        if not path.is_file() or path.is_symlink():
            missing(kind, target)
            return
        item = {'path': target, 'kind': kind, 'source': str(path.resolve()),
                'mode': path.stat().st_mode & 0o777, 'expected_sha256': expected}
        metadata['files'].append(item)
        metadata['paths'][str(path)] = target
        included[target] = item

    ledger_path = work / 'ledger.sqlite'
    _snapshot(root, ledger_path, project)
    used = {}
    with closing(_connect(ledger_path)) as db:
        metadata['campaign'] = [dict(r) for r in db.execute('SELECT * FROM storage_campaign')]
        projects = [dict(r) for r in db.execute('SELECT * FROM projects')]
        has_attempts = db.execute('SELECT 1 FROM attempts LIMIT 1').fetchone() is not None
        metadata['projects'] = [p['id'] for p in projects]
        tables = _tables(db)
        for kind, field in (('skills', 'skill_revision'), ('memory', 'memory_revision')):
            used[kind] = {r[0] for r in db.execute('SELECT DISTINCT ' + field + ' FROM storage_attempts WHERE ' + field + ' IS NOT NULL')} if 'storage_attempts' in tables else set()
        active_version = _load_json(config_path).get('skills', {}).get('activeVersion')
        if active_version:
            versions = {active_version}
            if 'storage_skill_versions' in tables:
                versions.update(r['version'] for r in db.execute('SELECT version,revision FROM storage_skill_versions')
                                if r['revision'] in used['skills'])
            for version in sorted(versions):
                directory = _local(root, Path('skills/versions') / _safe(version))
                if not (directory / 'skill-map.json').is_file():
                    missing('skill-version', version)
                    continue
                for path in _files(directory):
                    collect(path, path.relative_to(root), 'skill-version')
        for p in projects:
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
        if 'storage_retrievals' in tables:
            for row in db.execute("SELECT DISTINCT json_extract(j.value,'$.artifact') FROM storage_retrievals,json_each(citations) j LEFT JOIN storage_artifacts a ON a.id=json_extract(j.value,'$.artifact') WHERE a.id IS NULL AND json_extract(j.value,'$.artifact') IS NOT NULL"):
                missing('cited-entry', row[0])
        for record in db.execute('SELECT * FROM storage_artifacts'):
            row = dict(record)
            if row['kind'] == 'source-git' and not row['external']:
                repository = _local(root, row['path'])
                if not repository.is_dir() or not (repository / 'objects').is_dir():
                    missing('source-git', row['path'])
                    continue
                metadata['repositories'].append({'path': row['path'], 'kind': 'source-git'})
                for path in _repository_files(repository):
                    collect(path, path.relative_to(root), 'source-git')
                continue
            if row['external'] or row['status'] not in ('closed', 'archived', 'open'):
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
                staged = work / 'collected' / row['id']
                archived.append((descriptor, staged, row))
                continue
            collect(path, row['path'], row['kind'], row.get('sha256'))
            metadata['paths'][str(root / row['path'])] = row['path']

    groups = {}
    for manifest, path, row in archived:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        groups.setdefault((manifest['repository'], manifest['snapshot']), []).append((manifest, path, row))
    for group in groups.values():
        destinations = {m['path']: path for m, path, _ in group}
        restore_files([m for m, _, _ in group], lambda m: destinations[m['path']])
        for _, path, row in group:
            collect(path, row['path'], row['kind'], row.get('sha256'))
            metadata['paths'][str(root / row['path'])] = row['path']

    manifest = root / 'package-inputs.json'
    inputs = _load_json(manifest) if manifest.exists() else {}
    kinds = {{'source-git': 'source-history', 'cited-text': 'cited-entry'}.get(f['kind'], f['kind']) for f in metadata['files']}
    covered = {'skills': set(), 'memory': set()}
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
    knowledge = root / 'knowledge.git'
    if knowledge.is_dir():
        git_dir = _git_directory(knowledge)
        if not git_dir.is_relative_to(root):
            raise ValueError('Knowledge Git directory must belong to campaign')
        for kind in covered:
            for revision in sorted(used[kind] - covered[kind]):
                if _revision_files(git_dir, revision, kind, work, collect):
                    covered[kind].add(revision)
                    kinds.add(kind)
        if not project:
            metadata['repositories'].append({'path': 'knowledge.git', 'kind': 'knowledge-history'})
            for path in _repository_files(git_dir):
                collect(path, Path('knowledge.git') / path.relative_to(git_dir), 'knowledge-history')
            kinds.add('knowledge-history')
    if not project:
        machinery = root / 'machinery'
        for name in ('controller', 'cli', 'ascend_tools', 'web', 'templates', 'integrations'):
            directory = machinery / name
            if directory.is_dir():
                for path in _files(directory):
                    collect(path, path.relative_to(root), 'machinery')
                    kinds.add('machinery')
        for name in ('package.json', 'npm-shrinkwrap.json', 'package-lock.json', 'pyproject.toml',
                     'ascend_tools/requirements.txt', 'ascend_tools/requirements-embeddings.txt'):
            path = machinery / name
            if path.is_file():
                collect(path, path.relative_to(root), 'dependencies')
                kinds.add('dependencies')
    for kind in covered:
        for revision in sorted(used[kind] - covered[kind]):
            missing(kind + '-revision', revision)
    required = {'source-history'} if has_attempts else set()
    if not project:
        required |= {'knowledge-history', 'machinery', 'dependencies'}
        snapshot = work / 'collected' / 'knowledge.sqlite'
        if not snapshot.exists() and (root / 'knowledge.sqlite').is_file():
            snapshot.parent.mkdir(exist_ok=True)
            temporary = snapshot.with_suffix('.tmp')
            temporary.unlink(missing_ok=True)
            with closing(_connect(root / 'knowledge.sqlite')) as source, closing(sqlite3.connect(temporary)) as out:
                source.backup(out, pages=256)
            os.replace(temporary, snapshot)
        collect(snapshot, 'knowledge.sqlite', 'active-kdb')
    for kind in sorted(required - kinds):
        missing(kind, 'package-inputs.json')
    metadata['knowledge_revisions'] = {kind: sorted(revisions) for kind, revisions in covered.items()}
    collect(config_path, 'instance.json', 'config')
    collect(ledger_path, 'state.sqlite', 'ledger')
    metadata['complete'] = not metadata['missing']
    if require_complete and not metadata['complete']:
        _json(dest / 'inspection.json', metadata)
        raise ValueError('Package incomplete; inspect inspection.json')
    batch = archive_files(list(dict.fromkeys(item['source'] for item in metadata['files'])), dest)
    metadata['snapshot'] = batch['snapshot']
    archived_sources = {item['source']: item for item in batch['files']}
    for item in metadata['files']:
        archive = archived_sources[item.pop('source')]
        expected = item.pop('expected_sha256')
        if expected and archive['sha256'] != expected:
            raise ValueError('Ledger checksum mismatch: ' + item['path'])
        item['archive'] = {k: v for k, v in archive.items() if k not in ('repository', 'manifest_path')}
    _json(dest / 'inspection.json', metadata)
    if (dest / 'inspection.json').stat().st_size > MAX_JSON_BYTES:
        raise ValueError('Package manifest exceeds portable size limit; package.json not published')
    _json(dest / 'package.json', metadata)
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
    metadata = _load_json(package / 'package.json')
    if metadata.get('format') != 2 or metadata.get('engine') != 'restic':
        raise ValueError('Unsupported package')
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.unpack-', dir=dest.parent))
    try:
        manifests = []
        for item in metadata['files']:
            relative = _safe(item['path'])
            if _excluded(relative):
                raise ValueError('Excluded package input')
            (stage / relative).parent.mkdir(parents=True, exist_ok=True)
            manifests.append({**item['archive'], 'repository': str(package), 'logical_path': str(relative)})
        restore_files(manifests, lambda manifest: stage / manifest['logical_path'])
        for item in metadata['files']:
            path = stage / _safe(item['path'])
            path.chmod(item.get('mode', 0o600) & 0o777)
            if item['kind'] == 'source-git' and path.name == 'rsi-source.json':
                _json(path, _rebase(_load_json(path), metadata['paths'], dest))
        for repo in metadata.get('repositories', []):
            directory = stage / _safe(repo['path'])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'objects').mkdir(exist_ok=True)
            (directory / 'refs').mkdir(exist_ok=True)
            bare = 'false' if repo['kind'] == 'source-git' else 'true'
            (directory / 'config').write_text('[core]\n\tbare = ' + bare + '\n\tlogAllRefUpdates = true\n')
        # Snapshot already preserves the original schema, indexes and stable IDs.
        with closing(sqlite3.connect(stage / 'state.sqlite')) as db, db:
            db.execute('PRAGMA cache_size=-2048')
            db.execute('PRAGMA temp_store=FILE')
            db.create_function('rebase_text', 1, lambda value: _rebase(value, metadata['paths'], dest))
            for table in _tables(db) & set(PORTABLE_TABLES):
                columns = [r[1] for r in db.execute('PRAGMA table_info(' + _quote(table) + ')') if r[2].upper() == 'TEXT']
                if columns:
                    db.execute('UPDATE ' + _quote(table) + ' SET ' + ','.join(_quote(c) + '=rebase_text(' + _quote(c) + ')' for c in columns))
            db.execute("UPDATE storage_artifacts SET encoding='plain',descriptor=NULL")
            projects = [json.loads(r[0]) for r in db.execute('SELECT spec FROM projects')]
        result = {'format': 1, 'paused': True, 'binding_required': True,
                  'storage': metadata['campaign'],
                  'complete': metadata['complete'], 'missing': metadata['missing']}
        _json(stage / 'restore.json', result)
        config = _rebase(_load_json(stage / 'instance.json'), metadata['paths'], dest)
        config.update(format=1, paused=True, binding_required=True, environment=None,
                      projects=projects)
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
