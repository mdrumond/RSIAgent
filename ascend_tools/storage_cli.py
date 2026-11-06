"""Ledger-backed storage maintenance. Walk writes JSONL; inventory reads metadata only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from .log_walk import walk_records
from .storage_archive import compress_file, compress_files, finalize_source, iter_artifact_bytes, restore_file


class Storage:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.db = sqlite3.connect(f'file:{self.root / "state.sqlite"}?mode=rw', uri=True)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA busy_timeout=5000')

    def path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('path escapes campaign root')
        return path

    def rows(self, action):
        limit = int(action.get('limit', 100))
        if not 1 <= limit <= 1000:
            raise ValueError('limit must be 1..1000')
        clauses, args = ['f.id > ?'], [action.get('after', '')]
        if action.get('action') == 'compress':
            clauses.extend(["f.encoding='plain'", "f.kind!='source-git'"])
        fields = {'artifact': 'f.id', 'project': 'COALESCE(f.project,a.project)',
                  'role': 'f.role', 'phase': 'COALESCE(s.phase,ss.phase)', 'kind': 'f.kind',
                  'outcome': 'a.outcome', 'skill_revision': 's.skill_revision',
                  'memory_revision': 's.memory_revision'}
        for key, column in fields.items():
            value = action.get(key, action.get(key.replace('_', '-')))
            if value is not None:
                clauses.append(column + '=?')
                args.append(value)
        if 'stage' in action or 'iteration' in action:
            extra = ['i.attempt=f.attempt']
            for key, column in [('stage', 'stage'), ('iteration', 'ordinal')]:
                if key in action:
                    extra.append('i.' + column + '=?')
                    args.append(action[key])
            clauses.append('EXISTS(SELECT 1 FROM storage_iterations i WHERE ' + ' AND '.join(extra) + ')')
        return [dict(row) for row in self.db.execute('''SELECT f.*,a.outcome,
            COALESCE(f.project,a.project) AS project_id,COALESCE(s.phase,ss.phase) AS phase,s.memory_revision,s.skill_revision
            FROM storage_artifacts f LEFT JOIN attempts a ON a.id=f.attempt
            LEFT JOIN storage_attempts s ON s.attempt=f.attempt
            LEFT JOIN storage_sessions ss ON ss.session=f.path OR ss.provider=f.path WHERE ''' +
            ' AND '.join(clauses) + ' ORDER BY f.id LIMIT ?', [*args, limit])]

    def inspect(self, action):
        rows = self.rows(action)
        for row in rows:
            if not row['external']:
                self.path(row['path'])
            row['missing'] = bool(row['external']) or not (
                self.path(row['path']).exists() if row['encoding'] == 'plain' else
                self.path(json.loads(row['descriptor'])['manifest']).exists())
        return dict(campaign=dict(self.db.execute('SELECT * FROM storage_campaign').fetchone()),
                    usage=[dict(r) for r in self.db.execute('''SELECT status,encoding,COUNT(*) AS artifacts,
                        SUM(bytes) AS logical_bytes FROM storage_artifacts GROUP BY status,encoding''')],
                    revisions=[dict(r) for r in self.db.execute('SELECT * FROM storage_revisions')],
                    artifacts=rows, next_after=rows[-1]['id'] if rows else None)

    def manifest(self, row):
        descriptor = json.loads(row['descriptor'])
        manifest = json.loads(self.path(descriptor['manifest']).read_text())
        self.path(manifest['repository'])
        self.path(manifest['source'])
        return manifest

    def content(self, row):
        if row['external']:
            raise ValueError('external evidence is not collected')
        if row['encoding'] != 'plain':
            yield from iter_artifact_bytes(self.manifest(row))
        else:
            with self.path(row['path']).open('rb') as stream:
                while block := stream.read(65536):
                    yield block

    def verify(self, row):
        if row['kind'] == 'source-git' and not row['external']:
            subprocess.run(['git', '--git-dir', str(self.path(row['path'])),
                            'fsck', '--no-dangling'], check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return dict(id=row['id'], verified=True, kind='source-git')
        digest, size = hashlib.sha256(), 0
        for block in self.content(row):
            digest.update(block)
            size += len(block)
        if size != row['bytes'] or digest.hexdigest() != row['sha256']:
            raise ValueError('ledger digest mismatch: ' + row['id'])
        return dict(id=row['id'], verified=True, bytes=size)

    @staticmethod
    def stopped(owner):
        owner = json.loads(owner) if isinstance(owner, str) else owner
        if not owner or not all(owner.get(k) for k in ('pid', 'start', 'boot')):
            return False
        try:
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            stat = Path(f'/proc/{int(owner["pid"])}/stat').read_text()
            start = stat[stat.rfind(')') + 2:].split()[19]
            return boot != owner['boot'] or start != str(owner['start'])
        except FileNotFoundError:
            return True

    def quiescent(self, row):
        if row['attempt']:
            attempt = self.db.execute('SELECT * FROM attempts WHERE id=?', [row['attempt']]).fetchone()
            if not attempt or attempt['outcome'] is None or not self.stopped(attempt['owner']):
                raise ValueError('artifact owner is live or unresolved')
            events = self.db.execute('''SELECT type,data FROM events WHERE attempt=? AND
                type IN ('postprocess-completed','failure-settlement') ORDER BY sequence DESC''', [row['attempt']])
            completed, settled = False, None
            for event in events:
                if event['type'] == 'postprocess-completed':
                    completed = True
                elif settled is None:
                    settled = json.loads(event['data']).get('closed') is True
            if not completed or settled is False:
                raise ValueError('artifact postprocessing or settlement unresolved')
        elif row['kind'] == 'cited-text':
            return  # Immutable cache files are published before registration.
        elif self.db.execute('SELECT 1 FROM storage_sessions WHERE (session=? OR provider=?) AND closed=1', [row['path'], row['path']]).fetchone():
            return  # The role session disposed its writer before marking closure.
        else:
            # Unowned evidence requires all recorded campaign writers stopped.
            if self.db.execute('SELECT 1 FROM attempts WHERE outcome IS NULL LIMIT 1').fetchone():
                raise ValueError('campaign has active attempts')
            if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='control'").fetchone():
                for item in self.db.execute("SELECT value FROM control WHERE key IN ('owner','skill-writer')"):
                    if not self.stopped(item['value']):
                        raise ValueError('campaign owner is live or unresolved')

    def compression_candidate(self, row):
        if row['external'] or row['status'] != 'closed':
            return dict(id=row['id'], skipped='not closed local evidence')
        if row['encoding'] != 'plain':
            return dict(id=row['id'], skipped='already compressed')
        path = self.path(row['path'])
        if not path.is_file():
            return dict(id=row['id'], skipped='missing or not a file')
        try:
            self.quiescent(row)
        except ValueError as error:
            return dict(id=row['id'], skipped=str(error))
        return None

    def compress_batch(self, rows, delete_source=False):
        eligible = [row for row in rows if self.compression_candidate(row) is None]
        for row in eligible:
            self.verify(row)
        manifests = compress_files([self.path(row['path']) for row in eligible],
                                   self.path('artifacts/compressed')) if eligible else []
        by_id = dict(zip((row['id'] for row in eligible), manifests))
        return [self.compress(row, delete_source, by_id.get(row['id'])) for row in rows]

    def compress(self, row, delete_source=False, manifest=None):
        skipped = self.compression_candidate(row)
        if skipped is not None:
            return skipped
        self.verify(row)
        manifest = manifest or compress_file(self.path(row['path']), self.path('artifacts/compressed'))
        if manifest['sha256'] != row['sha256'] or manifest['source_identity']['size'] != row['bytes']:
            raise ValueError('source changed since ledger seal')
        self.quiescent(row)
        descriptor = json.dumps({'manifest': str(Path(manifest['manifest_path']).relative_to(self.root))})
        # Durable ledger publication must precede optional source finalization.
        with self.db:
            changed = self.db.execute('''UPDATE storage_artifacts SET encoding='restic',descriptor=?
                WHERE id=? AND status='closed' AND encoding='plain' AND sha256=?''',
                [descriptor, row['id'], row['sha256']]).rowcount
            if not changed:
                raise ValueError('ledger changed during compression')
        if delete_source:
            self.quiescent(row)
            finalize_source(manifest)
        return dict(id=row['id'], manifest=json.loads(descriptor), source_deleted=delete_source)

    def walk(self, action):
        action = dict(action)
        action.setdefault('kind', 'provider-telemetry' if action.get('raw_provider', action.get('raw-provider')) else 'pi-session')
        for row in self.rows(action):
            if row['external']:
                continue
            artifact = self.manifest(row) if row['encoding'] != 'plain' else self.path(row['path'])
            for item in walk_records(artifact, cursor=int(action.get('cursor', 0))):
                item['metadata']['artifact_id'] = row['id']
                print(json.dumps(item), flush=True)

    def plan(self, action):
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='control'").fetchone():
            for item in self.db.execute("SELECT value FROM control WHERE key IN ('owner','skill-writer')"):
                if not self.stopped(item['value']):
                    raise ValueError('campaign owner is live or unresolved')
        query, args = 'SELECT outcome,owner FROM attempts', []
        if action.get('project'):
            query += ' WHERE project=?'
            args.append(action['project'])
        for attempt in self.db.execute(query, args):
            if attempt['outcome'] is None or not self.stopped(attempt['owner']):
                raise ValueError('cleanup scope has live or unresolved owners')
        entries = []
        for row in self.rows(action):
            if row['external'] or row['encoding'] == 'plain':
                continue
            if not self.path(row['path']).exists():
                continue
            self.quiescent(row)
            self.verify(row)
            manifest = self.manifest(row)
            source = self.path(row['path'])
            identity = {key: getattr(source.stat(), 'st_' + key) for key in
                        ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')}
            if identity != manifest['source_identity']:
                raise ValueError('source changed since compression')
            entries.append(dict(row=row, identity={key: str(value) for key, value in identity.items()}))
        return dict(campaign=self.db.execute('SELECT id FROM storage_campaign').fetchone()['id'],
                    project=action.get('project'), entries=entries)

    def apply(self, inventory):
        if inventory['campaign'] != self.db.execute('SELECT id FROM storage_campaign').fetchone()['id']:
            raise ValueError('inventory belongs to another campaign')
        checked = []
        for entry in inventory['entries']:
            rows = self.rows({'artifact': entry['row']['id']})
            if len(rows) != 1 or rows[0] != entry['row']:
                raise ValueError('inventory ledger entry changed')
            row = rows[0]
            current = self.plan({'artifact': row['id'], 'project': inventory.get('project')})
            if not current['entries'] or current['entries'][0]['identity'] != entry['identity']:
                raise ValueError('inventory source changed')
            checked.append(row)
        # Validate every entry before the first deletion.
        for row in checked:
            self.quiescent(row)
            finalize_source(self.manifest(row))
        return dict(removed=[row['id'] for row in checked])


def execute(root, action):
    if action.get('action') == 'unpack':
        from .storage_package import unpack
        return unpack(Path(action['package']).resolve(), Path(action['destination']).resolve())
    storage = Storage(root)
    try:
        name = action.get('action', 'inspect')
        if name == 'inspect':
            return storage.inspect(action)
        if name == 'walk':
            storage.walk(action)
            return None
        if name == 'verify':
            return [storage.verify(row) for row in storage.rows(action)]
        if name == 'compress':
            storage.db.execute('CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            explicit = 'after' in action or any(key in action for key in ('artifact','project','phase','role','kind','outcome','stage','iteration','skill-revision','memory-revision','skill_revision','memory_revision'))
            if not explicit:
                row = storage.db.execute("SELECT value FROM control WHERE key='storage-compress-cursor'").fetchone()
                action = dict(action, after=json.loads(row[0]) if row else '')
            rows = storage.rows(action)
            result = storage.compress_batch(rows, bool(action.get('delete_source', action.get('delete-source', False))))
            if not explicit:
                with storage.db:
                    storage.db.execute("INSERT OR REPLACE INTO control VALUES ('storage-compress-cursor',?)", [json.dumps(rows[-1]['id'] if rows else '')])
            return result
        if name == 'prune':
            if action.get('apply'):
                return storage.apply(action['inventory'])
            return storage.plan(action)
        if name == 'pack':
            from .storage_package import pack
            result = pack(storage.root, Path(action['destination']).resolve(), project=action.get('project'))
            return dict(package=str(Path(action['destination']).resolve()), complete=result['complete'],
                        scope=result['scope'], files=len(result['files']), missing=result['missing'])
        if name == 'restore':
            row, = storage.rows(action)
            return dict(path=restore_file(storage.manifest(row), storage.path(action['destination'])))
        raise ValueError('unknown storage action')
    finally:
        storage.db.close()


def main(argv=None):
    # File IO is streamed; Restic owns its bounded-concurrency archive work.
    # Do not cap virtual address space: Go reserves a large virtual arena.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--action-json')
    parser.add_argument('action', nargs='?', default='inspect',
                        choices=['inspect', 'verify', 'walk', 'compress', 'pack', 'unpack', 'prune', 'restore'])
    parser.add_argument('--artifact')
    parser.add_argument('--project')
    parser.add_argument('--destination')
    parser.add_argument('--package')
    parser.add_argument('--raw-provider', action='store_true')
    parser.add_argument('--delete-source', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--inventory')
    for key in ('role', 'phase', 'kind', 'outcome', 'stage', 'iteration', 'skill-revision',
                'memory-revision', 'cursor', 'after', 'limit'):
        parser.add_argument('--' + key)
    args = parser.parse_args(argv)
    action = json.load(sys.stdin) if args.action_json == '-' else json.loads(args.action_json) if args.action_json else {k: v for k, v in vars(args).items() if v is not None}
    if isinstance(action.get('inventory'), str):
        storage = Storage(args.root)
        try:
            action['inventory'] = json.loads(storage.path(action['inventory']).read_text())
        finally:
            storage.db.close()
    mutating = action.get('action') in ('compress', 'pack', 'unpack', 'prune', 'restore')
    if action.get('action') == 'unpack':
        os.nice(10)
        result = execute(args.root, action)
    elif mutating:
        os.nice(10)
        with (Path(args.root) / '.storage-maintenance.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = execute(args.root, action)
    else:
        result = execute(args.root, action)
    if action.get('action') != 'walk':
        print(json.dumps(result))


if __name__ == '__main__':
    main()
