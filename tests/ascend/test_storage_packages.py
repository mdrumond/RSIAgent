from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ascend_tools.storage_package import pack, unpack


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'campaign'
        self.root.mkdir()
        self.source = self.root / 'projects/p/source'
        self.source.mkdir(parents=True)
        (self.source / 'kernel.py').write_text('answer = 42\n')
        self.verify = self.root / 'projects/p/verify.py'
        self.verify.write_text('assert True\n')
        self.log = self.root / 'attempt.log'
        self.log.write_text('iteration one\n')
        spec = {'source': str(self.source), 'goal': 'tiny fixture', 'actor_seconds': 1,
                'verifier': {'files': [str(self.verify)],
                             'hashes': {str(self.verify): hashlib.sha256(self.verify.read_bytes()).hexdigest()},
                             'command': 'python ' + str(self.verify)}}
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.executescript('''CREATE TABLE projects(id TEXT PRIMARY KEY,spec TEXT,status TEXT,frozen INTEGER);
                CREATE TABLE attempts(id TEXT PRIMARY KEY,project TEXT,session TEXT,elapsed REAL,outcome TEXT,owner TEXT,created TEXT);
                CREATE TABLE events(sequence INTEGER PRIMARY KEY,attempt TEXT,type TEXT,data TEXT,created TEXT);
                CREATE TABLE storage_campaign(id TEXT PRIMARY KEY,created TEXT,target TEXT,format INTEGER);
                CREATE TABLE storage_artifacts(id TEXT PRIMARY KEY,attempt TEXT,project TEXT,kind TEXT,role TEXT,path TEXT,external INTEGER,status TEXT,bytes INTEGER,sha256 TEXT,encoding TEXT,descriptor TEXT,created TEXT);
                CREATE TABLE storage_attempts(attempt TEXT PRIMARY KEY,phase TEXT,role TEXT,memory_revision TEXT,skill_revision TEXT);
                CREATE TABLE storage_iterations(id INTEGER PRIMARY KEY,attempt TEXT,stage TEXT,ordinal INTEGER,reflection TEXT);''')
            db.execute('INSERT INTO projects VALUES(?,?,?,?)', ('p', json.dumps(spec), 'running', 0))
            db.execute('INSERT INTO projects VALUES(?,?,?,?)', ('other', json.dumps(spec), 'queued', 0))
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?)', ('a', 'p', 'session-secret', 0.5, None, 'owner-secret', 'now'))
            db.execute('INSERT INTO storage_campaign VALUES(?,?,?,?)', ('campaign-id', 'now', 'a5', 1))
            db.execute('INSERT INTO storage_artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       ('log-id', 'a', 'p', 'log', 'actor', 'attempt.log', 0, 'closed', self.log.stat().st_size,
                        hashlib.sha256(self.log.read_bytes()).hexdigest(), 'plain', None, 'now'))
            db.execute('INSERT INTO storage_attempts VALUES(?,?,?,?,?)', ('a', 'phase1', 'actor', 'm1', 's1'))
            db.execute('INSERT INTO storage_iterations VALUES(?,?,?,?,?)', (1, 'a', 'edit', 1, 'reflection'))
        files = []
        for kind in ('skills', 'memory', 'cited-entry', 'machinery', 'dependencies'):
            name = kind + '.txt'
            (self.root / name).write_text(kind)
            files.append({'path': name, 'kind': kind, 'revision': {'skills': 's1', 'memory': 'm1'}.get(kind)})
        repo = self.root / 'history-repo'
        repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        (repo / 'code.py').write_text('old = True\n')
        subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        inputs = {'files': files, 'git': [{'path': 'history-repo', 'kind': kind} for kind in ('source-history', 'knowledge-history')]}
        (self.root / 'package-inputs.json').write_text(json.dumps(inputs))
        with closing(sqlite3.connect(self.root / 'knowledge.sqlite')) as db, db:
            db.execute('CREATE TABLE entries(id TEXT PRIMARY KEY)')
            db.execute("INSERT INTO entries VALUES('active-entry')")
        self.package = self.base / 'package'

    def test_project_roundtrip_relocation_and_identities(self):
        (self.source / '.env').write_text('do not export')
        result = pack(self.root, self.package, 'p')
        self.assertTrue(result['complete'])
        self.assertFalse(any(f['path'] == 'knowledge.sqlite' for f in result['files']))
        self.assertFalse({'knowledge-history', 'machinery', 'dependencies'} & {f['kind'] for f in result['files']})
        portable = self.base / 'moved-package'
        shutil.move(self.package, portable)
        restored = self.base / 'restored'
        unpack(portable, restored)
        with closing(sqlite3.connect(restored / 'state.sqlite')) as db:
            self.assertEqual(db.execute('SELECT id FROM storage_campaign').fetchone()[0], 'campaign-id')
            row = db.execute('SELECT spec,frozen FROM projects').fetchone()
            spec = json.loads(row[0])
            self.assertEqual(row[1], 1)
            self.assertEqual(spec['source'], str(restored / 'projects/p/source'))
            self.assertEqual(spec['verifier']['files'], [str(restored / 'projects/p/verify.py')])
            self.assertEqual(db.execute('SELECT owner,session,elapsed FROM attempts').fetchone(), (None, None, 0.5))
            self.assertEqual(db.execute('SELECT attempt FROM storage_iterations').fetchone(), ('a',))
        self.assertEqual((restored / 'attempt.log').read_text(), self.log.read_text())
        self.assertFalse((restored / 'projects/p/source/.env').exists())
        self.assertTrue(json.loads((restored / 'restore.json').read_text())['binding_required'])
        with self.assertRaises(FileExistsError):
            unpack(portable, restored)

    def test_scripts_execute_and_archived_project_artifact(self):
        from ascend_tools.storage_archive import compress_file
        archive = compress_file(self.log, self.root / 'archives', chunk_bytes=4)
        descriptor = {'manifest': str(Path(archive['manifest_path']).relative_to(self.root))}
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.execute("UPDATE storage_artifacts SET attempt=NULL,encoding='gzip-chunks',descriptor=?",
                       (json.dumps(descriptor),))
        self.log.unlink()
        scripts = Path(__file__).resolve().parents[2] / 'scripts'
        process = subprocess.run([sys.executable, str(scripts / 'pack_project.py'),
                                  str(self.root), str(self.package), '--project', 'p'],
                                 cwd=self.base, check=True, capture_output=True, text=True)
        self.assertTrue(json.loads(process.stdout)['complete'])
        dest = self.base / 'script-restored'
        subprocess.run([sys.executable, str(scripts / 'unpack_project.py'),
                        str(self.package), str(dest)], cwd=self.base, check=True,
                       capture_output=True)
        self.assertEqual((dest / 'attempt.log').read_text(), 'iteration one\n')
        with closing(sqlite3.connect(dest / 'state.sqlite')) as db:
            self.assertEqual(db.execute('SELECT encoding,descriptor FROM storage_artifacts').fetchone(), ('plain', None))

    def test_campaign_has_active_kdb_and_full_git_history(self):
        subprocess.run(['git', '-C', str(self.root / 'history-repo'), 'config', 'remote.origin.url',
                        'https://credential-fixture@example.invalid/repo'], check=True)
        result = pack(self.root, self.package)
        restored = self.base / 'restored'
        unpack(self.package, restored)
        with closing(sqlite3.connect(restored / 'knowledge.sqlite')) as db:
            self.assertEqual(db.execute('SELECT id FROM entries').fetchone(), ('active-entry',))
        history = next(r for r in result['repositories'] if r['kind'] == 'knowledge-history')
        actual = subprocess.check_output(['git', '--git-dir', str(restored / history['path']),
                                          'show', 'HEAD:code.py'], text=True)
        self.assertEqual(actual, 'old = True\n')
        self.assertNotIn('credential-fixture', (restored / history['path'] / 'config').read_text())

    def test_missing_refuses_publish_but_is_inspectable(self):
        self.log.unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            pack(self.root, self.package, 'p')
        self.assertFalse((self.package / 'package.json').exists())
        self.assertEqual(json.loads((self.package / 'inspection.json').read_text())['missing'][0]['kind'], 'log')
        result = pack(self.root, self.package, 'p', require_complete=False)
        self.assertFalse(result['complete'])

    def test_source_git_preserves_noncommit_refs_and_rebases_binding(self):
        repo = self.root / 'source-history/fixture'
        repo.parent.mkdir()
        subprocess.run(['git', 'clone', '--bare', '-q', str(self.root / 'history-repo'), str(repo)], check=True)
        tree = subprocess.check_output(['git', '--git-dir', str(repo), 'rev-parse', 'HEAD^{tree}'], text=True).strip()
        ref = 'refs/rsi/snapshots/attempt/' + tree
        subprocess.run(['git', '--git-dir', str(repo), 'update-ref', ref, tree], check=True)
        branch = subprocess.check_output(['git', '--git-dir', str(repo), 'symbolic-ref', 'HEAD'], text=True).strip()
        subprocess.run(['git', '--git-dir', str(repo), 'update-ref', '-d', branch], check=True)
        subprocess.run(['git', '--git-dir', str(repo), 'config', 'core.bare', 'false'], check=True)
        (repo / 'rsi-source.json').write_text(json.dumps({'source': str(self.source)}))
        subprocess.run(['git', '--git-dir', str(repo), 'gc', '--prune=now'], check=True, capture_output=True)
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.execute('INSERT INTO storage_artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       ('source-history-id', 'a', 'p', 'source-git', 'actor',
                        str(repo.relative_to(self.root)), 0, 'open', None, None, 'plain', None, 'now'))
        inputs = self.root / 'package-inputs.json'
        value = json.loads(inputs.read_text())
        value['git'] = []
        inputs.write_text(json.dumps(value))
        self.assertTrue(pack(self.root, self.package, 'p')['complete'])
        dest = self.base / 'tree-restored'
        unpack(self.package, dest)
        restored = dest / repo.relative_to(self.root)
        self.assertEqual(subprocess.check_output(['git', '--git-dir', str(restored), 'cat-file', '-t', ref], text=True).strip(), 'tree')
        self.assertEqual(json.loads((restored / 'rsi-source.json').read_text())['source'], str(dest / 'projects/p/source'))
        self.assertEqual(subprocess.check_output(['git', '--git-dir', str(restored), 'rev-parse', ref], text=True).strip(), tree)

    def test_used_revision_must_be_collected(self):
        path = self.root / 'package-inputs.json'
        inputs = json.loads(path.read_text())
        inputs['files'][0]['revision'] = 'different-version'
        path.write_text(json.dumps(inputs))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            pack(self.root, self.package, 'p')
        result = json.loads((self.package / 'inspection.json').read_text())
        self.assertIn({'kind': 'skills-revision', 'path': 's1'}, result['missing'])

    def test_corrupt_chunks_never_publish_restored_destination(self):
        result = pack(self.root, self.package, 'p')
        chunk = self.package / 'chunks' / result['files'][0]['archive']['chunks'][0]['file']
        chunk.write_bytes(b'broken')
        dest = self.base / 'restored'
        with self.assertRaises((ValueError, OSError, EOFError)):
            unpack(self.package, dest)
        self.assertFalse(dest.exists())

    def test_resume_after_interruption_and_checksum_failure(self):
        from ascend_tools import storage_package
        real = storage_package.compress_file
        calls = []
        def interrupt(*args, **kwargs):
            calls.append(args[0])
            if len(calls) == 2:
                raise InterruptedError('fixture interruption')
            return real(*args, **kwargs)
        with patch.object(storage_package, 'compress_file', interrupt):
            with self.assertRaises(InterruptedError):
                pack(self.root, self.package, 'p')
        result = pack(self.root, self.package, 'p')
        self.assertTrue(result['complete'])
        self.log.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            pack(self.root, self.base / 'bad', 'p')


if __name__ == '__main__':
    unittest.main()
