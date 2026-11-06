from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import tracemalloc
from unittest.mock import Mock, patch

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
            stopped = json.dumps(dict(pid=os.getpid(), start='old-process', boot='old-boot'))
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?)', ('a', 'p', 'session-secret', 0.5, None, stopped, 'now'))
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

    def test_campaign_export_refuses_unportable_background_state_but_project_export_remains_valid(self):
        for table in ('skill_evidence_queue', 'evaluation_archives'):
            with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
                db.execute('CREATE TABLE ' + table + '(id TEXT)')
                db.execute('INSERT INTO ' + table + ' VALUES (?)', ('retained-job',))
            with self.assertRaisesRegex(ValueError, 'cannot preserve background'):
                pack(self.root, self.package)
            self.assertFalse(self.package.exists())
            with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
                db.execute('DELETE FROM ' + table)
        (self.root / 'instance.json').write_text(json.dumps(dict(format=1, skills=dict(background=True))))
        (self.root / 'skills').mkdir();(self.root / 'skills/evolution.json').write_text(json.dumps(dict(history=[dict(id='retained')], pending=None)))
        with self.assertRaisesRegex(ValueError, 'cannot preserve background'):
            pack(self.root, self.package)
        self.assertFalse(self.package.exists())
        result = pack(self.root, self.package, 'p')
        self.assertTrue(result['complete']);self.assertEqual(result['scope'], 'project')

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
            self.assertEqual(db.execute('SELECT owner,session,elapsed FROM attempts').fetchone(), (None, 'session-secret', 0.5))
            self.assertEqual(db.execute('SELECT attempt FROM storage_iterations').fetchone(), ('a',))
        self.assertEqual((restored / 'attempt.log').read_text(), self.log.read_text())
        self.assertFalse((restored / 'projects/p/source/.env').exists())
        self.assertTrue(json.loads((restored / 'restore.json').read_text())['binding_required'])
        with self.assertRaises(FileExistsError):
            unpack(portable, restored)

    def test_scripts_execute_and_archived_project_artifact(self):
        from ascend_tools.storage_archive import compress_file
        archive = compress_file(self.log, self.root / 'archives')
        descriptor = {'manifest': str(Path(archive['manifest_path']).relative_to(self.root))}
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.execute("UPDATE storage_artifacts SET attempt=NULL,encoding='restic',descriptor=?",
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

    def test_sqlite_ledger_many_events_is_bounded_and_preserves_schema(self):
        count = 12000
        payload = json.dumps({'text': 'x' * 2048, 'owner': 'private-owner',
                              'executed_source': str(self.source / 'kernel.py')})
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.executemany('INSERT INTO events VALUES(?,?,?,?,?)',
                           ((i, 'a', 'meaningful-edit', payload, 'now') for i in range(count)))
            db.execute('CREATE UNIQUE INDEX fixture_event_order ON events(attempt,sequence)')
            db.executescript('CREATE TABLE storage_sessions(session TEXT PRIMARY KEY,attempt TEXT);'
                             'CREATE TABLE storage_retrievals(id INTEGER PRIMARY KEY,attempt TEXT,citations TEXT);'
                             'CREATE TABLE storage_skill_versions(version TEXT PRIMARY KEY,revision TEXT);')
            db.execute("INSERT INTO storage_sessions VALUES('sessions/native.jsonl','a')")
            db.execute("INSERT INTO storage_retrievals VALUES(1,'a','[]')")
            db.execute("INSERT INTO storage_skill_versions VALUES('baseline','s1')")
        tracemalloc.start()
        try:
            result = pack(self.root, self.package, 'p')
            dest = self.base / 'many-restored'
            unpack(self.package, dest)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 12 * 1024 * 1024)
        self.assertNotIn('ledger', result)
        self.assertLess((self.package / 'package.json').stat().st_size, 128 * 1024)
        with closing(sqlite3.connect(dest / 'state.sqlite')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM events').fetchone()[0], count)
            value = json.loads(db.execute('SELECT data FROM events LIMIT 1').fetchone()[0])
            self.assertNotIn('owner', value)
            self.assertEqual(value['executed_source'], str(dest / 'projects/p/source/kernel.py'))
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='fixture_event_order'").fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT session FROM storage_sessions').fetchone()[0], 'sessions/native.jsonl')
            self.assertEqual(db.execute('SELECT version FROM storage_skill_versions').fetchone()[0], 'baseline')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM projects').fetchone()[0], 1)
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM projects').fetchone()[0], 2)

    def automatic_fixture(self):
        (self.root / 'package-inputs.json').unlink()
        knowledge = self.root / 'knowledge.git'
        subprocess.run(['git', 'init', '-q', str(knowledge)], check=True)
        for kind in ('memory', 'skills'):
            (knowledge / kind).mkdir()
            (knowledge / kind / 'exact.txt').write_text(kind + ' exact snapshot')
        subprocess.run(['git', '-C', str(knowledge), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(knowledge), '-c', 'user.name=Fixture', '-c',
                        'user.email=fixture@example.invalid', 'commit', '-qm', 'knowledge'], check=True)
        revision = subprocess.check_output(['git', '-C', str(knowledge), 'rev-parse', 'HEAD'], text=True).strip()
        history = self.root / 'source-history/project'
        history.parent.mkdir()
        subprocess.run(['git', 'clone', '--bare', '-q', str(self.root / 'history-repo'), str(history)], check=True)
        machinery = self.root / 'machinery'
        (machinery / 'controller').mkdir(parents=True)
        (machinery / 'controller/core.mjs').write_text('export const fixture = true;')
        (machinery / 'package.json').write_text('{}')
        (machinery / 'npm-shrinkwrap.json').write_text('{}')
        (machinery / 'node_modules').mkdir()
        (machinery / 'node_modules/private').write_text('excluded')
        native = self.root / 'sessions/native.jsonl'
        native.parent.mkdir()
        native.write_text('{"session":"stable-native-id"}\n')
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.execute('CREATE TABLE storage_sessions(session TEXT PRIMARY KEY,attempt TEXT,role TEXT,phase TEXT,created TEXT)')
            db.execute('INSERT INTO storage_sessions VALUES(?,?,?,?,?)', (str(native), 'a', 'actor', 'phase1', 'now'))
            db.execute('UPDATE attempts SET session=?', (str(native),))
            db.execute('INSERT INTO storage_artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       ('native-session', 'a', 'p', 'pi-session', 'actor', str(native.relative_to(self.root)),
                        0, 'open', None, None, 'plain', None, 'now'))
            db.execute('UPDATE storage_attempts SET memory_revision=?,skill_revision=?', (revision, revision))
            db.execute('INSERT INTO storage_artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       ('git-id', 'a', 'p', 'source-git', 'actor', str(history.relative_to(self.root)),
                        0, 'open', None, None, 'plain', None, 'now'))
            db.execute('CREATE TABLE storage_retrievals(id INTEGER PRIMARY KEY,attempt TEXT,citations TEXT)')
            (self.root / 'artifacts/citations').mkdir(parents=True)
            for name in ('used', 'unused'):
                path = self.root / 'artifacts/citations' / (name + '.txt')
                path.write_text(name + ' citation')
                db.execute('INSERT INTO storage_artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                           (name, None, None, 'cited-text', 'kdb', str(path.relative_to(self.root)),
                            0, 'open', path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), 'plain', None, 'now'))
            db.execute('INSERT INTO storage_retrievals VALUES(?,?,?)', (1, 'a', json.dumps([{'artifact': 'used'}])))
        return revision

    def test_automatic_project_uses_exact_revisions_and_only_shared_citations(self):
        revision = self.automatic_fixture()
        result = pack(self.root, self.package, 'p')
        self.assertTrue(result['complete'])
        self.assertFalse(any(f['kind'] == 'knowledge-history' for f in result['files']))
        dest = self.base / 'automatic-restored'
        unpack(self.package, dest)
        for kind in ('memory', 'skills'):
            self.assertEqual((dest / 'exported' / revision / kind / 'exact.txt').read_text(), kind + ' exact snapshot')
        self.assertTrue((dest / 'artifacts/citations/used.txt').is_file())
        self.assertFalse((dest / 'artifacts/citations/unused.txt').exists())
        with closing(sqlite3.connect(dest / 'state.sqlite')) as db:
            self.assertEqual(db.execute('SELECT citations FROM storage_retrievals').fetchone()[0], '[{"artifact": "used"}]')
            session = str(dest / 'sessions/native.jsonl')
            self.assertEqual(db.execute('SELECT session FROM storage_sessions').fetchone()[0], session)
            self.assertEqual(db.execute('SELECT session FROM attempts').fetchone()[0], session)
            self.assertEqual(json.loads(Path(session).read_text())['session'], 'stable-native-id')

    def test_automatic_project_without_retrievals_needs_no_citation_input(self):
        self.automatic_fixture()
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.execute('DELETE FROM storage_retrievals')
        result = pack(self.root, self.package, 'p')
        self.assertTrue(result['complete'])
        self.assertFalse(any('citations/' in f['path'] for f in result['files']))

    def test_automatic_campaign_and_empty_attempt_baseline(self):
        self.automatic_fixture()
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            for table in ('storage_attempts', 'storage_iterations', 'storage_retrievals', 'storage_sessions', 'attempts'):
                db.execute('DELETE FROM ' + table)
            db.execute('DELETE FROM storage_artifacts')
        result = pack(self.root, self.package)
        self.assertTrue(result['complete'])
        dest = self.base / 'baseline-restored'
        unpack(self.package, dest)
        actual = subprocess.check_output(['git', '--git-dir', str(dest / 'knowledge.git'),
                                          'show', 'HEAD:memory/exact.txt'], text=True)
        self.assertEqual(actual, 'memory exact snapshot')
        self.assertTrue((dest / 'machinery/controller/core.mjs').is_file())
        self.assertTrue((dest / 'machinery/npm-shrinkwrap.json').is_file())
        self.assertFalse((dest / 'machinery/node_modules').exists())

    def test_pending_project_recovery_schedule_survives_export(self):
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.execute('CREATE TABLE recovery_queue(project TEXT PRIMARY KEY,origin TEXT,failed_rounds INTEGER,kind TEXT,eligible_wave INTEGER,attempt TEXT)')
            db.execute("INSERT INTO recovery_queue VALUES('p','a',2,'investigation',7,NULL)")
            db.execute("INSERT INTO recovery_queue VALUES('other','other-attempt',1,'investigation',8,NULL)")
        pack(self.root, self.package, 'p')
        dest = self.base / 'recovery-restored'
        unpack(self.package, dest)
        with closing(sqlite3.connect(dest / 'state.sqlite')) as db:
            self.assertEqual(db.execute('SELECT * FROM recovery_queue').fetchall(),
                             [('p', 'a', 2, 'investigation', 7, None)])

    def test_campaign_materialized_skills_work_with_actual_consumers(self):
        revision = self.automatic_fixture()
        scopes = ('global-kdb', 'preflight', 'task-construction', 'design', 'implementation',
                  'test', 'debug', 'optimisation', 'validation', 'acceptance')
        for version in ('active', 'used'):
            directory = self.root / 'skills/versions' / version
            (directory / 'assets').mkdir(parents=True)
            (directory / 'skill-map.json').write_text(json.dumps(dict.fromkeys(scopes, 'guidance.md')))
            (directory / 'guidance.md').write_text(version + ' skill; see assets/policy.txt')
            (directory / 'assets/policy.txt').write_text(version + ' policy')
        (self.root / 'instance.json').write_text(json.dumps({'format': 1, 'skills': {'activeVersion': 'active'}}))
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.execute('CREATE TABLE storage_skill_versions(version TEXT PRIMARY KEY,revision TEXT)')
            db.execute('INSERT INTO storage_skill_versions VALUES(?,?)', ('used', revision))
        pack(self.root, self.package)
        dest = self.base / 'skills-restored'
        unpack(self.package, dest)
        repository = Path(__file__).resolve().parents[2]
        code = ('import {readConfig} from ' + json.dumps((repository / 'cli/instance.mjs').as_uri()) + ';'
                'import {run} from ' + json.dumps((repository / 'cli/skill-input.mjs').as_uri()) + ';'
                'const config=await readConfig(process.argv[1]);'
                'console.log(JSON.stringify(await run(config,{scope:"design"})));')
        process = subprocess.run(['node', '--max-old-space-size=256', '--input-type=module', '-e', code, str(dest)],
                                 check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(process.stdout)['content'], 'active skill; see assets/policy.txt')
        self.assertEqual((dest / 'skills/versions/used/assets/policy.txt').read_text(), 'used policy')

    def test_pack_rejects_live_writers_in_both_scopes_before_capture(self):
        from ascend_tools import storage_package
        live = json.dumps(storage_package._identity(os.getpid()))
        for project in (None, 'p'):
            for key in ('owner', 'skill-writer', 'memory-writer', 'knowledge-startup', 'supervisor-owner'):
                with self.subTest(project=project, writer=key):
                    with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
                        db.execute('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
                        db.execute('INSERT OR REPLACE INTO control VALUES(?,?)', (key, live))
                    with patch.object(storage_package, '_pack') as capture:
                        with self.assertRaisesRegex(ValueError, 'stopped writers'):
                            pack(self.root, self.package, project)
                        capture.assert_not_called()
                    with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
                        db.execute('DELETE FROM control')
            with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
                db.execute('UPDATE attempts SET owner=?', [live])
            with self.assertRaisesRegex(ValueError, 'stopped attempt writers'):
                pack(self.root, self.package, project)
            with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
                db.execute('UPDATE attempts SET owner=?', [json.dumps(dict(pid=os.getpid(),start='old',boot='old'))])
        self.assertFalse(self.package.exists())

    def test_campaign_plan_restore_keeps_cursor_and_frozen_prefix(self):
        plan = [dict(id='warmup', phase='phase1', new_projects=2),
                dict(id='research', phase='phase2', new_projects=3)]
        config_path = self.root / 'instance.json'
        config = dict(format=1, campaign_plan=plan)
        config_path.write_text(json.dumps(config))
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.execute("UPDATE projects SET spec=json_set(spec,'$.id',id)")
            db.execute('PRAGMA user_version=1')
            db.execute('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.execute('INSERT INTO control VALUES(?,?)',
                       ('campaign-plan', json.dumps(dict(plan=plan, index=1, started=1))))
        pack(self.root, self.package)
        dest = self.base / 'planned-restored'
        unpack(self.package, dest)
        repository = Path(__file__).resolve().parents[2]
        code = '''import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const {campaignPlanStatus,syncCampaignPlan}=await import(process.argv[2]);
const {State}=await import(process.argv[3]);
const root=process.argv[1],config=JSON.parse(readFileSync(root+'/instance.json','utf8'));
const state=new State(root+'/state.sqlite');
try {
 const progress=campaignPlanStatus(state,config);
 assert.equal(progress.currentIndex,1);assert.equal(progress.lockedPrefix,2);
 assert.deepEqual(progress.entries.map(e=>e.state),['completed','started']);
 syncCampaignPlan(state,config);
 const changed={...config,campaign_plan:config.campaign_plan.map((e,i)=>i===0?{...e,new_projects:4}:e)};
 assert.throws(()=>syncCampaignPlan(state,changed),/Cannot edit or remove a started campaign entry/);
 assert.equal(campaignPlanStatus(state,config).currentIndex,1);
}finally{state.close();}'''
        result = subprocess.run(['node', '--input-type=module', '-e', code, str(dest),
                                 (repository / 'controller/progress.mjs').as_uri(),
                                 (repository / 'controller/state.mjs').as_uri()],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_capture_accepts_cleared_owner_but_rejects_unresolved_identity(self):
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.execute('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.execute("INSERT INTO control VALUES('owner','null')")
        self.assertTrue(pack(self.root, self.package)['complete'])
        for owner in ({}, {'pid': os.getpid()}):
            with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
                db.execute("UPDATE control SET value=? WHERE key='owner'", [json.dumps(owner)])
            with self.assertRaisesRegex(ValueError, 'stopped writers'):
                pack(self.root, self.package)

    def test_capture_claim_spans_copy_rejects_competitors_and_releases_on_failure(self):
        from ascend_tools import storage_package
        def interrupted(*args):
            with closing(sqlite3.connect(self.root / 'state.sqlite')) as db:
                owner = json.loads(db.execute("SELECT value FROM control WHERE key='pack-capture'").fetchone()[0])
                self.assertEqual(owner, storage_package._identity(os.getpid()))
            # A separate process cannot take the live claim.
            code = ('from ascend_tools.storage_package import pack; import sys; '
                    'pack(sys.argv[1],sys.argv[2])')
            result = subprocess.run([sys.executable, '-c', code, str(self.root), str(self.base/'competing')],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Pack capture already active', result.stderr)
            raise InterruptedError('copy interrupted')
        with patch.object(storage_package, '_pack', side_effect=interrupted):
            with self.assertRaises(InterruptedError):
                pack(self.root, self.package)
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            self.assertIsNone(db.execute("SELECT value FROM control WHERE key='pack-capture'").fetchone())
            db.execute("INSERT INTO control VALUES('pack-capture',?)",
                       [json.dumps(dict(pid=os.getpid(),start='old',boot='old'))])
        with patch.object(storage_package, '_pack', return_value={'complete': True}):
            self.assertTrue(pack(self.root, self.package)['complete'])

    def test_queued_and_running_background_writers_refuse_capture(self):
        from ascend_tools import storage_package
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
            db.execute('CREATE TABLE requests(id TEXT,kind TEXT,status TEXT,owner TEXT)')
        for status, owner in [('queued', None), ('running', json.dumps(storage_package._identity(os.getpid())))]:
            with self.subTest(status=status):
                with closing(sqlite3.connect(self.root / 'state.sqlite')) as db, db:
                    db.execute('DELETE FROM requests')
                    db.execute('INSERT INTO requests VALUES(?,?,?,?)', ('writer', 'kdb', status, owner))
                with patch.object(storage_package, '_pack') as capture:
                    with self.assertRaisesRegex(ValueError, 'stopped background writers'):
                        pack(self.root, self.package)
                    capture.assert_not_called()

    def test_capture_process_death_allows_recovery(self):
        from ascend_tools import storage_package
        code = ('from ascend_tools.storage_package import _capture; from pathlib import Path; import os,sys; '
                'gate=_capture(Path(sys.argv[1])); gate.__enter__(); os._exit(0)')
        subprocess.run([sys.executable, '-c', code, str(self.root)], check=True)
        with patch.object(storage_package, '_pack', return_value={'complete': True}):
            self.assertTrue(pack(self.root, self.package)['complete'])
        with closing(sqlite3.connect(self.root / 'state.sqlite')) as db:
            self.assertIsNone(db.execute("SELECT value FROM control WHERE key='pack-capture'").fetchone())

    def test_interrupted_campaign_kdb_backup_ignores_partial_scratch(self):
        self.automatic_fixture()
        from ascend_tools import storage_package
        connect = storage_package._connect
        def interrupted_connect(path):
            source = connect(path)
            if path != self.root / 'knowledge.sqlite':
                return source
            proxy = Mock(wraps=source)
            def interrupt(out, **kwargs):
                out.execute('CREATE TABLE partial_backup(id TEXT)')
                out.commit()
                raise InterruptedError('during SQLite backup')
            proxy.backup.side_effect = interrupt
            return proxy
        with patch.object(storage_package, '_connect', interrupted_connect):
            with self.assertRaises(InterruptedError):
                pack(self.root, self.package)
        self.assertFalse((self.package / 'package.json').exists())
        pack(self.root, self.package)
        dest = self.base / 'kdb-restored'
        unpack(self.package, dest)
        with closing(sqlite3.connect(dest / 'knowledge.sqlite')) as db:
            self.assertEqual(db.execute('PRAGMA quick_check').fetchone()[0], 'ok')
            self.assertEqual(db.execute('SELECT id FROM entries').fetchall(), [('active-entry',)])

    def test_oversized_manifest_rejected_before_publication(self):
        with patch('ascend_tools.storage_package.MAX_JSON_BYTES', 4096):
            with self.assertRaisesRegex(ValueError, 'Package manifest exceeds portable size limit'):
                pack(self.root, self.package, 'p')
        self.assertFalse((self.package / 'package.json').exists())
        self.assertTrue((self.package / 'inspection.json').exists())
        self.assertTrue(pack(self.root, self.package, 'p')['complete'])
        unpack(self.package, self.base / 'manifest-restored')

    def test_used_revision_must_be_collected(self):
        path = self.root / 'package-inputs.json'
        inputs = json.loads(path.read_text())
        inputs['files'][0]['revision'] = 'different-version'
        path.write_text(json.dumps(inputs))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            pack(self.root, self.package, 'p')
        result = json.loads((self.package / 'inspection.json').read_text())
        self.assertIn({'kind': 'skills-revision', 'path': 's1'}, result['missing'])

    def test_checksum_failure_never_publishes_restored_destination(self):
        result = pack(self.root, self.package, 'p')
        result['files'][0]['archive']['sha256'] = '0' * 64
        (self.package / 'package.json').write_text(json.dumps(result))
        dest = self.base / 'restored'
        with self.assertRaises(ValueError):
            unpack(self.package, dest)
        self.assertFalse(dest.exists())

    def test_restic_retry_reuses_repository_and_batches_all_files(self):
        from ascend_tools import storage_package
        real = storage_package.archive_files
        def interrupt(*args, **kwargs):
            real(*args, **kwargs)
            raise InterruptedError('after native backup')
        with patch.object(storage_package, 'archive_files', interrupt):
            with self.assertRaises(InterruptedError):
                pack(self.root, self.package, 'p')
        self.assertFalse((self.package / 'package.json').exists())
        with patch.object(storage_package, 'archive_files', wraps=real) as backup:
            result = pack(self.root, self.package, 'p')
            self.assertEqual(backup.call_count, 1)
            self.assertGreater(len(backup.call_args.args[0]), 10)
        self.assertTrue(result['complete'])
        self.assertEqual(result['engine'], 'restic')
        self.assertTrue((self.package / 'config').is_file())
        self.assertFalse((self.package / 'chunks').exists())
        restore = storage_package.restore_files
        with patch.object(storage_package, 'restore_files', wraps=restore) as restored:
            unpack(self.package, self.base / 'batch-restored')
            self.assertEqual(restored.call_count, 1)
        self.log.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            pack(self.root, self.base / 'bad', 'p')


if __name__ == '__main__':
    unittest.main()
