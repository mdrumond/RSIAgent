"""Small CPU fixtures for binary storage, provenance and snapshot selection."""
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ascend_tools.kdb import KDB
from ascend_tools.kdb_import import import_legacy
from ascend_tools.knowledge import KnowledgeDB, build_manifest, chunk_source


class Embeddings:
    model, revision = 'fixture', '1'

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.extend(texts)
        return [[1., .125, 0.] if 'first' in text else [0., .125, 1.] for text in texts]


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.embedding = Embeddings()
        self.kdb = KDB(self.root/'new.sqlite', self.embedding)
        self.addCleanup(self.kdb.close)

    def source(self, name, text):
        path = self.root/name
        path.write_text(text)
        return path

    def add(self, collection, name='a.py', text='first vector\n'):
        self.source(name, text)
        return self.kdb.add_repo(str(self.root), collection, 'python', paths=[name])

    def test_float32_roundtrip_ranking_dedup_and_provenance(self):
        manifest = self.add('one')
        self.add('two', 'copy.py')
        self.add('three', 'other.py', 'second vector\n')
        self.assertEqual(len(self.embedding.calls), 2)
        conn = self.kdb.db.connection
        self.assertEqual(conn.execute('SELECT count(*) FROM chunk_contents').fetchone()[0], 2)
        self.assertEqual(conn.execute('SELECT count(*) FROM chunk_vectors').fetchone()[0], 2)
        self.assertEqual(conn.execute('SELECT count(*) FROM chunk_provenance').fetchone()[0], 3)
        row = conn.execute('SELECT vector FROM chunks WHERE collection="one"').fetchone()
        self.assertEqual(len(row[0]), 12)
        self.assertEqual(np.frombuffer(row[0], dtype='<f4').dtype.itemsize, 4)
        with KnowledgeDB.open_read_only(self.kdb.db.path, self.embedding) as reopened:
            self.assertEqual(reopened.manifest('one'), build_manifest(self.root, [self.root/'a.py'],
                collection='one', language='python', embedding_model='fixture', embedding_revision='1'))
            self.assertEqual(reopened.manifest('one').fingerprint, manifest['fingerprint'])
            near = reopened.cosine_scores('one', [1,0,0])[0][1]
            far = reopened.cosine_scores('three', [1,0,0])[0][1]
            self.assertGreater(near, far)
        hits = self.kdb.search('first', limit=3)
        self.assertEqual({h['path'] for h in hits[:2]}, {'a.py','copy.py'})
        self.assertNotEqual(hits[0]['chunk_id'], hits[1]['chunk_id'])
        self.assertEqual(self.kdb.read_entry(hits[0]['collection'], hits[0]['chunk_id'])['text'], 'first vector\n')

    def test_snapshot_replacement_pins_and_old_citations(self):
        self.source('a.py','first vector\n')
        self.kdb.replace_repo('repo', str(self.root), 'old', 'python', paths=['a.py'])
        citation = self.kdb.search('first')[0]
        self.kdb.pin_generation('git-one', ['old'])
        self.kdb.pin_generation('git-one', ['old'])
        with self.assertRaises(ValueError):
            self.kdb.pin_generation('git-one', [])
        self.kdb.activate_generation('git-one')
        with self.assertRaises(ValueError):
            self.kdb.release_generation('git-one')
        self.source('a.py','second vector\n')
        self.kdb.replace_repo('repo', str(self.root), 'new', 'python', paths=['a.py'])
        self.assertEqual({h['collection'] for h in self.kdb.search('first')}, {'new'})
        self.assertEqual(self.kdb.read_entry('old',citation['chunk_id'])['text'], 'first vector\n')
        self.assertEqual(self.kdb.generation_collections('git-one'), ('old',))
        with self.assertRaises(ValueError):
            self.kdb.replace_repo('repo', str(self.root), 'new', 'python', paths=['missing'])
        self.assertEqual({h['collection'] for h in self.kdb.search('second')}, {'new'})

    def test_prune_keeps_cited_chunks_and_live_generation_then_reclaims_space(self):
        # Multiple distinct chunks allow partial citation retention in one snapshot.
        self.source('a.py', ''.join(f'first unique line {i}\n' for i in range(250)))
        self.kdb.replace_repo('repo', str(self.root), 'old', 'python', paths=['a.py'])
        rows = self.kdb.db.connection.execute(
            'SELECT chunk_id FROM chunks WHERE collection="old" ORDER BY start_line').fetchall()
        cited = rows[0][0]
        self.kdb.read_entry('old', cited)
        self.kdb.pin_generation('live', ['old'])
        self.source('a.py','second vector\n')
        self.kdb.replace_repo('repo', str(self.root), 'new', 'python', paths=['a.py'])
        self.assertEqual(self.kdb.prune_snapshots()['removed_chunks'], 0)
        self.assertEqual(self.kdb.db.connection.execute(
            'SELECT count(*) FROM chunks WHERE collection="old"').fetchone()[0], len(rows))
        self.kdb.release_generation('live')
        self.assertEqual(self.kdb.prune_snapshots()['removed_chunks'], len(rows)-1)
        with self.assertRaises(ValueError):
            self.kdb.pin_generation('too-late', ['old'])
        with self.assertRaises(ValueError):
            self.kdb.replace_repo('repo',str(self.root),'old','python',paths=['a.py'])
        retained = self.kdb.read_entry('old', cited)
        self.assertTrue(retained['text'].startswith('first unique line 0'))
        with self.assertRaises(ValueError):
            self.kdb.read_entry('old',rows[-1][0])
        self.assertEqual({h['collection'] for h in self.kdb.search('first')}, {'new'})
        self.assertEqual(self.kdb.db.connection.execute('SELECT count(*) FROM chunk_contents').fetchone()[0], 2)
        self.assertEqual(self.kdb.db.connection.execute('SELECT count(*) FROM chunk_vectors').fetchone()[0], 2)
        self.assertEqual(self.kdb.db.connection.execute('PRAGMA freelist_count').fetchone()[0], 0)
        self.assertEqual(self.kdb.prune_snapshots()['removed_chunks'], 0)

    def test_prune_removes_unreferenced_snapshot_and_retains_shared_content(self):
        self.source('a.py','first vector\n')
        for name in ('old','new'):
            self.kdb.replace_repo('repo',str(self.root),name,'python',paths=['a.py'])
        before = self.kdb.db.path.stat().st_size
        self.assertEqual(self.kdb.prune_snapshots()['removed_chunks'], 1)
        self.assertLess(self.kdb.db.path.stat().st_size, before)
        with self.assertRaises(KeyError):
            self.kdb.db.manifest('old')
        self.assertEqual(self.kdb.search('first')[0]['collection'], 'new')
        self.assertEqual(self.kdb.db.connection.execute('SELECT count(*) FROM chunk_contents').fetchone()[0], 1)

    def record(self, identity, lesson='first lesson', kind='good'):
        return dict(id=identity, delivery_id='delivery-'+identity, lesson=lesson,
                    evidence_refs=[{'path':'evidence.log','sha256':'abc'}], kind=kind, target='a3')

    def test_memory_stage_atomic_activation_reuse_and_exact_generation_queries(self):
        self.add('repository')
        record = dict(self.record('one'), evidence=[{'verbose':'retained in Git only'}])
        first = self.kdb.stage_memory_revision('git-one', [record])
        self.assertEqual({h['collection'] for h in self.kdb.search('first')}, {'repository'})
        count = len(self.embedding.calls)
        self.assertEqual(self.kdb.stage_memory_revision('git-one',[record]), first)
        self.assertEqual(len(self.embedding.calls), count)
        self.kdb.activate_memory_revision('git-one')
        second = self.kdb.stage_memory_revision('git-two', [record, self.record('two','second lesson','negative')])
        self.assertEqual(len(self.embedding.calls), count+1)
        self.assertTrue(set(first['collections']) < set(second['collections']))
        self.assertFalse(any(h['kind']=='negative' for h in self.kdb.search('lesson')))
        self.kdb.activate_memory_revision('git-two')
        hits = self.kdb.search('lesson', generation='git-one')
        self.assertEqual({h['generation'] for h in hits}, {'git-one'})
        self.assertEqual({h['collection'] for h in hits}, {'repository', *first['collections']})
        current = self.kdb.search('lesson')
        self.assertEqual({h['generation'] for h in current}, {'git-two'})
        self.assertTrue(any(h['kind']=='negative' for h in current))
        negative = next(h for h in current if h['kind']=='negative')
        with self.assertRaises(ValueError):
            self.kdb.read_entry(negative['collection'], negative['chunk_id'], generation='git-one')
        self.assertEqual(self.kdb.read_entry(negative['collection'],negative['chunk_id'],
                                           generation='git-two')['generation'], 'git-two')
        self.kdb.stage_memory_revision('git-empty', [])
        self.kdb.activate_memory_revision('git-empty')
        self.assertEqual({h['collection'] for h in self.kdb.search('lesson')}, {'repository'})
        self.assertEqual(self.kdb.read_entry(negative['collection'],negative['chunk_id'])['generation'], 'git-two')
        self.kdb.release_generation('git-one')
        with self.assertRaises(KeyError):
            self.kdb.search('lesson',generation='git-one')

    def test_memory_failed_stage_stays_hidden_and_revision_membership_is_immutable(self):
        record = self.record('one')
        with patch.object(self.embedding, 'embed', side_effect=RuntimeError('encoder failed')):
            with self.assertRaises(RuntimeError):
                self.kdb.stage_memory_revision('git-one',[record])
        self.assertEqual(self.kdb.search('first'), [])
        with self.assertRaises(KeyError):
            self.kdb.activate_memory_revision('git-one')
        self.kdb.stage_memory_revision('git-one',[record])
        with self.assertRaises(ValueError):
            self.kdb.stage_memory_revision('git-one',[self.record('two')])
        self.kdb.activate_memory_revision('git-one')
        self.assertEqual(len(self.kdb.search('first')), 1)

    def test_baseline_adoption_is_atomic_reuses_vectors_and_retires_after_publication(self):
        self.source('lesson.json','first baseline lesson')
        self.kdb.add_repo(str(self.root),'baseline','lesson',target='a3',paths=['lesson.json'])
        self.add('repository')
        with patch.object(self.embedding, 'embed', side_effect=AssertionError('must not embed')):
            with self.assertRaises(ValueError):
                self.kdb.adopt_memory_revision('bad',['baseline','repository'])
            self.assertIsNone(self.kdb.current_memory_revision())
            adopted = self.kdb.adopt_memory_revision('git-baseline',['baseline'])
            self.assertEqual(self.kdb.adopt_memory_revision('git-baseline',['baseline']), adopted)
        self.kdb.stage_memory_revision('git-new',[])
        self.kdb.activate_memory_revision('git-new')
        self.assertEqual({h['collection'] for h in self.kdb.search('first')}, {'repository'})
        old = self.kdb.search('first',generation='git-baseline')
        self.assertEqual({h['collection'] for h in old}, {'repository','baseline'})
        self.assertEqual({h['generation'] for h in old}, {'git-baseline'})

    def legacy(self):
        path = self.source('legacy.py', 'first vector\n')
        manifest = build_manifest(self.root,[path],collection='baseline',language='python',
                                  embedding_model='fixture',embedding_revision='1')
        legacy = self.root/'old.sqlite'
        with closing(sqlite3.connect(legacy)) as conn, conn:
            conn.executescript('''CREATE TABLE collections(name,language,fingerprint,manifest_json,fts_table);
                CREATE TABLE chunks(collection,chunk_id,path,start_line,end_line,text,vector_json);''')
            conn.execute('INSERT INTO collections VALUES(?,?,?,?,?)',
                         ('baseline','python',manifest.fingerprint,manifest.to_json(),'ignored'))
            chunk = chunk_source('legacy.py',path.read_bytes())[0]
            conn.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?)',
                ('baseline',chunk.chunk_id,chunk.path,chunk.start_line,chunk.end_line,chunk.text,'[1,0,0]'))
        return legacy, chunk, manifest

    def test_streaming_legacy_import_new_destination_only(self):
        legacy, chunk, manifest = self.legacy()
        original = legacy.read_bytes()
        cli_destination = self.root/'cli.sqlite'
        subprocess.run([sys.executable, '-m', 'ascend_tools.kdb_import', '--source',str(legacy),
                        '--destination',str(cli_destination)], check=True, capture_output=True)
        with KnowledgeDB.open_read_only(cli_destination, self.embedding) as view:
            self.assertEqual(view.manifest('baseline'), manifest)
        with self.assertRaises(ValueError):
            KnowledgeDB(legacy, self.embedding)
        destination = self.root/'import.sqlite'
        import_legacy(legacy,destination,self.embedding,batch_size=1)
        imported = KDB(destination, self.embedding)
        self.addCleanup(imported.close)
        self.assertEqual(imported.db.manifest('baseline'),manifest)
        self.assertEqual(imported.search('first')[0]['chunk_id'],chunk.chunk_id)
        self.assertEqual(imported.read_entry('baseline',chunk.chunk_id)['text'],chunk.text)
        self.assertEqual(legacy.read_bytes(),original)
        with self.assertRaises(FileExistsError):
            import_legacy(legacy,destination,self.embedding)
        with self.assertRaises(FileExistsError):
            import_legacy(legacy,legacy,self.embedding)

    def test_bad_import_removes_partial_destination(self):
        legacy, _, _ = self.legacy()
        with closing(sqlite3.connect(legacy)) as conn, conn:
            conn.execute('UPDATE chunks SET vector_json=?', ('[NaN]',))
        destination = self.root/'partial.sqlite'
        with self.assertRaises(ValueError):
            import_legacy(legacy,destination,self.embedding)
        self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
