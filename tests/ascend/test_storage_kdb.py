"""Small CPU fixtures for binary storage, provenance and snapshot selection."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

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

    def legacy(self):
        path = self.source('legacy.py', 'first vector\n')
        manifest = build_manifest(self.root,[path],collection='baseline',language='python',
                                  embedding_model='fixture',embedding_revision='1')
        legacy = self.root/'old.sqlite'
        with sqlite3.connect(legacy) as conn:
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
        with sqlite3.connect(legacy) as conn:
            conn.execute('UPDATE chunks SET vector_json=?', ('[NaN]',))
        destination = self.root/'partial.sqlite'
        with self.assertRaises(ValueError):
            import_legacy(legacy,destination,self.embedding)
        self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
