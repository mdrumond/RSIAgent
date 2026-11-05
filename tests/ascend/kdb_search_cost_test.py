"""Exact retrieval reference comparisons; --benchmark emits elapsed telemetry only."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

import numpy as np
from unittest.mock import patch

from ascend_tools.kdb import KDB
from ascend_tools.knowledge import RRF_K, _fts_query


class Embeddings:
    model, revision = 'deterministic-cost-fixture', '1'

    def __init__(self, dimensions=16):
        self.dimensions = dimensions

    def embed(self, texts):
        return [[(hashlib.sha256(text.encode()).digest()[i % 32] - 127) / 127
                 for i in range(self.dimensions)] for text in texts]


def cosine(left, right):
    norm = math.sqrt(sum(x*x for x in left)) * math.sqrt(sum(x*x for x in right))
    return sum(a*b for a, b in zip(left, right)) / norm if norm else 0.0


def reference(db, query, kind='all', target=None, limit=10):
    """Pre-optimization scalar full-corpus cosine + global lexical RRF contract."""
    vector = db.db.embeddings.embed([query])[0]
    norm = math.sqrt(sum(x*x for x in vector))
    vector = [x/norm for x in vector] if norm else vector
    candidates, keywords = [], set()
    scopes = db.db.connection.execute('''SELECT c.name,s.kind,s.target FROM collections c
        LEFT JOIN knowledge_scopes s ON c.name=s.collection''').fetchall()
    for scope in scopes:
        name, scoped_kind = scope['name'], scope['kind'] or 'good'
        if kind != 'all' and kind != scoped_kind:
            continue
        if target is not None and scope['target'] is not None and target != scope['target']:
            continue
        table, lexical = db.db._fts_table_name(name), _fts_query(query)
        if lexical:
            keywords.update((name, row[0]) for row in db.db.connection.execute(
                f'SELECT c.chunk_id FROM {table} CROSS JOIN chunks c ON c.rowid={table}.rowid '
                f'WHERE {table} MATCH ? AND c.collection=?', (lexical, name)))
        for row in db.db.connection.execute('SELECT * FROM chunks WHERE collection=?', (name,)):
            candidates.append((name, row, cosine(vector, np.frombuffer(row['vector'], dtype='<f4').astype(float)), scoped_kind, scope['target']))
    candidates.sort(key=lambda row: (-row[2], row[0], row[1]['chunk_id']))
    lexical_ranks = {key: i for i, key in enumerate(
        ((name, row['chunk_id']) for name, row, *_ in candidates if (name, row['chunk_id']) in keywords), 1)}
    result = []
    for rank, (name, row, _, scoped_kind, scoped_target) in enumerate(candidates, 1):
        lexical_rank = lexical_ranks.get((name, row['chunk_id']))
        score = 1/(RRF_K+rank) + (1/(RRF_K+lexical_rank) if lexical_rank else 0)
        manifest = db.db.manifest(name)
        result.append(dict({key: row[key] for key in ('chunk_id', 'path', 'start_line', 'end_line', 'text')},
            collection=name, kind=scoped_kind, target=scoped_target, language=manifest.language,
            kernel_languages=list(manifest.kernel_languages), score=score, vector_rank=rank,
            lexical_rank=lexical_rank, lexical_score=None,
            source_sha256=db.db.source_fingerprint(name, row['path']).sha256))
    return sorted(result, key=lambda hit: (-hit['score'], hit['collection'], hit['chunk_id']))[:limit]


def add(db, root, name, kind='good', target=None, lines=400):
    source = root / (name + '.py')
    source.write_text(''.join(f'{name} aligned loads observation {i}\n' for i in range(lines)))
    db.add_repo(str(root), name, 'python', target=target, kind=kind, paths=[source.name])
    return source


class SearchCostTests(unittest.TestCase):
    def test_global_reference_all_scopes_backfill_and_top_only_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = KDB(root/'db.sqlite', Embeddings())
            self.addCleanup(db.close)
            for name, kind, target in [('universal', 'good', None), ('universal-neg', 'negative', None),
                                        ('a3', 'good', 'a3'), ('a5', 'negative', 'a5')]:
                add(db, root, name, kind, target)
            for query in ['aligned loads', 'no-keyword-overlap', '?']:
                for kind in ['all', 'good', 'negative']:
                    for target in [None, 'a3', 'a5']:
                        for limit in [1, 7, 50]:
                            with self.subTest(query=query, kind=kind, target=target, limit=limit):
                                expected = reference(db, query, kind, target, limit)
                                with patch.object(db.db, 'source_fingerprint', wraps=db.db.source_fingerprint) as fingerprints:
                                    self.assertEqual(db.search(query, kind, target, limit), expected)
                                    self.assertEqual(fingerprints.call_count, len(expected))
            evidence = db.search('aligned loads', limit=1)[0]
            self.assertEqual(db.read_entry(evidence['collection'], evidence['chunk_id'])['evidence'][0]['sha256'],
                             evidence['source_sha256'])

    def test_warm_cache_survives_own_events_and_new_repo_lesson_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = KDB(root/'db.sqlite', Embeddings())
            self.addCleanup(db.close)
            source = add(db, root, 'base')
            expected = db.search('aligned loads')
            db.event('search', 'one', status='completed')
            with patch('ascend_tools.knowledge.json.loads', side_effect=AssertionError('Unexpected warm vector parsing')):
                self.assertEqual(db.search('aligned loads'), expected)
            db.record_lesson('Avoid aligned loads on this case', 'negative', 'a3',
                             [{'path': str(source)}], 'lesson-one')
            add(db, root, 'new-repo', target='a3')
            self.assertEqual(db.search('loads', 'all', 'a3', 50), reference(db, 'loads', 'all', 'a3', 50))
            self.assertTrue(any(hit['kind'] == 'negative' for hit in db.search('loads', 'negative', 'a3')))
            source.unlink()  # Search and citations use immutable indexed evidence, not the checkout.
            self.assertEqual(db.search('loads', 'good', 'a3'), reference(db, 'loads', 'good', 'a3'))

    def test_external_content_write_invalidates_warm_vectors_and_readonly_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = KDB(root/'db.sqlite', Embeddings())
            self.addCleanup(db.close)
            add(db, root, 'base')
            db.search('aligned')
            view = db.db.read_only_view()
            self.addCleanup(view.close)
            before = view.cosine_scores('base', [1]*16)
            with sqlite3.connect(db.db.path) as other:
                other.execute('UPDATE chunk_vectors SET vector=? WHERE content_id IN '
                              '(SELECT content_id FROM chunk_provenance WHERE collection=?)',
                              (np.zeros(16, dtype='<f4').tobytes(), 'base'))
            self.assertNotEqual(view.cosine_scores('base', [1]*16), before)
            self.assertEqual(db.search('aligned'), reference(db, 'aligned'))

    def test_zero_vectors_ties_dimensions_and_bounded_eviction(self):
        class Zero(Embeddings):
            def embed(self, texts): return [[0]*16 for _ in texts]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = KDB(root/'db.sqlite', Zero())
            self.addCleanup(db.close)
            add(db, root, 'b'); add(db, root, 'a', kind='negative')
            self.assertEqual(db.search('?', limit=50), reference(db, '?', limit=50))
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                db.db.cosine_scores('a', [0])
            budget = max(matrix.nbytes for matrix in db.db._vectors.values())
            db.db._vectors.clear(); db.db._vector_bytes = 0
            with patch('ascend_tools.knowledge.VECTOR_CACHE_BYTES', budget):
                self.assertEqual(db.search('loads', limit=50), reference(db, 'loads', limit=50))
                self.assertLessEqual(db.db._vector_bytes, budget)
                self.assertEqual(len(db.db._vectors), 1)

    def test_identical_nonzero_cosines_use_collection_and_chunk_ties(self):
        class Constant(Embeddings):
            def embed(self, texts): return [[1, 0, 0] for _ in texts]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = KDB(root/'db.sqlite', Constant())
            self.addCleanup(db.close)
            add(db, root, 'z'); add(db, root, 'a', kind='negative')
            self.assertEqual(db.search('?', limit=50), reference(db, '?', limit=50))


def benchmark(chunks=12000, dimensions=384):
    """Synthetic elapsed telemetry, not a host-speed-dependent unit-test gate."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        db = KDB(root/'db.sqlite', Embeddings(dimensions))
        try:
            for name in ['positive', 'negative']:
                add(db, root, name, kind='negative' if name == 'negative' else 'good', lines=chunks*30)
            report = {'chunks': db.db.connection.execute('SELECT count(*) FROM chunks').fetchone()[0],
                      'dimensions': dimensions, 'matrix_bytes': 0, 'elapsed_seconds': {}}
            results = {}
            for name, operation in [('reference', lambda: reference(db, 'aligned loads')),
                                    ('cold', lambda: db.search('aligned loads')),
                                    ('warm', lambda: db.search('aligned loads'))]:
                started = time.perf_counter(); results[name] = operation()
                report['elapsed_seconds'][name] = time.perf_counter() - started
            assert results['reference'] == results['cold'] == results['warm']
            report['matrix_bytes'] = db.db._vector_bytes
            return report
        finally:
            db.close()


if __name__ == '__main__':
    import sys
    if '--benchmark' in sys.argv:
        print(json.dumps(benchmark(), indent=2))
    else:
        unittest.main()
