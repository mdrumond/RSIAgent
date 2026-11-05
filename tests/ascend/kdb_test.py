"""Behavioral retrieval and native MCP transport integration tests."""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
import httpx

from ascend_tools.kdb import KDB
import kdb_fixture
from kdb_fixture import Embeddings


class KnowledgeTests(unittest.TestCase):
    def test_imported_language_baseline_is_universal_but_explicit_target_is_filtered(self):
        from ascend_tools.knowledge import KnowledgeDB

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'kernel.py'
            source.write_text('aligned vector memory loads\n')
            path = root / 'knowledge.sqlite'
            with KnowledgeDB(path, Embeddings()) as baseline:
                baseline.index(root, [source], collection='imported', language='triton',
                               kernel_languages=('triton',))
            source.unlink()
            db = KDB(path, Embeddings())
            self.assertEqual(db.db.connection.execute(
                'SELECT count(*) FROM knowledge_scopes').fetchone()[0], 0)
            hits = db.search('aligned loads', target='a3-triton')
            self.assertEqual([hit['collection'] for hit in hits], ['imported'])
            self.assertEqual(hits[0]['language'], 'triton')
            self.assertEqual(hits[0]['kernel_languages'], ['triton'])
            self.assertIsNone(hits[0]['target'])
            citation = db.read_entry('imported', hits[0]['chunk_id'])['evidence']
            lesson = db.record_lesson('A5-specific alignment advice', 'negative',
                                      'a5-catlass', citation, 'attempt-a5')
            self.assertEqual([hit['collection'] for hit in db.search(
                'aligned loads', target='a3-triton')], ['imported'])
            self.assertEqual({hit['collection'] for hit in db.search(
                'aligned loads', target='a5-catlass')}, {'imported', lesson['collection']})
            db.close()

    def test_scope_lessons_evidence_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'kernel.py'
            source.write_text('vector addition uses aligned loads\n')
            db = KDB(root / 'knowledge.sqlite', Embeddings())
            db.add_repo(str(root), 'baseline', 'triton', target='a3', paths=[source.name])
            evidence = db.search('aligned loads', kind='good', target='a3')[0]
            self.assertEqual(evidence['collection'], 'baseline')
            self.assertEqual(db.search('loads', target='a5'), [])
            record = db.record_lesson('avoid misaligned loads', 'negative', 'a3',
                                     [{'path': str(source)}], 'attempt-1')
            self.assertEqual(len(record['evidence'][0]['sha256']), 64)
            self.assertEqual(len(db.search('loads', kind='negative', target='a3')), 1)
            with self.assertRaises(ValueError):
                db.record_lesson('bad', 'negative', 'a3',
                                 [{'path': str(source), 'raw_refs': []}], 'attempt-1')
            with self.assertRaises(ValueError):
                db.record_lesson('bad', 'negative', 'a3',
                                 [{'path': str(source), 'sha256': 'wrong'}], 'attempt-1')
            db.close()
            db = KDB(root / 'knowledge.sqlite', Embeddings())
            self.assertEqual(len(db.search('loads', target='a3')), 2)
            db.close()

    def test_repeated_outbox_delivery_survives_reopen_without_duplicate_embeddings(self):
        class CountingEmbeddings(Embeddings):
            calls = 0

            def embed(self, texts):
                self.calls += 1
                return super().embed(texts)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'evidence.log'
            evidence.write_text('observed numeric verification failure\n')
            backend = CountingEmbeddings()
            arguments = dict(lesson='use aligned loads', outcome='good', applicability='a3',
                             evidence=[{'path': str(evidence)}], attempt_id='attempt-1')
            db = KDB(root / 'knowledge.sqlite', backend)
            first = db.record_lesson(**arguments)
            calls = backend.calls
            db.close()
            db = KDB(root / 'knowledge.sqlite', backend)
            self.assertEqual(db.record_lesson(**arguments), first)
            self.assertEqual(backend.calls, calls)
            other = db.record_lesson(**dict(arguments, lesson='avoid naive reductions'))
            self.assertNotEqual(other['entry_id'], first['entry_id'])
            self.assertEqual(db.db.connection.execute(
                'SELECT count(*) FROM collections').fetchone()[0], 2)
            db.close()

    def test_replay_finishes_file_staged_before_failed_index(self):
        class UnavailableEmbeddings(Embeddings):
            def embed(self, texts):
                raise RuntimeError('embedding process interrupted')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'evidence.log'
            evidence.write_text('direct verifier output\n')
            arguments = dict(lesson='avoid unaligned loads', outcome='negative',
                             applicability='a3', evidence=[{'path': str(evidence)}],
                             attempt_id='attempt-2')
            path = root / 'knowledge.sqlite'
            db = KDB(path, UnavailableEmbeddings())
            with self.assertRaises(RuntimeError):
                db.record_lesson(**arguments)
            staged = list((root / 'knowledge.sqlite.lessons').glob('*.json'))
            self.assertEqual(len(staged), 1)
            self.assertEqual(db.db.connection.execute(
                'SELECT count(*) FROM collections').fetchone()[0], 0)
            db.close()
            db = KDB(path, Embeddings())
            record = db.record_lesson(**arguments)
            self.assertEqual(staged[0].stem, record['delivery_id'])
            self.assertEqual(db.record_lesson(**arguments), record)
            hits = db.search('unaligned loads', kind='negative', target='a3')
            self.assertEqual(len(hits), 1)
            db.close()

    def test_retrieved_citation_chains_lessons_after_original_source_is_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'kernel.py'
            source.write_text('aligned vector memory loads\n')
            db = KDB(root / 'knowledge.sqlite', Embeddings())
            db.add_repo(str(root), 'baseline', 'triton', target='a3', paths=[source.name])
            hit = db.search('aligned loads', target='a3')[0]
            reference = dict(collection=hit['collection'], entry_id=hit['chunk_id'],
                             path=hit['path'], start_line=hit['start_line'],
                             end_line=hit['end_line'], sha256=hit['source_sha256'])
            source.unlink()
            lesson = db.record_lesson('avoid scalar loads', 'negative', 'a3',
                                      [reference], 'attempt-1')
            self.assertNotEqual(lesson['delivery_id'], lesson['entry_id'])
            chained = db.record_lesson('use vector alignment', 'good', 'a3',
                                       lesson['references'], 'attempt-2')
            self.assertEqual(chained['evidence'], lesson['references'])
            self.assertEqual(len(db.search('scalar', kind='negative', target='a3')), 1)
            with self.assertRaises(ValueError):
                db.record_lesson('wrong line', 'good', 'a3',
                                 [dict(reference, start_line=2)], 'attempt-3')
            with self.assertRaises(ValueError):
                db.record_lesson('wrong hash', 'good', 'a3',
                                 [dict(reference, sha256='wrong')], 'attempt-3')
            db.close()

    def test_lesson_returns_all_real_chunk_references_for_chaining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [root / f'kernel-{index}.py' for index in range(12)]
            for source in sources:
                source.write_text('aligned vector loads\n')
            db = KDB(root / 'knowledge.sqlite', Embeddings())
            db.add_repo(str(root), 'baseline', 'python', paths=[p.name for p in sources])
            references = [dict(collection=hit['collection'], entry_id=hit['chunk_id'],
                               path=hit['path'], start_line=hit['start_line'],
                               end_line=hit['end_line'], sha256=hit['source_sha256'])
                          for hit in db.search('vector', limit=12)]
            for source in sources:
                source.unlink()
            lesson = db.record_lesson('prefer aligned loads', 'good', 'a3',
                                      references, 'attempt-1')
            self.assertGreater(len(lesson['references']), 1)
            self.assertEqual(lesson['entry_id'], lesson['references'][0]['entry_id'])
            chained = db.record_lesson('retain complete evidence', 'good', 'a3',
                                       lesson['references'], 'attempt-2')
            self.assertEqual({ref['entry_id'] for ref in chained['evidence']},
                             {ref['entry_id'] for ref in lesson['references']})
            db.close()

    def test_read_entry_after_source_unlink_is_bounded_and_does_not_embed(self):
        class NoEmbeddings(Embeddings):
            def embed(self, texts):
                raise AssertionError('entry lookup must not embed')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'kernel.py'
            original = 'é中' * 5000
            source.write_text(original)
            path = root / 'knowledge.sqlite'
            db = KDB(path, Embeddings())
            db.add_repo(str(root), 'baseline', 'python', paths=[source.name])
            hit = db.search('loads')[0]
            source.unlink()
            db.close()
            db = KDB(path, NoEmbeddings())
            entry = db.read_entry('baseline', hit['chunk_id'])
            self.assertTrue(entry['truncated'])
            self.assertLessEqual(len(entry['text'].encode('utf-8')), 16384)
            self.assertTrue(original.startswith(entry['text']))
            self.assertEqual(entry['total_bytes'], len(original.encode('utf-8')))
            self.assertEqual(entry['sha256'], hit['source_sha256'])
            self.assertEqual(entry['evidence'][0]['entry_id'], hit['chunk_id'])
            self.assertEqual(entry['evidence'][0]['path'], 'kernel.py')
            with self.assertRaises(ValueError):
                db.read_entry('baseline', 'missing-id')
            cli_env = dict(os.environ, PYTHONPATH=os.pathsep.join([
                str(Path(__file__).resolve().parents[2]),
                str(Path(kdb_fixture.__file__).resolve().parent),
                os.environ.get('PYTHONPATH', '')]))
            cli = subprocess.run([sys.executable, '-m', 'ascend_tools.kdb_service',
                                  '--db', str(path), '--embedding-factory',
                                  'kdb_fixture:Embeddings', 'read-entry', '--arguments',
                                  json.dumps({'collection': 'baseline', 'entry_id': hit['chunk_id']})],
                                 check=True, capture_output=True, text=True, env=cli_env)
            self.assertEqual(json.loads(cli.stdout), entry)
            db.db.embeddings = Embeddings()
            lesson = db.record_lesson('retain indexed citations', 'good', 'a3',
                                      entry['evidence'], 'attempt-1')
            self.assertEqual(lesson['evidence'], entry['evidence'])
            db.close()

    def test_global_ranking_beats_unrelated_singleton_collections_and_embeds_query_once(self):
        class SemanticEmbeddings(Embeddings):
            queries = 0

            def embed(self, texts):
                def vector(text):
                    if text.startswith('prime'): return [1, 0]
                    if text.startswith('useful'): return [.99, .1]
                    if text.startswith('related'): return [.98, .2]
                    return [0, 1]
                return [vector(text) for text in texts]

            def embed_query(self, texts):
                self.queries += 1
                return [[1, 0] for text in texts]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = SemanticEmbeddings()
            db = KDB(root / 'knowledge.sqlite', backend)
            for collection, texts in [('baseline-a', ['prime context', 'useful context']),
                                      ('baseline-b', ['related context'])]:
                paths = []
                for index, text in enumerate(texts):
                    file = root / f'{collection}-{index}.txt'
                    file.write_text(text)
                    paths.append(file.name)
                db.add_repo(str(root), collection, 'python', target='a3', paths=paths)
            for index in range(4):
                file = root / f'unrelated-{index}.txt'
                file.write_text('unrelated context')
                db.add_repo(str(root), f'lesson-{index}', 'python', target='a3', paths=[file.name])
            for count, query in enumerate(('paraphrase', 'context'), 1):
                hits = db.search(query, target='a3', limit=3)
                self.assertEqual([hit['text'] for hit in hits],
                                 ['prime context', 'useful context', 'related context'])
                self.assertEqual(backend.queries, count)
                self.assertEqual([hit['vector_rank'] for hit in hits], [1, 2, 3])
            db.close()


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, read, write, root):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            self.assertEqual({tool.name for tool in tools.tools},
                             {'search', 'read_entry', 'add_repo', 'record_lesson',
                              'stage_memory_revision', 'activate_memory_revision', 'adopt_memory_revision'})
            result = await session.call_tool('add_repo', {
                'path': str(root), 'collection': 'cpu', 'language': 'triton',
                'target': 'a3', 'paths': ['kernel.py']})
            self.assertFalse(result.isError)
            result = await session.call_tool('search', {
                'query': 'vector addition', 'kind': 'good', 'target': 'a3'})
            self.assertFalse(result.isError)
            hit = result.structuredContent['result'][0]
            self.assertEqual(hit['collection'], 'cpu')
            reference = dict(collection=hit['collection'], entry_id=hit['chunk_id'],
                             path=hit['path'], start_line=hit['start_line'],
                             end_line=hit['end_line'], sha256=hit['source_sha256'])
            (root / 'kernel.py').unlink()
            restored = await session.call_tool('read_entry', {
                'collection': 'cpu', 'entry_id': hit['chunk_id']})
            self.assertFalse(restored.isError)
            self.assertEqual(restored.structuredContent['text'], 'vector addition\n')
            self.assertEqual(restored.structuredContent['evidence'], [reference])
            missing = await session.call_tool('read_entry', {
                'collection': 'cpu', 'entry_id': 'absent'})
            self.assertTrue(missing.isError)
            result = await session.call_tool('record_lesson', {
                'lesson': 'aligned loads', 'outcome': 'negative', 'applicability': 'a3',
                'evidence': [reference], 'attempt_id': 'cpu-1'})
            self.assertFalse(result.isError)
            lesson = result.structuredContent
            chained = await session.call_tool('record_lesson', {
                'lesson': 'remember aligned vector loads', 'outcome': 'good',
                'applicability': 'a3', 'evidence': lesson['references'], 'attempt_id': 'cpu-1'})
            self.assertFalse(chained.isError)
            await session.read_resource('kdb://knowledge/skill')
            adopted = await session.call_tool('adopt_memory_revision', {'revision':'baseline','collections':[]})
            self.assertEqual(adopted.structuredContent['generation'], 'baseline')
            staged = await session.call_tool('stage_memory_revision', {
                'revision': 'git-one', 'records': [dict(id='one', delivery_id='delivery-one',
                    lesson='memory vector alignment', evidence_refs=[reference], kind='good', target='a3')]})
            self.assertFalse(staged.isError)
            self.assertEqual(staged.structuredContent['generation'], 'git-one')
            names = set(staged.structuredContent['collections'])
            hidden = await session.call_tool('search', {'query':'memory', 'limit':50})
            self.assertTrue(names.isdisjoint(hit['collection'] for hit in hidden.structuredContent['result']))
            activated = await session.call_tool('activate_memory_revision', {'revision':'git-one'})
            self.assertEqual(activated.structuredContent['generation'], 'git-one')
            visible = await session.call_tool('search', {'query':'memory', 'limit':50, 'generation':'git-one'})
            self.assertTrue(names.issubset(hit['collection'] for hit in visible.structuredContent['result']))
            self.assertEqual({hit['generation'] for hit in visible.structuredContent['result']}, {'git-one'})

    async def test_real_stdio_and_http(self):
        for transport in ('stdio', 'streamable-http'):
            with self.subTest(transport=transport), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'kernel.py').write_text('vector addition\n')
                env = dict(os.environ, ASCEND_ATTEMPT_ID='cpu-1')
                env['PYTHONPATH'] = os.pathsep.join([
                    str(Path(__file__).parent), str(Path.cwd()), env.get('PYTHONPATH', '')])
                args = ['-m', 'ascend_tools.kdb_service', '--db', str(root / 'k.sqlite'),
                        '--embedding-factory', 'kdb_fixture:Embeddings', '--transport', transport]
                if transport == 'stdio':
                    async with stdio_client(StdioServerParameters(
                            command=sys.executable, args=args, env=env)) as (read, write):
                        await self.exercise(read, write, root)
                else:
                    with socket.socket() as sock:
                        sock.bind(('127.0.0.1', 0))
                        port = sock.getsockname()[1]
                    process = subprocess.Popen([sys.executable, *args, '--port', str(port)],
                                               env=env, stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL)
                    try:
                        async with httpx.AsyncClient() as client:
                            async with asyncio.timeout(15):
                                while True:
                                    if process.poll() is not None:
                                        self.fail('MCP service exited before readiness')
                                    try:
                                        response = await client.get(f'http://127.0.0.1:{port}/health')
                                        if response.status_code == 200:
                                            break
                                    except httpx.ConnectError:
                                        pass
                                    await asyncio.sleep(.05)
                        async with httpx.AsyncClient(headers={'X-Attempt-Id': 'cpu-1'}) as http:
                            async with streamable_http_client(
                                    f'http://127.0.0.1:{port}/mcp',
                                    http_client=http) as (read, write, _):
                                await self.exercise(read, write, root)
                    finally:
                        process.terminate()
                        await asyncio.to_thread(process.wait, timeout=10)
                db = KDB(root / 'k.sqlite', Embeddings())
                events = db.db.connection.execute(
                    'SELECT attempt_id,operation FROM knowledge_events').fetchall()
                self.assertEqual([row['operation'] for row in events],
                                 ['add_repo', 'search', 'read_entry', 'read_entry',
                                  'record_lesson', 'record_lesson', 'skill.read',
                                  'adopt_memory_revision', 'stage_memory_revision', 'search', 'activate_memory_revision', 'search'])
                self.assertTrue(all(row['attempt_id'] == 'cpu-1' for row in events))
                staged_event = json.loads(db.db.connection.execute(
                    "SELECT data_json FROM knowledge_events WHERE operation='stage_memory_revision'").fetchone()[0])
                self.assertEqual(staged_event['arguments'], {'revision':'git-one','record_count':1})
                last_event = json.loads(db.db.connection.execute(
                    'SELECT data_json FROM knowledge_events ORDER BY id DESC LIMIT 1').fetchone()[0])
                self.assertEqual(last_event['arguments']['generation'], 'git-one')
                db.close()


if __name__ == '__main__':
    unittest.main()
