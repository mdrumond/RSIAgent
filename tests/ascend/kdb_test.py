"""Behavioral retrieval and native MCP transport integration tests."""
import asyncio
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
from kdb_fixture import Embeddings


class KnowledgeTests(unittest.TestCase):
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


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, read, write, root):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            self.assertEqual({tool.name for tool in tools.tools},
                             {'search', 'add_repo', 'record_lesson'})
            result = await session.call_tool('add_repo', {
                'path': str(root), 'collection': 'cpu', 'language': 'triton',
                'target': 'a3', 'paths': ['kernel.py']})
            self.assertFalse(result.isError)
            result = await session.call_tool('search', {
                'query': 'vector addition', 'kind': 'good', 'target': 'a3'})
            self.assertFalse(result.isError)
            self.assertEqual(result.structuredContent['result'][0]['collection'], 'cpu')
            result = await session.call_tool('record_lesson', {
                'lesson': 'aligned loads', 'outcome': 'negative', 'applicability': 'a3',
                'evidence': [{'path': str(root / 'kernel.py')}], 'attempt_id': 'cpu-1'})
            self.assertFalse(result.isError)
            await session.read_resource('kdb://skill')

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
                        async with streamable_http_client(
                                f'http://127.0.0.1:{port}/mcp',
                                headers={'X-Attempt-Id': 'cpu-1'}) as (read, write, _):
                            await self.exercise(read, write, root)
                    finally:
                        process.terminate()
                        await asyncio.to_thread(process.wait, timeout=10)
                db = KDB(root / 'k.sqlite', Embeddings())
                events = db.db.connection.execute(
                    'SELECT attempt_id,operation FROM knowledge_events').fetchall()
                self.assertEqual([row['operation'] for row in events],
                                 ['add_repo', 'search', 'record_lesson', 'skill.read'])
                self.assertTrue(all(row['attempt_id'] == 'cpu-1' for row in events))
                db.close()


if __name__ == '__main__':
    unittest.main()
