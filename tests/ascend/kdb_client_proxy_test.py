"""Real MCP self-requests must stay local even on a proxy-configured host."""
import asyncio
import http.server
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

import uvicorn

from ascend_tools.kdb import KDB
from ascend_tools.kdb_client import invoke
from ascend_tools.kdb_service import create_server


class Embeddings:
    model = 'proxy-fixture'
    revision = '1'

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_loopback_mcp_bypasses_proxy_but_remote_urls_retain_it(self):
        requests = []

        class RejectProxy(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(self.path)
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                self.send_response(502)
                self.send_header('Connection', 'close')
                self.end_headers()

            def log_message(self, *args):
                pass

        proxy = http.server.ThreadingHTTPServer(('127.0.0.1', 0), RejectProxy)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'answer.py').write_text('answer = 42\n')

                def backend():
                    db = KDB(root / 'knowledge.sqlite', Embeddings())
                    db.add_repo(str(root), 'proxy-smoke', 'python', kind='good', paths=['answer.py'])
                    return db

                app = create_server(backend).streamable_http_app()
                sock = socket.socket()
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
                server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
                serving = asyncio.create_task(server.serve(sockets=[sock]))
                try:
                    async with asyncio.timeout(15):
                        while not server.started:
                            if serving.done():
                                await serving
                            await asyncio.sleep(.01)
                    address = f'http://127.0.0.1:{proxy.server_port}'
                    environment = {key: address for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                                                            'http_proxy', 'https_proxy', 'all_proxy')}
                    environment.update(NO_PROXY='', no_proxy='')
                    with patch.dict(os.environ, environment):
                        result = await invoke(f'http://127.0.0.1:{port}/mcp', 'search',
                                              {'query': 'answer', 'kind': 'good', 'limit': 1})
                        self.assertEqual(result['structuredContent']['result'][0]['collection'], 'proxy-smoke')
                        self.assertEqual(requests, [], 'Instance self-requests must not reach the proxy')
                        with self.assertRaises(Exception):
                            await invoke('http://remote-kdb.invalid/mcp', 'search', {'query': 'answer'})
                        self.assertTrue(requests)
                        self.assertEqual(requests[0], 'http://remote-kdb.invalid/mcp')
                finally:
                    server.should_exit = True
                    await asyncio.wait_for(serving, 15)
                    sock.close()
        finally:
            proxy.shutdown()
            proxy.server_close()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
