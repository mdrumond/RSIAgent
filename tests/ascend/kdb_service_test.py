"""Real MCP requests against a blocked, thread-owned SQLite backend."""
import asyncio
from contextlib import asynccontextmanager
import sqlite3
import threading
import unittest

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from ascend_tools.kdb_service import create_server


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_search_keeps_protocol_responsive_and_serializes_backend(self):
        started, release, closed = (threading.Event() for _ in range(3))
        calls = []
        instances = []

        class Backend:
            def __init__(self):
                self.db = sqlite3.connect(':memory:')
                self.db.execute('create table events(operation, attempt)')
                calls.append(('construct', threading.get_ident()))
                instances.append(self)

            def search(self, **kwargs):
                calls.append(('search', threading.get_ident()))
                started.set()
                if not release.wait(15):
                    raise TimeoutError('test did not release search')
                return [{'text': kwargs['query']}]

            def event(self, operation, attempt_id, **kwargs):
                self.db.execute('insert into events values (?, ?)', (operation, attempt_id))
                calls.append((operation, threading.get_ident()))

            def close(self):
                self.events = self.db.execute('select * from events').fetchall()
                self.db.close()
                calls.append(('close', threading.get_ident()))
                closed.set()

        server = create_server(Backend)
        app = server.streamable_http_app()

        @asynccontextmanager
        async def connection(attempt_id=None):
            headers = {'x-attempt-id': attempt_id} if attempt_id else {}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         headers=headers) as client:
                async with streamable_http_client(
                        'http://localhost/mcp', http_client=client) as streams:
                    yield streams

        async def session_work():
            async with connection('slow') as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool('search', {'query': 'retained text'})

        try:
            async with app.router.lifespan_context(app):
                search = asyncio.create_task(session_work())
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                try:
                    # These must finish BEFORE releasing the blocked backend.
                    async with asyncio.timeout(5):
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
                            response = await client.get('http://localhost/health')
                            self.assertTrue(response.json()['ready'])
                        async with connection() as (read, write, _):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                tools = await session.list_tools()
                                self.assertEqual(len(tools.tools), 4)
                    self.assertFalse(closed.is_set())
                finally:
                    release.set()
                result = await search
                self.assertFalse(result.isError)
                async with connection('resource') as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        await session.read_resource('kdb://knowledge/skill')
        finally:
            release.set()
        self.assertTrue(closed.is_set())
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].events, [('search', 'slow'), ('skill.read', 'resource')])
        self.assertEqual(len({thread for _, thread in calls}), 1)
        self.assertNotEqual(calls[0][1], threading.get_ident())

    async def test_cancelled_search_drains_before_backend_close(self):
        started, release, closed = (threading.Event() for _ in range(3))
        order = []

        class Backend:
            def __init__(self):
                self.db = sqlite3.connect(':memory:')

            def search(self, **kwargs):
                started.set()
                if not release.wait(15):
                    raise TimeoutError('test did not release search')
                self.db.execute('select 1')
                order.append('finished')
                return []

            def event(self, *args, **kwargs):
                self.db.execute('select 1')
                order.append('event')

            def close(self):
                self.db.close()
                order.append('closed')
                closed.set()

        server = create_server(Backend)
        app = server.streamable_http_app()
        async with app.router.lifespan_context(app):
            # SDK tool invocation exercises the same handler/executor without
            # an HTTP disconnect hiding the request task's cancellation.
            from unittest.mock import patch
            with patch.object(server, 'get_context') as context:
                context.return_value.request_context.request = None
                request = asyncio.create_task(server.call_tool('search', {'query': 'slow'}))
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                request.cancel()
                try:
                    await request
                except asyncio.CancelledError:
                    pass
                self.assertFalse(closed.is_set())

                async def release_during_shutdown():
                    await asyncio.sleep(0)
                    self.assertFalse(closed.is_set())
                    release.set()

                releaser = asyncio.create_task(release_during_shutdown())
        await releaser
        self.assertTrue(closed.is_set())
        self.assertEqual(order, ['finished', 'event', 'closed'])
