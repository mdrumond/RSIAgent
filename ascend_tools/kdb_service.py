"""Standard MCP SDK adapter and CLI for one instance-owned KDB process."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from .kdb import KDB


def create_server(backend_factory, *, port=8765):
    from starlette.responses import JSONResponse

    executor = None
    backend = None

    async def dispatch(function, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            executor, partial(function, *args, **kwargs))

    @asynccontextmanager
    async def backend_lifespan():
        nonlocal executor, backend
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ascend-kdb')
        construction = executor.submit(backend_factory)
        try:
            backend = await asyncio.shield(asyncio.wrap_future(construction))
            yield
        finally:
            # FIFO drains running operations before closing SQLite. Retain the
            # constructor even when startup's caller stops waiting for it.
            def close():
                if not construction.cancelled() and construction.exception() is None:
                    construction.result().close()
            closing = executor.submit(close)
            try:
                await asyncio.shield(asyncio.wrap_future(closing))
            finally:
                executor.shutdown(wait=True)
                backend = executor = None

    class BackendServer(FastMCP):
        # SDK low-level lifespan runs per stateless request. The shared backend
        # belongs to the application/transport lifespan outside those scopes.
        def streamable_http_app(self):
            app = super().streamable_http_app()
            sdk_lifespan = app.router.lifespan_context

            @asynccontextmanager
            async def lifespan(app):
                async with backend_lifespan(), sdk_lifespan(app):
                    yield

            app.router.lifespan_context = lifespan
            return app

        async def run_stdio_async(self):
            async with backend_lifespan():
                await super().run_stdio_async()

    server = BackendServer('ascend-kdb', host='127.0.0.1', port=port,
                     stateless_http=True, json_response=True)

    def attempt(ctx):
        request = ctx.request_context.request
        return (request.headers.get('x-attempt-id') if request is not None
                else os.environ.get('ASCEND_ATTEMPT_ID'))

    def invoke_backend(attempt_id, operation, args):
        try:
            result = getattr(backend, operation)(**args)
        except Exception as exc:
            backend.event(operation, attempt_id, status='failed', error=type(exc).__name__)
            raise
        event_args = dict(args)
        if operation in ('search', 'read_entry') and hasattr(backend, 'current_memory_revision'):
            event_args['generation'] = args.get('generation') or backend.current_memory_revision()
        if operation == 'stage_memory_revision':
            event_args = {'revision': args['revision'], 'record_count': len(args['records'])}
        backend.event(operation, attempt_id, status='completed', arguments=event_args,
                      result_count=len(result) if isinstance(result, list) else 1)
        return result

    async def invoke(ctx, operation, **args):
        return await dispatch(invoke_backend, attempt(ctx), operation, args)

    @server.tool()
    async def search(query: str, ctx: Context, kind: str = 'all',
                     target: str | None = None, limit: int = 10,
                     generation: str | None = None) -> list[dict]:
        """Search positive/negative source knowledge within a target scope."""
        return await invoke(ctx, 'search', query=query, kind=kind, target=target, limit=limit,
                            generation=generation)

    @server.tool()
    async def read_entry(collection: str, entry_id: str, ctx: Context, generation: str | None = None) -> dict[str, Any]:
        """Read stored chunk text (up to16KiB UTF-8) and direct validated evidence."""
        return await invoke(ctx, 'read_entry', collection=collection, entry_id=entry_id, generation=generation)

    @server.tool()
    async def add_repo(path: str, collection: str, language: str, ctx: Context,
                       target: str | None = None, kind: str = 'good',
                       paths: list[str] | None = None) -> dict[str, Any]:
        """Index selected repository files as an immutable collection."""
        return await invoke(ctx, 'add_repo', path=path, collection=collection,
                      language=language, target=target, kind=kind, paths=paths)

    @server.tool()
    async def record_lesson(lesson: str, outcome: str, applicability: str,
                            evidence: list[dict], attempt_id: str, ctx: Context) -> dict[str, Any]:
        """Store a compact lesson backed by direct evidence references."""
        correlation = attempt(ctx)
        if correlation is not None and correlation != attempt_id:
            raise ValueError('attempt_id differs from request correlation')
        return await invoke(ctx, 'record_lesson', lesson=lesson, outcome=outcome,
                      applicability=applicability, evidence=evidence, attempt_id=attempt_id)

    @server.tool()
    async def stage_memory_revision(revision: str, records: list[dict], ctx: Context) -> dict[str, Any]:
        """Controller publication: index a memory revision without activating it."""
        return await invoke(ctx, 'stage_memory_revision', revision=revision, records=records)

    @server.tool()
    async def activate_memory_revision(revision: str, ctx: Context) -> dict[str, Any]:
        """Controller publication: atomically activate a staged memory revision."""
        return await invoke(ctx, 'activate_memory_revision', revision=revision)

    @server.resource('kdb://knowledge/skill')
    async def skill() -> str:
        correlation = attempt(server.get_context())

        def read():
            backend.event('skill.read', correlation, resource='kdb://knowledge/skill')
            return (Path(__file__).parent / 'templates/kdb/SKILL.md').read_text()

        return await dispatch(read)

    @server.custom_route('/health', methods=['GET'])
    async def health(request):
        return JSONResponse({'ready': True, 'service': 'ascend-kdb'})

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--transport', choices=['stdio', 'streamable-http'],
                        default='streamable-http')
    parser.add_argument('--embedding-cache', type=Path, help='Existing offline BGE model cache')
    parser.add_argument('--embedding-factory', help='Optional module:factory for CPU tests')
    parser.add_argument('command', nargs='?', choices=['serve', 'search', 'read-entry', 'add-repo',
                                                     'record-lesson'], default='serve')
    parser.add_argument('--arguments', default='{}', help='JSON backend arguments')
    args = parser.parse_args()
    def backend_factory():
        if args.embedding_factory:
            from importlib import import_module
            module, factory = args.embedding_factory.split(':', 1)
            embeddings = getattr(import_module(module), factory)()
        else:
            from .embeddings import PinnedBGEEmbeddings
            embeddings = PinnedBGEEmbeddings(cache_dir=args.embedding_cache, local_files_only=True)
        return KDB(args.db, embeddings)

    if args.command == 'serve':
        create_server(backend_factory, port=args.port).run(transport=args.transport)
    else:
        backend = backend_factory()
        try:
            print(json.dumps(getattr(backend, args.command.replace('-', '_'))(
                **json.loads(args.arguments))))
        finally:
            backend.close()



if __name__ == '__main__':
    main()
