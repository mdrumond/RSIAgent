"""Standard MCP SDK adapter and CLI for one instance-owned KDB process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from .kdb import KDB


def create_server(backend: KDB, *, port=8765):
    from starlette.responses import JSONResponse

    server = FastMCP('ascend-kdb', host='127.0.0.1', port=port,
                     stateless_http=True, json_response=True)

    def attempt(ctx):
        request = ctx.request_context.request
        return (request.headers.get('x-attempt-id') if request is not None
                else os.environ.get('ASCEND_ATTEMPT_ID'))

    def invoke(ctx, operation, **args):
        attempt_id = attempt(ctx)
        try:
            result = getattr(backend, operation)(**args)
        except Exception as exc:
            backend.event(operation, attempt_id, status='failed', error=type(exc).__name__)
            raise
        backend.event(operation, attempt_id, status='completed', arguments=args,
                      result_count=len(result) if isinstance(result, list) else 1)
        return result

    @server.tool()
    async def search(query: str, ctx: Context, kind: str = 'all',
                     target: str | None = None, limit: int = 10) -> list[dict]:
        """Search positive/negative source knowledge within a target scope."""
        return invoke(ctx, 'search', query=query, kind=kind, target=target, limit=limit)

    @server.tool()
    async def add_repo(path: str, collection: str, language: str, ctx: Context,
                       target: str | None = None, kind: str = 'good',
                       paths: list[str] | None = None) -> dict:
        """Index selected repository files as an immutable collection."""
        return invoke(ctx, 'add_repo', path=path, collection=collection,
                      language=language, target=target, kind=kind, paths=paths)

    @server.tool()
    async def record_lesson(lesson: str, outcome: str, applicability: str,
                            evidence: list[dict], attempt_id: str, ctx: Context) -> dict:
        """Store a compact lesson backed by direct evidence references."""
        correlation = attempt(ctx)
        if correlation is not None and correlation != attempt_id:
            raise ValueError('attempt_id differs from request correlation')
        return invoke(ctx, 'record_lesson', lesson=lesson, outcome=outcome,
                      applicability=applicability, evidence=evidence, attempt_id=attempt_id)

    @server.resource('kdb://skill')
    async def skill(ctx: Context) -> str:
        backend.event('skill.read', attempt(ctx), resource='kdb://skill')
        return (Path(__file__).parent / 'templates/kdb/SKILL.md').read_text()

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
    parser.add_argument('--embedding-factory', help='Optional module:factory for CPU tests')
    parser.add_argument('command', nargs='?', choices=['serve', 'search', 'add-repo',
                                                     'record-lesson'], default='serve')
    parser.add_argument('--arguments', default='{}', help='JSON backend arguments')
    args = parser.parse_args()
    if args.embedding_factory:
        from importlib import import_module
        module, factory = args.embedding_factory.split(':', 1)
        embeddings = getattr(import_module(module), factory)()
    else:
        from .embeddings import PinnedBGEEmbeddings
        embeddings = PinnedBGEEmbeddings()
    backend = KDB(args.db, embeddings)
    try:
        if args.command == 'serve':
            create_server(backend, port=args.port).run(transport=args.transport)
        else:
            print(json.dumps(getattr(backend, args.command.replace('-', '_'))(
                **json.loads(args.arguments))))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
