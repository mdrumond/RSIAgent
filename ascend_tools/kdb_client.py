"""Single standard MCP SDK invocation for the CLI and dashboard."""
import argparse
import asyncio
import json
from urllib.parse import urlsplit
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def invoke(url, tool, arguments, attempt=None):
    headers = {'X-Attempt-Id': attempt} if attempt else {}
    endpoint = urlsplit(url)
    loopback = endpoint.scheme == 'http' and endpoint.hostname in {'127.0.0.1', 'localhost', '::1'}
    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(30, read=None),
                                 trust_env=not loopback) as client, streamable_http_client(url, http_client=client) as (reader, writer, _):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
            if result.isError:
                raise RuntimeError(result.model_dump_json())
            return result.model_dump(mode='json')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--attempt')
    parser.add_argument('--tool', required=True, choices=['search', 'read_entry', 'add_repo', 'record_lesson',
        'stage_memory_revision', 'activate_memory_revision', 'adopt_memory_revision'])
    parser.add_argument('--arguments', default='{}')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(invoke(args.url, args.tool, json.loads(args.arguments), args.attempt))))

if __name__ == '__main__':
    main()
