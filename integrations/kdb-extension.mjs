/** Native Pi MCP configuration and workflow retrieval adapter. No transport client. */
import { createMcpExtension } from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { join } from 'node:path';

const searchTool = 'mcp__kdb__search';

function decode(result) {
  // Pi wraps the complete MCP CallToolResult in structuredContent. FastMCP
  // wraps a list return value once more in {result:[...]}.
  const wire = result?.structuredContent;
  if (!wire || wire.isError) throw new Error('KDB returned no successful structured MCP result');
  const hits = wire.structuredContent?.result;
  if (!Array.isArray(hits)) throw new Error('KDB search returned an invalid result');
  return hits;
}

export function createKdbBridge({url, attemptId, agentDir, target, generation, recordRetrieval, event = async () => {}}) {
  const endpoint = new URL(url);
  if (endpoint.protocol !== 'http:' || !['127.0.0.1', '[::1]', 'localhost'].includes(endpoint.hostname)) {
    throw new Error('Instance KDB must use loopback HTTP');
  }
  if (!attemptId || !agentDir) throw new Error('KDB requires attemptId and attempt agentDir');
  const config = {
    servers: [{name: 'kdb', source: join(agentDir, 'mcp.json'), scope: 'extension', config: {
      url: endpoint.href, headers: {'X-Attempt-Id': attemptId}, exposure: 'direct',
    }}], errors: [], autoEnableCodemode: false,
  };
  const nativeExtension = createMcpExtension({loadConfig: () => config, logPath: join(agentDir, 'mcp.log')});
  const extension = async pi => {
    await nativeExtension(pi);
    pi.on('tool_result', async result => {
      if (!result.toolName.startsWith('mcp__kdb__')) return;
      await event('knowledge-tool-result', {
        attempt_id: attemptId, tool: result.toolName, tool_call_id: result.toolCallId,
        parent_tool_call_id: result.parentToolCallId, failed: result.isError,
      });
      if(recordRetrieval&&!result.isError&&['mcp__kdb__search','mcp__kdb__read_entry'].includes(result.toolName)) {
        const wire=result.structuredContent??result.result?.structuredContent;
        const body=wire?.structuredContent??wire;
        const entries=Array.isArray(body?.result)?body.result:typeof body?.text==='string'?[body]:null;
        if(wire?.isError||entries===null){
          await event('knowledge-retrieval-unavailable',{tool_call_id:result.toolCallId,tool:result.toolName});return;
        }
        await recordRetrieval({toolCall:result.toolCallId,operation:result.toolName,
          arguments:result.input??{},generation:entries[0]?.generation??generation??null,entries});
        await event('knowledge-retrieval-recorded',{tool_call_id:result.toolCallId,tool:result.toolName,entries:entries.length});
      }
    });
  };
  // The native /mcp commands await discovery and own reconnect behavior. No
  // provider request is made by these commands. Keep headless native diagnostics.
  const prepareSession = async (session, {signal} = {}) => {
    const writes = [];
    const record = (type, data) => {
      const write = Promise.resolve().then(() => event(type, {attempt_id:attemptId, ...data}));
      writes.push(write); void write.catch(() => {});
    };
    let cancel;
    const cancelled = new Promise((_resolve, reject) => {
      cancel = () => reject(signal.reason ?? new Error('KDB setup cancelled'));
    });
    // Binding can await asynchronous session_start hooks before command() races it.
    void cancelled.catch(() => {});
    signal?.addEventListener('abort', cancel, {once:true});
    const command = async text => {
      signal?.throwIfAborted();
      await Promise.race([session.prompt(text), cancelled]);
      signal?.throwIfAborted();
    };
    try {
      signal?.throwIfAborted();
      const defaults = session.extensionRunner.getUIContext();
      await session.bindExtensions({mode:'print',uiContext:{...defaults,
        notify:(message, level) => record('knowledge-notification', {message, level})},
        onError:error => record('knowledge-extension-error', error)});
      if (!session.extensionRunner.getCommand('mcp')) throw new Error('Required native MCP command is not registered');
      await command('/mcp');
      if (!session.getCallableToolNames().includes(searchTool)) await command('/mcp reconnect kdb');
      if (!session.getCallableToolNames().includes(searchTool)) {
        throw new Error('Required native KDB search unavailable after MCP startup/reconnect; inspect knowledge diagnostics');
      }
      await event('knowledge-ready', {attempt_id:attemptId, tool:searchTool});
    } finally {
      signal?.removeEventListener('abort', cancel);
      await Promise.all(writes);
    }
  };
  const knowledge = {
    prepareSession,
    async search(request, ctx) {
      if (!ctx?.executeTool) throw new Error('KDB retrieval requires a live Pi tool context');
      if (request.attempt_id !== undefined && request.attempt_id !== attemptId) {
        throw new Error('KDB attempt correlation mismatch');
      }
      const limit = request.limit ?? 8;
      if (!Number.isInteger(limit) || limit < 1 || limit > 10) {
        throw new Error('Workflow KDB limit must be 1..10');
      }
      const args = {query: request.query, kind: request.kind ?? 'all', limit};
      if (target !== undefined) args.target = target;
      if (generation !== undefined && generation !== null) args.generation = generation;
      const outcome = await ctx.executeTool(searchTool, args);
      if (outcome.isError) throw new Error('Native KDB search failed or was blocked');
      let remaining = 12000;
      const content = decode(outcome.result).slice(0,limit).map(hit => {
        const text = hit.text.slice(0,Math.min(2000,remaining));
        remaining -= text.length;
        return {collection:hit.collection, chunk_id:hit.chunk_id, path:hit.path,
          start_line:hit.start_line, end_line:hit.end_line, source_sha256:hit.source_sha256,
          kind:hit.kind, target:hit.target, language:hit.language,
          kernel_languages:hit.kernel_languages, score:hit.score, text,
          truncated:text.length !== hit.text.length};
      });
      const evidence = content.map(hit => ({
        collection: hit.collection, entry_id: hit.chunk_id, path: hit.path,
        start_line: hit.start_line, end_line: hit.end_line, sha256: hit.source_sha256,
      }));
      return {content, evidence};
    },
  };
  // PRIMARY may register this instead of its existing rsi_knowledge definition.
  // query forwards ctx through Workflow.query into knowledge.search(request,ctx).
  const knowledgeTool = (query = (text, kind, ctx) => knowledge.search({query:text, kind},ctx)) => ({
    name:'rsi_knowledge', label:'Retrieve knowledge',
    description:'Retrieve positive or negative evidence using native MCP tools.',
    reachesTools:() => [searchTool],
    parameters:Type.Object({query:Type.String({minLength:1}),
      kind:Type.Union([Type.Literal('good'),Type.Literal('negative')])}),
    execute:async (_id, args, _signal, _onUpdate, ctx) => {
      const value = await query(args.query, args.kind, ctx);
      return {content:[{type:'text',text:JSON.stringify(value)}],details:{}};
    },
  });
  return {extension, knowledge, knowledgeTool, prepareSession};
}
