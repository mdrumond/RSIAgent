/** RSI is a native Pi tool extension. Pi owns all history and compaction. */
import { readFileSync } from "node:fs";
import { convertToLlm, serializeConversation, defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

async function host(name: string, args: unknown) {
  const response = await fetch(process.env.RSI_PI_TOOL_URL!, {
    method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.RSI_PI_TOOL_TOKEN}` },
    body: JSON.stringify({ name, arguments: args }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Host rejected tool");
  return result;
}
export default function (pi: ExtensionAPI) {
  let batch = 0;
  const schemas = JSON.parse(readFileSync(process.env.RSI_PI_SCHEMAS!, "utf8"));
  for (const [name, parameters] of Object.entries(schemas)) {
    pi.registerTool(defineTool({
      name, label: name, parameters: parameters as any, executionMode: "sequential",
      description: name === "read_context" ? "Read an admitted resource page." : `Execute ${name}; receive the completed host result.`,
      async execute(id, _coerced, _signal) {
        const result = await host("execute_call", { batch, id });
        if (result.state) pi.setActiveTools(result.state.allowed);
        return { content: [{ type: "text" as const, text: JSON.stringify(result) }],
          details: undefined, isError: result.isError, terminate: true };
      },
    }));
  }
  pi.on("before_agent_start", async () => {
    const state = await host("snapshot", {});
    pi.setActiveTools(state.allowed);
  });
  pi.on("message_end", async event => {
    if (event.message.role === "toolResult") {
      await host("tool_result", { batch, id: event.message.toolCallId });
      return;
    }
    if (event.message.role !== "assistant") return;
    batch += 1;
    await host("propose_batch", { batch, truncated: event.message.stopReason === "length",
      calls: event.message.content.filter(c => c.type === "toolCall") });
  });
  pi.on("agent_before_settle", async event => {
    if (event.outcome !== "completed") return;
    const state = await host("snapshot", {});
    pi.setActiveTools(state.allowed);
    if (!state.done) return { continue: true };
  });
  pi.on("session_before_compact", async (event, ctx) => {
    const p = event.preparation;
    if (!ctx.model) return { cancel: true };
    // Same combined span as Pi's official custom-compaction example. Preserve
    // Pi's cutpoint; never leave the split-turn prefix on the verbose path.
    const messages = [...p.messagesToSummarize, ...p.turnPrefixMessages];
    const conversation = serializeConversation(convertToLlm(messages));
    try {
    const response = await ctx.modelRegistry.complete(ctx.model, {
      systemPrompt: "Write a concise RSI continuation checkpoint, at most about 1000 tokens. " +
        "Preserve current stage, source path/hash, current operation/receipt IDs, unresolved failures, " +
        "applicable positive/negative evidence and the next guarded action. " +
        "Omit obsolete receipt IDs, duplicate history, full skill text, source bodies and tool schemas. " +
        "Never invent evidence or report an uncommitted stage as accepted.",
      messages: [{ role: "user", content: [{ type: "text", text:
        `<previous-summary>\n${p.previousSummary ?? ""}\n</previous-summary>\n` +
        `<conversation>\n${conversation}\n</conversation>` }], timestamp: Date.now() }],
    }, { maxTokens: 1536, signal: event.signal, cacheRetention: "none" });
    const summary = response.content.filter(c => c.type === "text").map(c => c.text).join("\n");
    const successful = response.stopReason === "stop" && !!summary.trim() && Buffer.byteLength(summary) <= 6144;
    await host("native_event", { kind: "compaction-summary", successful,
      stopReason: response.stopReason, summary, usage: response.usage,
      firstKeptEntryId: p.firstKeptEntryId, tokensBefore: p.tokensBefore });
    if (!successful) return { cancel: true }; // Never silently use a truncated checkpoint.
    const modifiedFiles = [...new Set([...p.fileOps.written, ...p.fileOps.edited])];
    return { compaction: { summary, firstKeptEntryId: p.firstKeptEntryId,
      tokensBefore: p.tokensBefore, usage: response.usage,
      details: { readFiles: [...p.fileOps.read].filter(path => !modifiedFiles.includes(path)), modifiedFiles } } };
    } catch (error) {
      if (!event.signal.aborted) await host("native_event", { kind: "compaction-summary", successful: false,
        error: error instanceof Error ? error.message : String(error), usage: null });
      return { cancel: true };
    }
  });
  // No context hook: native tool exchanges, recovery and compaction remain Pi's.
}
