/** Typed action selection only. The Python host owns every execution effect. */
import { readFileSync } from "node:fs";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

async function host(name: string, args: unknown, signal?: AbortSignal) {
  const response = await fetch(process.env.RSI_PI_TOOL_URL!, {
    method: "POST", signal,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.RSI_PI_TOOL_TOKEN}` },
    body: JSON.stringify({ name, arguments: args }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Host rejected tool");
  return body;
}

export default function (pi: ExtensionAPI) {
  const ordered = process.env.RSI_PI_FIRST_ACTION_ONLY === "1";
  let batch = 0;
  const schemas = JSON.parse(readFileSync(process.env.RSI_PI_SCHEMAS!, "utf8"));
  const requestBytes = Number(process.env.RSI_PI_REQUEST_BYTES ?? 49152);
  for (const [name, parameters] of Object.entries(schemas)) {
    pi.registerTool(defineTool({
      name, label: name,
      ...(ordered ? { executionMode: "sequential" as const } : {}),
      description: name === "read_context" ? "Page an admitted host context record using its digest and next_offset."
        : ordered && name === "read_skill" ? "Select a skill read; its completed text and receipt appear in the next host snapshot. Use current_skill_content for already-read text or read_context for paged text, instead of repeating completed reads."
        : name === "edit_source" ? "Queue an exact, unambiguous source edit against the current SHA256."
        : `Queue the host action ${name}. This receipt does not execute or validate anything.`,
      parameters: parameters as any,
      async execute(id, args, signal) {
        try {
          const result = ordered
            ? await host("execute_call", { batch, id }, signal)
            : await host(name, args, signal);
          return { content: [{ type: "text" as const, text: JSON.stringify(result) }],
            details: undefined, terminate: name !== "read_context" };
        } catch (error) {
          return { content: [{ type: "text" as const, text: error instanceof Error ? error.message : "Host tool request failed." }],
            details: undefined, isError: true, terminate: true };
        }
      },
    }));
  }
  pi.on("before_agent_start", async (_event, ctx) => {
    const snapshot = await host("snapshot", {}, ctx.signal);
    pi.setActiveTools(snapshot.allowed);
  });
  pi.on("message_end", async (event, ctx) => {
    if (!ordered || event.message.role !== "assistant") return;
    batch += 1;
    // Pinned Pi emits this before argument coercion/tool execution. Python
    // selects by original model order, independent of HTTP scheduling.
    await host("propose_batch", { batch, truncated: event.message.stopReason === "length",
      calls: event.message.content.filter(c => c.type === "toolCall") }, ctx.signal);
  });
  pi.on("turn_end", (event, ctx) => {
    if (event.message.role === "assistant" && event.message.stopReason === "length") ctx.abort();
  });
  pi.on("context", async (event, ctx) => {
    const snapshot = await host("snapshot", {}, ctx.signal);
    // A host turn is a new authoritative snapshot. Keep the pages read within
    // that turn together, so reading a later page does not erase an earlier one.
    let boundary = -1;
    for (let i = event.messages.length - 1; i >= 0; i--) {
      if (event.messages[i].role === "user") { boundary = i; break; }
    }
    const exchanges: (typeof event.messages)[] = [];
    const pages = new Set<string>();
    const snapshotMessage = { role: "user" as const, content: JSON.stringify(snapshot),
      timestamp: Date.now() };
    // Provider messages wrap JSON text inside JSON strings. Double serialization
    // conservatively covers argument/result escaping in the pinned SDK's chat
    // and Responses projections. Include active schemas and the system prompt;
    // reserve 8 KiB for provider envelopes and projection metadata as well.
    const wireBound = (value: unknown) => Buffer.byteLength(JSON.stringify(JSON.stringify(value)));
    const active = new Set(pi.getActiveTools());
    const tools = pi.getAllTools().filter(tool => active.has(tool.name))
      .map(({ name, description, parameters }) => ({ name, description, parameters }));
    let bytes = wireBound({ system: ctx.getSystemPrompt(), tools }) + 8192
      + Buffer.byteLength(JSON.stringify(snapshotMessage));
    for (let i = event.messages.length - 1; i > boundary; i--) {
      const message = event.messages[i];
      if (message.role !== "assistant" || message.stopReason !== "toolUse") continue;
      const calls = message.content.filter(c => c.type === "toolCall");
      const results = event.messages.slice(i + 1).filter(m => m.role === "toolResult"
        && calls.some(c => c.id === m.toolCallId));
      if (calls.length && calls.every(c => results.some(r => r.role === "toolResult" && r.toolCallId === c.id))) {
        const keys = calls.map(c => c.name === "read_context"
          ? JSON.stringify([c.name, c.arguments.record, c.arguments.offset ?? 0, c.arguments.limit ?? 4096])
          : c.name + JSON.stringify(c.arguments));
        if (keys.every(key => pages.has(key))) continue;
        const exchange = [message, ...results];
        const size = wireBound(exchange);
        // The current snapshot and latest complete exchange stay mandatory.
        // Admit older exchanges only below the gateway's complete request cap.
        if (exchanges.length && bytes + size > requestBytes) continue;
        exchanges.push(exchange);
        bytes += size;
        keys.forEach(key => pages.add(key));
      }
    }
    return { messages: [snapshotMessage, ...exchanges.reverse().flat()] };
  });
}
