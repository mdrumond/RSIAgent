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
  const schemas = JSON.parse(readFileSync(process.env.RSI_PI_SCHEMAS!, "utf8"));
  for (const [name, parameters] of Object.entries(schemas)) {
    pi.registerTool(defineTool({
      name, label: name,
      description: name === "read_context" ? "Page an admitted host context record using its digest and next_offset."
        : name === "edit_source" ? "Queue an exact, unambiguous source edit against the current SHA256."
        : `Queue the host action ${name}. This receipt does not execute or validate anything.`,
      parameters: parameters as any,
      async execute(_id, args, signal) {
        try {
          const result = await host(name, args, signal);
          return { content: [{ type: "text" as const, text: JSON.stringify(result) }],
            details: undefined, terminate: name !== "read_context" };
        } catch {
          return { content: [{ type: "text" as const, text: "Host rejected invalid, stale or disallowed arguments." }],
            details: undefined, isError: true, terminate: true };
        }
      },
    }));
  }
  pi.on("before_agent_start", async (_event, ctx) => {
    const snapshot = await host("snapshot", {}, ctx.signal);
    pi.setActiveTools(snapshot.allowed);
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
    let bytes = Buffer.byteLength(JSON.stringify(snapshot));
    for (let i = event.messages.length - 1; i > boundary; i--) {
      const message = event.messages[i];
      if (message.role !== "assistant" || message.stopReason !== "toolUse") continue;
      const calls = message.content.filter(c => c.type === "toolCall");
      const results = event.messages.slice(i + 1).filter(m => m.role === "toolResult"
        && calls.some(c => c.id === m.toolCallId));
      if (calls.length && calls.every(c => results.some(r => r.role === "toolResult" && r.toolCallId === c.id))) {
        const keys = calls.map(c => c.name + JSON.stringify(c.arguments));
        if (keys.every(key => pages.has(key))) continue;
        const exchange = [message, ...results];
        const size = Buffer.byteLength(JSON.stringify(exchange));
        // Preserve the latest complete exchange as before. Older pages fit in a
        // bounded window below the provider's separately enforced request cap.
        if (exchanges.length && bytes + size > 32768) continue;
        exchanges.push(exchange);
        bytes += size;
        keys.forEach(key => pages.add(key));
      }
    }
    return { messages: [{ role: "user" as const, content: JSON.stringify(snapshot),
      timestamp: Date.now() }, ...exchanges.reverse().flat()] };
  });
}
