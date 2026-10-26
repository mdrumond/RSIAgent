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
  pi.on("turn_end", (event, ctx) => {
    if (event.message.role === "assistant" && event.message.stopReason === "length") ctx.abort();
  });
  pi.on("context", async (event, ctx) => {
    const snapshot = await host("snapshot", {}, ctx.signal);
    // Pi owns the tool loop. Refresh host state at the user boundary, while
    // preserving every complete exchange selected during this host turn.
    let boundary = -1;
    for (let i = event.messages.length - 1; i >= 0; i--) {
      if (event.messages[i].role === "user") { boundary = i; break; }
    }
    const exchange: typeof event.messages = [];
    for (let i = boundary + 1; i < event.messages.length; i++) {
      const message = event.messages[i];
      if (message.role !== "assistant") continue;
      const calls = message.content.filter(c => c.type === "toolCall");
      if (!calls.length) { exchange.push(message); continue; }
      const results = [];
      for (let j = i + 1; j < event.messages.length; j++) {
        const candidate = event.messages[j];
        if (candidate.role !== "toolResult") break;
        if (calls.some(c => c.id === candidate.toolCallId)) results.push(candidate);
      }
      if (calls.every(c => results.some(r => r.toolCallId === c.id))) {
        exchange.push(message, ...results);
      }
    }
    return { messages: [{ role: "user" as const, content: JSON.stringify(snapshot),
      timestamp: Date.now() }, ...exchange] };
  });
}
