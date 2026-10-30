/** Native review tools: Python validates original arguments before every effect. */
import { readFileSync } from "node:fs";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

async function host(name: string, args: unknown, signal?: AbortSignal) {
  const response = await fetch(process.env.RSI_PI_TOOL_URL!, {
    method: "POST", signal,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.RSI_PI_TOOL_TOKEN}` },
    body: JSON.stringify({ name, arguments: args }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Host rejected review request");
  return body;
}

export default function (pi: ExtensionAPI) {
  let batch = 0;
  const schemas = JSON.parse(readFileSync(process.env.RSI_PI_SCHEMAS!, "utf8"));
  for (const [name, parameters] of Object.entries(schemas)) {
    if (name === "edit_source") continue;
    pi.registerTool(defineTool({
      name, label: name, parameters: parameters as any, executionMode: "sequential",
      description: name === "submit_review" ? "Submit a standalone host-validated review. Never combine with another call."
        : name === "request_check" ? "Request a standalone host check. Never combine with another call."
        : `Execute ${name} and receive the authoritative result immediately.`,
      async execute(id, _coercedArgs, signal) {
        const result = await host("execute_call", { batch, id }, signal);
        return { content: [{ type: "text" as const, text: JSON.stringify(result.observation) }],
          details: undefined, isError: result.isError, terminate: result.terminate };
      },
    }));
  }
  pi.on("before_agent_start", async (_event, ctx) => {
    const snapshot = await host("snapshot", {}, ctx.signal);
    pi.setActiveTools(snapshot.allowed);
  });
  pi.on("message_end", async (event, ctx) => {
    // message_end precedes prepareToolCall/argument coercion in pinned Pi 1.3.
    if (event.message.role === "assistant") batch += 1;
    await host("transcript", { batch, message: event.message });
    if (event.message.role !== "assistant") return;
    await host("propose_batch", { batch, truncated: event.message.stopReason === "length",
      calls: event.message.content.filter(c => c.type === "toolCall") });
  });
  pi.on("context", async (event, ctx) => {
    const snapshot = await host("snapshot", {}, ctx.signal);
    return { messages: [{ role: "user" as const, content: JSON.stringify(snapshot),
      timestamp: Date.now() }, ...event.messages] };
  });
  pi.on("session_compact_failed", async (event) => {
    await host("compaction", { error: event.errorMessage, aborted: event.aborted });
  });
  pi.on("session_compact", async (event) => {
    await host("compaction", { fromExtension: event.fromExtension });
  });
}
