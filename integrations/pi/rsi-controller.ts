/** RSI is a native Pi tool extension. Pi owns all history and compaction. */
import { readFileSync } from "node:fs";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

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
  // No context hook: native tool exchanges, recovery and compaction remain Pi's.
}
