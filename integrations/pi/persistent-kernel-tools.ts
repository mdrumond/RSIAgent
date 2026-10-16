/** Persistent Pi owns iteration; the host owns files, validation, and limits. */
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

async function host(name: string, args: unknown, signal?: AbortSignal) {
  const response = await fetch(process.env.RSI_PI_TOOL_URL!, {
    method: "POST", signal,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.RSI_PI_TOOL_TOKEN}` },
    body: JSON.stringify({ name, arguments: args }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Host operation failed");
  return data;
}

export default function (pi: ExtensionAPI) {
  pi.on("turn_end", (event, ctx) => {
    // A partial tool call cannot safely change the source. The host resumes this
    // same session with a clean context, counting the truncated provider call.
    if (event.message.role === "assistant" && event.message.stopReason === "length") ctx.abort();
  });
  pi.on("context", async (event, ctx) => {
    const snapshot = await host("context_snapshot", {}, ctx.signal);
    const first = event.messages.find(m => m.role === "user");
    if (!first) return;
    const original = typeof first.content === "string" ? first.content
      : first.content.filter(c => c.type === "text").map(c => c.text).join("\n");
    const task = original.split("\nStarter source:\n")[0];
    // Retain the most recent COMPLETE tool exchange, including opaque reasoning
    // blocks, without altering its call IDs or argument/result pairing.
    let exchange: typeof event.messages = [];
    for (let i = event.messages.length - 1; i >= 0; i--) {
      const message = event.messages[i];
      if (message.role !== "assistant" || message.stopReason !== "toolUse") continue;
      const calls = message.content.filter(c => c.type === "toolCall");
      const results = event.messages.slice(i + 1).filter(m => m.role === "toolResult");
      if (calls.length && calls.every(c => results.some(r => r.role === "toolResult" && r.toolCallId === c.id))) {
        exchange = [message, ...results.filter(r => r.role === "toolResult" && calls.some(c => c.id === r.toolCallId))];
        break;
      }
    }
    const text = task + "\nCurrent authoritative snapshot:\n" + JSON.stringify(snapshot)
      + "\nContinue until the host verifies a pass or a limit is reached. Use concise tool calls; no progress prose.";
    return { messages: [{ role: "user" as const, content: text, timestamp: first.timestamp }, ...exchange] };
  });

  for (const [name, description, parameters] of [
    ["read_candidate", "Read the current kernel.cpp. It is also in the current snapshot.", Type.Object({})],
    ["submit_candidate", "Replace kernel.cpp with complete source and immediately compile and validate on A2.", Type.Object({ source: Type.String() })],
    ["edit_and_validate", "Replace exactly one matching old_text in kernel.cpp with new_text, then compile and validate. Use for small fixes; ambiguous edits leave the file unchanged.",
      Type.Object({ old_text: Type.String(), new_text: Type.String() })],
  ] as const) {
    pi.registerTool(defineTool({
      name, label: name, description, parameters,
      async execute(_id, args, signal) {
        try {
          const result = await host(name, args, signal);
          const terminal = result.passed || result.status === "infrastructure_unavailable" || result.remaining_revisions === 0;
          return { content: [{ type: "text" as const, text: JSON.stringify(result) }], details: undefined, terminate: !!terminal };
        } catch (error) {
          const text = error instanceof Error ? error.message : "Host tool failed";
          return { content: [{ type: "text" as const, text }], details: undefined, isError: true,
            terminate: /exhausted|deadline/.test(text) };
        }
      },
    }));
  }
}
