/** Only candidate operations cross into the Python host. No provider keys here. */
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const efficient = process.env.RSI_PI_EFFICIENT === "1";
  let submitted = false;
  // Pi's tool-result termination flag ends the turn cleanly without scheduling
  // another model call. The Python host starts the next bounded revision.
  if (efficient) pi.on("turn_end", (event, ctx) => {
    // Truncated tool arguments fail before execute(), so they cannot return the
    // termination flag. Stop recovery here; the host can retry a clean snapshot.
    if (event.message.role === "assistant" && event.message.stopReason === "length") ctx.abort();
  });
  for (const [name, description, parameters] of [
    ["read_candidate", "Read the current kernel.cpp candidate.", Type.Object({})],
    ["write_candidate", "Replace kernel.cpp with complete source. Subject to the host revision limit.",
      Type.Object({ source: Type.String() })],
    ["validate_candidate", "Compile and check the candidate on A2 using the independent host validator. Subject to the host validation limit. An infrastructure failure is not a correctness result.", Type.Object({})],
    ["submit_candidate", "Submit complete kernel.cpp source. The host writes it and automatically compiles and validates it. Call exactly once, with no separate read or validation call.",
      Type.Object({ source: Type.String() })],
  ] as const) {
    if (efficient !== (name === "submit_candidate")) continue;
    pi.registerTool(defineTool({
      name, label: name, description, parameters,
      async execute(_id, args, signal) {
        const failure = (text: string) => ({ content: [{ type: "text" as const, text }],
          details: undefined, isError: true, terminate: true });
        try {
          if (efficient) {
            if (submitted) return failure("Only one submission is allowed in this revision.");
            submitted = true;
          }
          const response = await fetch(process.env.RSI_PI_TOOL_URL!, {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.RSI_PI_TOOL_TOKEN}` },
            body: JSON.stringify({ name, arguments: args }), signal,
          });
          const payload = await response.json();
          if (!response.ok) throw new Error(JSON.stringify(payload));
          return { content: [{ type: "text" as const, text: JSON.stringify(payload) }],
            details: undefined, terminate: efficient };
        } catch (error) {
          if (efficient) return failure(error instanceof Error ? error.message : "Host tool failed");
          throw error;
        }
      },
    }));
  }
}
