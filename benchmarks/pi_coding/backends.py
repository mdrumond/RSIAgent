"""RSI action-loop and Pi RPC backends; the Python host owns verification."""

import hashlib
import json
import os
from pathlib import Path
import time

import requests

from benchmarks.a3_experiments import BackendModel
from benchmarks.a3_model_profiles import A3TransportResult, complete_a3
from benchmarks.a3kernels.trial import parse_action
from .config import ROOT, RULES, EFFICIENT_RULES
from .gateway import PROVIDERS
from .observations import compact_observation
from .rpc import PiError, PiRPC
from .tools import ToolServer

def rsi_attempt(provider, gateway, tools, prompt, trace, *, api="chat-completions"):
    comparison = api == "responses" or tools.max_revisions > 3
    model = BackendModel.GPT_5_6_SOL if provider == "openai" else BackendModel.DEEPSEEK_FLASH
    system = ('Reply with exactly one JSON action: {"action":"write_source","source":"complete kernel.cpp"}. '
              'The host validates every write and returns its observation. No Markdown fences.')

    def transport(*, request, **_kwargs):
        request = dict(request)
        request.update(request.pop("extra_body", {}))
        if api == "responses":
            request = {"model": gateway.spec["model"], "input": request["messages"]}
        response = requests.post(gateway.url + ("/responses" if api == "responses" else "/chat/completions"), json=request,
                                 headers={"Authorization": "Bearer " + gateway.token}, timeout=max(1, min(610, tools.deadline - time.monotonic())))
        if response.status_code != 200:
            raise RuntimeError(f"RSI gateway HTTP {response.status_code}")
        body = response.json()
        if api == "responses":
            if body.get("error"):
                raise RuntimeError("RSI provider error")
            text = "".join(part.get("text", "") for item in body.get("output", [])
                           if item.get("type") == "message" for part in item.get("content", [])
                           if part.get("type") == "output_text")
            if body.get("status") == "incomplete":
                trace.append({"type": "rsi_revision_truncated"})
                text = "{}"
            return A3TransportResult(text, body["usage"]["output_tokens"])
        text = body["choices"][0]["message"].get("content") or ""
        if comparison and body["choices"][0].get("finish_reason") == "length":
            trace.append({"type": "rsi_revision_truncated"})
            text = "{}"
        return A3TransportResult(text, body["usage"]["completion_tokens"])

    context = prompt
    for _ in range(gateway.max_calls if comparison else tools.max_revisions):
        if tools.writes >= tools.max_revisions:
            break
        completion = complete_a3(system, context, model=model,
                                 environ={PROVIDERS[provider]["key"]: "gateway-only"}, transport=transport)
        # These are smoke overrides; original A3 profile provenance is not reported as executed.
        trace.append({"type": "completion", "text": completion.text})
        try:
            action = parse_action(completion.text)
            if action.kind != "write_source":
                raise ValueError("RSI arm expects a write_source action")
        except ValueError:
            if not comparison:
                raise
            context = revision_prompt(prompt, tools) + "\nPrevious response was incomplete or invalid. Return one complete JSON write_source action."
            continue
        tools.execute("write_candidate", {"source": action.source})
        observation = tools.execute("validate_candidate", {})
        trace.append({"type": "validation", "result": observation})
        if observation["passed"] or observation["status"] == "infrastructure_unavailable":
            break
        if comparison:
            context = revision_prompt(prompt, tools)
            continue
        context = prompt + "\nCurrent source:\n" + tools.execute("read_candidate", {})["source"]
        context += "\nHost observation:\n" + json.dumps(observation)



def revision_prompt(original, tools):
    if not tools.validator.results:
        return original
    # Keep the entire task and user request, but replace the starter and old
    # trajectories with the current source and latest authoritative observation.
    base = original.rsplit("\nStarter source:\n", 1)[0]
    context = base + "\nCurrent source:\n" + (tools.root / "kernel.cpp").read_text()
    context += "\nLatest host observation:\n" + json.dumps(compact_observation(tools.validator.results[-1]))
    earlier = []
    for observation in tools.validator.results[:-1]:
        compact = compact_observation(observation)
        earlier.append({"status": compact["status"], "diagnostic": compact.get("diagnostic", "")[:250]})
    if earlier:
        context += "\nEarlier failure summaries (avoid repeating):\n" + json.dumps(earlier)
    context += f"\nRemaining source revisions: {tools.max_revisions - tools.writes}."
    return context


def pi_attempt(provider, gateway, tools, prompt, run_dir, trace, *, workflow="efficient"):
    if workflow not in ("efficient", "native", "persistent"):
        raise ValueError("unknown Pi workflow")
    efficient = workflow == "efficient"
    persistent = workflow == "persistent"
    pi_dir = run_dir / "pi-agent"
    pi_dir.mkdir()
    spec = gateway.spec
    (pi_dir / "settings.json").write_text(json.dumps({
        "retry": {"enabled": False}, "compaction": {"enabled": False},
        "cacheWarming": "off", "quietStartup": True,
    }))
    (pi_dir / "models.json").write_text(json.dumps({"providers": {"rsi-smoke": {
        "baseUrl": gateway.url, "api": "openai-responses" if provider == "openai" else "openai-completions",
        "apiKey": "${RSI_PI_GATEWAY_TOKEN}",
        "models": [{"id": spec["model"], "name": spec["model"], "reasoning": provider == "openai",
                    "input": ["text"], "contextWindow": 65536, "maxTokens": spec["max_tokens"],
                    "cost": {"input": spec["rates"][0], "output": spec["rates"][1], "cacheRead": 0, "cacheWrite": 0}}],
    }}}))
    with ToolServer(tools, compact=persistent) as server:
        # Explicit environment: provider keys and machine connection configuration are absent.
        env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(Path.home()),
               "PI_CODING_AGENT_DIR": str(pi_dir), "PI_OFFLINE": "1",
               "RSI_PI_GATEWAY_TOKEN": gateway.token,
               "RSI_PI_TOOL_TOKEN": server.token, "RSI_PI_TOOL_URL": server.url}
        if efficient:
            env["RSI_PI_EFFICIENT"] = "1"
        argv = [str(ROOT / "integrations/pi/node_modules/.bin/pi"), "--mode", "rpc", "--no-session",
                "--provider", "rsi-smoke", "--model", spec["model"], "--thinking", "high" if provider == "openai" else "off",
                "--no-builtin-tools", "--no-extensions", "--no-skills", "--no-prompt-templates",
                "--no-themes", "--no-context-files", "--no-mcp", "--offline", "--no-approve",
                "--extension", str(ROOT / "integrations/pi/kernel-tools.ts"),
                "--tools", "submit_candidate" if efficient else "read_candidate,write_candidate,validate_candidate",
                "--system-prompt", (EFFICIENT_RULES if efficient else RULES).replace("three", str(tools.max_revisions))]
        if persistent:
            argv.remove("--no-session")
            argv[argv.index("--extension") + 1] = str(ROOT / "integrations/pi/persistent-kernel-tools.ts")
            argv[argv.index("--tools") + 1] = "read_candidate,submit_candidate,edit_and_validate"
            argv[argv.index("--system-prompt") + 1] = (
                "Implement the requested kernel. Treat host validation as authoritative. "
                "Use submit_candidate for the initial implementation, then edit_and_validate for small fixes. "
                "Keep debugging after failures. Stop only on a verified pass, infrastructure failure, or exhausted limits. "
                "Use concise tool calls without progress commentary or closing explanations. "
                "The current source and compact failure history are provided every turn.")
            argv += ["--session-dir", str(run_dir / "sessions")]
            with PiRPC(argv, cwd=run_dir, env=env, timeout=max(1, tools.deadline - time.monotonic())) as rpc:
                try:
                    while gateway.calls < gateway.max_calls and tools.writes < tools.max_revisions:
                        before = gateway.calls
                        start = len(rpc.events)
                        rpc.timeout = max(1, tools.deadline - time.monotonic())
                        try:
                            rpc.prompt(prompt if before == 0 else "Continue correcting the kernel using the latest authoritative snapshot.")
                        except PiError:
                            messages = [e.get("message", {}) for e in rpc.events[start:] if e.get("type") == "message_end"]
                            truncated = any(m.get("stopReason") == "length" for m in messages)
                            unexpected = any(m.get("stopReason") == "error" and m.get("errorMessage") != "This operation was aborted" for m in messages)
                            if not truncated or unexpected:
                                raise
                            trace.append({"type": "pi_revision_truncated"})
                        if tools.validator.results and (tools.validator.results[-1]["passed"] or tools.validator.results[-1]["status"] == "infrastructure_unavailable"):
                            break
                        if gateway.calls == before or gateway.stop_reason:
                            break
                finally:
                    trace.extend(rpc.events)
            return
        for revision in range(tools.max_revisions if efficient else 1):
            if efficient and gateway.calls >= gateway.max_calls:
                break
            before = tools.writes
            validations_before = len(tools.validator.results)
            context = revision_prompt(prompt, tools) if efficient else prompt
            if efficient and revision:
                context += "\nEmit a concise complete implementation; keep the tool call under the output-token limit."
            if efficient:
                context += f"\nRemaining provider requests including this one: {gateway.max_calls - gateway.calls}."
            trace.append({"type": "pi_revision", "revision": revision + 1, "workflow": workflow,
                          "prompt_sha256": hashlib.sha256(context.encode()).hexdigest()})
            # A fresh ephemeral runtime per revision prevents accidental replay of
            # earlier source/tool transcripts. Pi still owns provider/tool execution.
            with PiRPC(argv, cwd=run_dir, env=env, timeout=max(1, tools.deadline - time.monotonic())) as rpc:
                truncated = False
                try:
                    rpc.prompt(context)
                except PiError:
                    # Only the explicit post-length stop is expected. Preserve
                    # provider errors and unrelated aborts as real failures.
                    messages = [e.get("message", {}) for e in rpc.events if e.get("type") == "message_end"]
                    truncated = efficient and any(m.get("role") == "assistant" and m.get("stopReason") == "length" for m in messages)
                    unexpected_error = any(m.get("stopReason") == "error" and
                                           m.get("errorMessage") != "This operation was aborted"
                                           for m in messages)
                    if not truncated or unexpected_error:
                        raise
                finally:
                    trace.extend(rpc.events)
            if not efficient:
                break
            if truncated or any(e.get("type") == "message_end" and e.get("message", {}).get("stopReason") == "length" for e in rpc.events):
                if tools.writes != before:
                    raise ValueError("truncated generation unexpectedly modified the candidate")
                trace.append({"type": "pi_revision_truncated", "revision": revision + 1})
                continue
            if tools.writes != before + 1 or len(tools.validator.results) != validations_before + 1:
                raise ValueError("efficient Pi revision must submit exactly one candidate")
            observation = tools.validator.results[-1]
            if observation["passed"] or observation["status"] == "infrastructure_unavailable":
                break
