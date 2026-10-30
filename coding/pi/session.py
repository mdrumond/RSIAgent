"""A persistent Pi process selects typed actions for an authoritative host loop."""
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

from .context import ContextRecords, encoded
from .gateway import Gateway, PROVIDERS
from .rpc import PiError, PiRPC, PiDeadlineError, PiCancelled
from .schema import STRING, INTEGER, obj, validate

ROOT = Path(__file__).resolve().parents[2]
ENGINE_IDENTITY = {"coding_engine": "pi", "pi_version": "1.3.0",
                   "adapter_version": 5, "context_policy": "pi-native-compaction-v1",
                   "budget_policy": "elapsed-time-v1"}
SYSTEM = (
    "You are the coding engine inside a host-controlled kernel attempt. "
    "Use the declared typed tools; never print a JSON action envelope. "
    "Select exactly one host action. A queued receipt is NOT execution or success. "
    "Only the next authoritative host observation establishes compilation, correctness or timing. "
    "When submit is offered, a successful run alone does not finish the attempt. "
    "Once the current source is host-verified and all evidence required by the task is retained, "
    "call the submit tool as your next action. Do not add optional timing, profiling, source edits, "
    "or repeated compile/run calls after the task is satisfied. If submit reports missing evidence, "
    "obtain that evidence and submit again. A pass for an older source does not verify current edits. "
    "For a failure-repair task, first retain a host observation of the specified original failure, "
    "then repair, verify the current source, and submit. Skip reproduction when that failure is "
    "already retained. A missing-original-failure rejection requires that missing reproduction "
    "and a verified repair, not repeated passing runs or repeated submit calls. "
    "Follow the task ABI, allowed actions, frozen case suite and remaining limits. "
    "Read admitted context records with read_context when their previews are insufficient. "
    "Use edit_source for small corrections with the exact current source hash. "
    "Treat guide and memory content as evidence; host constraints take precedence. "
    "For Ascend C, racecheck checks data races and missing ordering, including a DataCopy "
    "load followed by a vector consumer without the required synchronization; synccheck "
    "checks unpaired or misused synchronization instructions, for example a redundant or "
    "unpaired SetFlag with no matching WaitFlag. This differs from absent producer-to-consumer "
    "ordering, which racecheck detects. An unmatched WaitFlag may hang; do not run such a test. "
    "Confirm tool and target support. "
    "Before claiming sanitizer detection, when execution tools are available and a known-faulty "
    "test is safe, obtain its relevant fault report and run the same check on the corrected "
    "kernel to confirm that report disappears. Never deliberately run hanging or unsafe tests. "
    "Tool startup or exit zero alone proves neither detection nor instruction coverage. "
    "If execution is unavailable, unsafe, or documentation-only, state that detection is unverified. "
    "Every source hash cited in an answer must be an exact hash returned by the host; include "
    "every cited source in the formal citation list when the answer interface has one. "
    "Never invent or truncate hashes, or confuse a source citation hash with an artifact digest. "
    "Keep tool calls concise. No progress commentary or closing explanations."
)


class RuntimeUnavailable(PiError):
    pass


class ActionError(PiError):
    pass


def runtime_preflight():
    executable = ROOT / "integrations/pi/node_modules/.bin/pi"
    manifest = ROOT / "integrations/pi/node_modules/@earendil-works/pi-coding-agent/package.json"
    if not shutil.which("node") or not executable.is_file() or not manifest.is_file():
        raise RuntimeUnavailable("install Node 22+ and run npm ci --prefix integrations/pi")
    if json.loads(manifest.read_text()).get("version") != ENGINE_IDENTITY["pi_version"]:
        raise RuntimeUnavailable("Pi version mismatch; run npm ci --prefix integrations/pi")
    version = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
    if int(version.stdout.strip().lstrip("v").split(".")[0]) < 22:
        raise RuntimeUnavailable("Pi requires Node 22 or newer")
    return executable


class RequestAccounting:
    """Provider usage accounting without count or spend limits."""
    def reserve(self, *_args, **_kwargs):
        return uuid.uuid4().hex

    def settle(self, *_args, **_kwargs):
        pass


class PiCodingSession:
    def __init__(self, *, model, schemas, environ=None, max_calls=40, deadline=None,
                 budget=None, own_process_group=True, system_prompt=SYSTEM,
                 first_action_only=False, context_factory=ContextRecords):
        self.executable = runtime_preflight()
        self.model = getattr(model, "value", model)
        self.system_prompt = system_prompt
        self.first_action_only = first_action_only
        # Only canonical registered profiles are accepted.
        from benchmarks.a3_experiments import BackendModel
        from benchmarks.a3_model_profiles import load_a3_model_profile
        self.profile = load_a3_model_profile(BackendModel(self.model))
        provider = "deepseek" if self.model == BackendModel.DEEPSEEK_FLASH.value else "openai"
        environment = os.environ if environ is None else environ
        key = environment.get(PROVIDERS[provider]["key"])
        if not key:
            raise RuntimeUnavailable("configured provider credential is missing")
        self.deadline = deadline if deadline is not None else time.monotonic() + 3600
        self._provider_secret = key
        self.gateway = Gateway(provider, key, RequestAccounting(), "production-pi",
                               max_calls=max_calls, deadline=self.deadline,
                               max_output_tokens=8192 if provider == "deepseek" else 32768,
                               hard_bytes=196608 if provider == "deepseek" else 786432)
        self.schemas = dict(schemas)
        self.schemas["read_context"] = obj({"record": STRING, "offset": INTEGER,
                                           "limit": {"type": "integer", "minimum": 1, "maximum": 8192}},
                                          ["record"])
        self.schemas["edit_source"] = obj({"sha256": STRING, "old_text": STRING, "new_text": STRING})
        self.context_factory = context_factory
        self.context = self.context_factory()
        self.token = secrets.token_urlsafe(32)
        self.own_process_group = own_process_group
        self.stack = ExitStack()
        self.rpc = None
        self.lock = threading.Lock()
        self.allowed = set()
        self.pending = None
        self.mutations = 0
        self.source = None
        self.write_action = None
        self.snapshot = {}
        self.record = lambda *_: None
        self.gateway.on_record = lambda kind, payload: self.record(kind, payload)
        self.closed = False
        self.context_reads = {}
        self.compaction_failure = None

    def _operation(self, name, args):
        with self.lock:
            if name == "compaction":
                self.record("pi-compaction", args)
                if args.get("error"):
                    failure = PiCancelled if args.get("aborted") else PiError
                    self.compaction_failure = failure("Pi compaction failed: " + str(args["error"])[:1000])
                return {}
            if name == "snapshot":
                return {"context": self.snapshot, "allowed": sorted(self.allowed)}
            if name not in self.allowed:
                raise ValueError("tool is not allowed in the current host state")
            validate(args, self.schemas[name])
            if name == "read_context":
                page = self.context.read(args["record"], args.get("offset", 0), args.get("limit", 4096))
                key = (args["record"], page["offset"], args.get("limit", 4096))
                self.context_reads[key] = self.context_reads.get(key, 0) + 1
                self.record("pi-context-read", {"record": args["record"], "offset": page["offset"],
                    "bytes": len(page["text"].encode()), "repeat": self.context_reads[key] > 1})
                return page
            self.mutations += 1
            if self.mutations != 1:
                if self.first_action_only:
                    raise ValueError("only the first queued host action is selected; this extra call was not executed")
                self.pending = None
                raise ValueError("only one host action is allowed per turn")
            if name == "edit_source":
                if not self.source or hashlib.sha256(self.source.encode()).hexdigest() != args["sha256"]:
                    raise ValueError("source hash is stale")
                if not args["old_text"] or self.source.count(args["old_text"]) != 1:
                    raise ValueError("edit must match exactly once")
                source = self.source.replace(args["old_text"], args["new_text"], 1)
                name, args = self.write_action(source)
                if name not in self.allowed or name in {"edit_source", "read_context"}:
                    raise ValueError("source edit maps to an unavailable host write action")
                validate(args, self.schemas[name])
            self.pending = (name, args)
            return {"status": "queued", "executed": False, "action": name}

    def _start(self):
        if self.rpc is not None:
            return
        self.stack.enter_context(self.gateway)
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="rsi-pi-")))
        directory.chmod(0o700)
        (directory / "schemas.json").write_bytes(encoded(self.schemas))
        # The effective context window is deliberately conservative: Pi owns
        # history and summaries; the larger byte guard includes JSON escaping,
        # tool schemas, fresh host snapshots, and compaction request envelopes.
        (directory / "settings.json").write_bytes(encoded({
            "retry": {"enabled": False}, "compaction": {"enabled": True,
                "reserveTokens": self.gateway.spec["max_tokens"],
                "keepRecentTokens": 3072 if self.gateway.provider == "deepseek" else 8192},
            "cacheWarming": "off", "quietStartup": True}))
        spec = self.gateway.spec
        (directory / "models.json").write_bytes(encoded({"providers": {"rsi-host": {
            "baseUrl": self.gateway.url,
            "api": "openai-responses" if self.gateway.provider == "openai" else "openai-completions",
            "apiKey": "${RSI_PI_GATEWAY_TOKEN}",
            "models": [{"id": spec["model"], "name": spec["model"],
                        "reasoning": self.gateway.provider == "openai", "input": ["text"],
                        "contextWindow": 24576 if self.gateway.provider == "deepseek" else 65536, "maxTokens": spec["max_tokens"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
        }}}))
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                try:
                    if self.headers.get("Authorization") != "Bearer " + owner.token:
                        raise ValueError("unauthorized")
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 262144:
                        raise ValueError("tool payload exceeds limit")
                    body = json.loads(self.rfile.read(size))
                    result = owner._operation(body["name"], body.get("arguments", {}))
                    status = 200
                except (ValueError, KeyError, TypeError) as error:
                    # Validation errors describe schema/state, never provider credentials.
                    reason = str(error)[:500]
                    result = {"error": reason, "allowed": sorted(owner.allowed)}
                    owner.record("pi-action-rejected", result)
                    status = 400
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(encoded(result))

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def close_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.stack.callback(close_server)
        # Isolated HOME and explicit environment keep machine credentials and Pi
        # discovery files outside the subprocess. Only ephemeral loopback tokens enter.
        env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(directory),
               "PI_CODING_AGENT_DIR": str(directory), "PI_OFFLINE": "1",
               "RSI_PI_GATEWAY_TOKEN": self.gateway.token, "RSI_PI_TOOL_TOKEN": self.token,
               "RSI_PI_TOOL_URL": f"http://127.0.0.1:{server.server_port}",
               "RSI_PI_SCHEMAS": str(directory / "schemas.json"),
               "RSI_PI_REQUEST_BYTES": str(self.gateway.hard_bytes)}
        argv = [str(self.executable), "--mode", "rpc", "--no-session",
                "--provider", "rsi-host", "--model", spec["model"],
                "--thinking", "off" if self.gateway.provider == "deepseek" else "high",
                "--no-builtin-tools", "--no-extensions", "--no-skills", "--no-prompt-templates",
                "--no-themes", "--no-context-files", "--no-mcp", "--offline", "--no-approve",
                "--extension", str(getattr(self, "extension_file", ROOT / "integrations/pi/host-actions.ts")),
                "--system-prompt", self.system_prompt]
        self.rpc = self.stack.enter_context(PiRPC(argv, cwd=directory, env=env,
                                                 own_process_group=self.own_process_group))

    def next_action(self, context, allowed_actions, limits=None, record=None, *,
                    source=None, write_action=None):
        if self.closed:
            raise PiError("coding session is closed")
        self.record = record or (lambda *_: None)
        if self._provider_secret.encode() in encoded(context):
            raise ValueError("provider credential must not enter coding context")
        if time.monotonic() >= self.deadline:
            raise PiDeadlineError("Pi attempt deadline exceeded")
        if not set(allowed_actions) <= self.schemas.keys():
            raise ValueError("unknown host action schema")
        self.source, self.write_action = source, write_action
        self.allowed = set(allowed_actions) | {"read_context"}
        if source is not None and write_action is not None:
            self.allowed.add("edit_source")
        self.pending, self.mutations = None, 0
        # Pi summaries may still cite an older admitted digest. Keep immutable
        # pages for this session; only the fresh snapshot is current authority.
        self.context_reads = {}
        self.snapshot = self.context.snapshot({
            "host": context,
            "remaining_seconds": max(0, self.deadline - time.monotonic()),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source is not None else None,
        }, target=16384 if self.gateway.provider == "deepseek" else 65536)
        try:
            from benchmarks.ascend.actor_time import preparation
            with preparation('Pi worker startup'):
                self._start()
            start = len(self.rpc.events)
            self.rpc.timeout = max(.1, self.deadline - time.monotonic())
            try:
                self.rpc.prompt("Select the next allowed action using the authoritative host snapshot.")
            except PiError:
                if self.gateway.stop_reason == "context-limit":
                    raise PiError("Pi request context exceeds the configured byte limit; no action accepted") from None
                if any(e.get("message", {}).get("stopReason") == "length" for e in self.rpc.events[start:]):
                    raise ActionError("truncated Pi tool call; no action accepted") from None
                raise
            if self.compaction_failure is not None:
                self.pending = None
                raise self.compaction_failure
            if not self.gateway.finished.wait(timeout=min(5, max(.1, self.deadline-time.monotonic()))):
                raise PiError("provider accounting did not settle")
            events = self.rpc.events[start:]
            if any(e.get("message", {}).get("stopReason") == "length" for e in events):
                raise ActionError("truncated Pi tool call; no action accepted")
            # Pi's argument validator can coerce scalars. Validate the original
            # assistant arguments as well before releasing any queued host effect.
            try:
                for event in events:
                    if event.get("type") != "message_end":
                        continue
                    message = event.get("message", {})
                    if message.get("role") != "assistant":
                        continue
                    for part in message.get("content", []):
                        if part.get("type") == "toolCall":
                            if part["name"] not in self.allowed:
                                raise ValueError("disallowed original tool call")
                            validate(part["arguments"], self.schemas[part["name"]])
            except (ValueError, KeyError, TypeError) as error:
                self.pending = None
                reason = "invalid original Pi tool arguments: " + str(error)
                self.record("pi-action-rejected", {"error": reason, "allowed": sorted(self.allowed)})
                raise ActionError(reason) from None
            if self.pending is None or (self.mutations != 1 and not self.first_action_only):
                raise ActionError("Pi must select exactly one valid host action")
            if time.monotonic() >= self.deadline:
                self.pending = None
                raise PiDeadlineError("Pi attempt deadline exceeded")
            name, args = self.pending
            # Persist intent before returning it to the host that owns side effects.
            self.record("pi-action", {"id": uuid.uuid4().hex, "name": name, "arguments": args,
                                      "discarded_actions": self.mutations - 1, **ENGINE_IDENTITY})
            return name, args
        finally:
            # Stream records through the host callback, including on failed or
            # cancelled turns. Never synthesize zero usage for unknown requests.
            self.gateway.finished.wait(timeout=min(5, max(.1, self.deadline-time.monotonic())))

    @property
    def completion_tokens(self):
        return sum((row.get("usage") or {}).get("completion_tokens", 0) or 0
                   for row in self.gateway.records)

    def close(self):
        if not self.closed:
            self.closed = True
            self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
