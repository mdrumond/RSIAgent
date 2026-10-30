"""Loopback-only provider gateway: keys stay outside Pi; attempt deadlines remain authoritative.

Only the configured model and endpoint are forwarded. Reservations happen before
HTTP. The host can retain request evidence; credentials never enter that evidence.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import secrets
import threading
import time

import requests

from .context import ContextLimitError


class RequestContextLimit(ContextLimitError):
    """Rejected prepared request sizes; no upstream dispatch or ledger entry."""
    def __init__(self, request, raw, hard_bytes):
        super().__init__('provider request exceeds context hard byte budget')
        size = lambda value:len(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode())
        messages = request.get('messages',request.get('input',[]))
        self.details = dict(hard_bytes=hard_bytes, request_bytes=len(raw),
            request_sha256=hashlib.sha256(raw).hexdigest(), summary_request=not bool(request.get('tools')),
            messages_bytes=size(messages),tools_bytes=size(request.get('tools',[])),
            message_count=len(messages),tool_count=len(request.get('tools',[])),
            message_sizes=[dict(role=message.get('role',message.get('type')),bytes=size(message)) for message in messages],
            max_output_tokens=request.get('max_tokens',request.get('max_output_tokens',request.get('max_completion_tokens'))))


PROVIDERS = {
    "openai": {"model": "gpt-5.6-sol", "url": "https://api.openai.com/v1/chat/completions",
               "key": "OPENAI_API_KEY", "rates": (8.0, 40.0), "max_tokens": 8192},
    "deepseek": {"model": "deepseek-flash", "url": "https://api.deepseek.com/chat/completions",
                 "key": "DEEPSEEK_API_KEY", "rates": (2.0, 4.0), "max_tokens": 4096},
}


def normalize_usage(usage):
    """Keep provider breakdowns; cached input and reasoning are subsets of totals."""
    result = {"prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens")}
    if "input_tokens_details" in usage:
        result["prompt_tokens_details"] = usage["input_tokens_details"]
    if "output_tokens_details" in usage:
        result["completion_tokens_details"] = usage["output_tokens_details"]
    return result


def normalize_chat_usage(usage, provider):
    if not usage:
        return usage
    result = dict(usage)
    if provider == "deepseek":
        if type(usage.get("prompt_cache_hit_tokens")) is int:
            result["prompt_tokens_details"] = {"cached_tokens": usage["prompt_cache_hit_tokens"]}
        # The gateway pins thinking=disabled on every DeepSeek request.
        result.setdefault("completion_tokens_details", {"reasoning_tokens": 0})
    return result


def safe_provider_error(error, api_key, token):
    if not isinstance(error, dict):
        return {}
    return {k: str(error[k]).replace(api_key, "[redacted]").replace(token, "[redacted]")[:500]
            for k in ("type", "code", "param", "message") if k in error}


class Gateway:
    def __init__(self, provider, api_key, budget, label, *, max_calls=9, deadline=None, max_output_tokens=None, records_path=None, hard_bytes=49152, on_record=None, request_projector=None):
        self.provider, self.spec = provider, dict(PROVIDERS[provider])
        if max_output_tokens is not None:
            self.spec["max_tokens"] = max_output_tokens
        self.records_path = records_path
        self.request_projector = request_projector
        self.hard_bytes = hard_bytes
        self.on_record = on_record or (lambda *_: None)
        self.finished = threading.Event()
        self.finished.set()
        self.stop_reason = None
        self.api_key, self.budget, self.label = api_key, budget, label
        self.token = secrets.token_urlsafe(32)
        self.max_calls, self.deadline = max_calls, deadline if deadline is not None else time.monotonic() + 600
        self.calls, self.records = 0, []
        self.lock = threading.Lock()

    def prepare(self, payload, *, responses=False):
        if payload.get("model") != self.spec["model"]:
            raise ValueError("unexpected model")
        # Keep only the fields needed by the two harnesses; no user-supplied URLs.
        fields = (
            "model", "messages", "tools", "tool_choice", "stream", "stream_options",
        ) if not responses else ("model", "input", "instructions", "tools", "tool_choice", "stream", "include")
        request = {k: v for k, v in payload.items() if k in fields}
        messages = request.get("input" if responses else "messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages are required")
        output_limit = self.spec["max_tokens"]
        if not payload.get("tools"):
            # Pi's native compactor supplies its own history/prefix output budget.
            # Respect it without exceeding the host's registered profile ceiling.
            # The loopback provider uses OpenAI compatibility naming even for
            # DeepSeek; translate either SDK chat key to the real provider key.
            keys = ("max_output_tokens",) if responses else ("max_completion_tokens", "max_tokens")
            requested = next((payload[key] for key in keys if key in payload),None)
            if requested is not None:
                if type(requested) is not int or requested <= 0:
                    raise ValueError("invalid requested summary output limit")
                output_limit = min(output_limit,requested)
        if responses:
            if self.provider != "openai":
                raise ValueError("Responses API is configured only for OpenAI")
            request.update(max_output_tokens=output_limit, reasoning={"effort": "high"},
                           service_tier="default", store=False, parallel_tool_calls=False)
        elif self.provider == "openai":
            request.update(max_completion_tokens=output_limit, reasoning_effort="high",
                           service_tier="default")
        else:
            request.update(max_tokens=output_limit, thinking={"type": "disabled"}, top_p=1.0)
        if request.get("stream") and not responses:
            request["stream_options"] = {"include_usage": True}
        request["parallel_tool_calls"] = False
        if self.request_projector is not None:
            request = self.request_projector(request, responses=responses, hard_bytes=self.hard_bytes)
        encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
        if len(encoded) > self.hard_bytes:
            raise RequestContextLimit(request,encoded,self.hard_bytes)
        return request

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def error(self, status, message, *, context_overflow=False):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error = {"message": message, "type": "invalid_request_error" if context_overflow else "smoke_gateway"}
                if context_overflow:
                    error["code"] = "context_length_exceeded"
                self.wfile.write(json.dumps({"error": error}).encode())

            def do_POST(self):
                if self.path not in ("/v1/chat/completions", "/v1/responses") or self.headers.get("Authorization") != "Bearer " + owner.token:
                    return self.error(403, "gateway access denied")
                started = time.monotonic()
                started_epoch = time.time()
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= (8388608 if owner.request_projector is not None else 524288):
                        return self.error(413, "smoke request size limit")
                    payload = json.loads(self.rfile.read(size))
                    responses = self.path == "/v1/responses"
                    request = owner.prepare(payload, responses=responses)
                    with owner.lock:
                        if time.monotonic() >= owner.deadline:
                            owner.stop_reason = "deadline"
                            raise TimeoutError("attempt deadline reached")
                        rid = owner.budget.reserve(request, output_tokens=owner.spec["max_tokens"],
                                                   rates=owner.spec["rates"], label=owner.label)
                        owner.calls += 1
                except TimeoutError as exc:
                    owner.stop_reason = owner.stop_reason or "deadline"
                    return self.error(408, str(exc))
                except (ValueError, TypeError, AttributeError) as error:
                    if isinstance(error, ContextLimitError) or "hard byte" in str(error):
                        owner.stop_reason = "context-limit"
                        owner.on_record("pi-context-limit", {"hard_bytes": owner.hard_bytes, **getattr(error,"details",{})})
                        return self.error(413, "context_length_exceeded: request exceeds the context window byte budget", context_overflow=True)
                    return self.error(400, "invalid smoke request")
                row = {"reservation_id": rid, "usage": None, "status": "unknown", "request_bytes": len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode())}
                owner.records.append(row)
                owner.finished.clear()
                try:
                    # Intent is durable before HTTP. Cancellation/unknown usage
                    # still consumes the request and its conservative reservation.
                    owner.on_record("pi-provider-request", {
                        "reservation_id": rid, "request_bytes": row["request_bytes"], "request": request})
                    url = "https://api.openai.com/v1/responses" if responses else owner.spec["url"]
                    with requests.post(url, data=json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(), headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + owner.api_key,
                    }, timeout=(min(15, max(0.1, owner.deadline - time.monotonic())),
                                min(600, max(0.1, owner.deadline - time.monotonic()))),
                       stream=True, allow_redirects=False) as response:
                        row["http_status"] = response.status_code
                        if response.status_code != 200:
                            row["status"] = "provider_error"
                            try:
                                error = response.json().get("error", {})
                                row["provider_error"] = safe_provider_error(error, owner.api_key, owner.token)
                            except (ValueError, AttributeError, TypeError):
                                pass
                            return self.error(502, f"provider HTTP {response.status_code}")
                        if request.get("stream"):
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.end_headers()
                            for line in response.iter_lines():
                                if time.monotonic() >= owner.deadline:
                                    raise TimeoutError("attempt deadline")
                                if line.startswith(b"data: ") and line != b"data: [DONE]":
                                    data = json.loads(line[6:])
                                    if data.get("type") in ("error", "response.failed") or data.get("error"):
                                        row["provider_error"] = safe_provider_error(
                                            data.get("error") or data.get("response", {}).get("error") or data, owner.api_key, owner.token)
                                    if data.get("usage"):
                                        row["usage"] = normalize_chat_usage(data["usage"], owner.provider)
                                    if responses and data.get("response", {}).get("usage"):
                                        usage = data["response"]["usage"]
                                        row["usage"] = normalize_usage(usage)
                                self.wfile.write(line + b"\n")
                                self.wfile.flush()
                        else:
                            data = response.json()
                            row["usage"] = normalize_chat_usage(data.get("usage"), owner.provider)
                            if responses and row["usage"]:
                                row["usage"] = normalize_usage(row["usage"])
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(json.dumps(data).encode())
                        row["status"] = "provider_error" if row.get("provider_error") else "completed"
                except (requests.RequestException, OSError, ValueError) as exc:
                    row["status"] = "provider_error" if row.get("provider_error") else "transport_error"
                    row["error_type"] = type(exc).__name__
                    # The reservation stays charged when the provider outcome is unknown.
                finally:
                    try:
                        if row["usage"]:
                            owner.budget.settle(rid, row["usage"], owner.spec["rates"])
                        row["seconds"] = round(time.monotonic() - started, 3)
                        row.update(started_epoch=started_epoch, ended_epoch=time.time())
                        owner.on_record("pi-provider", dict(row))
                        if owner.records_path:
                            owner.records_path.write_text(json.dumps(owner.records, indent=2))
                    finally:
                        owner.finished.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
