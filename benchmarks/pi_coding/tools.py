"""Host-owned candidate tools; no shell, arbitrary paths, or connection settings."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
import time

from .observations import compact_observation


class CandidateTools:
    def __init__(self, root, validator, *, deadline, max_revisions=3, on_validation=None):
        self.root, self.validator, self.deadline = root, validator, deadline
        self.validator.deadline = deadline
        self.writes = 0
        self.max_revisions = max_revisions
        self.on_validation = on_validation
        self.lock = threading.Lock()

    def execute(self, name, arguments):
        with self.lock:
            if time.monotonic() >= self.deadline:
                raise TimeoutError("attempt deadline exceeded")
            path = self.root / "kernel.cpp"
            if path.is_symlink():
                raise ValueError("candidate symlinks are forbidden")
            if name == "context_snapshot" and arguments == {}:
                observations = self.validator.results
                return {"source": path.read_text(),
                        "latest": compact_observation(observations[-1]) if observations else None,
                        "earlier": [{"status": r["status"], "diagnostic": compact_observation(r).get("diagnostic", "")[:250]}
                                    for r in observations[:-1]],
                        "remaining_revisions": self.max_revisions - self.writes}
            if name == "read_candidate" and arguments == {}:
                return {"source": path.read_text()}
            edit = name == "edit_and_validate" and set(arguments) == {"old_text", "new_text"}
            if edit or (name in ("write_candidate", "submit_candidate") and set(arguments) == {"source"}):
                if edit:
                    old, new = arguments["old_text"], arguments["new_text"]
                    if not isinstance(old, str) or not old or not isinstance(new, str):
                        raise ValueError("edit requires nonempty old_text and string new_text")
                    current = path.read_text()
                    if current.count(old) != 1:
                        raise ValueError("old_text must match exactly once; candidate unchanged")
                    source = current.replace(old, new, 1)
                else:
                    source = arguments["source"]
                if not isinstance(source, str) or not source.strip() or len(source.encode()) > 65536:
                    raise ValueError("source must be nonempty text of at most 64 KiB")
                if self.writes >= self.max_revisions:
                    raise ValueError("candidate writes exhausted")
                if (name == "submit_candidate" or edit) and len(self.validator.results) >= self.validator.max_validations:
                    raise ValueError("validation cycles exhausted")
                path.write_text(source)
                self.writes += 1
                revisions = self.root / "revisions"
                revisions.mkdir(exist_ok=True)
                (revisions / f"{self.writes:03d}.cpp").write_text(source)
                if name == "submit_candidate" or edit:
                    return self._validate()
                return {"source_sha256": hashlib.sha256(source.encode()).hexdigest()}
            if name == "validate_candidate" and arguments == {}:
                return self._validate()
            raise ValueError("unknown tool or invalid arguments")

    def _validate(self):
        import sys
        print(f"Validating candidate (cycle {len(self.validator.results) + 1}/{self.validator.max_validations})...", file=sys.stderr, flush=True)
        result = self.validator.validate()
        if self.on_validation:
            self.on_validation(result)
        return result


class ToolServer:
    def __init__(self, tools, *, compact=False):
        self.tools, self.token = tools, secrets.token_urlsafe(32)
        self.compact = compact

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                status = 200
                try:
                    if self.path != "/tool" or self.headers.get("Authorization") != "Bearer " + owner.token:
                        raise ValueError("tool access denied")
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 262144:
                        raise ValueError("invalid tool request size")
                    data = json.loads(self.rfile.read(size))
                    payload = owner.tools.execute(data["name"], data["arguments"])
                    if owner.compact and "status" in payload:
                        payload = compact_observation(payload)
                        payload["remaining_revisions"] = owner.tools.max_revisions - owner.tools.writes
                except (ValueError, TypeError, KeyError, OSError, TimeoutError) as exc:
                    status, payload = 400, {"error": str(exc)}
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(json.dumps(payload).encode())
                except BrokenPipeError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/tool"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
