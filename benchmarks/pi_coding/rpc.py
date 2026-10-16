"""Small synchronous Pi RPC client with continuous pipe draining and deadlines."""

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import uuid


class PiError(RuntimeError):
    pass


class PiRPC:
    def __init__(self, argv, *, cwd: Path, env: dict, timeout=600):
        self.timeout = timeout
        self.events = []
        self.records = queue.Queue()
        self.process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        start_new_session=True)
        self.readers = []
        for stream, kind in ((self.process.stdout, "stdout"), (self.process.stderr, "stderr")):
            thread = threading.Thread(target=self._drain, args=(stream, kind), daemon=True)
            thread.start()
            self.readers.append(thread)

    def _drain(self, stream, kind):
        try:
            # Binary readline splits only on LF, not Unicode paragraph separators.
            for line in iter(stream.readline, b""):
                if kind == "stdout":
                    self.records.put((kind, line))
                # Drain stderr without persisting arbitrary subprocess diagnostics.
        finally:
            if kind == "stdout":
                self.records.put(("eof", b""))

    def send(self, command):
        self.process.stdin.write(json.dumps(command).encode() + b"\n")
        self.process.stdin.flush()

    def prompt(self, message):
        rid = uuid.uuid4().hex
        self.send({"id": rid, "type": "prompt", "message": message})
        deadline = time.monotonic() + self.timeout
        accepted = settled = False
        failure = None
        while not (accepted and settled):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Pi attempt deadline exceeded")
            try:
                kind, line = self.records.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError("Pi attempt deadline exceeded") from None
            if kind == "eof":
                raise PiError("Pi exited before completing the RPC request")
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError()
            except (ValueError, UnicodeError):
                raise PiError("invalid Pi JSONL record") from None
            self.events.append(event)
            if event.get("type") == "response" and event.get("id") == rid:
                if not event.get("success"):
                    raise PiError("Pi rejected the prompt")
                accepted = True
                if event.get("data", {}).get("disposition") == "handled":
                    settled = True
            if event.get("type") == "message_end":
                msg = event.get("message", {})
                if msg.get("role") == "assistant" and msg.get("stopReason") in ("error", "aborted"):
                    failure = "Pi provider request failed or was aborted"
            if event.get("type") == "agent_settled":
                settled = True
        if failure:
            raise PiError(failure)
        return self.events

    def close(self):
        if self.process.poll() is None:
            try:
                self.send({"type": "abort"})
                self.process.stdin.close()
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        for thread in self.readers:
            thread.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if not stream.closed:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
