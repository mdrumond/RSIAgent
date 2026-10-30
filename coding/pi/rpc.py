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


class PiProviderError(PiError):
    pass


class PiDeadlineError(PiError, TimeoutError):
    pass


class PiCancelled(PiError):
    pass


class PiRPC:
    def __init__(self, argv, *, cwd: Path, env: dict, timeout=600, own_process_group=True):
        self.timeout = timeout
        self.own_process_group = own_process_group
        self.events = []
        self.records = queue.Queue()
        self.process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        start_new_session=own_process_group)
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
        try:
            self.process.stdin.write(json.dumps(command).encode() + b"\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            raise PiError("Pi RPC input closed") from None

    def prompt(self, message, *, pump=None, remaining_seconds=None, on_event=None):
        rid = uuid.uuid4().hex
        self.send({"id": rid, "type": "prompt", "message": message})
        deadline = time.monotonic() + self.timeout
        accepted = settled = False
        failure = None
        while not (accepted and settled):
            if pump is not None:
                pump()
            remaining = deadline - time.monotonic()
            if remaining_seconds is not None:
                dynamic = remaining_seconds()
                remaining = dynamic if dynamic is not None else self.timeout
            if remaining <= 0:
                raise PiDeadlineError("Pi attempt deadline exceeded")
            try:
                kind, line = self.records.get(timeout=min(remaining, .02) if pump is not None else remaining)
            except queue.Empty:
                if pump is not None:
                    continue
                raise PiDeadlineError("Pi attempt deadline exceeded") from None
            if kind == "eof":
                raise PiError("Pi exited before completing the RPC request")
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError()
            except (ValueError, UnicodeError):
                raise PiError("invalid Pi JSONL record") from None
            self.events.append(event)
            if on_event is not None:
                on_event(event)
            if event.get("type") == "response" and event.get("id") == rid:
                if not event.get("success"):
                    raise PiError("Pi rejected the prompt")
                accepted = True
                if event.get("data", {}).get("disposition") == "handled":
                    settled = True
            if event.get("type") == "message_end":
                msg = event.get("message", {})
                if msg.get("role") == "assistant" and msg.get("stopReason") in ("error", "aborted"):
                    failure = PiCancelled if msg.get("stopReason") == "aborted" else PiProviderError
            if event.get("type") == "agent_settled":
                settled = True
        if failure:
            raise failure("Pi provider request failed or was aborted")
        return self.events

    def _kill(self, sig):
        try:
            if self.own_process_group:
                os.killpg(self.process.pid, sig)
            else:
                self.process.send_signal(sig)
        except ProcessLookupError:
            pass

    def close(self):
        if self.process.poll() is None:
            try:
                self.send({"type": "abort"})
                self.process.stdin.close()
                self.process.wait(timeout=3)
            except (PiError, OSError, subprocess.TimeoutExpired):
                self._kill(signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._kill(signal.SIGKILL)
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
