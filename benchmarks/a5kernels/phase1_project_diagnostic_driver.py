"""Staged host-owned diagnostic runner (stdlib only; no Actor-selected commands)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

_LIMIT = 48_000
_TIMEOUT = 120
_TOOLS = {"msdebug", "memcheck", "racecheck", "initcheck", "synccheck"}


def _capture(command, *, environment, timeout=_TIMEOUT):
    """Bound output and the entire tool process group, including its inferior."""
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, env=environment,
                                   start_new_session=True)
        reason = None
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if time.monotonic() >= deadline:
                reason = "timeout"
            elif os.fstat(output.fileno()).st_size > _LIMIT:
                reason = "output-limit"
            if reason:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # Kill descendants even if the launcher handled SIGINT and exited.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                break
            time.sleep(0.05)
        # A tool can exit before its inferior; never leave that process group running.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output.seek(0)
        raw = output.read(_LIMIT + 1)
        return {"command": list(command), "exit_code": process.returncode,
                "termination": reason, "output": raw[:_LIMIT].decode("utf-8", "replace"),
                "output_truncated": len(raw) > _LIMIT}


def diagnose(arguments):
    tool = arguments[3].removeprefix("diagnostic-")
    if tool not in _TOOLS:
        raise ValueError("unsupported diagnostic tool")
    candidate = Path(arguments[1]).read_bytes()
    manifest = json.loads(Path(arguments[2]).read_text(encoding="utf-8"))
    candidate_sha256 = hashlib.sha256(candidate).hexdigest()
    if candidate_sha256 != manifest["candidate_sha256"]:
        raise ValueError("diagnostic source differs from manifest")
    executable = shutil.which("msdebug" if tool == "msdebug" else "mssanitizer")
    record = {"status": "diagnostic-observation", "diagnostic": "diagnostic execution only",
              "tool": tool, "available": executable is not None,
              "candidate_sha256": candidate_sha256,
              "coverage": "unconfirmed", "host_verified": False,
              "scope": "application replay, including framework setup kernels"}
    if executable is None:
        record["diagnostic"] = "tool executable unavailable in checked runtime"
        return record
    environment = dict(os.environ)
    environment["PYTORCH_NO_NPU_MEMORY_CACHING"] = "1"
    version = _capture((executable, "--version"), environment=environment, timeout=10)
    record["version"] = version
    application = (sys.executable, "-B", arguments[0], arguments[1], arguments[2],
                   "run", *arguments[4:])
    if tool == "msdebug":
        name = manifest["abi"]["kernel_name"]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise ValueError("invalid kernel breakpoint name")
        # LLDB batch flags are checked against installed msdebug --help. Pending
        # regex breakpoint can resolve JIT symbols; a missing hit is not success.
        command = (executable, "--batch", "--no-lldbinit",
                   "-o", "breakpoint set -r " + name, "-o", "run",
                   "-o", "ascend info cores", "-o", "thread backtrace", "--", *application)
    else:
        command = (executable, "--tool=" + tool, "--max-debuglog-size=1", "--", *application)
    execution = _capture(command, environment=environment)
    record["execution"] = execution
    record["diagnostic"] = (
        "bounded diagnostic terminated: " + execution["termination"]
        if execution["termination"] else
        "inspect retained transcript; tool exit status does not prove instruction coverage"
    )
    # Instrumentation requirements differ by compiler and chip. Do not inject
    # older AscendC sanitizer/-O0 flags into A5 Catlass or claim a clean check.
    return record
