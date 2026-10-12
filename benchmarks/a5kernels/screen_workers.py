"""Durable local GNU Screen envelopes for campaign cell workers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, Sequence

from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.priority_controller import (
    Job, JobKind, Observation, RetainedAssignment,
)


REQUEST_SCHEMA = "a5-screen-worker-request-v1"
RESULT_SCHEMA = "a5-screen-worker-result-v1"


class ScreenWorkerPool:
    """Start and observe one deterministic Screen session per worker attempt."""

    def __init__(
        self, root: Path, *, script: Path,
        screen_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.root = Path(root).resolve()
        self.script = Path(script).resolve()
        self.run = screen_runner

    @staticmethod
    def _handle(job: Job, device: int, ordinal: int) -> str:
        token = hashlib.sha256(
            f"{job.job_id}:{device}:{ordinal}".encode("utf-8")
        ).hexdigest()[:12]
        return f"rsi-a5-{token}"

    def recover(self) -> tuple[RetainedAssignment, ...]:
        """Recover unrecorded live Screen envelopes after controller restart."""

        recovered = []
        workers = self.root / "workers"
        for path in sorted(workers.glob("*/*/attempt-*/request.json")):
            if (path.parent / "result.json").exists():
                continue
            request = _read_request(path)
            job = Job(JobKind(request["job_kind"]), request["cell_id"])
            assignment = RetainedAssignment(
                job, request["physical_device"],
                self._handle(job, request["physical_device"], request["attempt"]),
                request["attempt"],
            )
            if self.observe(assignment).state == "active":
                recovered.append(assignment)
        return tuple(recovered)

    def start(
        self, job: Job, device: int, worker_root: Path, command: Sequence[str],
    ) -> RetainedAssignment:
        worker_root = Path(worker_root).resolve()
        expected = self.root / "workers" / job.kind.value / job.cell_id
        if worker_root != expected:
            raise ValueError("worker root does not match scheduled identity")
        attempts = [
            int(item.name.removeprefix("attempt-"))
            for item in worker_root.glob("attempt-[0-9][0-9][0-9][0-9]")
            if item.is_dir()
        ]
        ordinal = max(attempts, default=0) + 1
        attempt = worker_root / f"attempt-{ordinal:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        body = {
            "schema": REQUEST_SCHEMA,
            "job_kind": job.kind.value,
            "cell_id": job.cell_id,
            "physical_device": device,
            "attempt": ordinal,
            "command": list(command),
        }
        request = {**body, "request_sha256": canonical_digest(body)}
        request_path = attempt / "request.json"
        request_path.write_bytes(canonical_bytes(request) + b"\n")
        handle = self._handle(job, device, ordinal)
        # A detached Screen server inherits its launcher's file descriptors.
        # Regular files let subprocess.run return when the Screen client exits;
        # capture pipes would stay open until the detached worker finishes.
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as diagnostics:
            completed = self.run(
                ("screen", "-DmS", handle, sys.executable, str(self.script),
                 "worker", "--request", str(request_path)),
                text=True, stdout=diagnostics, stderr=subprocess.STDOUT,
                check=False,
            )
            diagnostics.seek(0)
            detail = diagnostics.read().strip()
        if completed.returncode != 0:
            message = "GNU Screen worker could not start"
            raise RuntimeError(f"{message}: {detail}" if detail else message)
        return RetainedAssignment(job, device, handle, ordinal)

    def observe(self, assignment: RetainedAssignment) -> Observation:
        attempt = (
            self.root / "workers" / assignment.job.kind.value
            / assignment.job.cell_id / f"attempt-{assignment.start_identity:04d}"
        )
        result_path = attempt / "result.json"
        if result_path.exists():
            request = _read_request(attempt / "request.json")
            try:
                result = json.loads(result_path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("invalid Screen worker result") from exc
            body = {key: value for key, value in result.items()
                    if key != "record_sha256"}
            if (
                set(result) != {"schema", "request_sha256", "exit_code", "record_sha256"}
                or result.get("schema") != RESULT_SCHEMA
                or result.get("request_sha256") != request["request_sha256"]
                or isinstance(result.get("exit_code"), bool)
                or not isinstance(result.get("exit_code"), int)
                or result.get("record_sha256") != canonical_digest(body)
            ):
                raise ValueError("invalid Screen worker result")
            return Observation.completed(result["exit_code"])
        completed = self.run(
            ("screen", "-ls", assignment.worker_handle),
            text=True, capture_output=True, check=False,
        )
        if completed.returncode == 0 and assignment.worker_handle in completed.stdout:
            return Observation.active()
        return Observation.incomplete()


def _read_request(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Screen worker request") from exc
    body = {key: item for key, item in value.items() if key != "request_sha256"}
    if (
        set(value) != {"schema", "job_kind", "cell_id", "physical_device",
                       "attempt", "command", "request_sha256"}
        or value.get("schema") != REQUEST_SCHEMA
        or value.get("request_sha256") != canonical_digest(body)
        or value.get("job_kind") not in {item.value for item in JobKind}
        or not isinstance(value.get("cell_id"), str) or not value["cell_id"]
        or isinstance(value.get("physical_device"), bool)
        or not isinstance(value.get("physical_device"), int)
        or not 1 <= value["physical_device"] <= 7
        or isinstance(value.get("attempt"), bool)
        or not isinstance(value.get("attempt"), int) or value["attempt"] < 1
        or not isinstance(value.get("command"), list) or not value["command"]
        or any(not isinstance(item, str) or not item for item in value["command"])
    ):
        raise ValueError("invalid Screen worker request")
    return value


def run_request(
    path: Path, *, process_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    """Run one immutable request and publish its exact process exit code."""

    path = Path(path).resolve()
    request = _read_request(path)
    completed = process_runner(request["command"], check=False)
    body = {
        "schema": RESULT_SCHEMA,
        "request_sha256": request["request_sha256"],
        "exit_code": completed.returncode,
    }
    result = {**body, "record_sha256": canonical_digest(body)}
    destination = path.parent / "result.json"
    if destination.exists():
        if destination.read_bytes() != canonical_bytes(result) + b"\n":
            raise ValueError("immutable Screen worker result conflicts")
    else:
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(canonical_bytes(result) + b"\n")
        os.replace(temporary, destination)
    return completed.returncode


__all__ = ["ScreenWorkerPool", "run_request"]
