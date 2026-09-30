"""Stage, execute, and independently verify the A3 hello kernel."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Callable

from benchmarks.a3kernels.fixture import source_files
from benchmarks.a3kernels.protocol import (
    ExecutionPlan,
    RunRequest,
    VerifiedResult,
    canonical_hash,
)


OUTPUT_MARKER = "A3KERNEL_OUTPUT="
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class A3KernelRunner:
    def __init__(self, command_runner: CommandRunner = subprocess.run, *, atol=1e-5):
        self._command_runner = command_runner
        self._atol = float(atol)

    def prepare(self, request: RunRequest) -> ExecutionPlan:
        rng = random.Random(request.seed)
        a = tuple(rng.uniform(-1.0, 1.0) for _ in range(request.length))
        b = tuple(rng.uniform(-1.0, 1.0) for _ in range(request.length))
        return ExecutionPlan(
            request_id=request.request_id,
            files=source_files(),
            argv=(sys.executable, "host_driver.py", "input.json"),
            input_a=a,
            input_b=b,
        )

    def run(self, request: RunRequest, workdir: Path) -> VerifiedResult:
        plan = self.prepare(request)
        workdir.mkdir(parents=True, exist_ok=True)
        for source in plan.files:
            (workdir / source.relative_path).write_text(source.content, encoding="utf-8")
        (workdir / "input.json").write_text(
            json.dumps(
                {"input_a": plan.input_a, "input_b": plan.input_b},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        completed = self._command_runner(
            plan.argv, cwd=workdir, text=True, capture_output=True, check=False
        )
        output: tuple[float, ...] = ()
        parse_error = None
        if completed.returncode == 0:
            try:
                output = _parse_output(completed.stdout)
            except (TypeError, ValueError) as exc:
                parse_error = str(exc)
        expected = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
        valid = len(output) == len(expected) and all(math.isfinite(x) for x in output)
        max_error = (
            max((abs(got - want) for got, want in zip(output, expected)), default=0.0)
            if valid
            else None
        )
        passed = (
            completed.returncode == 0
            and valid
            and max_error is not None
            and max_error <= self._atol
        )
        output_sha = hashlib.sha256(
            json.dumps(output, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        evidence_sha = canonical_hash(
            {
                "exit_code": completed.returncode,
                "stderr": completed.stderr,
                "stdout": completed.stdout,
                "output_sha256": output_sha,
                "output_parse_error": parse_error,
            }
        )
        fields = {
            "request_id": plan.request_id,
            "execution_id": plan.execution_id,
            "passed": passed,
            "max_abs_error": max_error,
            "exit_code": completed.returncode,
            "source_fingerprint": plan.source_fingerprint,
            "output_sha256": output_sha,
            "a3_evidence_sha256": evidence_sha,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "output_parse_error": parse_error,
        }
        return VerifiedResult(**fields, attestation_sha256=canonical_hash(fields))


def _parse_output(stdout: str) -> tuple[float, ...]:
    records = [
        line.removeprefix(OUTPUT_MARKER)
        for line in stdout.splitlines()
        if line.startswith(OUTPUT_MARKER)
    ]
    if len(records) != 1:
        raise ValueError("runtime must emit exactly one A3KERNEL_OUTPUT record")
    payload = json.loads(records[0])
    if not isinstance(payload, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in payload
    ):
        raise ValueError("A3KERNEL_OUTPUT must be a JSON numeric array")
    return tuple(float(value) for value in payload)
