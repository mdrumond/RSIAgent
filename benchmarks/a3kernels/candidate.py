"""Host-owned execution boundary for one agent-authored A3 Ascend C source."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path
import random
import re
import subprocess
from typing import Callable

from benchmarks.a3kernels.phase1_protocol import (
    ExecutionPlan,
    ExecutionReceipt,
    FailedEvidence,
    SourceFile,
    VerifiedResult,
    attest,
)


_PACKAGE = "benchmarks.a3kernels.candidate_runtime"
_HOST_FILES = ("build.json", "host_driver.py", "host_wrapper.inc")
_COMPILE_ARGV = ("python", "host_driver.py", "--compile-only")
_RUN_ARGV = ("python", "host_driver.py", "input.json")
_COMPILE_MARKER = "A3CANDIDATE_COMPILED="
_OUTPUT_MARKER = "A3KERNEL_OUTPUT="
_SIGNATURE = re.compile(
    r'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+vector_add\s*\(\s*'
    r'GM_ADDR\s+input_a\s*,\s*GM_ADDR\s+input_b\s*,\s*GM_ADDR\s+output\s*,\s*'
    r'uint32_t\s+count\s*,\s*uint32_t\s+buffer_bytes\s*\)',
    re.MULTILINE,
)
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _host_source_files() -> tuple[SourceFile, ...]:
    root = files(_PACKAGE)
    return tuple(
        SourceFile(name, root.joinpath(name).read_text(encoding="utf-8"))
        for name in _HOST_FILES
    )


def validate_candidate_source(source: str) -> None:
    if type(source) is not str or not source.strip():
        raise ValueError("candidate must contain the exact exported vector_add signature")
    if len(_SIGNATURE.findall(source)) != 1 or len(re.findall(r"\bvector_add\s*\(", source)) != 1:
        raise ValueError("candidate must contain the exact exported vector_add signature once")
    if any(token in source.lower() for token in ("catlass", "@tla", "torch_library", "torch::")):
        raise ValueError("candidate source must contain only the Ascend C device implementation")


@dataclass(frozen=True)
class CandidateCompilation:
    plan: ExecutionPlan
    library_sha256: str
    stdout: str
    stderr: str
    attestation_sha256: str

    def attestation_payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "attestation_sha256"
        }


class A3CandidateBackend:
    def __init__(self, command_runner: CommandRunner = subprocess.run, *, atol=1e-5):
        self._command_runner = command_runner
        self._atol = float(atol)

    def plan(
        self,
        source: str,
        *,
        request_id: str,
        attempt_id: str,
        length: int,
        padded_length: int | None = None,
        block_count: int = 1,
        seed: int,
        project_id: str = "vector-add",
    ) -> ExecutionPlan:
        validate_candidate_source(source)
        if type(length) is not int or not 1 <= length <= 4096:
            raise ValueError("length must be an integer in [1, 4096]")
        if padded_length is None:
            padded_length = length
        if type(padded_length) is not int or not length <= padded_length <= 4096:
            raise ValueError("padded_length must be an integer in [length, 4096]")
        if type(block_count) is not int or not 1 <= block_count <= 32:
            raise ValueError("block_count must be an integer in [1, 32]")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        rng = random.Random(seed)
        input_a = tuple(rng.uniform(-1.0, 1.0) for _ in range(length))
        input_b = tuple(rng.uniform(-1.0, 1.0) for _ in range(length))
        padding = (0.0,) * (padded_length - length)
        return ExecutionPlan(
            request_id=request_id,
            attempt_id=attempt_id,
            project_id=project_id,
            files=(SourceFile("candidate.cpp", source), *_host_source_files()),
            argv=_RUN_ARGV,
            input_a=input_a + padding,
            input_b=input_b + padding,
            logical_length=length,
            padded_length=padded_length,
            block_count=block_count,
        )

    def compile(
        self,
        source: str,
        workdir: Path,
        *,
        request_id: str,
        attempt_id: str,
        length: int = 1,
        padded_length: int | None = None,
        block_count: int = 1,
        seed: int = 0,
        project_id: str = "vector-add",
    ) -> CandidateCompilation | FailedEvidence:
        plan = self.plan(
            source, request_id=request_id, attempt_id=attempt_id,
            length=length, padded_length=padded_length, block_count=block_count,
            seed=seed, project_id=project_id,
        )
        self._stage(plan, workdir)
        return self._compile(plan, workdir)

    def run(
        self,
        source: str,
        workdir: Path,
        *,
        request_id: str,
        attempt_id: str,
        length: int,
        padded_length: int | None = None,
        block_count: int = 1,
        seed: int = 0,
        project_id: str = "vector-add",
    ) -> VerifiedResult | FailedEvidence:
        plan = self.plan(
            source, request_id=request_id, attempt_id=attempt_id,
            length=length, padded_length=padded_length, block_count=block_count,
            seed=seed, project_id=project_id,
        )
        self._stage(plan, workdir)
        compilation = self._compile(plan, workdir)
        if isinstance(compilation, FailedEvidence):
            return compilation
        completed = self._command_runner(
            _RUN_ARGV, cwd=workdir, text=True, capture_output=True, check=False
        )
        if completed.returncode != 0:
            return FailedEvidence.create(
                plan, stage="execute", error_type="RuntimeError",
                detail=_detail(completed),
            )
        try:
            output = _parse_output(completed.stdout, len(plan.input_a))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return FailedEvidence.create(
                plan, stage="verify", error_type="OutputError", detail=str(exc)
            )
        expected = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
        max_error = max(abs(got - want) for got, want in zip(output, expected))
        receipt = ExecutionReceipt(
            exit_code=0,
            output=output,
            stdout=completed.stdout,
            stderr=completed.stderr,
            metadata=(("library_sha256", compilation.library_sha256),),
        )
        return VerifiedResult.from_receipt(
            plan, receipt, max_abs_error=max_error, tolerance=self._atol
        )

    @staticmethod
    def _stage(plan: ExecutionPlan, workdir: Path) -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        for source in plan.files:
            (workdir / source.relative_path).write_text(
                source.content, encoding="utf-8"
            )
        (workdir / "input.json").write_text(
            json.dumps(
                {
                    "input_a": plan.input_a,
                    "input_b": plan.input_b,
                    "logical_length": plan.logical_length,
                    "padded_length": plan.padded_length,
                    "block_count": plan.block_count,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    def _compile(
        self, plan: ExecutionPlan, workdir: Path
    ) -> CandidateCompilation | FailedEvidence:
        completed = self._command_runner(
            _COMPILE_ARGV, cwd=workdir, text=True, capture_output=True, check=False
        )
        if completed.returncode != 0:
            return FailedEvidence.create(
                plan, stage="compile", error_type="CompileError",
                detail=_detail(completed),
            )
        records = [
            line.removeprefix(_COMPILE_MARKER)
            for line in completed.stdout.splitlines()
            if line.startswith(_COMPILE_MARKER)
        ]
        if len(records) != 1 or re.fullmatch(r"[0-9a-f]{64}", records[0]) is None:
            return FailedEvidence.create(
                plan, stage="compile", error_type="CompileOutputError",
                detail="compiler must emit one lowercase A3CANDIDATE_COMPILED digest",
            )
        body = {
            "plan": plan,
            "library_sha256": records[0],
            "stderr": completed.stderr,
            "stdout": completed.stdout,
        }
        return CandidateCompilation(
            plan, records[0], completed.stdout, completed.stderr, attest(body)
        )


def _parse_output(stdout: str, expected_length: int) -> tuple[float, ...]:
    records = [
        line.removeprefix(_OUTPUT_MARKER)
        for line in stdout.splitlines()
        if line.startswith(_OUTPUT_MARKER)
    ]
    if len(records) != 1:
        raise ValueError("runtime must emit exactly one A3KERNEL_OUTPUT record")
    value = json.loads(records[0])
    if (
        not isinstance(value, list)
        or len(value) != expected_length
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value)
    ):
        raise ValueError("A3KERNEL_OUTPUT must be a finite numeric array of the requested length")
    return tuple(float(item) for item in value)


def _detail(completed: subprocess.CompletedProcess[str]) -> str:
    return (completed.stderr or completed.stdout or f"process exited {completed.returncode}").strip()


__all__ = [
    "A3CandidateBackend", "CandidateCompilation", "FailedEvidence",
    "VerifiedResult", "validate_candidate_source",
]
