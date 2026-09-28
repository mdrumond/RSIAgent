"""Concrete host-owned runtime for agent-authored Catlass candidates."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.fixtures import (
    Language,
    catlass_candidate_fixture,
)
from benchmarks.a5kernels.matrix import Workload
from benchmarks.a5kernels.profiling import (
    ProfileRequest,
    ProfilingTreatmentController,
)
from benchmarks.a5kernels.protocol import ExecutionPlan, ExecutionReceipt, RunRequest
from benchmarks.a5kernels.runner import A5KernelRunner, ExecutionBackend
from benchmarks.a5kernels.trial import CandidateRun


EXACT_KERNEL_NAME = "vector_add__kernel0"
_KERNEL_MARKER = "A5KERNEL_NAME="
_VECTOR_ARGUMENTS = ("gm_a", "gm_b", "gm_c")


@dataclass(frozen=True)
class CompileDiagnostics:
    """Stable host evidence from a real Catlass compiler invocation."""

    passed: bool
    exit_code: int
    execution_id: str
    source_fingerprint: str
    attestation_sha256: str
    kernel_name: str | None
    diagnostics: str


def validate_candidate_source(source: str) -> None:
    """Require the one agent-owned ABI while leaving launch policy host-owned."""

    if not source.strip():
        raise ValueError("candidate source must not be empty")
    try:
        module = ast.parse(source, filename="kernel.py")
    except SyntaxError as exc:
        raise ValueError(f"candidate source is not valid Python: {exc.msg}") from exc
    functions = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise ValueError("candidate must define exactly one synchronous kernel")
    kernel = functions[0]
    arguments = kernel.args
    if (
        kernel.name != "vector_add"
        or tuple(argument.arg for argument in arguments.args) != _VECTOR_ARGUMENTS
        or arguments.posonlyargs
        or arguments.kwonlyargs
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.defaults
        or arguments.kw_defaults
    ):
        raise ValueError("candidate kernel must use vector_add(gm_a, gm_b, gm_c)")
    if len(kernel.decorator_list) != 1 or ast.dump(kernel.decorator_list[0]) != ast.dump(
        ast.Attribute(value=ast.Name(id="tla", ctx=ast.Load()), attr="kernel", ctx=ast.Load())
    ):
        raise ValueError("candidate vector_add must use exactly @tla.kernel")
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef)
    if any(not isinstance(node, allowed) for node in module.body):
        raise ValueError("candidate module may contain only imports, constants, and vector_add")
    for node in module.body:
        if isinstance(node, ast.Import):
            if [(alias.name, alias.asname) for alias in node.names] != [("catlass.tla", "tla")]:
                raise ValueError("candidate may import only catlass.tla as tla")
        elif isinstance(node, ast.ImportFrom):
            raise ValueError("candidate may import only catlass.tla as tla")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None or not isinstance(value, ast.Constant) or isinstance(value.value, (bool, str, bytes)):
                raise ValueError("candidate module constants must be numeric literals")


class CatlassCandidateBackend:
    """Turn workspace source into an immutable plan and execute it on BZ-A5."""

    def __init__(self, execution_backend: ExecutionBackend, *, length: int = 32,
                 seed: int = 0, device: int = 0):
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError("device must be a non-negative integer")
        self._execution_backend = execution_backend
        self._length = length
        self._seed = seed
        self._device = device

    def compile(self, workspace: Path, language: str,
                attempt_id: str) -> Mapping[str, object]:
        source = self._source(workspace, language)
        request = self._request()
        capture = _CaptureExecution(self._execution_backend)
        runner = A5KernelRunner(capture)
        plan = self._prepare(
            runner, request, source, attempt_id,
            compile_only=True, device=self._device
        )
        result = runner.run_plan(request, plan)
        kernel_name = _discovered_kernel(capture.receipt) if result.passed else None
        diagnostics = CompileDiagnostics(
            result.passed,
            result.exit_code,
            result.execution_id,
            result.source_fingerprint,
            result.attestation_sha256,
            kernel_name,
            _compile_diagnostics(capture.receipt, passed=result.passed),
        )
        return asdict(diagnostics)

    def run(self, workspace: Path, language: str, workload: Workload,
            attempt_id: str, ledger: EvidenceLedger) -> CandidateRun:
        if workload is not Workload.SMOKE_VECTOR_ADD:
            raise ValueError("Catlass candidate runtime supports only smoke-vector-add")
        source = self._source(workspace, language)
        request = self._request()
        capture = _CaptureExecution(self._execution_backend)
        runner = A5KernelRunner(capture, evidence_ledger=ledger)
        plan = self._prepare(
            runner, request, source, attempt_id, compile_only=False, device=self._device
        )
        verified = runner.run_plan(request, plan)
        kernel_name = _discovered_kernel(capture.receipt) if verified.passed else None
        return CandidateRun(
            plan, verified, kernel_name, workload,
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )

    def _request(self) -> RunRequest:
        return RunRequest(
            Language.CATLASS_DSL.value, length=self._length, seed=self._seed,
            padded_length=((self._length + 63) // 64) * 64,
        )

    @staticmethod
    def _source(workspace: Path, language: str) -> str:
        if language != Language.CATLASS_DSL.value:
            raise ValueError("Catlass candidate runtime requires catlass-dsl")
        source = (workspace / "kernel.py").read_bytes().decode("utf-8")
        validate_candidate_source(source)
        return source

    @staticmethod
    def _prepare(runner: A5KernelRunner, request: RunRequest, source: str,
                 attempt_id: str, *, compile_only: bool, device: int) -> ExecutionPlan:
        plan = runner.prepare(request, attempt_id=attempt_id)
        fixture = catlass_candidate_fixture(source)
        argv = fixture.argv
        assert argv is not None
        argv = ("env", f"BZ_A5_PROFILE_PHYSICAL_DEVICE={device}", *argv)
        if compile_only:
            argv = ("env", "A5KERNEL_COMPILE_ONLY=1", *argv)
        return replace(plan, files=fixture.files, argv=argv)


class _CaptureExecution:
    def __init__(self, backend: ExecutionBackend):
        self._backend = backend
        self.receipt: ExecutionReceipt | None = None

    @property
    def runtime_provenance(self):
        return tuple(getattr(self._backend, "runtime_provenance", ()))

    def execute(self, plan: ExecutionPlan) -> ExecutionReceipt:
        self.receipt = self._backend.execute(plan)
        return self.receipt


def _discovered_kernel(receipt: ExecutionReceipt | None) -> str:
    if receipt is None:
        raise RuntimeError("candidate execution returned no receipt")
    names = [line.removeprefix(_KERNEL_MARKER) for line in receipt.stdout.splitlines()
             if line.startswith(_KERNEL_MARKER)]
    if names != [EXACT_KERNEL_NAME]:
        raise RuntimeError("candidate execution did not report one exact kernel name")
    return names[0]


def _compile_diagnostics(receipt: ExecutionReceipt | None, *, passed: bool) -> str:
    if receipt is None:
        raise RuntimeError("candidate execution returned no receipt")
    if passed:
        return "Catlass compilation succeeded"
    detail = receipt.stderr.strip() or receipt.stdout.strip() or "Catlass compilation failed"
    return detail[-4000:]


class CandidateProfileEvaluation:
    """Bind treatment and final campaigns to the exact candidate execution."""

    def __init__(self, controller: ProfilingTreatmentController, *, device: int):
        self._controller = controller
        self._device = device

    def intermediate(self, run: CandidateRun) -> Mapping[str, object]:
        result = self._controller.run_intermediate(run.verified, self._request(run))
        return {"available": False} if result is None else asdict(result)

    def final(self, run: CandidateRun) -> Mapping[str, object]:
        return asdict(self._controller.run_final(run.verified, self._request(run)))

    def _request(self, run: CandidateRun) -> ProfileRequest:
        if run.kernel_name != EXACT_KERNEL_NAME:
            raise ValueError("candidate run did not report the host-discovered kernel")
        return ProfileRequest.from_execution_plan(
            run.plan,
            implementation="catlass-dsl",
            expected_kernel=run.kernel_name,
            device=self._device,
        )
