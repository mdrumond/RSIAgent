"""Host-authoritative execution contracts for agent-authored Catlass DSL."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
from typing import Mapping

from benchmarks.a5kernels.fixtures import catlass_source_fixture
from benchmarks.a5kernels.protocol import ExecutionPlan, canonical_hash


SCHEMA = "a5-catlass-harness-result-v1"
CATLASS_REVISION = "9a6ac627b5f4078060287844189730cf0d184800"


class HarnessContract(str, Enum):
    PADDED_SIMD = "padded-simd"
    MULTIBLOCK_SIMT = "multiblock-simt"
    CUBE_MATMUL = "cube-matmul"


@dataclass(frozen=True)
class ContractSpec:
    kernel_name: str
    arguments: tuple[str, ...]
    logical_shape: tuple[int, ...]
    physical_length: int
    block_count: int
    atol: float


SPECS = {
    HarnessContract.PADDED_SIMD: ContractSpec(
        "padded_add", ("gm_a", "gm_b", "gm_out"), (65,), 128, 1, 1e-5,
    ),
    HarnessContract.MULTIBLOCK_SIMT: ContractSpec(
        "indexed_add", ("gm_a", "gm_b", "gm_out"), (257,), 320, 4, 1e-5,
    ),
    HarnessContract.CUBE_MATMUL: ContractSpec(
        "matrix_multiply", ("gm_a", "gm_b", "gm_out"), (16, 16, 16), 256, 1, 2e-2,
    ),
}


@dataclass(frozen=True)
class HarnessResult:
    schema: str
    contract: str
    attempt_id: str
    source_sha256: str
    source_fingerprint: str
    catlass_revision: str
    execution_profile: str
    device: int
    session_handle: str | None
    host_verdict: str
    exit_code: int
    max_abs_error: float | None
    mismatch_index: int | None
    mismatch_expected: float | None
    mismatch_actual: float | None
    output_sha256: str
    retained_evidence_sha256: str
    record_sha256: str


def validate_contract_source(source: str, contract: HarnessContract | str) -> None:
    """Enforce the sole agent-controlled boundary: one exact kernel definition."""

    selected = HarnessContract(contract)
    spec = SPECS[selected]
    if not source.strip():
        raise ValueError("candidate source must not be empty")
    try:
        module = ast.parse(source, filename="kernel.py")
    except SyntaxError as exc:
        raise ValueError(f"candidate source is not valid Python: {exc.msg}") from exc
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError("candidate must define exactly one synchronous kernel")
    kernel = functions[0]
    args = kernel.args
    if (
        kernel.name != spec.kernel_name
        or tuple(item.arg for item in args.args) != spec.arguments
        or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
        or args.defaults or args.kw_defaults
    ):
        signature = f"{spec.kernel_name}({', '.join(spec.arguments)})"
        raise ValueError(f"candidate kernel must use {signature}")
    decorator = ast.Attribute(value=ast.Name(id="tla", ctx=ast.Load()), attr="kernel", ctx=ast.Load())
    if len(kernel.decorator_list) != 1 or ast.dump(kernel.decorator_list[0]) != ast.dump(decorator):
        raise ValueError("candidate kernel must use exactly @tla.kernel")
    allowed = (ast.Import, ast.Assign, ast.AnnAssign, ast.FunctionDef)
    if any(not isinstance(node, allowed) for node in module.body):
        raise ValueError("candidate module may contain only the tla import, numeric constants, and kernel")
    for node in module.body:
        if isinstance(node, ast.Import):
            if [(item.name, item.asname) for item in node.names] != [("catlass.tla", "tla")]:
                raise ValueError("candidate may import only catlass.tla as tla")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None or not isinstance(value, ast.Constant) or isinstance(value.value, (bool, str, bytes)):
                raise ValueError("candidate module constants must be numeric literals")
    calls = {_call_name(node.func) for node in ast.walk(kernel) if isinstance(node, ast.Call)}
    modes = {
        keyword.value.value
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call) and _call_name(node.func) == "tla.vec.func"
        for keyword in node.keywords
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant)
    }
    required_calls = {
        HarnessContract.PADDED_SIMD: {"tla.copy"},
        HarnessContract.MULTIBLOCK_SIMT: {
            "tla.arch.block_idx", "tla.arch.block_num", "tla.arch.thread_idx",
        },
        HarnessContract.CUBE_MATMUL: {"tla.copy", "tla.mmad"},
    }[selected]
    missing = sorted(required_calls - calls)
    required_mode = {
        HarnessContract.PADDED_SIMD: "simd",
        HarnessContract.MULTIBLOCK_SIMT: "simt",
        HarnessContract.CUBE_MATMUL: None,
    }[selected]
    has_cube = any(
        isinstance(node, ast.With)
        and any(_call_name(item.context_expr.func) == "tla.cube"
                for item in node.items if isinstance(item.context_expr, ast.Call))
        for node in ast.walk(kernel)
    )
    if missing or (required_mode is not None and required_mode not in modes) or (
        selected is HarnessContract.CUBE_MATMUL and not has_cube
    ):
        technique = "Cube movement and tla.mmad" if selected is HarnessContract.CUBE_MATMUL else f"{required_mode} vector execution"
        raise ValueError(f"candidate must implement the contract with {technique}")


class CatlassContractHarness:
    """Compose and grade a fixed contract through an existing BZ-A5 backend."""

    def __init__(self, backend, *, device: int = 0):
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError("device must be a non-negative integer")
        self._backend = backend
        self._device = device

    def preflight(self, source: str, contract: HarnessContract | str) -> Mapping[str, object]:
        selected = HarnessContract(contract)
        validate_contract_source(source, selected)
        provenance = self._provenance()
        return {
            "ready": True,
            "contract": selected.value,
            "source_sha256": _sha(source.encode()),
            "catlass_revision": provenance["catlass_revision"],
            "execution_profile": provenance["execution_profile"],
            "device": self._device,
        }

    def run(self, source: str, contract: HarnessContract | str, attempt_id: str) -> HarnessResult:
        selected = HarnessContract(contract)
        validate_contract_source(source, selected)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", attempt_id) is None:
            raise ValueError("attempt_id must be a safe 1-64 character identifier")
        spec = SPECS[selected]
        input_a, input_b, expected = _case(selected)
        fixture = catlass_source_fixture(source, _host_runtime(selected))
        self._provenance()
        provenance = tuple(self._backend.runtime_provenance)
        plan = ExecutionPlan(
            request_id=canonical_hash({
                "contract": selected.value,
                "logical_shape": spec.logical_shape,
                "physical_length": spec.physical_length,
            }),
            attempt_id=attempt_id,
            language="catlass-dsl",
            files=fixture.files,
            argv=fixture.argv,
            input_a=input_a,
            input_b=input_b,
            runtime_provenance=provenance,
            environment=(fixture.environment
                         .with_binding("A5KERNEL_BLOCK_NUM", str(spec.block_count))
                         .with_binding("BZ_A5_PROFILE_PHYSICAL_DEVICE", str(self._device))),
        )
        prepare = getattr(self._backend, "prepare_execution", None)
        if prepare is not None:
            plan = prepare(plan)
        receipt = self._backend.execute(plan)
        mismatch, max_error = _compare(receipt.output, expected, spec.atol)
        passed = receipt.exit_code == 0 and mismatch is None
        provenance_map = dict(provenance)
        output_sha = _sha(json.dumps(receipt.output, separators=(",", ":")).encode())
        evidence_sha = canonical_hash({
            "execution_id": plan.execution_id,
            "exit_code": receipt.exit_code,
            "stdout": receipt.stdout,
            "stderr": receipt.stderr,
            "session_handle": receipt.session_handle,
            "output_sha256": output_sha,
        })
        fields = {
            "schema": SCHEMA,
            "contract": selected.value,
            "attempt_id": attempt_id,
            "source_sha256": _sha(source.encode()),
            "source_fingerprint": plan.source_fingerprint,
            "catlass_revision": provenance_map["catlass_revision"],
            "execution_profile": provenance_map["execution_profile"],
            "device": self._device,
            "session_handle": receipt.session_handle,
            "host_verdict": "PASS" if passed else "FAIL",
            "exit_code": receipt.exit_code,
            "max_abs_error": max_error,
            "mismatch_index": None if mismatch is None else mismatch[0],
            "mismatch_expected": None if mismatch is None else mismatch[1],
            "mismatch_actual": None if mismatch is None else mismatch[2],
            "output_sha256": output_sha,
            "retained_evidence_sha256": evidence_sha,
        }
        return HarnessResult(**fields, record_sha256=canonical_hash(fields))

    def _provenance(self) -> Mapping[str, str]:
        provenance = dict(self._backend.runtime_provenance)
        if provenance.get("execution_profile") != "bz-a5":
            raise ValueError("harness requires bz-a5 runtime provenance")
        revision = provenance.get("catlass_revision", "")
        if revision != CATLASS_REVISION:
            raise ValueError(f"harness requires Catlass revision {CATLASS_REVISION}")
        return provenance


def write_result(result: HarnessResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


def report_results(root: Path) -> Mapping[str, object]:
    paths = [root] if root.is_file() else sorted(root.glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for record in records:
        digest = record.pop("record_sha256", None)
        if record.get("schema") != SCHEMA or digest != canonical_hash(record):
            raise ValueError("result record failed schema or digest validation")
        record["record_sha256"] = digest
    return {
        "schema": SCHEMA,
        "total": len(records),
        "passed": sum(record["host_verdict"] == "PASS" for record in records),
        "failed": sum(record["host_verdict"] == "FAIL" for record in records),
        "contracts": {contract.value: sum(record["contract"] == contract.value for record in records)
                      for contract in HarnessContract},
    }


def _case(contract: HarnessContract) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    spec = SPECS[contract]
    rng = random.Random(0xA5)
    if contract is HarnessContract.CUBE_MATMUL:
        a = tuple(_f16(rng.uniform(-1, 1)) for _ in range(256))
        b = tuple(_f16(rng.uniform(-1, 1)) for _ in range(256))
        expected = tuple(sum(a[row * 16 + k] * b[k * 16 + col] for k in range(16))
                         for row in range(16) for col in range(16))
        return a, b, expected
    logical = spec.logical_shape[0]
    a = tuple(rng.uniform(-1, 1) for _ in range(logical))
    b = (tuple(rng.uniform(-1, 1) for _ in range(logical))
         if contract is HarnessContract.PADDED_SIMD
         else tuple(index / 257.0 for index in range(logical)))
    padding = (0.0,) * (spec.physical_length - logical)
    return a + padding, b + padding, tuple(left + right for left, right in zip(a + padding, b + padding))


def _compare(actual: tuple[float, ...], expected: tuple[float, ...], atol: float):
    if len(actual) != len(expected):
        return (min(len(actual), len(expected)), float(len(expected)), float(len(actual))), None
    max_error = 0.0
    first = None
    for index, (got, want) in enumerate(zip(actual, expected)):
        error = abs(got - want) if math.isfinite(got) else math.inf
        max_error = max(max_error, error)
        if first is None and error > atol:
            first = (index, want, got)
    return first, max_error


def _host_runtime(contract: HarnessContract) -> str:
    spec = SPECS[contract]
    shape = spec.logical_shape
    matrix = contract is HarnessContract.CUBE_MATMUL
    dtype = "torch.float16" if matrix else "torch.float32"
    reshape = ".reshape(16, 16)" if matrix else ""
    output_dtype = "torch.float32" if matrix else dtype
    compile_args = "tla_a, tla_b, tla_out"
    return f'''\
def run(input_a, input_b):
    import os
    import torch
    import torch_npu
    from catlass.tla.runtime import from_dlpack

    device = int(os.environ["BZ_A5_PROFILE_PHYSICAL_DEVICE"])
    torch.npu.set_device(device)
    a = torch.tensor(input_a, dtype={dtype}, device="npu"){reshape}
    b = torch.tensor(input_b, dtype={dtype}, device="npu"){reshape}
    out = torch.full({shape[:2] if matrix else (spec.physical_length,)}, float("nan"), dtype={output_dtype}, device="npu")

    def as_tla(tensor):
        return from_dlpack(tensor.contiguous(), layout_tag=tla.arch.RowMajor)

    tla_a, tla_b, tla_out = (as_tla(item) for item in (a, b, out))
    artifact = tla.compile({spec.kernel_name}, {compile_args}, options="--npu-arch 3510")
    print("A5KERNEL_NAME={spec.kernel_name}__kernel0")
    artifact({compile_args}, block_num=int(os.environ["A5KERNEL_BLOCK_NUM"]))
    torch.npu.synchronize()
    return out.flatten().cpu().tolist()
'''


def _f16(value: float) -> float:
    return struct.unpack("e", struct.pack("e", value))[0]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _call_name(node: ast.expr) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))
