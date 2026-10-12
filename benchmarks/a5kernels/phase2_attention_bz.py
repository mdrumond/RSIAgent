"""Checked BZ-A5 execution backend for the Phase 2 attention target.

Candidate modules own one TLA kernel and nothing else.  Case constants, tensors,
launch policy, retained-session observation, and output parsing remain host-owned.
"""

from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from benchmarks.a5kernels.bz import BZCommandExecutor, CommandInvocation
from benchmarks.a5kernels.fixtures import _CATLASS_DRIVER
from benchmarks.a5kernels.phase2_attention import (
    AttentionExecution,
    AttentionInfrastructureError,
    AttentionReceipt,
)
from benchmarks.a5kernels.protocol import SourceFile


OUTPUT_MARKER = "A5ATTENTION_OUTPUT="
KERNEL_NAME = "dense_attention"
MAX_OBSERVED_BYTES = 4 * 1024 * 1024
_ARGUMENTS = ("gm_q", "gm_k", "gm_v", "gm_output")


# This transport-only fixture demonstrates the exact ABI with already-proven TLA
# tensor/copy constructs.  It merely copies V: it neither implements attention
# nor establishes compiler, runtime, numerical, or performance validity.
ATTENTION_CANDIDATE_FIXTURE = """\
import catlass.tla as tla

@tla.kernel
def dense_attention(gm_q: tla.Tensor, gm_k: tla.Tensor,
                    gm_v: tla.Tensor, gm_output: tla.Tensor) -> None:
    elements = BATCH * HEADS * SEQUENCE * HEAD_DIM
    padded = ((elements + 127) // 128) * 128
    local_ptr = tla.allocate(padded, tla.Float16, tla.AddressSpace.ub, 256)
    local_v = tla.make_tensor_like(local_ptr, gm_v, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(local_v, gm_v)
        tla.copy(gm_output, local_v)
        tla.pipe_barrier(tla.pipes.ALL)
"""


def validate_attention_candidate(source: str) -> None:
    """Validate the narrow source/export ABI before any remote dispatch."""

    if not source.strip():
        raise ValueError("candidate source must not be empty")
    try:
        module = ast.parse(source, filename="kernel.py")
    except SyntaxError as exc:
        raise ValueError(f"candidate source is not valid Python: {exc.msg}") from exc
    functions = [node for node in module.body if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef)
    )]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise ValueError("candidate must define exactly one synchronous kernel")
    function = functions[0]
    arguments = function.args
    if (
        function.name != KERNEL_NAME
        or tuple(argument.arg for argument in arguments.args) != _ARGUMENTS
        or arguments.posonlyargs
        or arguments.kwonlyargs
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.defaults
        or arguments.kw_defaults
    ):
        raise ValueError(
            "candidate kernel must use dense_attention(gm_q, gm_k, gm_v, gm_output)"
        )
    decorator = ast.Attribute(
        value=ast.Name(id="tla", ctx=ast.Load()), attr="kernel", ctx=ast.Load()
    )
    if len(function.decorator_list) != 1 or ast.dump(
        function.decorator_list[0]
    ) != ast.dump(decorator):
        raise ValueError("candidate dense_attention must use exactly @tla.kernel")
    if any(not isinstance(node, (ast.Import, ast.FunctionDef)) for node in module.body):
        raise ValueError("candidate module may contain only its import and kernel")
    imports = [node for node in module.body if isinstance(node, ast.Import)]
    if len(imports) != 1 or [
        (alias.name, alias.asname) for alias in imports[0].names
    ] != [("catlass.tla", "tla")]:
        raise ValueError("candidate may import only catlass.tla as tla")


class CatlassAttentionBackend:
    """Stage and observe one dense-attention case through checked BZ wrappers."""

    def __init__(
        self,
        executor: BZCommandExecutor,
        *,
        session_wrapper: str,
        device: int = 0,
        observe_timeout: int = 600,
    ) -> None:
        if Path(session_wrapper).name != "session.sh":
            raise ValueError("session_wrapper must identify the checked-in session.sh")
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError("device must be a non-negative integer")
        if type(observe_timeout) is not int or observe_timeout < 1:
            raise ValueError("observe_timeout must be a positive integer")
        self._executor = executor
        self._wrapper = session_wrapper
        self._device = device
        self._observe_timeout = observe_timeout

    def execute(
        self,
        candidate_source: str,
        execution: AttentionExecution,
        workdir: Path,
    ) -> AttentionReceipt:
        validate_attention_candidate(candidate_source)
        if not isinstance(execution, AttentionExecution):
            raise TypeError("execution must be an AttentionExecution")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        if not workdir.is_dir():
            raise ValueError("workdir must be a directory")

        payload = _input_payload(execution)
        payload_text = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        composed_source = _case_constants(execution) + candidate_source
        files = (
            SourceFile("kernel.py", composed_source),
            SourceFile("host_driver.py", _ATTENTION_DRIVER),
            SourceFile("input.json", payload_text),
        )
        for source in files:
            destination = workdir / source.relative_path
            destination.write_text(source.content, encoding="utf-8")

        identity = hashlib.sha256(json.dumps(
            {
                "candidate": hashlib.sha256(candidate_source.encode()).hexdigest(),
                "case": asdict(execution.case),
                "physical_device": self._device,
                "input": hashlib.sha256(payload_text.encode()).hexdigest(),
            }, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        session_name = f"codex-a5attn-{identity[:32]}"
        remote_directory = f".a5kernels/attention/{identity}"
        remote_argv = (
            "env",
            "-u", "PYTHONPYCACHEPREFIX",
            f"BZ_A5_PROFILE_PHYSICAL_DEVICE={self._device}",
            "A5KERNEL_BLOCK_NUM=1",
            "python", "-B",
            f"{remote_directory}/host_driver.py",
            f"{remote_directory}/kernel.py",
            f"{remote_directory}/input.json",
        )
        invocation = CommandInvocation(
            argv=(
                self._wrapper, "--name", session_name, "run", "--wait",
                "--observe-timeout", str(self._observe_timeout), "--", *remote_argv,
            ),
            files=files,
            stdin="",
            remote_directory=remote_directory,
            physical_device=self._device,
        )
        try:
            dispatch = self._executor.run(invocation)
        except Exception as exc:
            raise AttentionInfrastructureError("A5 attention dispatch failed") from exc

        uncertain = _observation_uncertain(
            dispatch.exit_code, dispatch.stdout, session_name
        )
        confirmed = _dispatch_confirmed(
            dispatch.exit_code, dispatch.stdout, session_name
        )
        if dispatch.exit_code != 0 and not (uncertain or confirmed):
            raise AttentionInfrastructureError("A5 attention dispatch was unavailable")

        try:
            logs = self._executor.inspect(
                (self._wrapper, "--name", session_name, "logs")
            )
            result = self._executor.inspect(
                (self._wrapper, "--name", session_name, "result")
            )
        except Exception as exc:
            raise AttentionInfrastructureError(
                "A5 attention retained operation could not be observed"
            ) from exc
        if logs.exit_code != 0:
            raise AttentionInfrastructureError("A5 attention retained logs are unavailable")
        if _observation_uncertain(result.exit_code, result.stdout, session_name):
            raise AttentionInfrastructureError("A5 attention result is indeterminate")
        if not _completed_result(result.exit_code, result.stdout, session_name):
            raise AttentionInfrastructureError("A5 attention terminal result is unavailable")

        stdout = _bounded_join(dispatch.stdout, logs.stdout)
        stderr = _bounded_join(dispatch.stderr, logs.stderr, result.stderr)
        output = (
            _parse_output(logs.stdout, execution.case.output_elements)
            if result.exit_code == 0 else ()
        )
        handle = (
            result.session_handle or logs.session_handle or dispatch.session_handle
            or f"bz-a5:{session_name}"
        )
        return AttentionReceipt(
            exit_code=result.exit_code,
            output=output,
            stdout=stdout,
            stderr=stderr,
            session_handle=handle,
            metadata=(
                ("case_id", execution.case.case_id),
                ("dispatch_exit_code", str(dispatch.exit_code)),
                ("logs_exit_code", str(logs.exit_code)),
                ("result_exit_code", str(result.exit_code)),
                ("remote_directory", remote_directory),
            ),
        )


def _input_payload(execution: AttentionExecution) -> dict[str, object]:
    case = execution.case
    return {
        "case": asdict(case),
        "q": list(execution.q),
        "k": list(execution.k),
        "v": list(execution.v),
        "scale": execution.scale,
    }


def _case_constants(execution: AttentionExecution) -> str:
    case = execution.case
    values = {
        "BATCH": case.batch,
        "HEADS": case.heads,
        "SEQUENCE": case.sequence,
        "HEAD_DIM": case.head_dim,
        "SCALE": execution.scale,
        "CAUSAL": case.causal,
        "DTYPE": case.dtype,
    }
    return "\n".join(f"{name} = {value!r}" for name, value in values.items()) + "\n"


def _parse_output(stdout: str, expected_elements: int) -> tuple[float, ...]:
    _bounded(stdout)
    records = [
        line.removeprefix(OUTPUT_MARKER)
        for line in stdout.splitlines() if line.startswith(OUTPUT_MARKER)
    ]
    if len(records) != 1:
        raise ValueError("remote output must contain exactly one A5ATTENTION_OUTPUT record")
    try:
        decoded = json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise ValueError("A5ATTENTION_OUTPUT must be valid JSON") from exc
    if not isinstance(decoded, list) or len(decoded) > expected_elements or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) for value in decoded
    ):
        raise ValueError("A5ATTENTION_OUTPUT must be a bounded finite numeric array")
    return tuple(float(value) for value in decoded)


def _bounded(value: str) -> str:
    if len(value.encode("utf-8")) > MAX_OBSERVED_BYTES:
        raise AttentionInfrastructureError("A5 attention observed output exceeded its bound")
    return value


def _bounded_join(*values: str) -> str:
    return _bounded("\n".join(value.rstrip("\n") for value in values if value))


def _observation_uncertain(exit_code: int, stdout: str, session_name: str) -> bool:
    return (
        exit_code == 75
        and "CATLASS_VALIDATION_STATE=observation-unavailable" in stdout.splitlines()
        and f"CATLASS_VALIDATION_HANDLE=bz-a5:{session_name}" in stdout.splitlines()
    )


def _dispatch_confirmed(exit_code: int, stdout: str, session_name: str) -> bool:
    prefix = f"SESSION_DISPATCH=confirmed session={session_name}"
    legacy = any(line == prefix or line.startswith(prefix + " ")
                 for line in stdout.splitlines())
    return legacy or _completed_result(exit_code, stdout, session_name)


def _completed_result(exit_code: int, stdout: str, session_name: str) -> bool:
    prefix = f"SESSION_STATE=completed session={session_name} exit="
    values = [line.removeprefix(prefix) for line in stdout.splitlines()
              if line.startswith(prefix)]
    return (
        len(values) == 1
        and values[0].isascii()
        and values[0].isdigit()
        and str(int(values[0])) == values[0]
        and int(values[0]) == exit_code
    )


_DRIVER_PREFIX, _SEPARATOR, _UNUSED = _CATLASS_DRIVER.partition("def _numbers")
if not _SEPARATOR:  # pragma: no cover - guards an internal fixture integration seam
    raise RuntimeError("Catlass host runtime prefix is unavailable")

_ATTENTION_DRIVER = _DRIVER_PREFIX + f'''\
import math

def _numbers(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{{name}} must be a non-empty JSON list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{{name}} must contain only numbers")
    return [float(item) for item in value]

def main():
    if len(sys.argv) != 9:
        raise SystemExit("usage: host_driver.py KERNEL.py INPUT.json REVISION MANIFEST_SHA BRIDGE_SHA GITLINK INSTALL_COMMIT CANN")
    payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    required = {{"case", "q", "k", "v", "scale"}}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("attention input fields do not match the host contract")
    case = payload["case"]
    shape = (case["batch"], case["heads"], case["sequence"], case["head_dim"])
    count = math.prod(shape)
    q, k, v = (_numbers(payload[name], name) for name in ("q", "k", "v"))
    if any(len(tensor) != count for tensor in (q, k, v)):
        raise ValueError("Q, K, and V must have exact [B,H,S,D] shape")
    source, native_location, native_sha256 = _preflight_runtime(*sys.argv[3:])
    kernel = _load_kernel(sys.argv[1])
    _verify_loaded_catlass_modules(source, sys.argv[3], native_location, native_sha256)
    import catlass.tla as tla
    import torch
    import torch_npu
    from catlass.tla.runtime import from_dlpack
    dtype = {{"float16": torch.float16, "bfloat16": torch.bfloat16}}[case["dtype"]]
    torch.npu.set_device(0)
    tensors = [torch.tensor(item, dtype=dtype, device="npu").reshape(shape)
               for item in (q, k, v)]
    output = torch.empty(shape, dtype=dtype, device="npu")
    as_tla = lambda tensor: from_dlpack(tensor.contiguous().reshape(-1), layout_tag=tla.arch.RowMajor)
    tla_q, tla_k, tla_v, tla_output = [as_tla(item) for item in (*tensors, output)]
    artifact = tla.compile(kernel.dense_attention, tla_q, tla_k, tla_v, tla_output,
                           options="--npu-arch 3510")
    print("A5KERNEL_NAME=dense_attention")
    artifact(tla_q, tla_k, tla_v, tla_output, block_num=1)
    torch.npu.synchronize()
    values = output.float().cpu().reshape(-1).tolist()
    print("{OUTPUT_MARKER}" + json.dumps(values, separators=(",", ":"), allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''
