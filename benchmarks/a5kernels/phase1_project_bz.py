"""Checked BZ-A5 backend for generic Phase-1 Catlass projects."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from benchmarks.a5kernels.bz import BZCommandExecutor, CommandInvocation
from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.fixtures import _CATLASS_DRIVER
from benchmarks.a5kernels.phase1_project_executor import (
    CompileObservation,
    ProjectExecutionRequest,
    RuntimeObservation,
)
from benchmarks.a5kernels.protocol import SourceFile


PROJECT_RESULT_MARKER = "A5PROJECT_RESULT="
MAX_OBSERVED_BYTES = 4 * 1024 * 1024


class ProjectInfrastructureError(RuntimeError):
    """The checked transport, Catlass environment, or device was unavailable."""


class CatlassProjectBackend:
    """Compile and launch exact candidates through the checked BZ command seam."""

    def __init__(
        self,
        executor: BZCommandExecutor,
        *,
        session_wrapper: str,
        evidence_root: Path,
        device: int = 0,
        observe_timeout: int = 600,
    ) -> None:
        if Path(session_wrapper).name != "session.sh":
            raise ValueError("session_wrapper must identify the checked-in session.sh")
        if type(device) is not int or device < 0:
            raise ValueError("device must be a non-negative integer")
        if type(observe_timeout) is not int or observe_timeout < 1:
            raise ValueError("observe_timeout must be a positive integer")
        self._executor = executor
        self._wrapper = session_wrapper
        self._root = Path(evidence_root)
        self._device = device
        self._timeout = observe_timeout

    def compile(self, request: ProjectExecutionRequest) -> CompileObservation:
        identity, files = self._bundle(request)
        record, evidence = self._execute(identity, files, "compile")
        if record["status"] == "infrastructure-failure":
            raise ProjectInfrastructureError(record["diagnostic"])
        if record["status"] == "compile-failure":
            return CompileObservation(False, evidence, record["diagnostic"], None)
        if record["status"] != "compiled":
            raise ProjectInfrastructureError("compile returned an invalid host status")
        return CompileObservation(True, evidence, record["diagnostic"], identity)

    def run(
        self, request: ProjectExecutionRequest, compilation: CompileObservation
    ) -> RuntimeObservation:
        identity, files = self._bundle(request)
        if not compilation.passed or compilation.artifact_id != identity:
            raise ValueError("compilation does not match the immutable project bundle")
        record, evidence = self._execute(identity, files, "run")
        status = record["status"]
        if status in {"infrastructure-failure", "compile-failure"}:
            raise ProjectInfrastructureError(record["diagnostic"])
        if status == "runtime-failure":
            return RuntimeObservation(False, evidence, record["diagnostic"], None)
        if status != "completed":
            raise ProjectInfrastructureError("launch returned an invalid host status")
        output = record.get("output")
        expected = math.prod(request.expected.physical_shape)
        if not isinstance(output, list) or len(output) != expected or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in output
        ):
            raise ProjectInfrastructureError(
                "host output does not match the declared physical storage"
            )
        return RuntimeObservation(
            True, evidence, record["diagnostic"], tuple(float(value) for value in output)
        )

    def _bundle(
        self, request: ProjectExecutionRequest
    ) -> tuple[str, tuple[SourceFile, ...]]:
        if not isinstance(request, ProjectExecutionRequest):
            raise TypeError("request must be a ProjectExecutionRequest")
        manifest = {
            "schema": "a5-phase1-project-v1",
            "project_id": request.spec.project_id,
            "project_sha256": canonical_digest(request.spec.as_dict()),
            "family": request.family.value,
            "abi": request.abi.as_dict(),
            "dtype": request.spec.dtype.value,
            "block_count": request.spec.block_count,
            "inputs_sha256": request.inputs_sha256,
            "inputs": [
                {
                    "shape": list(item.shape),
                    "physical_shape": list(item.physical_shape),
                    "storage": list(item.storage),
                }
                for item in request.inputs
            ],
            "output": {
                "shape": list(request.expected.shape),
                "physical_shape": list(request.expected.physical_shape),
                "storage_elements": len(request.expected.storage),
            },
            "candidate_sha256": hashlib.sha256(
                request.candidate_source.encode("utf-8")
            ).hexdigest(),
            "runtime_provenance": [list(item) for item in self._provenance()],
        }
        manifest_text = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        files = (
            SourceFile("kernel.py", request.candidate_source),
            SourceFile("host_driver.py", _PROJECT_DRIVER),
            SourceFile("manifest.json", manifest_text),
        )
        identity = canonical_digest({
            "files": {item.relative_path: item.sha256 for item in files},
            "device": self._device,
        })
        directory = self._root / identity
        try:
            directory.mkdir(parents=True, exist_ok=True)
            for item in files:
                path = directory / item.relative_path
                if path.exists() and path.read_text(encoding="utf-8") != item.content:
                    raise ProjectInfrastructureError("retained project bundle conflicts")
                if not path.exists():
                    path.write_text(item.content, encoding="utf-8")
        except OSError as exc:
            raise ProjectInfrastructureError("project bundle could not be retained") from exc
        return identity, files

    def _provenance(self) -> tuple[tuple[str, str], ...]:
        try:
            return tuple(self._executor.runtime_provenance)
        except Exception as exc:
            raise ProjectInfrastructureError(
                "Catlass runtime provenance is unavailable"
            ) from exc

    def _execute(
        self, identity: str, files: tuple[SourceFile, ...], mode: str
    ) -> tuple[dict[str, object], str]:
        session = f"codex-a5project-{mode}-{identity[:24]}"
        remote = f".a5kernels/projects/{identity}"
        command = (
            "env", "-u", "PYTHONPYCACHEPREFIX",
            f"BZ_A5_PROFILE_PHYSICAL_DEVICE={self._device}",
            "python", "-B", f"{remote}/host_driver.py",
            f"{remote}/kernel.py", f"{remote}/manifest.json", mode,
        )
        invocation = CommandInvocation(
            argv=(
                self._wrapper, "--name", session, "run", "--wait",
                "--observe-timeout", str(self._timeout), "--", *command,
            ),
            files=files,
            stdin="",
            remote_directory=remote,
            physical_device=self._device,
        )
        try:
            dispatch = self._executor.run(invocation)
            logs = self._executor.inspect((self._wrapper, "--name", session, "logs"))
            result = self._executor.inspect((self._wrapper, "--name", session, "result"))
        except Exception as exc:
            raise ProjectInfrastructureError("BZ-A5 project operation unavailable") from exc
        if logs.exit_code != 0 or not _completed(result, session):
            raise ProjectInfrastructureError("BZ-A5 project result is unavailable")
        observed = "\n".join(
            item.rstrip("\n") for item in (dispatch.stdout, logs.stdout, dispatch.stderr,
                                             logs.stderr, result.stderr) if item
        )
        if len(observed.encode("utf-8")) > MAX_OBSERVED_BYTES:
            raise ProjectInfrastructureError("BZ-A5 project evidence exceeded its bound")
        records = [
            line.removeprefix(PROJECT_RESULT_MARKER)
            for line in logs.stdout.splitlines()
            if line.startswith(PROJECT_RESULT_MARKER)
        ]
        if len(records) != 1:
            raise ProjectInfrastructureError("host driver returned no unique result")
        try:
            record = json.loads(records[0])
        except json.JSONDecodeError as exc:
            raise ProjectInfrastructureError("host driver returned invalid JSON") from exc
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("status"), str)
            or not isinstance(record.get("diagnostic"), str)
            or not record["diagnostic"]
        ):
            raise ProjectInfrastructureError("host driver returned an invalid record")
        evidence = canonical_digest({
            "identity": identity,
            "mode": mode,
            "record": record,
            "observed": observed,
            "handle": dispatch.session_handle or f"bz-a5:{session}",
        })
        return record, evidence


def _completed(result, session: str) -> bool:
    prefix = f"SESSION_STATE=completed session={session} exit="
    values = [line.removeprefix(prefix).split()[0] for line in result.stdout.splitlines()
              if line.startswith(prefix)]
    return result.exit_code == 0 and values == ["0"]


_DRIVER_PREFIX, _SEPARATOR, _UNUSED = _CATLASS_DRIVER.partition("def _numbers")
if not _SEPARATOR:  # pragma: no cover
    raise RuntimeError("Catlass host runtime prefix is unavailable")

_PROJECT_DRIVER = _DRIVER_PREFIX + f'''\
import math

RESULT_MARKER = {PROJECT_RESULT_MARKER!r}

def _emit(status, diagnostic, output=None):
    record = {{"status": status, "diagnostic": diagnostic}}
    if output is not None:
        record["output"] = output
    print(RESULT_MARKER + json.dumps(record, separators=(",", ":"), allow_nan=False))

def _main():
    if len(sys.argv) != 10 or sys.argv[3] not in {{"compile", "run"}}:
        raise RuntimeError("invalid project host-driver invocation")
    manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    if manifest.get("schema") != "a5-phase1-project-v1":
        raise RuntimeError("invalid project manifest")
    source, native_location, native_sha256, generated = _preflight_runtime(*sys.argv[4:])
    import catlass.tla as tla
    import torch
    import torch_npu
    from catlass.tla.runtime import from_dlpack
    torch.npu.set_device(0)
    dtype = {{"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}}[manifest["dtype"]]
    tla_dtype = {{"float16": tla.Float16, "bfloat16": tla.BFloat16,
                 "float32": tla.Float32}}[manifest["dtype"]]
    tensors = [torch.tensor(item["storage"], dtype=dtype, device="npu").reshape(
               tuple(item["physical_shape"])) for item in manifest["inputs"]]
    output_shape = tuple(manifest["output"]["physical_shape"])
    output = torch.full(output_shape, -991.0, dtype=dtype, device="npu")
    try:
        kernel = _load_kernel(sys.argv[1])
        _verify_loaded_catlass_modules(
            source, sys.argv[4], native_location, native_sha256, generated,
        )
        kernel.LOGICAL_INPUT_SHAPES = tuple(tuple(item["shape"])
                                            for item in manifest["inputs"])
        kernel.PHYSICAL_INPUT_SHAPES = tuple(tuple(item["physical_shape"])
                                             for item in manifest["inputs"])
        kernel.LOGICAL_OUTPUT_SHAPE = tuple(manifest["output"]["shape"])
        kernel.PHYSICAL_OUTPUT_SHAPE = output_shape
        kernel.DTYPE = tla_dtype
        entry = getattr(kernel, manifest["abi"]["kernel_name"])
        as_tla = lambda tensor: from_dlpack(tensor.contiguous(),
                                            layout_tag=tla.arch.RowMajor)
        tla_inputs = [as_tla(item) for item in tensors]
        tla_output = as_tla(output)
        artifact = tla.compile(entry, *tla_inputs, tla_output,
                               options="--npu-arch 3510")
    except Exception as exc:
        _emit("compile-failure", f"{{type(exc).__name__}}: {{exc}}")
        return 0
    if sys.argv[3] == "compile":
        _emit("compiled", "candidate compiled")
        return 0
    try:
        artifact(*tla_inputs, tla_output, block_num=manifest["block_count"])
        torch.npu.synchronize()
        values = output.float().cpu().reshape(-1).tolist()
        if len(values) != manifest["output"]["storage_elements"] or not all(
            math.isfinite(value) for value in values
        ):
            raise RuntimeError("physical output is incomplete or non-finite")
    except Exception as exc:
        _emit("runtime-failure", f"{{type(exc).__name__}}: {{exc}}")
        return 0
    _emit("completed", "launch synchronized", values)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except SystemExit:
        raise
    except Exception as exc:
        _emit("infrastructure-failure", f"{{type(exc).__name__}}: {{exc}}")
'''


__all__ = [
    "CatlassProjectBackend", "PROJECT_RESULT_MARKER", "ProjectInfrastructureError",
]
