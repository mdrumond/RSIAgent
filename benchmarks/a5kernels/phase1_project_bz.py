"""Checked BZ-A5 backend for generic Phase-1 Catlass projects."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import tempfile

from benchmarks.a5kernels.bz import BZCommandExecutor, CommandInvocation, CommandResult
from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.fixtures import _CATLASS_DRIVER
from benchmarks.a5kernels.phase1_low_precision import MX_DTYPES, mx_quantize
from benchmarks.a5kernels.phase1_project_executor import (
    CompileObservation,
    ProjectExecutionRequest,
    RuntimeObservation,
)
from benchmarks.a5kernels.protocol import SourceFile
from benchmarks.a5kernels.protocol import ExecutionEnvironment, ExecutionPlan


PROJECT_RESULT_MARKER = "A5PROJECT_RESULT="
_BASE_EVIDENCE_BYTES = 1024 * 1024
_EVIDENCE_BYTES_PER_OUTPUT = 32
_SESSION_IDENTITY_CHARS = 12


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
        self._session_namespace = secrets.token_hex(8)

    def compile(self, request: ProjectExecutionRequest) -> CompileObservation:
        identity, files = self._bundle(request)
        record, evidence = self._execute(identity, files, "compile", 0)
        if record["status"] == "infrastructure-failure":
            raise ProjectInfrastructureError(record["diagnostic"])
        if record["status"] == "compile-failure":
            return CompileObservation(False, evidence, record["diagnostic"], None)
        if record["status"] != "compiled":
            raise ProjectInfrastructureError("compile returned an invalid host status")
        return CompileObservation(
            True,
            evidence,
            record["diagnostic"] + "; run performs a fresh exact-bundle compilation",
            f"bundle:{identity}",
        )

    def run(
        self, request: ProjectExecutionRequest, compilation: CompileObservation
    ) -> RuntimeObservation:
        identity, files = self._bundle(request)
        if not compilation.passed or compilation.artifact_id != f"bundle:{identity}":
            raise ValueError("compilation does not match the immutable project bundle")
        record, evidence = self._execute(
            identity, files, "run", math.prod(request.expected.physical_shape)
        )
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
        input_payloads = [
            {"shape": list(item.shape), "physical_shape": list(item.physical_shape),
             "storage": list(item.storage)} for item in request.inputs
        ]
        scales = []
        if request.spec.dtype.value in MX_DTYPES:
            for index, item in enumerate(input_payloads):
                item["storage"], scale, _ = mx_quantize(
                    item["shape"], item["storage"], request.spec.dtype.value, index)
                scales.append(scale)
        manifest = {
            "schema": "a5-phase1-project-v1",
            "project_id": request.spec.project_id,
            "project_sha256": canonical_digest(request.spec.as_dict()),
            "family": request.family.value,
            "abi": request.abi.as_dict(),
            "dtype": request.spec.dtype.value,
            "block_count": request.spec.block_count,
            "inputs_sha256": request.inputs_sha256,
            "inputs": input_payloads,
            "scales": scales,
            "output_dtype": request.expected.dtype,
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
            SourceFile("project_diagnostics.py", _diagnostic_driver()),
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

    def measurement_plan(self, request: ProjectExecutionRequest) -> ExecutionPlan:
        """Bind a replay to the exact bundle already staged by correctness execution."""
        identity, files = self._bundle(request)
        remote = f".a5kernels/projects/{identity}"
        return ExecutionPlan(
            identity, f"measure-{secrets.token_hex(8)}", "catlass-dsl", files,
            ("python", "-B", f"{remote}/host_driver.py", f"{remote}/kernel.py",
             f"{remote}/manifest.json", "run"), (), (), self._provenance(),
            ExecutionEnvironment(bindings=(("A5PROJECT_MASKED_TIMING", "1"),
                                           ("PYTHONDONTWRITEBYTECODE", "1")),
                                 unset=("PYTHONPYCACHEPREFIX",)),
        )

    def diagnostic(self, request: ProjectExecutionRequest, *, tool: str) -> dict[str, object]:
        """Run fixed diagnostic commands; their result never verifies the kernel."""
        if tool not in {"msdebug", "memcheck", "racecheck", "initcheck", "synccheck"}:
            raise ValueError("unsupported project diagnostic tool")
        identity, files = self._bundle(request)
        record, digest = self._execute(identity, files, "diagnostic-" + tool, 0)
        if (record.get("status") != "diagnostic-observation"
                or record.get("candidate_sha256") != hashlib.sha256(
                    request.candidate_source.encode("utf-8")).hexdigest()
                or type(record.get("available")) is not bool
                or record.get("host_verified") is not False
                or record.get("coverage") != "unconfirmed"):
            raise ProjectInfrastructureError("host driver returned invalid diagnostic evidence")
        return {"record": record, "evidence_sha256": digest,
                "artifact": self.observation_payload(digest)}

    def observation_payload(self, digest: str) -> dict[str, object]:
        path = self._root / "cas" / digest[:2] / f"{digest}.json"
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ProjectInfrastructureError("project observation hash mismatch")
        return json.loads(data)

    @property
    def device(self) -> int:
        return self._device

    def _provenance(self) -> tuple[tuple[str, str], ...]:
        try:
            return tuple(self._executor.runtime_provenance)
        except Exception as exc:
            raise ProjectInfrastructureError(
                "Catlass runtime provenance is unavailable"
            ) from exc

    def _execute(
        self,
        identity: str,
        files: tuple[SourceFile, ...],
        mode: str,
        output_elements: int,
    ) -> tuple[dict[str, object], str]:
        attempt = self._reserve_attempt(identity, mode)
        # The checked session wrapper allows at most 63 characters. Diagnostic
        # tool names already distinguish modes; omit their redundant prefix.
        session_mode = mode.removeprefix("diagnostic-")
        session = (
            f"codex-a5project-{session_mode}-{identity[:_SESSION_IDENTITY_CHARS]}-"
            f"{self._session_namespace}-{attempt:06d}"
        )
        remote = f".a5kernels/projects/{identity}"
        command = (
            "env", "-u", "PYTHONPYCACHEPREFIX",
            f"BZ_A5_PROFILE_PHYSICAL_DEVICE={self._device}",
            "PYTHONDONTWRITEBYTECODE=1",
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
            handle = f"bz-a5:{session}"
            if dispatch.session_handle not in {None, handle}:
                raise ProjectInfrastructureError(
                    "BZ-A5 project dispatch returned a mismatched retained handle"
                )
            if _observation_uncertain(dispatch, session):
                self._observe_terminal(handle, session)
            elif dispatch.exit_code != 0:
                raise ProjectInfrastructureError(
                    "BZ-A5 project dispatch failed before a current session was confirmed"
                )
            logs = self._executor.inspect((self._wrapper, "--name", session, "logs"))
            result = self._executor.inspect((self._wrapper, "--name", session, "result"))
            if _inspection_uncertain(logs, session) or _inspection_uncertain(result, session):
                self._observe_terminal(handle, session)
                logs = self._executor.inspect((self._wrapper, "--name", session, "logs"))
                result = self._executor.inspect((self._wrapper, "--name", session, "result"))
        except Exception as exc:
            if isinstance(exc, ProjectInfrastructureError):
                raise
            raise ProjectInfrastructureError("BZ-A5 project operation unavailable") from exc
        if logs.exit_code != 0 or not _completed(result, session):
            raise ProjectInfrastructureError("BZ-A5 project result is unavailable")
        observed = "\n".join(
            item.rstrip("\n") for item in (dispatch.stdout, logs.stdout, dispatch.stderr,
                                             logs.stderr, result.stderr) if item
        )
        maximum = _BASE_EVIDENCE_BYTES + output_elements * _EVIDENCE_BYTES_PER_OUTPUT
        if len(observed.encode("utf-8")) > maximum:
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
        retained = {
            "schema": "a5-phase1-project-observation-v1",
            "identity": identity,
            "mode": mode,
            "attempt": attempt,
            "record": record,
            "handle": handle,
            "command_results": {
                "dispatch": _command_result_payload(dispatch),
                "logs": _command_result_payload(logs),
                "result": _command_result_payload(result),
            },
        }
        evidence = self._retain_evidence(retained)
        return record, evidence

    def _reserve_attempt(self, identity: str, mode: str) -> int:
        """Reserve a fresh identity after a confirmed operation is retried."""

        attempts = self._root / identity / "attempts"
        try:
            attempts.mkdir(parents=True, exist_ok=True)
            for ordinal in range(1, 1_000_000):
                try:
                    (attempts / f"{mode}-{ordinal:06d}").mkdir()
                except FileExistsError:
                    continue
                return ordinal
        except OSError as exc:
            raise ProjectInfrastructureError("project attempt could not be retained") from exc
        raise ProjectInfrastructureError("project attempt space is exhausted")

    def _observe_terminal(self, handle: str, session: str) -> None:
        observe = getattr(self._executor, "observe", None)
        if observe is None:
            raise ProjectInfrastructureError(
                "BZ-A5 project retained observation is unavailable"
            )
        observed = observe(handle)
        if not _terminal_observation(observed, session):
            raise ProjectInfrastructureError(
                "BZ-A5 project retained observation returned no terminal state"
            )

    def _retain_evidence(self, payload: dict[str, object]) -> str:
        """Store reconstructible raw command evidence in a local content-addressed store."""

        content = canonical_bytes(payload)
        digest = hashlib.sha256(content).hexdigest()
        directory = self._root / "cas" / digest[:2]
        destination = directory / f"{digest}.json"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.read_bytes() != content:
                    raise ProjectInfrastructureError("retained evidence CAS conflicts")
                return digest
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except OSError as exc:
            raise ProjectInfrastructureError("project evidence could not be retained") from exc
        return digest


def _completed(result, session: str) -> bool:
    prefix = f"SESSION_STATE=completed session={session} exit="
    values = [line.removeprefix(prefix).split()[0] for line in result.stdout.splitlines()
              if line.startswith(prefix)]
    return result.exit_code == 0 and values == ["0"]


def _observation_uncertain(result: CommandResult, session: str) -> bool:
    lines = result.stdout.splitlines()
    return (
        "CATLASS_VALIDATION_STATE=observation-unavailable" in lines
        and f"CATLASS_VALIDATION_HANDLE=bz-a5:{session}" in lines
    )


def _inspection_uncertain(result: CommandResult, session: str) -> bool:
    if result.exit_code in {75, 255}:
        return True
    states = {"running", "reconnecting", "observation-unavailable", "missing"}
    return any(
        line in {f"REMOTE_STATE={state}" for state in states}
        or any(
            line == f"SESSION_STATE={state} session={session}"
            or line.startswith(f"SESSION_STATE={state} session={session} ")
            for state in states
        )
        for line in result.stdout.splitlines()
    )


def _terminal_observation(result: CommandResult, session: str) -> bool:
    lines = result.stdout.splitlines()
    return (
        f"CATLASS_VALIDATION_HANDLE=bz-a5:{session}" in lines
        and any(f"CATLASS_VALIDATION_STATE={state}" in lines
                for state in ("completed", "failed"))
    )


def _command_result_payload(result: CommandResult) -> dict[str, object]:
    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "session_handle": result.session_handle,
    }


_DRIVER_PREFIX, _SEPARATOR, _UNUSED = _CATLASS_DRIVER.partition("def _numbers")
if not _SEPARATOR:  # pragma: no cover
    raise RuntimeError("Catlass host runtime prefix is unavailable")

_PROJECT_DRIVER = _DRIVER_PREFIX + f'''\
import math
import os
import time

RESULT_MARKER = {PROJECT_RESULT_MARKER!r}

def _emit(status, diagnostic, output=None):
    record = {{"status": status, "diagnostic": diagnostic}}
    if output is not None:
        record["output"] = output
    print(RESULT_MARKER + json.dumps(record, separators=(",", ":"), allow_nan=False))

def _main():
    if len(sys.argv) != 10 or (sys.argv[3] not in {{"compile", "run"}}
                              and not sys.argv[3].startswith("diagnostic-")):
        raise RuntimeError("invalid project host-driver invocation")
    manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    if manifest.get("schema") != "a5-phase1-project-v1":
        raise RuntimeError("invalid project manifest")
    if sys.argv[3].startswith("diagnostic-"):
        from project_diagnostics import diagnose
        record = diagnose(sys.argv)
        print(RESULT_MARKER + json.dumps(record, separators=(",", ":"), allow_nan=False))
        return 0
    source, native_location, native_sha256, generated = _preflight_runtime(*sys.argv[4:])
    import catlass.tla as tla
    import torch
    import torch_npu
    from catlass.tla.runtime import from_dlpack
    device = int(os.environ["BZ_A5_PROFILE_PHYSICAL_DEVICE"]) if os.environ.get(
        "A5KERNEL_PROFILE_DIRECT_PHYSICAL_DEVICE") == "1" else 0
    torch.npu.set_device(device)
    storage_name = manifest["dtype"].replace("mxfp8_", "float8_")
    dtype = getattr(torch, storage_name)
    tla_names = {{"float16": "Float16", "bfloat16": "BFloat16", "float32": "Float32",
                 "float8_e4m3fn": "Float8E4M3FN", "float8_e5m2": "Float8E5M2"}}
    tla_dtype = getattr(tla, tla_names[storage_name])
    output_name = manifest.get("output_dtype", storage_name)
    output_dtype = getattr(torch, output_name)
    low_precision = storage_name.startswith("float8_")
    mx = manifest["dtype"].startswith("mxfp8_")
    # CPU casts preserve byte-exact FP8 and avoid framework Cast kernels in traces.
    tensors = [torch.tensor(item["storage"], dtype=torch.float32).to(dtype).reshape(
               tuple(item["physical_shape"])) for item in manifest["inputs"]]
    if low_precision:
        tensors = [item.view(torch.int8) for item in tensors]
    if mx:
        tensors[1] = tensors[1].transpose(0, 1).contiguous()
    tensors = [item.to("npu") for item in tensors]
    output_shape = tuple(manifest["output"]["physical_shape"])
    output = torch.full(output_shape, -7.0 if low_precision else -991.0,
                        dtype=torch.float32).to(output_dtype)
    if output_name.startswith("float8_"):
        output = output.view(torch.int8)
    output = output.to("npu")
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
        kernel.OUTPUT_DTYPE = getattr(tla, tla_names[output_name])
        entry = getattr(kernel, manifest["abi"]["kernel_name"])
        as_tla = lambda tensor: from_dlpack(tensor.contiguous(),
                                            layout_tag=tla.arch.RowMajor)
        if low_precision:
            tla_inputs = [from_dlpack(item, layout_tag=(tla.arch.ColumnMajor if mx and index == 1
                          else tla.arch.RowMajor), origin_shape=tuple(contract["physical_shape"]),
                          element_type=tla_dtype)
                          for index, (item, contract) in enumerate(zip(tensors, manifest["inputs"]))]
        else:
            tla_inputs = [as_tla(item) for item in tensors]
        if mx:
            for index, scale in enumerate(manifest["scales"]):
                scale_tensor = torch.tensor(scale["storage"], dtype=torch.uint8).reshape(
                    tuple(scale["shape"])).view(torch.int8).to("npu")
                shape = tuple(scale["shape"]) if index == 0 else tuple(reversed(scale["shape"]))
                tag = tla.arch.RowMajorMxScaleA if index == 0 else tla.arch.ColMajorMxScaleB
                tla_inputs.append(from_dlpack(scale_tensor, layout_tag=tag,
                                  origin_shape=shape, element_type=tla.Float8E8M0).mark_layout_dynamic())
        tla_output = (from_dlpack(output, layout_tag=tla.arch.RowMajor,
                                 element_type=tla_dtype) if output_name.startswith("float8_")
                      else as_tla(output))
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
        host_output = output.cpu()
        if output_name.startswith("float8_"):
            host_output = host_output.view(output_dtype)
        values = host_output.float().reshape(-1).tolist()
        if len(values) != manifest["output"]["storage_elements"] or not all(
            math.isfinite(value) for value in values
        ):
            raise RuntimeError("physical output is incomplete or non-finite")
    except Exception as exc:
        _emit("runtime-failure", f"{{type(exc).__name__}}: {{exc}}")
        return 0
    if os.environ.get("A5KERNEL_EMIT_TIMING") == "1":
        warmups = int(os.environ["A5KERNEL_WARM_UP"])
        launches = int(os.environ["A5KERNEL_LAUNCH_COUNT"])
        for _ in range(warmups):
            artifact(*tla_inputs, tla_output, block_num=manifest["block_count"])
            torch.npu.synchronize()
        start = time.perf_counter_ns()
        for _ in range(launches):
            artifact(*tla_inputs, tla_output, block_num=manifest["block_count"])
            torch.npu.synchronize()
        duration_us = (time.perf_counter_ns() - start) / launches / 1000
        print("A5KERNEL_TIMING_US=" + str(duration_us))
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


def _diagnostic_driver() -> str:
    return Path(__file__).with_name("phase1_project_diagnostic_driver.py").read_text(encoding="utf-8")


__all__ = [
    "CatlassProjectBackend", "PROJECT_RESULT_MARKER", "ProjectInfrastructureError",
]
