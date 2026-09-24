"""Host-owned hello/vector-add fixtures for supported A5 languages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from benchmarks.a5kernels.protocol import SourceFile


class Language(str, Enum):
    CATLASS_DSL = "catlass-dsl"
    ASCEND_C = "ascend-c"
    TRITON_ASCEND = "triton-ascend"


@dataclass(frozen=True)
class Fixture:
    language: Language
    files: tuple[SourceFile, ...]
    # Concrete runtime drivers are supplied by language-specific integrations.
    # A missing argv is an explicit, fail-closed support boundary.
    argv: tuple[str, ...] | None = None
    max_length: int | None = None


_CATLASS_SOURCE = '''\
import catlass.tla as tla

VECTOR_ELE = 400
PADDED_VECTOR_ELE = 448
VL_ELE = 64

@tla.kernel
def vector_add(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    n_ele = gm_a.origin_shape[0]
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    ptr_a = tla.allocate(PADDED_VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ptr_b = tla.allocate(PADDED_VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ptr_c = tla.allocate(PADDED_VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_a = tla.make_tensor_like(ptr_a, gm_a, tla.arch.RowMajor)
    ub_b = tla.make_tensor_like(ptr_b, gm_b, tla.arch.RowMajor)
    ub_c = tla.make_tensor_like(ptr_c, gm_c, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(ub_a, gm_a)
        tla.copy(ub_b, gm_b)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                tile_a = tla.tile_view(ub_a, tla.make_shape(VL_ELE), tla.make_coord(i))
                tile_b = tla.tile_view(ub_b, tla.make_shape(VL_ELE), tla.make_coord(i))
                tile_c = tla.tile_view(ub_c, tla.make_shape(VL_ELE), tla.make_coord(i))
                tile_c.store(tla.add(tile_a.load(), tile_b.load()))
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_c, ub_c)
        tla.pipe_barrier(tla.pipes.ALL)


def run(input_a, input_b):
    import torch
    import torch_npu
    from catlass.tla.runtime import from_dlpack

    if not 0 < len(input_a) <= VECTOR_ELE or len(input_a) != len(input_b):
        raise ValueError(f"vector length must be in [1, {VECTOR_ELE}]")
    original_length = len(input_a)
    padded_length = ((original_length + VL_ELE - 1) // VL_ELE) * VL_ELE
    input_a = [*input_a, *([0.0] * (padded_length - original_length))]
    input_b = [*input_b, *([0.0] * (padded_length - original_length))]
    torch.npu.set_device(0)
    a = torch.tensor(input_a, dtype=torch.float32, device="npu")
    b = torch.tensor(input_b, dtype=torch.float32, device="npu")
    out = torch.empty_like(a)

    def as_tla(tensor):
        return from_dlpack(
            tensor.contiguous(), layout_tag=tla.arch.RowMajor
        ).mark_compact_shape_dynamic(0)

    tla_a, tla_b, tla_out = (as_tla(item) for item in (a, b, out))
    artifact = tla.compile(
        vector_add, tla_a, tla_b, tla_out, options="--npu-arch 3510"
    )
    artifact(tla_a, tla_b, tla_out, block_num=1)
    torch.npu.synchronize()
    return out[:original_length].cpu().tolist()
'''

_CATLASS_DRIVER = '''\
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUTPUT_MARKER = "A5KERNEL_OUTPUT="
REQUIRED_TLA_API = (
    "AddressSpace",
    "allocate",
    "compile",
    "flag",
    "kernel",
    "vector",
)
NATIVE_MANIFEST = "python/tla_dsl/csrc/mlir/build/.codex-native-provenance.json"
NATIVE_MODULE = "catlass._tla_type_bridge_native"


def _provenance_value(path: Path, names: tuple[str, ...]) -> str:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in names:
            values[key] = value
    for name in names:
        if values.get(name):
            return values[name]
    return ""


def _native_artifact(source: Path, expected_revision: str) -> tuple[Path, str]:
    manifest_path = source / NATIVE_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Catlass native provenance is unavailable: {manifest_path}") from exc
    expected_keys = {
        "schema",
        "catlass_revision",
        "ascendnpu_ir_gitlink",
        "ascendnpu_ir_install_commit",
        "cann_version",
        "artifacts",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise RuntimeError("Catlass native provenance manifest has an invalid schema")
    artifacts = manifest["artifacts"]
    if (
        manifest["schema"] != "catlass-native-build-v1"
        or manifest["catlass_revision"] != expected_revision
        or not isinstance(artifacts, list)
        or len(artifacts) != 1
        or not isinstance(artifacts[0], dict)
        or set(artifacts[0]) != {"module", "path", "sha256"}
        or artifacts[0]["module"] != NATIVE_MODULE
    ):
        raise RuntimeError("Catlass native provenance manifest has an invalid schema")
    gitlink_result = subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "ls-tree",
            expected_revision,
            "python/tla_dsl/3rdparty/AscendNPU-IR",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    fields = gitlink_result.stdout.split()
    gitlink = fields[2] if gitlink_result.returncode == 0 and len(fields) >= 3 else ""
    dep_root_value = os.environ.get("CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR")
    if not dep_root_value:
        raise RuntimeError("Catlass dependency provenance root is unavailable")
    dep_provenance = Path(dep_root_value) / ".cpl-build-provenance"
    try:
        install_commit = _provenance_value(
            dep_provenance, ("CAT_DEP_ASCENDNPU_IR_COMMIT",)
        )
        cann_version = _provenance_value(
            dep_provenance, ("ASCEND_CANN_VERSION", "CANN_VERSION")
        )
    except OSError as exc:
        raise RuntimeError(f"Catlass dependency provenance is unavailable: {dep_provenance}") from exc
    if (
        not gitlink
        or manifest["ascendnpu_ir_gitlink"] != gitlink
        or manifest["ascendnpu_ir_install_commit"] != install_commit
        or manifest["cann_version"] != cann_version
        or install_commit != gitlink
        or not cann_version
    ):
        raise RuntimeError("Catlass native provenance does not match the configured runtime")
    artifact = artifacts[0]
    relative_value = artifact["path"]
    digest = artifact["sha256"]
    if not isinstance(relative_value, str) or not isinstance(digest, str):
        raise RuntimeError("Catlass native provenance artifact is invalid")
    relative = Path(relative_value)
    required_parent = Path("python/tla_dsl/csrc/mlir/build/python/catlass")
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parent != required_parent
        or not relative.name.startswith("_tla_type_bridge_native")
        or not relative.name.endswith(".so")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("Catlass native provenance artifact is invalid")
    location = (source / relative).resolve()
    try:
        location.relative_to(source)
    except ValueError as exc:
        raise RuntimeError("Catlass native provenance artifact escapes the retained source") from exc
    return location, digest


def _preflight_runtime(expected_revision: str) -> None:
    source_value = os.environ.get("CATLASS_SRC")
    if not source_value or not Path(source_value).is_absolute():
        raise RuntimeError("CATLASS_SRC must name the explicit retained Catlass source")
    source = Path(source_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    actual_revision = revision.stdout.strip()
    if revision.returncode != 0 or actual_revision != expected_revision:
        actual = actual_revision or "<unavailable>"
        raise RuntimeError(
            "Catlass source revision mismatch: "
            f"expected {expected_revision}, found {actual} at {source}"
        )
    worktree = subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignore-submodules=all",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    changes = worktree.stdout.strip()
    if worktree.returncode != 0 or changes:
        detail = changes or worktree.stderr.strip() or "status unavailable"
        raise RuntimeError(
            f"Catlass retained source is not clean at {source}: {detail}"
        )

    import catlass.tla as tla

    native_location, native_sha256 = _native_artifact(source, expected_revision)
    native_seen = False
    for name, module in sorted(sys.modules.items()):
        if name != "catlass" and not name.startswith("catlass."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        location = Path(module_file).resolve()
        if name == NATIVE_MODULE:
            if native_seen or location != native_location:
                raise RuntimeError(
                    f"Catlass native import provenance mismatch: {name} at {location}"
                )
            actual_sha256 = hashlib.sha256(location.read_bytes()).hexdigest()
            if actual_sha256 != native_sha256:
                raise RuntimeError(
                    f"Catlass native import provenance hash mismatch: {name} at {location}"
                )
            native_seen = True
            continue
        try:
            relative = location.relative_to(source)
        except ValueError as exc:
            raise RuntimeError(
                f"Catlass import provenance mismatch: {name} at {location} "
                f"is not under {source}"
            ) from exc
        expected_blob = subprocess.run(
            ["git", "-C", str(source), "rev-parse", f"{expected_revision}:{relative.as_posix()}"],
            text=True,
            capture_output=True,
            check=False,
        )
        actual_blob = subprocess.run(
            ["git", "-C", str(source), "hash-object", "--no-filters", str(location)],
            text=True,
            capture_output=True,
            check=False,
        )
        expected = expected_blob.stdout.strip()
        actual = actual_blob.stdout.strip()
        if expected_blob.returncode != 0 or actual_blob.returncode != 0 or actual != expected:
            raise RuntimeError(
                "Catlass import provenance mismatch: "
                f"{name} at {location} is not the pinned "
                f"{expected_revision}:{relative.as_posix()} blob"
            )
    if not native_seen:
        raise RuntimeError("Catlass native import provenance is missing")
    location = Path(getattr(tla, "__file__", "<unknown>")).resolve()
    missing = [name for name in REQUIRED_TLA_API if not hasattr(tla, name)]
    if missing:
        raise RuntimeError(
            "incompatible Catlass DSL runtime: "
            f"{location} is missing imperative catlass.tla APIs "
            f"{','.join(missing)}; install a Catlass revision that provides "
            "the @tla.kernel frontend and tla.compile before A5 dispatch"
        )


def _load_kernel(path: str):
    source = Path(path)
    spec = importlib.util.spec_from_file_location("a5kernel_fixture", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load kernel source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _numbers(value, name: str):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain only numbers")
    return [float(item) for item in value]


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: host_driver.py KERNEL.py INPUT.json CATLASS_REVISION")
    with Path(sys.argv[2]).open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict) or set(payload) != {"input_a", "input_b"}:
        raise ValueError("input must contain exactly input_a and input_b")
    input_a = _numbers(payload["input_a"], "input_a")
    input_b = _numbers(payload["input_b"], "input_b")
    if len(input_a) != len(input_b):
        raise ValueError("input vectors must have equal length")
    _preflight_runtime(sys.argv[3])
    output = _numbers(_load_kernel(sys.argv[1]).run(input_a, input_b), "output")
    print(OUTPUT_MARKER + json.dumps(output, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_ASCEND_C_SOURCE = '''\
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR a, GM_ADDR b, GM_ADDR out, uint32_t n) {
  GlobalTensor<float> ga, gb, gout;
  ga.SetGlobalBuffer((__gm__ float*)a, n);
  gb.SetGlobalBuffer((__gm__ float*)b, n);
  gout.SetGlobalBuffer((__gm__ float*)out, n);
  for (uint32_t i = GetBlockIdx(); i < n; i += GetBlockNum()) {
    gout.SetValue(i, ga.GetValue(i) + gb.GetValue(i));
  }
}
'''

_TRITON_SOURCE = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add(a, b, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(a + offsets, mask=mask) + tl.load(b + offsets, mask=mask), mask=mask)

def run(a, b):
    out = torch.empty_like(a)
    vector_add[(triton.cdiv(a.numel(), 256),)](a, b, out, a.numel(), BLOCK=256)
    return out
'''


_FIXTURES = {
    Language.CATLASS_DSL: Fixture(
        Language.CATLASS_DSL,
        (
            SourceFile("kernel.py", _CATLASS_SOURCE),
            SourceFile("host_driver.py", _CATLASS_DRIVER),
        ),
        ("python", "host_driver.py", "kernel.py", "input.json"),
        400,
    ),
    Language.ASCEND_C: Fixture(
        Language.ASCEND_C,
        (SourceFile("kernel.cpp", _ASCEND_C_SOURCE),),
    ),
    Language.TRITON_ASCEND: Fixture(
        Language.TRITON_ASCEND,
        (SourceFile("kernel.py", _TRITON_SOURCE),),
    ),
}


def fixture_for(language: Language | str) -> Fixture:
    try:
        return _FIXTURES[Language(language)]
    except ValueError as exc:
        supported = ", ".join(item.value for item in Language)
        raise ValueError(f"unsupported language {language!r}; choose {supported}") from exc
