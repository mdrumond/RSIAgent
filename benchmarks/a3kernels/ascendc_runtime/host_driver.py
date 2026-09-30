"""Compile and launch the registry-owned A3 Ascend C fixture."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


OUTPUT_MARKER = "A3KERNEL_OUTPUT="
_EXPECTED_BUILD = {
    "arch": "dav-2201",
    "compiler": "bisheng",
    "logical_device": 0,
    "max_elements": 4096,
    "output": "a3_kernel.so",
    "runtime": "py311-torch",
    "source": "kernel.cpp",
    "target": "Ascend910B4",
}


def _read_vectors(stream, max_elements: int) -> tuple[list[float], list[float]]:
    payload = json.load(stream)
    if not isinstance(payload, dict) or set(payload) != {"input_a", "input_b"}:
        raise ValueError("input must contain only input_a and input_b")
    vectors = []
    for name in ("input_a", "input_b"):
        value = payload[name]
        if (
            not isinstance(value, list)
            or not value
            or len(value) > max_elements
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value
            )
        ):
            raise ValueError(f"{name} must be a bounded non-empty numeric array")
        vectors.append([float(item) for item in value])
    if len(vectors[0]) != len(vectors[1]):
        raise ValueError("input vectors must have equal lengths")
    return vectors[0], vectors[1]


def _toolchain_paths(torch, torch_npu):
    torch_root = Path(torch.__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
    includes = (
        torch_root / "include",
        torch_root / "include" / "torch" / "csrc" / "api" / "include",
        npu_root / "include",
    )
    libraries = (torch_root / "lib", npu_root / "lib")
    return includes, libraries


def _compile(root: Path, spec: dict, torch, torch_npu) -> Path:
    compiler = shutil.which(spec["compiler"])
    if compiler is None:
        raise RuntimeError("bisheng compiler not found on PATH")
    output = root / spec["output"]
    includes, libraries = _toolchain_paths(torch, torch_npu)
    abi = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
    argv = [
        compiler,
        "-x", "asc",
        f"--npu-arch={spec['arch']}",
        "-shared", "-fPIC", "-std=c++17",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        "-ltorch_npu", "-ltorch", "-lc10",
        str(root / spec["source"]), "-o", str(output),
        *(f"-I{path}" for path in includes),
        *(f"-L{path}" for path in libraries),
    ]
    subprocess.run(argv, cwd=root, check=True)
    if not output.is_file():
        raise RuntimeError("bisheng succeeded without producing a3_kernel.so")
    return output


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: host_driver.py INPUT.json")
    root = Path(__file__).resolve().parent
    spec = json.loads((root / "build.json").read_text(encoding="utf-8"))
    if spec != _EXPECTED_BUILD:
        raise RuntimeError("unrecognized A3 Ascend C build descriptor")
    with Path(sys.argv[1]).open(encoding="utf-8") as stream:
        a_values, b_values = _read_vectors(stream, spec["max_elements"])

    import torch
    import torch_npu

    library = _compile(root, spec, torch, torch_npu)
    torch.ops.load_library(str(library))
    torch.npu.set_device(0)
    a = torch.tensor(a_values, dtype=torch.float32, device="npu:0")
    b = torch.tensor(b_values, dtype=torch.float32, device="npu:0")
    output = torch.ops.rsi_a3kernels.vector_add(a, b)
    torch.npu.synchronize()
    print(OUTPUT_MARKER + json.dumps(output.cpu().tolist(), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
