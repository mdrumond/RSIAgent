"""Build and run the registry-owned AscendC vector-add fixture."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


OUTPUT_MARKER = "A5KERNEL_OUTPUT="
_EXPECTED_BUILD = {
    "arch": "dav-3510",
    "compiler": "bisheng",
    "max_elements": 4096,
    "output": "kernel.so",
    "source": "kernel.cpp",
}


def _read_vectors(stream, *, max_elements: int) -> tuple[list[float], list[float]]:
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
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        ):
            raise ValueError(f"{name} must be a non-empty numeric JSON array")
        vectors.append([float(item) for item in value])
    if len(vectors[0]) != len(vectors[1]):
        raise ValueError("input vectors must have equal lengths")
    return vectors[0], vectors[1]


def _toolchain_paths(torch, torch_npu) -> tuple[list[str], list[str]]:
    torch_root = Path(torch.__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
    includes = [
        torch_root / "include",
        torch_root / "include" / "torch" / "csrc" / "api" / "include",
        npu_root / "include",
    ]
    libraries = [torch_root / "lib", npu_root / "lib"]
    return [str(path) for path in includes], [str(path) for path in libraries]


def _compile(root: Path, spec: dict, torch, torch_npu) -> Path:
    compiler = shutil.which(spec["compiler"])
    if compiler is None:
        raise RuntimeError("bisheng compiler not found on PATH")
    source = root / spec["source"]
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
        str(source), "-o", str(output),
        *(f"-I{path}" for path in includes),
        *(f"-L{path}" for path in libraries),
    ]
    subprocess.run(argv, cwd=root, check=True)
    if not output.is_file():
        raise RuntimeError("bisheng succeeded without producing kernel.so")
    return output


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: host_driver.py INPUT.json")
    root = Path(__file__).resolve().parent
    spec = json.loads((root / "build.json").read_text(encoding="utf-8"))
    if spec != _EXPECTED_BUILD:
        raise RuntimeError("unrecognized AscendC build descriptor")
    with Path(sys.argv[1]).open(encoding="utf-8") as stream:
        input_a, input_b = _read_vectors(stream, max_elements=spec["max_elements"])

    import torch
    import torch_npu

    library = _compile(root, spec, torch, torch_npu)
    torch.ops.load_library(str(library))
    device = int(os.environ.get("BZ_A5_PROFILE_PHYSICAL_DEVICE", "0"))
    if device < 0:
        raise ValueError("BZ_A5_PROFILE_PHYSICAL_DEVICE must be non-negative")
    torch.npu.set_device(device)
    a = torch.tensor(input_a, dtype=torch.float32, device="npu")
    b = torch.tensor(input_b, dtype=torch.float32, device="npu")
    output = torch.ops.a5kernels.vector_add(a, b)
    torch.npu.synchronize()
    values = output.cpu().tolist()
    print(OUTPUT_MARKER + json.dumps(values, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
