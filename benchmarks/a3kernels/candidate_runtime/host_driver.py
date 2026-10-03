"""Fixed compiler and launcher for one staged A3 candidate source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


_EXPECTED_BUILD = {
    "arch": "dav-2201",
    "compiler": "bisheng",
    "logical_device": 0,
    "max_elements": 4096,
    "output": "a3_candidate.so",
    "source": "kernel.cpp",
    "target": "Ascend910B4",
}
_CANDIDATE_SOURCE_SLOT = "// RSI_A3_CANDIDATE_SOURCE_SLOT"


def _toolchain_paths(torch, torch_npu):
    torch_root = Path(torch.__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
    return (
        torch_root / "include",
        torch_root / "include" / "torch" / "csrc" / "api" / "include",
        npu_root / "include",
    ), (torch_root / "lib", npu_root / "lib")


def _assemble_source(candidate: str, wrapper: str) -> str:
    if wrapper.count(_CANDIDATE_SOURCE_SLOT) != 1:
        raise RuntimeError("host wrapper must contain exactly one candidate source slot")
    return wrapper.replace(_CANDIDATE_SOURCE_SLOT, candidate)


def _compile(root: Path, spec: dict, torch, torch_npu) -> Path:
    compiler = shutil.which(spec["compiler"])
    if compiler is None:
        raise RuntimeError("bisheng compiler not found on PATH")
    combined = _assemble_source(
        (root / "candidate.cpp").read_text(encoding="utf-8"),
        (root / "host_wrapper.inc").read_text(encoding="utf-8"),
    )
    source = root / spec["source"]
    source.write_text(combined, encoding="utf-8")
    output = root / spec["output"]
    includes, libraries = _toolchain_paths(torch, torch_npu)
    abi = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
    argv = [
        compiler, "-x", "asc", f"--npu-arch={spec['arch']}",
        "-shared", "-fPIC", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        "-ltorch_npu", "-ltorch", "-lc10", str(source), "-o", str(output),
        *(f"-I{path}" for path in includes), *(f"-L{path}" for path in libraries),
    ]
    subprocess.run(argv, cwd=root, check=True)
    if not output.is_file():
        raise RuntimeError("bisheng succeeded without producing a3_candidate.so")
    return output


def _read_input(
    path: Path, maximum: int
) -> tuple[list[float], list[float], int, int, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "input_a", "input_b", "logical_length", "padded_length", "block_count"
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("input does not match the fixed execution schema")
    a, b = value["input_a"], value["input_b"]
    logical = value["logical_length"]
    padded = value["padded_length"]
    blocks = value["block_count"]
    if (
        not isinstance(a, list) or not isinstance(b, list) or not a
        or type(logical) is not int or type(padded) is not int
        or type(blocks) is not int or not 1 <= blocks <= 32
        or len(a) != len(b) or len(a) != padded or len(a) > maximum
        or not 1 <= logical <= padded
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in a + b)
    ):
        raise ValueError("input dimensions must describe bounded numeric arrays")
    return [float(item) for item in a], [float(item) for item in b], logical, padded, blocks


def _runtime_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one staged A3 candidate")
    parser.add_argument("--mode", choices=("run", "benchmark", "replay"), default="run")
    parser.add_argument("--warm-up", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("input")
    args = parser.parse_args(argv)
    if args.warm_up < 0 or args.launch_count < 1:
        parser.error("warm-up must be non-negative and launch-count must be positive")
    if args.mode == "run" and (args.warm_up != 0 or args.launch_count != 1):
        parser.error("run mode uses exactly one launch and no warm-up")
    return args


def main(argv: list[str] | None = None, *, clock_ns=time.perf_counter_ns) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parent
    spec = json.loads((root / "build.json").read_text(encoding="utf-8"))
    if spec != _EXPECTED_BUILD:
        raise RuntimeError("unrecognized A3 candidate build descriptor")
    import torch
    import torch_npu

    if args == ["--compile-only"]:
        library = _compile(root, spec, torch, torch_npu)
        print("A3CANDIDATE_COMPILED=" + hashlib.sha256(library.read_bytes()).hexdigest())
        return 0
    runtime = _runtime_args(args)
    library = root / spec["output"]
    if not library.is_file():
        raise RuntimeError("candidate library has not been compiled")
    input_path = Path(runtime.input)
    if not input_path.is_absolute():
        input_path = root / input_path
    a_values, b_values, logical, padded, blocks = _read_input(
        input_path, spec["max_elements"]
    )
    torch.ops.load_library(str(library))
    torch.npu.set_device(0)
    a = torch.tensor(a_values, dtype=torch.float32, device="npu:0")
    b = torch.tensor(b_values, dtype=torch.float32, device="npu:0")
    def launch():
        result = torch.ops.rsi_a3candidates.vector_add(
            a, b, logical, padded, blocks
        )
        torch.npu.synchronize()
        return result

    output = None
    for _ in range(runtime.warm_up):
        output = launch()
    samples = []
    for _ in range(runtime.launch_count):
        start = clock_ns() if runtime.mode == "benchmark" else None
        output = launch()
        if start is not None:
            elapsed = (clock_ns() - start) / 1000.0
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("timing clock returned a non-positive sample")
            samples.append(elapsed)
    values = output.cpu().tolist()
    expected = [
        a_values[index] + b_values[index] for index in range(logical)
    ]
    if (
        not isinstance(values, list) or len(values) != padded
        or any(
            not math.isfinite(float(value)) or abs(float(value) - want) > 1e-5
            for value, want in zip(values[:logical], expected)
        )
    ):
        raise RuntimeError("candidate output failed logical host verification")
    # Device padding is allocation-only and deliberately excluded from candidate
    # correctness. Canonicalize it before emitting auditable padded output.
    values = [float(value) for value in values[:logical]] + [0.0] * (
        padded - logical
    )
    for sample in samples:
        print(f"A3INNER_TIMING_US={sample:.6f}")
    print("A3KERNEL_OUTPUT=" + json.dumps(values, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
