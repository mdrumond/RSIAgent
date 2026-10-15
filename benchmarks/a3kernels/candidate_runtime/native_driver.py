"""Host-controlled generic native candidate compiler and synchronized launcher."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from legacy_driver import _compile


def main(*, clock_ns=time.perf_counter_ns):
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--mode", choices=("run", "benchmark", "replay"), default="run")
    parser.add_argument("--warm-up", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("input", nargs="?")
    args = parser.parse_args()
    if args.warm_up < 0 or args.launch_count < 1:
        parser.error("warm-up must be non-negative and launch-count positive")
    if args.mode == "run" and (args.warm_up or args.launch_count != 1):
        parser.error("run mode requires one launch and no warm-up")
    root = Path(__file__).resolve().parent
    import torch
    import torch_npu

    if args.compile_only:
        if args.input:
            parser.error("compile-only does not accept runtime input")
        spec = json.loads((root / "build.json").read_text())
        library = _compile(root, spec, torch, torch_npu)
        print("A3CANDIDATE_COMPILED=" + hashlib.sha256(library.read_bytes()).hexdigest())
        return
    if not args.input:
        parser.error("runtime input is required")
    payload = json.loads((root / args.input).read_text())
    torch.ops.load_library(str(root / "a3_candidate.so"))
    torch.npu.set_device(0)
    inputs = [torch.tensor(t["values"], dtype=getattr(torch, t["dtype"]), device="npu:0").reshape(t["shape"]) for t in payload["inputs"]]
    def launch():
        output = torch.ops.rsi_a3native.launch(inputs)
        torch.npu.synchronize()
        return output

    for _ in range(args.warm_up):
        launch()
    samples = []
    for _ in range(args.launch_count):
        start = clock_ns() if args.mode == "benchmark" else None
        output = launch()
        if start is not None:
            elapsed = (clock_ns() - start) / 1000.0
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("timing clock returned a non-positive sample")
            samples.append(elapsed)
    if list(output.shape) != payload["output"]["shape"]:
        raise RuntimeError("native output shape differs from host contract")
    values = output.cpu().float().reshape(-1).tolist()
    if any(not math.isfinite(v) for v in values):
        raise RuntimeError("native output contains nonfinite values")
    for sample in samples:
        print(f"A3TIMING_US={sample:.6f}")
    print("A3KERNEL_OUTPUT=" + json.dumps(values))


if __name__ == "__main__":
    main()
