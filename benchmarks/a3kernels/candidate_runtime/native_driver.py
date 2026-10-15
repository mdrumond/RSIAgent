"""Host-controlled generic native candidate compiler and synchronized launcher."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from legacy_driver import _compile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("input", nargs="?")
    args = parser.parse_args()
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
    output = torch.ops.rsi_a3native.launch(inputs)
    torch.npu.synchronize()
    if list(output.shape) != payload["output"]["shape"]:
        raise RuntimeError("native output shape differs from host contract")
    values = output.cpu().float().reshape(-1).tolist()
    if any(not math.isfinite(v) for v in values):
        raise RuntimeError("native output contains nonfinite values")
    print("A3KERNEL_OUTPUT=" + json.dumps(values))


if __name__ == "__main__":
    main()
