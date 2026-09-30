#!/usr/bin/env python3
"""Run the deterministic A3 Ascend C hello kernel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from typing import Sequence

from benchmarks.a3kernels import A3KernelRunner, RunRequest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args(argv)
    request = RunRequest(length=args.length, seed=args.seed)
    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix="rsi-a3kernels-") as directory:
            result = A3KernelRunner().run(request, Path(directory))
    else:
        result = A3KernelRunner().run(request, args.workdir)
    print(json.dumps(asdict(result), indent=2, allow_nan=False))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
