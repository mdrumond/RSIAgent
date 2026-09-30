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


VALIDATION_LENGTHS = (1, 33, 4096)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--length", type=int)
    size.add_argument(
        "--validation-suite",
        action="store_true",
        help="run native boundary coverage at lengths 1, 33, and 4096",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args(argv)
    lengths = (
        VALIDATION_LENGTHS
        if args.validation_suite
        else (args.length if args.length is not None else 32,)
    )

    def run_in(root: Path):
        runner = A3KernelRunner()
        return tuple(
            runner.run(
                RunRequest(length=length, seed=args.seed),
                root / f"length-{length}" if args.validation_suite else root,
            )
            for length in lengths
        )

    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix="rsi-a3kernels-") as directory:
            results = run_in(Path(directory))
    else:
        results = run_in(args.workdir)
    payload = (
        {
            "validation_lengths": list(lengths),
            "results": [asdict(result) for result in results],
        }
        if args.validation_suite
        else asdict(results[0])
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
