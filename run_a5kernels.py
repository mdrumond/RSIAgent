#!/usr/bin/env python3
"""Plan A5 kernel experiments and aggregate host-recorded results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from benchmarks.a5kernels.matrix import aggregate_report, initial_matrix, load_metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("matrix", help="print the immutable initial experiment plan")
    report = commands.add_parser("report", help="aggregate host-recorded JSON metrics")
    report.add_argument("input", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        payload = initial_matrix().as_dict()
    else:
        payload = aggregate_report(load_metrics(args.input))
    # Mapping insertion order is the public report priority: correctness first.
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
