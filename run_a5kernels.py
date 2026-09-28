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
    for name, help_text in (
        ("preflight", "check production pilot inputs without running a model or BZ job"),
        ("trial", "run the preregistered Catlass smoke-vector-add pilot"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--tla-root", type=Path, required=True)
        command.add_argument("--profiling-skill-root", type=Path, required=True)
        command.add_argument("--catlass-src", required=True)
        command.add_argument("--catlass-revision", required=True)
        command.add_argument("--bge-cache", type=Path, required=True)
        command.add_argument("--kdb", type=Path, required=True)
        command.add_argument("--collection", required=True)
        command.add_argument("--results-root", type=Path, required=True)
        command.add_argument("--device", type=int, required=True)
    return parser


def _production_paths(args):
    from benchmarks.a5kernels.production import ProductionPaths

    return ProductionPaths(
        tla_root=args.tla_root.resolve(),
        profiling_skill_root=args.profiling_skill_root.resolve(),
        catlass_source=args.catlass_src,
        catlass_revision=args.catlass_revision,
        bge_cache=args.bge_cache.resolve(),
        kdb=args.kdb.resolve(),
        collection=args.collection,
        results_root=args.results_root.resolve(),
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        payload = initial_matrix().as_dict()
    elif args.command == "report":
        payload = aggregate_report(load_metrics(args.input))
    else:
        # Production dependencies stay lazy for matrix/report users.
        from benchmarks.a5kernels.production import preflight, run_pilot

        paths = _production_paths(args)
        payload = preflight(paths) if args.command == "preflight" else run_pilot(paths)
    # Mapping insertion order is the public report priority: correctness first.
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
