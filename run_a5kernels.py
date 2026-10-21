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
    phase1 = commands.add_parser(
        "phase1-plan", help="print the deterministic host-owned Phase1 dry-run plan"
    )
    phase1.add_argument("--proposals", type=Path)
    for name, help_text in (
        (
            "preflight",
            "check inputs with local BGE inference and a bounded BZ device probe; no Actor/LLM inference",
        ),
        ("trial", "run the preregistered Catlass smoke-vector-add pilot"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--tla-root", type=Path)
        command.add_argument("--worker-config", type=Path)
        command.add_argument("--profiling-skill-root", type=Path)
        command.add_argument("--catlass-src")
        command.add_argument("--catlass-revision")
        command.add_argument("--bge-cache", type=Path, required=True)
        command.add_argument("--kdb", type=Path, required=True)
        command.add_argument("--collection", action="append", required=True)
        command.add_argument("--results-root", type=Path, required=True)
        command.add_argument("--device", type=int)
    return parser


def _production_paths(args):
    from benchmarks.a5kernels.production import ProductionPaths

    from benchmarks.ascend.execution import apply_worker_config, execution_mode
    worker = apply_worker_config({}, getattr(args, "worker_config", None)) if getattr(args, "worker_config", None) else None
    local = worker is not None and execution_mode(worker) == "local"
    if local:
        args.tla_root = args.tla_root or Path.cwd()
        args.profiling_skill_root = args.profiling_skill_root or Path.cwd()
        args.device = worker["physical_device"]
        args.catlass_src = worker["catlass_source"]
        args.catlass_revision = worker["catlass_revision"]
    if args.profiling_skill_root is None or args.tla_root is None or args.device is None or args.catlass_src is None or args.catlass_revision is None:
        raise ValueError("remote execution requires explicit toolchain paths and device")
    return ProductionPaths(
        worker_config=worker,
        tla_root=args.tla_root.resolve(),
        profiling_skill_root=args.profiling_skill_root.resolve(),
        catlass_source=args.catlass_src,
        catlass_revision=args.catlass_revision,
        bge_cache=args.bge_cache.resolve(),
        kdb=args.kdb.resolve(),
        collection=args.collection[0] if len(args.collection) == 1 else tuple(args.collection),
        results_root=args.results_root.resolve(),
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        payload = initial_matrix().as_dict()
    elif args.command == "report":
        payload = aggregate_report(load_metrics(args.input))
    elif args.command == "phase1-plan":
        from benchmarks.a5kernels.phase1_registry import dry_run_plan, load_proposals

        proposals = load_proposals(args.proposals) if args.proposals is not None else None
        payload = dry_run_plan() if proposals is None else dry_run_plan(proposals)
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
