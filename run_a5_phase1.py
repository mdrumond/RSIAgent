#!/usr/bin/env python3
"""Plan, execute, resume, and report qualified A5 Catlass Phase 1 cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from benchmarks.a5kernels.phase1_composition import (
    LiveDependencies,
    Phase1CellComposition,
)
from benchmarks.a5kernels.phase1_live import load_gate_report
from benchmarks.a5kernels.phase1_production import build_phase1_live_dependencies
from benchmarks.a5kernels.production import ProductionPaths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "resume", "report"):
        command = commands.add_parser(name)
        command.add_argument("--gate", type=Path, required=True)
        command.add_argument("--guide", type=Path, required=True)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument(
            "--one-project-smoke",
            action="store_true",
            help="run exactly the first registered project in every runnable cell",
        )
        if name in {"run", "resume"}:
            command.add_argument("--tla-root", type=Path, required=True)
            command.add_argument("--profiling-skill-root", type=Path, required=True)
            command.add_argument("--catlass-source", required=True)
            command.add_argument("--bge-cache", type=Path, required=True)
            command.add_argument("--kdb", type=Path, required=True)
            command.add_argument("--collection", required=True)
            command.add_argument("--device", type=int, required=True)
            command.add_argument("--env-file", type=Path)
    return parser


def _unavailable_dependencies() -> LiveDependencies:
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("live dependencies are unavailable in read-only mode")

    return LiveDependencies(unavailable, unavailable)


def _paths(args, revision: str) -> ProductionPaths:
    return ProductionPaths(
        tla_root=args.tla_root,
        profiling_skill_root=args.profiling_skill_root,
        catlass_source=args.catlass_source,
        catlass_revision=revision,
        bge_cache=args.bge_cache,
        kdb=args.kdb,
        collection=args.collection,
        results_root=args.root,
        device=args.device,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    dependency_builder: Callable[..., LiveDependencies] = (
        build_phase1_live_dependencies
    ),
) -> int:
    args = _parser().parse_args(argv)
    gate = load_gate_report(args.gate)
    dependencies = _unavailable_dependencies()
    if args.command in {"run", "resume"}:
        dependencies = dependency_builder(
            _paths(args, gate.guide.catlass_revision),
            env_file=args.env_file,
        )
    factory = (
        Phase1CellComposition.one_project_smoke
        if args.one_project_smoke
        else Phase1CellComposition
    )
    composition = factory(
        args.root, args.guide, gate.guide, gate, dependencies,
    )
    if args.command == "plan":
        value = composition.plan()
    elif args.command == "run":
        composition.run()
        value = composition.report()
    elif args.command == "resume":
        composition.resume()
        value = composition.report()
    else:
        value = composition.report()
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
