#!/usr/bin/env python3
"""Plan, execute, resume, and report qualified A5 Catlass Phase 1 cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Callable, Sequence

from benchmarks.a5kernels.bz import RuntimeUnavailableError
from benchmarks.a5kernels.phase1_composition import (
    LiveDependencies,
    Phase1CellComposition,
    select_worker_cell,
)
from benchmarks.a5kernels.phase1_live import (
    BACKOFF_SECONDS,
    MAX_INFRASTRUCTURE_RETRIES,
    InfrastructureFailure,
    load_gate_report,
)
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
            help=(
                "run the first registered project in each selected runnable cell; "
                "use a root distinct from every full campaign"
            ),
        )
        if name in {"run", "resume"}:
            command.add_argument(
                "--cell-id",
                action="append",
                help="run or resume exactly one registered runnable cell",
            )
            command.add_argument("--tla-root", type=Path, required=True)
            command.add_argument("--profiling-skill-root", type=Path, required=True)
            command.add_argument("--catlass-source", required=True)
            command.add_argument("--bge-cache", type=Path, required=True)
            command.add_argument("--kdb", type=Path, required=True)
            command.add_argument("--collection", action="append", required=True)
            command.add_argument("--device", type=int, required=True)
            command.add_argument("--env-file", type=Path)
    return parser


def _unavailable_dependencies() -> LiveDependencies:
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("live dependencies are unavailable in read-only mode")

    return LiveDependencies(unavailable, unavailable)


def _paths(
    args,
    revision: str,
    *,
    results_root: Path | None = None,
) -> ProductionPaths:
    return ProductionPaths(
        tla_root=args.tla_root,
        profiling_skill_root=args.profiling_skill_root,
        catlass_source=args.catlass_source,
        catlass_revision=revision,
        bge_cache=args.bge_cache,
        kdb=args.kdb,
        collection=args.collection[0] if len(args.collection) == 1 else tuple(args.collection),
        results_root=args.root if results_root is None else results_root,
        device=args.device,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    dependency_builder: Callable[..., LiveDependencies] = (
        build_phase1_live_dependencies
    ),
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    args = _parser().parse_args(argv)
    gate = load_gate_report(args.gate)
    selected = select_worker_cell(
        gate.guide, getattr(args, "cell_id", None),
    )
    dependencies = _unavailable_dependencies()
    if args.command in {"run", "resume"}:
        worker_root = None
        if selected is not None:
            worker_root = args.root / "cells" / selected.cell_id
            worker_root.mkdir(parents=True, exist_ok=True)
        failures = 0
        while True:
            try:
                dependencies = dependency_builder(
                    _paths(
                        args,
                        gate.guide.catlass_revision,
                        results_root=worker_root,
                    ),
                    env_file=args.env_file,
                )
                break
            except (InfrastructureFailure, RuntimeUnavailableError):
                if failures >= MAX_INFRASTRUCTURE_RETRIES:
                    raise
                failures += 1
                sleeper(BACKOFF_SECONDS)
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
        if selected is None:
            composition.run()
        else:
            composition.run_cell(selected.cell_id)
        value = composition.report()
    elif args.command == "resume":
        if selected is None:
            composition.resume()
        else:
            composition.run_cell(selected.cell_id)
        value = composition.report()
    else:
        value = composition.report()
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


def _entry() -> int:
    try:
        return main()
    except (InfrastructureFailure, RuntimeUnavailableError) as exc:
        _write_infrastructure_diagnostic(exc)
        return 75


def _write_infrastructure_diagnostic(exc: Exception) -> None:
    detail = " ".join(str(exc).split()) or "unavailable"
    message = f"{type(exc).__name__}: {detail}"
    encoded = message.encode("utf-8")
    if len(encoded) > 512:
        message = encoded[:509].decode("utf-8", errors="ignore") + "..."
    print(message, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(_entry())
