#!/usr/bin/env python3
"""Run or resume exactly one admitted A5 Phase 2 treatment cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Callable, Sequence

from benchmarks.a5kernels.bz import RuntimeUnavailableError
from benchmarks.a5kernels.phase1_live import InfrastructureFailure, load_gate_report
from benchmarks.a5kernels.phase2_attention import AttentionInfrastructureError
from benchmarks.a5kernels.phase2_live import Phase2Dependencies, Phase2LiveRunner
from benchmarks.a5kernels.phase2_memory import A5Phase1Snapshot
from benchmarks.a5kernels.phase2_production import (
    Phase2ProductionConfig,
    build_phase2_production,
)


RuntimeFactory = Callable[
    [argparse.Namespace], tuple[A5Phase1Snapshot, Phase2Dependencies]
]
MAX_INFRASTRUCTURE_RETRIES = 3
BACKOFF_SECONDS = 120


class _OneCell(argparse.Action):
    """Reject repeated cell selectors instead of silently taking the last one."""

    def __call__(self, parser, namespace, values, option_string=None):
        del parser
        if getattr(namespace, self.dest, None) is not None:
            raise argparse.ArgumentError(self, f"{option_string} may be supplied once")
        setattr(namespace, self.dest, values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("qualify", "run"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--snapshot", type=Path, required=True)
        command.add_argument("--cell-id", action=_OneCell, required=True)
        command.add_argument("--phase1-root", type=Path)
        command.add_argument("--gate", type=Path)
        command.add_argument("--guide", type=Path)
        command.add_argument("--tla-root", type=Path)
        command.add_argument("--catlass-source")
        command.add_argument("--bge-cache", type=Path)
        command.add_argument("--kdb", type=Path)
        command.add_argument("--collection")
        command.add_argument("--device", type=int)
        command.add_argument("--env-file", type=Path)
    return parser


def _production_factory(
    args: argparse.Namespace,
) -> tuple[A5Phase1Snapshot, Phase2Dependencies]:
    required = {
        "phase1_root": args.phase1_root,
        "gate": args.gate,
        "guide": args.guide,
        "tla_root": args.tla_root,
        "catlass_source": args.catlass_source,
        "device": args.device,
    }
    missing = [name.replace("_", "-") for name, value in required.items()
               if value is None]
    if missing:
        raise RuntimeError(
            "A5 Phase 2 production dependencies are missing: " + ", ".join(missing)
        )
    gate = load_gate_report(args.gate)
    return build_phase2_production(Phase2ProductionConfig(
        phase1_root=args.phase1_root,
        gate=gate,
        guide_path=args.guide,
        phase2_root=args.root,
        snapshot_path=args.snapshot,
        cell_id=args.cell_id,
        tla_root=args.tla_root,
        catlass_source=args.catlass_source,
        device=args.device,
        bge_cache=args.bge_cache,
        kdb=args.kdb,
        collection=args.collection,
        env_file=args.env_file,
    ))


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: RuntimeFactory = _production_factory,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    args = _parser().parse_args(argv)
    failures = 0
    while True:
        try:
            snapshot, dependencies = runtime_factory(args)
            if not isinstance(snapshot, A5Phase1Snapshot):
                raise TypeError("runtime factory must return an A5Phase1Snapshot")
            if not isinstance(dependencies, Phase2Dependencies):
                raise TypeError("runtime factory must return Phase2Dependencies")
            if args.snapshot.resolve() != snapshot.root.resolve():
                raise ValueError("requested snapshot path does not match admitted snapshot")
            if args.cell_id != snapshot.destination.cell_id:
                raise ValueError("requested cell does not match admitted snapshot")
            if dependencies.cell != snapshot.destination:
                raise ValueError("runtime cell does not match admitted snapshot")

            if args.command == "qualify":
                path = args.root.resolve() / "cells" / args.cell_id / "qualification.json"
                try:
                    qualification = json.loads(path.read_bytes())
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError("production factory did not retain qualification") from exc
                if qualification.get("status") != "transport-qualified":
                    raise ValueError("production factory retained invalid qualification")
                terminal = {
                    "cell_id": args.cell_id, "status": "transport-qualified",
                }
            else:
                terminal = Phase2LiveRunner(
                    phase2_root=args.root,
                    snapshot=snapshot,
                    dependencies=dependencies,
                ).run()
            break
        except (
            InfrastructureFailure,
            AttentionInfrastructureError,
            RuntimeUnavailableError,
        ):
            if failures >= MAX_INFRASTRUCTURE_RETRIES:
                raise
            failures += 1
            sleeper(BACKOFF_SECONDS)
    print(json.dumps(terminal, sort_keys=True, separators=(",", ":")))
    return 0


def _entry() -> int:
    try:
        return main()
    except (
        InfrastructureFailure,
        AttentionInfrastructureError,
        RuntimeUnavailableError,
    ):
        return 75


if __name__ == "__main__":
    raise SystemExit(_entry())
