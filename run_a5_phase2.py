#!/usr/bin/env python3
"""Run or resume exactly one admitted A5 Phase 2 treatment cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from benchmarks.a5kernels.phase2_live import Phase2Dependencies, Phase2LiveRunner
from benchmarks.a5kernels.phase2_memory import A5Phase1Snapshot


RuntimeFactory = Callable[[argparse.Namespace], tuple[A5Phase1Snapshot, Phase2Dependencies]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--snapshot", type=Path, required=True)
    command.add_argument("--cell-id", required=True)
    return parser


def _production_not_composed(
    _args: argparse.Namespace,
) -> tuple[A5Phase1Snapshot, Phase2Dependencies]:
    raise RuntimeError("A5 Phase 2 production dependencies are not composed")


def main(
    argv: Sequence[str] | None = None, *,
    runtime_factory: RuntimeFactory = _production_not_composed,
) -> int:
    args = _parser().parse_args(argv)
    snapshot, dependencies = runtime_factory(args)
    if args.cell_id != snapshot.destination.cell_id:
        raise ValueError("requested cell does not match admitted snapshot")
    if dependencies.cell != snapshot.destination:
        raise ValueError("runtime cell does not match admitted snapshot")
    terminal = Phase2LiveRunner(
        phase2_root=args.root,
        snapshot=snapshot,
        dependencies=dependencies,
    ).run()
    print(json.dumps(terminal, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
