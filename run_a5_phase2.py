#!/usr/bin/env python3
"""Run or resume exactly one admitted A5 Phase 2 treatment cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from benchmarks.a5kernels.phase2_live import Phase2Dependencies, Phase2LiveRunner
from benchmarks.a5kernels.phase2_memory import A5Phase1Snapshot


RuntimeFactory = Callable[
    [argparse.Namespace], tuple[A5Phase1Snapshot, Phase2Dependencies]
]


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
    command = commands.add_parser("run")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--snapshot", type=Path, required=True)
    command.add_argument("--cell-id", action=_OneCell, required=True)
    return parser


def _production_not_composed(
    _args: argparse.Namespace,
) -> tuple[A5Phase1Snapshot, Phase2Dependencies]:
    raise RuntimeError("A5 Phase 2 production dependencies are not composed")


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: RuntimeFactory = _production_not_composed,
) -> int:
    args = _parser().parse_args(argv)
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

    terminal = Phase2LiveRunner(
        phase2_root=args.root,
        snapshot=snapshot,
        dependencies=dependencies,
    ).run()
    print(json.dumps(terminal, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
