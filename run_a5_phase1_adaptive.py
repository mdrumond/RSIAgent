#!/usr/bin/env python3
"""Run, resume, or report an explicitly bound adaptive A5 Phase-1 campaign."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable, Sequence

from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "resume", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--factory", required=True, help="module:callable production binding")
    return parser


def _factory(reference: str) -> Callable[[argparse.Namespace], AdaptivePhase1Coordinator]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("adaptive factory must be callable")
    return factory


def main(
    argv: Sequence[str] | None = None,
    *,
    factory_loader: Callable[[str], Callable] = _factory,
) -> int:
    args = _parser().parse_args(argv)
    coordinator = factory_loader(args.factory)(args)
    if not isinstance(coordinator, AdaptivePhase1Coordinator):
        raise TypeError("adaptive factory must return AdaptivePhase1Coordinator")
    result = getattr(coordinator, args.command)()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
