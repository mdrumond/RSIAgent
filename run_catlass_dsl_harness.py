#!/usr/bin/env python3
"""Preflight, run, and report host-authoritative A5 Catlass contracts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from benchmarks.a5kernels.catlass_harness import (
    CatlassContractHarness,
    HarnessContract,
    report_results,
    write_result,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report")
    report.add_argument("results", type=Path)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--contract", choices=[item.value for item in HarnessContract], required=True)
        command.add_argument("--tla-root", type=Path, required=True)
        command.add_argument("--catlass-src", required=True)
        command.add_argument("--catlass-revision", required=True)
        command.add_argument("--device", type=int, required=True)
        if name == "run":
            command.add_argument("--attempt-id", required=True)
            command.add_argument("--output", type=Path, required=True)
    return parser


def _harness(args):
    from benchmarks.a5kernels.bz import BZSessionAdapter, CatlassValidationExecutor

    executor = CatlassValidationExecutor(
        upload_wrapper=str(args.tla_root / "execution-profiles/bz-a5/upload.sh"),
        validation_wrapper=str(args.tla_root / "execution-profiles/catlass-validation.sh"),
        catlass_source=args.catlass_src,
        catlass_revision=args.catlass_revision,
    )
    if args.command == "preflight":
        executor.probe_device(args.device)
    backend = BZSessionAdapter(
        executor,
        session_wrapper=str(args.tla_root / "execution-profiles/bz-a5/session.sh"),
    )
    return CatlassContractHarness(backend, device=args.device)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "report":
        payload = report_results(args.results)
    else:
        source = args.source.read_text(encoding="utf-8")
        harness = _harness(args)
        if args.command == "preflight":
            payload = harness.preflight(source, args.contract)
        else:
            result = harness.run(source, args.contract, args.attempt_id)
            write_result(result, args.output)
            payload = asdict(result)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
