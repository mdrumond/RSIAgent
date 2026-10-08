#!/usr/bin/env python3
"""Plan and inspect the fail-closed A5 Catlass Phase 1 lifecycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from benchmarks.a5kernels.phase1_experiments import (
    GUIDE_SCHEMA,
    PINNED_CATLASS_REVISION,
    ProgrammingGuideIdentity,
)
from benchmarks.a5kernels.phase1_live import (
    dry_run_manifest,
    load_gate_report,
    phase1_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--guide-sha256", required=True)
    plan.add_argument("--cpl-skills-revision", required=True)
    report = commands.add_parser("report")
    report.add_argument("--gate", type=Path, required=True)
    for name in ("run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--gate", type=Path, required=True)
        command.add_argument("--completed-cell", action="append", default=[])
    return parser


def _guide(args) -> ProgrammingGuideIdentity:
    return ProgrammingGuideIdentity(
        GUIDE_SCHEMA,
        args.guide_sha256,
        args.cpl_skills_revision,
        PINNED_CATLASS_REVISION,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        value = dry_run_manifest(_guide(args))
    else:
        report = load_gate_report(args.gate)
        if args.command == "report":
            value = {
                "schema": report.schema,
                "ready": report.ready,
                "admitted": sum(item.admitted for item in report.records),
                "guide_sha256": report.guide.guide_sha256,
                "qualification_sha256": report.report_sha256,
            }
        else:
            value = phase1_manifest(
                report.guide, report,
                completed_cell_ids=args.completed_cell,
            )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
