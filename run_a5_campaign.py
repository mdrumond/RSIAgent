#!/usr/bin/env python3
"""Reconcile the resource-aware A5 Phase 1/Phase 2 campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from benchmarks.a5kernels.device_inventory import (
    BZDeviceInventory, DeviceInventoryUnavailable,
)
from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.phase1_composition import LiveDependencies, Phase1CellComposition
from benchmarks.a5kernels.phase1_experiments import admit_programming_guide
from benchmarks.a5kernels.phase1_live import load_gate_report
from benchmarks.a5kernels.phase2_live import Phase2Dependencies, Phase2LiveRunner
from benchmarks.a5kernels.phase2_memory import (
    A5Phase1Snapshot, admit_phase1_snapshot, phase2_cells,
)
from benchmarks.a5kernels.phase2_production import QUALIFICATION_SCHEMA
from benchmarks.a5kernels.priority_controller import CampaignController, JobKind
from benchmarks.a5kernels.screen_workers import ScreenWorkerPool, run_request


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reconcile = commands.add_parser("reconcile")
    for flag in ("controller-root", "phase1-root", "phase2-root", "gate", "guide",
                 "tla-root", "profiling-skill-root", "bge-cache", "kdb"):
        reconcile.add_argument(f"--{flag}", type=Path, required=True)
    reconcile.add_argument("--catlass-source", required=True)
    reconcile.add_argument("--runtime-catlass-revision")
    reconcile.add_argument("--phase2-guide", type=Path)
    reconcile.add_argument("--phase2-cpl-skills-revision")
    reconcile.add_argument("--collection", required=True)
    reconcile.add_argument("--env-file", type=Path)
    reconcile.add_argument(
        "--compile-probes",
        type=_positive_int,
        metavar="LIMIT",
        help=(
            "give Phase 1 compile actions a separate positive LIMIT instead of "
            "charging their ordinary agent action budget"
        ),
    )
    reconcile.add_argument("--until-complete", action="store_true")
    reconcile.add_argument("--poll-seconds", type=float, default=30.0)
    reconcile.add_argument(
        "--retry-job", action="append", default=[],
        help="explicitly retry one blocked logical job ID",
    )
    worker = commands.add_parser("worker")
    worker.add_argument("--request", type=Path, required=True)
    return parser


def _unavailable(*_args, **_kwargs):
    raise RuntimeError("campaign controller cannot execute cell dependencies")


def _option(argv: list[str], name: str, value) -> None:
    argv.extend((f"--{name}", str(value)))


def _command_factory(args, destinations):
    phase1_script = Path(__file__).resolve().with_name("run_a5_phase1.py")
    phase2_script = Path(__file__).resolve().with_name("run_a5_phase2.py")
    by_id = {item.cell_id: item for item in destinations}

    def command(job, device):
        if job.kind in {JobKind.PHASE1_RUN, JobKind.PHASE1_RESUME}:
            argv = [
                sys.executable, str(phase1_script),
                "resume" if job.kind is JobKind.PHASE1_RESUME else "run",
            ]
            values = (
                ("gate", args.gate), ("guide", args.guide),
                ("root", args.phase1_root), ("cell-id", job.cell_id),
                ("tla-root", args.tla_root),
                ("profiling-skill-root", args.profiling_skill_root),
                ("catlass-source", args.catlass_source),
                ("bge-cache", args.bge_cache), ("kdb", args.kdb),
                ("collection", args.collection), ("device", device),
            )
        else:
            destination = by_id[job.cell_id]
            argv = [
                sys.executable, str(phase2_script),
                "qualify" if job.kind is JobKind.PHASE2_QUALIFY else "run",
            ]
            values = (
                ("root", args.phase2_root),
                ("snapshot", args.phase2_root / "cells" / destination.cell_id / "snapshot"),
                ("cell-id", destination.cell_id),
                ("phase1-root", args.phase1_root), ("gate", args.gate),
                ("guide", args.guide), ("tla-root", args.tla_root),
                ("catlass-source", args.catlass_source),
                ("bge-cache", args.bge_cache), ("kdb", args.kdb),
                ("collection", args.collection), ("device", device),
            )
            if args.phase2_guide is not None:
                values += (
                    ("phase2-guide", args.phase2_guide),
                    ("phase2-cpl-skills-revision", args.phase2_cpl_skills_revision),
                )
            runtime_revision = getattr(args, "runtime_catlass_revision", None)
            if runtime_revision is not None:
                values += (("runtime-catlass-revision", runtime_revision),)
        for name, value in values:
            _option(argv, name, value)
        if args.env_file is not None:
            _option(argv, "env-file", args.env_file)
        if (
            job.kind in {JobKind.PHASE1_RUN, JobKind.PHASE1_RESUME}
            and getattr(args, "compile_probes", None) is not None
        ):
            _option(
                argv,
                "compile-probes",
                args.compile_probes,
            )
        return tuple(argv)

    return command


def _qualification(args, destination) -> int | None:
    path = args.phase2_root / "cells" / destination.cell_id / "qualification.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid retained Phase 2 qualification") from exc
    body = {key: item for key, item in value.items() if key != "record_sha256"}
    if (
        value.get("schema") != QUALIFICATION_SCHEMA
        or value.get("status") != "transport-qualified"
        or value.get("cell_id") != destination.cell_id
        or value.get("catlass_revision") != destination.catlass_revision
        or value.get("catlass_source") != args.catlass_source
        or isinstance(value.get("physical_device"), bool)
        or not isinstance(value.get("physical_device"), int)
        or not 1 <= value["physical_device"] <= 7
        or value.get("record_sha256") != canonical_digest(body)
    ):
        raise ValueError("invalid retained Phase 2 qualification")
    return value["physical_device"]


def _terminal(args, destination):
    root = args.phase2_root / "cells" / destination.cell_id
    if not (root / "terminal.json").exists():
        return None
    snapshot = A5Phase1Snapshot.open(root / "snapshot", destination)
    def unavailable(*_args, **_kwargs):
        return _unavailable()
    runner = Phase2LiveRunner(
        phase2_root=args.phase2_root, snapshot=snapshot,
        dependencies=Phase2Dependencies(destination, unavailable, unavailable),
    )
    return runner.load_terminal()


def _controller(args) -> CampaignController:
    gate = load_gate_report(args.gate)
    phase1 = Phase1CellComposition(
        args.phase1_root, args.guide, gate.guide, gate,
        LiveDependencies(_unavailable, _unavailable),
    )
    if (args.phase2_guide is None) != (args.phase2_cpl_skills_revision is None):
        raise ValueError(
            "Phase 2 guide path and CPL skills revision must be supplied together"
        )
    phase2_guide = gate.guide
    if args.phase2_guide is not None:
        phase2_guide = admit_programming_guide(
            args.phase2_guide,
            cpl_skills_revision=args.phase2_cpl_skills_revision,
            catlass_revision=gate.guide.catlass_revision,
        )
    destinations = phase2_cells(
        gate.guide,
        gate.report_sha256,
        destination_guide=phase2_guide,
    )
    return CampaignController(
        root=args.controller_root,
        phase1=phase1,
        phase2_root=args.phase2_root,
        destinations=destinations,
        inventory=BZDeviceInventory(selector_wrapper=str(
            args.tla_root / "execution-profiles/bz-a5/select-device.sh"
        )),
        workers=ScreenWorkerPool(
            args.controller_root, script=Path(__file__).resolve(),
        ),
        command=_command_factory(args, destinations),
        phase1_progress=lambda cell_id: any(
            (args.phase1_root / "cells" / cell_id).glob("*.jsonl")
        ),
        snapshot=lambda destination: admit_phase1_snapshot(
            phase1, destination, args.phase2_root,
        ).root,
        qualified=lambda destination: _qualification(args, destination),
        phase2_terminal=lambda destination: _terminal(args, destination),
    )


def main(argv=None, *, sleeper=time.sleep) -> int:
    args = _parser().parse_args(argv)
    if args.command == "worker":
        return run_request(args.request)
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    controller = _controller(args)
    retry_job_ids = tuple(args.retry_job)
    while True:
        try:
            report = (
                controller.reconcile(retry_job_ids=retry_job_ids)
                if retry_job_ids else controller.reconcile()
            )
        except DeviceInventoryUnavailable:
            if not args.until_complete:
                raise
            print(json.dumps({
                "reason": "device-inventory-unavailable",
                "retry_in_seconds": args.poll_seconds,
                "status": "retrying",
            }, sort_keys=True, separators=(",", ":")), flush=True)
            sleeper(args.poll_seconds)
            continue
        retry_job_ids = ()
        print(json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":")),
              flush=True)
        if not args.until_complete or report.status in {"complete", "blocked"}:
            break
        sleeper(args.poll_seconds)
    return 2 if report.status == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
