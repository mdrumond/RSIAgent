#!/usr/bin/env python3
"""Run or resume exactly one admitted A5 Phase 2 treatment cell."""

from __future__ import annotations

import argparse
import json
from benchmarks.ascend.status_descriptions import describe_statuses
from pathlib import Path
import sys
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
        command.add_argument("--adaptive-review", type=Path)
        command.add_argument("--initial-memory", type=Path)
        command.add_argument("--profiling-guidance", type=Path)
        command.add_argument("--guide", type=Path)
        command.add_argument("--phase2-guide", type=Path)
        command.add_argument("--phase2-cpl-skills-revision")
        command.add_argument("--tla-root", type=Path)
        command.add_argument("--worker-config", type=Path)
        command.add_argument("--catlass-source")
        command.add_argument("--runtime-catlass-revision")
        command.add_argument("--bge-cache", type=Path)
        command.add_argument("--kdb", type=Path)
        command.add_argument("--collection", action="append")
        command.add_argument("--device", type=int)
        command.add_argument("--env-file", type=Path)
        command.add_argument(
            "--compile-probes", type=int, default=0, metavar="N",
            help=(
                "allow N host compile-only probes per semantic attempt; probes "
                "do not consume ordinary model turns or establish correctness"
            ),
        )
    return parser


def _production_factory(
    args: argparse.Namespace,
) -> tuple[A5Phase1Snapshot, Phase2Dependencies]:
    from benchmarks.ascend.execution import apply_worker_config, execution_mode
    worker = apply_worker_config({}, getattr(args, "worker_config", None)) if getattr(args, "worker_config", None) else None
    from benchmarks.ascend.worker_bindings import apply_a5_worker_arguments
    apply_a5_worker_arguments(args, worker, source="catlass_source", revision="runtime_catlass_revision")
    required = {
        "phase1_root": args.phase1_root,
        "guide": args.guide,
        "tla_root": args.tla_root,
        "catlass_source": args.catlass_source,
        "device": args.device,
    }
    if args.adaptive_review is not None:
        if args.gate is not None:
            raise ValueError("choose adaptive-review or legacy gate, not both")
        required["initial_memory"] = args.initial_memory
    else:
        required["gate"] = args.gate
    missing = [name.replace("_", "-") for name, value in required.items()
               if value is None]
    if missing:
        raise RuntimeError(
            "A5 Phase 2 production dependencies are missing: " + ", ".join(missing)
        )
    gate = load_gate_report(args.gate) if args.gate else None
    return build_phase2_production(Phase2ProductionConfig(
        worker_config=worker,
        phase1_root=args.phase1_root,
        gate=gate,
        guide_path=args.guide,
        phase2_root=args.root,
        snapshot_path=args.snapshot,
        cell_id=args.cell_id,
        tla_root=args.tla_root,
        catlass_source=args.catlass_source,
        runtime_catlass_revision=args.runtime_catlass_revision,
        device=args.device,
        bge_cache=args.bge_cache,
        kdb=args.kdb,
        collection=(args.collection[0] if len(args.collection) == 1 else tuple(args.collection))
        if args.collection else None,
        env_file=args.env_file,
        phase2_guide_path=args.phase2_guide,
        phase2_cpl_skills_revision=args.phase2_cpl_skills_revision,
        compile_probes=args.compile_probes,
        adaptive_review=args.adaptive_review,
        initial_memory=args.initial_memory,
        profiling_guidance_text=(args.profiling_guidance.read_text(encoding="utf-8")
                                 if args.profiling_guidance else None),
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
    print(json.dumps(describe_statuses(terminal), sort_keys=True, separators=(",", ":")))
    return 0


def _entry() -> int:
    try:
        return main()
    except (
        InfrastructureFailure,
        AttentionInfrastructureError,
        RuntimeUnavailableError,
    ) as exc:
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
