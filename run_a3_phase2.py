"""Argparse entrypoint for the concrete A3 Phase 2 experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks.a3kernels.live_composition import bz_live_dependencies
from benchmarks.a3kernels.phase1_evidence import canonical_bytes
from benchmarks.a3kernels.phase1_wave import Phase1Config
from benchmarks.a3kernels.phase2_composition import Phase2Composition, qualification_cells
from benchmarks.a3kernels.phase2_live import Phase2LiveRunner, phase2_cells
from benchmarks.a3kernels.phase2_memory import admit_phase1_snapshot, phase2_cell_pairs
from run_a3_phase1 import _bz_preflight


_ACK = "I_ACCEPT_A3_PHASE2_EXECUTION"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-run")
    report = sub.add_parser("report")
    report.add_argument("--state-root", type=Path, required=True)
    for name in ("preflight", "qualify", "run", "resume"):
        command = sub.add_parser(name)
        command.add_argument("--state-root", type=Path, required=True)
        command.add_argument("--phase1-root", action="append", type=Path, default=[])
        command.add_argument("--validation-wrapper", type=Path, required=True)
        command.add_argument("--cpl-remote", required=True)
        command.add_argument("--profile", choices=("bz-a3-1", "bz-a3-2"), required=True)
        command.add_argument("--remote-workspace", required=True)
        command.add_argument("--physical-device", type=int, required=True)
        command.add_argument("--embedding-cache", type=Path, default=Path("/unconfigured"))
        command.add_argument("--corpus-artifacts", type=Path, default=Path("/unconfigured"))
        command.add_argument("--knowledge-database", type=Path, default=Path("/unconfigured"))
        command.add_argument("--knowledge-manifest", type=Path, default=Path("/unconfigured"))
        if name in ("qualify", "run", "resume"):
            command.add_argument("--acknowledge-execution", choices=(_ACK,), required=True)
            command.add_argument(
                "--target-direction", default="Write a reliable tiled A3 vector-add kernel.",
            )
        if name in ("run", "resume"):
            command.add_argument("--cell-id", action="append", default=[])
    return parser


def _selected(cell_ids: list[str]) -> tuple:
    available = {cell.cell_id: cell for cell in phase2_cells()}
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("duplicate Phase 2 cell ID")
    unknown = set(cell_ids) - set(available)
    if unknown:
        raise ValueError("unknown Phase 2 cell ID: " + ", ".join(sorted(unknown)))
    return tuple(
        cell for cell in phase2_cells() if not cell_ids or cell.cell_id in cell_ids
    )


def _source_cell_root(roots: list[Path], source_id: str) -> Path:
    matches = []
    for root in map(Path.resolve, roots):
        for candidate in (root / "cells" / source_id, root / source_id, root):
            terminal = candidate / "terminal.json"
            if terminal.is_file():
                try:
                    cell_id = json.loads(terminal.read_text())["cell_id"]
                except (KeyError, json.JSONDecodeError) as exc:
                    raise ValueError("Phase 1 terminal is invalid") from exc
                if cell_id == source_id:
                    matches.append(candidate)
    unique = tuple(dict.fromkeys(matches))
    if len(unique) != 1:
        raise ValueError(f"expected exactly one Phase 1 source for {source_id}")
    return unique[0]


def _terminal(result, snapshot, profile: str) -> dict[str, object]:
    return {
        "schema": "a3-phase2-cell-terminal-v1",
        "cell_id": result.identity.cell_id,
        "snapshot_id": snapshot.snapshot_id,
        "execution_profile": profile,
        "verdict": result.verdict.value,
        "report": result.report,
        "target_attempts": result.target_attempts,
        "practice_projects": result.practice_projects,
        "termination": result.termination,
        "evidence_sha256": result.evidence_sha256,
    }


def _run(args, cells) -> tuple[dict[str, object], ...]:
    if not args.phase1_root:
        raise ValueError("execution requires at least one --phase1-root")
    config = Phase1Config(
        args.state_root, args.validation_wrapper, args.embedding_cache,
        args.corpus_artifacts, args.knowledge_database, args.knowledge_manifest,
    )
    sources = {destination.cell_id: source for source, destination in phase2_cell_pairs()}
    config.preflight(os.environ, cells=tuple(sources[cell.cell_id] for cell in cells))
    _bz_preflight(config.validation_wrapper, args.profile, args.cpl_remote)
    live = bz_live_dependencies(
        config, cpl_remote=args.cpl_remote, profile=args.profile,
        remote_workspace=args.remote_workspace,
        physical_device=args.physical_device, environ=os.environ,
    )
    composition = Phase2Composition(
        config=config, dependencies=live, execution_profile=args.profile,
    )
    records = []
    for cell in cells:
        root = args.state_root.resolve() / "cells" / cell.cell_id
        source_root = _source_cell_root(args.phase1_root, sources[cell.cell_id].cell_id)
        snapshot = admit_phase1_snapshot(source_root, root / "snapshot")
        terminal_path = root / "terminal.json"
        if terminal_path.exists():
            retained = json.loads(terminal_path.read_text())
            expected = {
                "cell_id": cell.cell_id, "snapshot_id": snapshot.snapshot_id,
                "execution_profile": args.profile,
            }
            if any(retained.get(key) != value for key, value in expected.items()):
                raise ValueError("retained Phase 2 terminal conflicts with this run")
            records.append(retained)
            continue
        result = Phase2LiveRunner(
            root=root, snapshot=snapshot,
            dependencies=composition.dependencies(cell, snapshot, root),
        ).run(args.target_direction)
        record = _terminal(result, snapshot, args.profile)
        terminal_path.parent.mkdir(parents=True, exist_ok=True)
        terminal_path.write_bytes(canonical_bytes(record) + b"\n")
        records.append(record)
    return tuple(records)


def _report(root: Path) -> dict[str, object]:
    records = [
        json.loads(path.read_text())
        for path in sorted((root / "cells").glob("*/terminal.json"))
    ] if (root / "cells").is_dir() else []
    return {
        "schema": "a3-phase2-report-v1", "completed_cells": len(records),
        "records": records,
    }


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dry-run":
        value = {"schema": "a3-phase2-dry-run-v1",
                 "cells": [cell.as_dict() for cell in phase2_cells()]}
    elif args.command == "report":
        value = _report(args.state_root)
    elif args.command == "preflight":
        config = Phase1Config(
            args.state_root, args.validation_wrapper, args.embedding_cache,
            args.corpus_artifacts, args.knowledge_database, args.knowledge_manifest,
        )
        value = {**config.preflight(os.environ), "remote_preflight": _bz_preflight(
            config.validation_wrapper, args.profile, args.cpl_remote,
        )}
    else:
        cells = qualification_cells() if args.command == "qualify" else _selected(args.cell_id)
        value = _run(args, cells)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
