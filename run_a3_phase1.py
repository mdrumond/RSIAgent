"""CLI for deterministic A3 Phase 1 planning and retained reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.a3kernels.phase1_wave import (
    Phase1Config,
    Phase1Wave,
    full_dry_run,
    select_foundation_cells,
)
from benchmarks.a3kernels.live_composition import LiveComposition, managed_live_dependencies


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-run")
    report = sub.add_parser("report")
    report.add_argument("--state-root", type=Path, required=True)
    for name in ("preflight", "run", "resume"):
        command = sub.add_parser(name)
        command.add_argument("--state-root", type=Path, required=True)
        command.add_argument("--validation-wrapper", type=Path, required=True)
        command.add_argument("--embedding-cache", type=Path, required=True)
        command.add_argument("--corpus-artifacts", type=Path, required=True)
        command.add_argument("--knowledge-database", type=Path, required=True)
        command.add_argument("--knowledge-manifest", type=Path, required=True)
        if name in ("run", "resume"):
            command.add_argument("--remote-client", required=True)
            command.add_argument("--server", required=True)
            command.add_argument("--remote", required=True)
            command.add_argument("--remote-workspace", required=True)
            command.add_argument("--physical-device", type=int, required=True)
            command.add_argument(
                "--cell-id", action="append", default=[],
                help="run only this foundation cell; repeat to select more",
            )
            command.add_argument("--shard-count", type=int)
            command.add_argument("--shard-index", type=int)
    return parser


def _config(args) -> Phase1Config:
    return Phase1Config(
        args.state_root, args.validation_wrapper, args.embedding_cache,
        args.corpus_artifacts, args.knowledge_database, args.knowledge_manifest,
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dry-run":
        value = full_dry_run()
    elif args.command == "report":
        cfg = Phase1Config(args.state_root, *(Path("/unconfigured") for _ in range(5)))
        value = Phase1Wave(cfg, lambda *_: {}).report()
    elif args.command == "preflight":
        import os
        value = _config(args).preflight(os.environ)
    else:
        import os
        cells = select_foundation_cells(
            cell_ids=args.cell_id,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
        cfg = _config(args)
        cfg.preflight(os.environ)
        dependencies = managed_live_dependencies(
            cfg, client=args.remote_client, server=args.server,
            remote=args.remote, remote_workspace=args.remote_workspace,
            physical_device=args.physical_device, environ=os.environ,
        )
        wave = Phase1Wave(
            cfg,
            LiveComposition(cfg, dependencies).execute,
            cells=cells,
        )
        value = wave.run() if args.command == "run" else wave.resume()
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
