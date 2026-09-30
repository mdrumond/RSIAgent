"""CLI for deterministic A3 Phase 1 planning and retained reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.a3kernels.phase1_wave import Phase1Config, Phase1Wave, full_dry_run


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
        raise SystemExit(
            "live run/resume composition is intentionally deferred; use the injected Phase1Wave API"
        )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
