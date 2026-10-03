"""CLI for deterministic A3 Phase 1 planning and retained reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from benchmarks.a3kernels.artifact_prepare import prepare_phase1_artifacts
from benchmarks.a3kernels.phase1_wave import (
    Phase1Config,
    Phase1Wave,
    SMOKE_PROPOSALS,
    TRIAL_RELIABILITY_SMOKE_PROPOSALS,
    SmokeWave,
    full_dry_run,
    registered_credential_envs,
    select_foundation_cells,
)
from benchmarks.a3kernels.live_composition import LiveComposition, bz_live_dependencies


_SUPPORTED_PROVIDER_CREDENTIAL_ENVS = frozenset({
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-run")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--embedding-cache", type=Path, required=True)
    prepare.add_argument("--corpus-artifacts", type=Path, required=True)
    prepare.add_argument("--knowledge-database", type=Path, required=True)
    prepare.add_argument("--knowledge-manifest", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--state-root", type=Path, required=True)
    for name in (
        "preflight", "run", "resume", "smoke-preflight", "smoke", "smoke-resume",
    ):
        command = sub.add_parser(name)
        command.add_argument("--state-root", type=Path, required=True)
        command.add_argument("--validation-wrapper", type=Path, required=True)
        command.add_argument("--embedding-cache", type=Path, default=Path("/unconfigured"))
        command.add_argument("--corpus-artifacts", type=Path, default=Path("/unconfigured"))
        command.add_argument("--knowledge-database", type=Path, default=Path("/unconfigured"))
        command.add_argument("--knowledge-manifest", type=Path, default=Path("/unconfigured"))
        if name in (
            "preflight", "run", "resume", "smoke-preflight", "smoke", "smoke-resume",
        ):
            command.add_argument(
                "--profile", choices=("bz-a3-1", "bz-a3-2"), required=True,
            )
            command.add_argument("--cpl-remote", required=True)
        if name in ("run", "resume", "smoke", "smoke-resume"):
            command.add_argument("--remote-workspace", required=True)
            command.add_argument("--physical-device", type=int, required=True)
            command.add_argument(
                "--cell-id", action="append", default=[],
                help="run only this foundation cell; repeat to select more",
            )
            command.add_argument("--shard-count", type=int)
            command.add_argument("--shard-index", type=int)
        if name in ("smoke-preflight", "smoke", "smoke-resume"):
            command.add_argument(
                "--smoke-project",
                choices=("baseline", "runtime-recovery"),
                default="baseline",
            )
    smoke_report = sub.add_parser("smoke-report")
    smoke_report.add_argument("--state-root", type=Path, required=True)
    smoke_report.add_argument(
        "--smoke-project", choices=("baseline", "runtime-recovery"),
        default="baseline",
    )
    return parser


def _config(args) -> Phase1Config:
    return Phase1Config(
        args.state_root, args.validation_wrapper, args.embedding_cache,
        args.corpus_artifacts, args.knowledge_database, args.knowledge_manifest,
    )


def _smoke_proposals(name: str):
    return (
        TRIAL_RELIABILITY_SMOKE_PROPOSALS
        if name == "runtime-recovery" else SMOKE_PROPOSALS
    )


def _marker(stdout: str, prefix: str) -> list[str]:
    return [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]


def _bz_preflight(
    validation_wrapper: Path, profile: str, cpl_remote: str,
) -> dict[str, str]:
    remote = Path(cpl_remote)
    if not remote.is_file() or not os.access(remote, os.X_OK) or remote.name != "cpl-remote":
        raise ValueError("cpl_remote must be the user-wide executable cpl-remote")
    credential_envs = (
        _SUPPORTED_PROVIDER_CREDENTIAL_ENVS | set(registered_credential_envs())
    )
    environment = {
        key: value for key, value in os.environ.items()
        if key not in credential_envs
    }
    environment["CPL_REMOTE"] = str(remote)
    argv = [str(validation_wrapper), "--profile", profile, "preflight"]
    completed = subprocess.run(
        argv, capture_output=True, text=True, check=False,
        timeout=120, env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"BZ-A3 adapter preflight failed with exit {completed.returncode}"
        )
    expected = {
        "CATLASS_VALIDATION_PROFILE=": profile,
        "CATLASS_VALIDATION_OPERATION=": "-",
        "CATLASS_VALIDATION_STATE=": "completed",
        "CATLASS_VALIDATION_EXIT=": "0",
    }
    if any(_marker(completed.stdout, prefix) != [value] for prefix, value in expected.items()):
        raise RuntimeError("BZ-A3 adapter preflight returned invalid terminal markers")
    if _marker(completed.stdout, "CATLASS_VALIDATION_HANDLE="):
        raise RuntimeError("BZ-A3 adapter preflight returned invalid terminal markers")
    return {"profile": profile, "state": "completed"}


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dry-run":
        value = full_dry_run()
    elif args.command == "prepare":
        value = prepare_phase1_artifacts(
            embedding_cache=args.embedding_cache,
            corpus_artifacts=args.corpus_artifacts,
            knowledge_database=args.knowledge_database,
            knowledge_manifest=args.knowledge_manifest,
        )
    elif args.command == "report":
        cfg = Phase1Config(args.state_root, *(Path("/unconfigured") for _ in range(5)))
        value = Phase1Wave(cfg, lambda *_: {}).report()
    elif args.command == "smoke-report":
        cfg = Phase1Config(args.state_root, *(Path("/unconfigured") for _ in range(5)))
        proposals = _smoke_proposals(args.smoke_project)
        value = SmokeWave(
            cfg, lambda *_: {}, execution_profile="bz-a3-1",
            proposals=proposals,
        ).report()
    elif args.command in ("preflight", "smoke-preflight"):
        cfg = _config(args)
        local = (
            cfg.smoke_preflight(
                os.environ, proposals=_smoke_proposals(args.smoke_project)
            )
            if args.command == "smoke-preflight"
            else cfg.preflight(os.environ)
        )
        value = {
            **local,
            "remote_preflight": _bz_preflight(
                cfg.validation_wrapper, args.profile, args.cpl_remote,
            ),
        }
    else:
        cells = select_foundation_cells(
            cell_ids=args.cell_id,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
        cfg = _config(args)
        smoke = args.command in ("smoke", "smoke-resume")
        proposals = _smoke_proposals(args.smoke_project) if smoke else None
        local_preflight = (
            cfg.smoke_preflight(
                os.environ, cells=cells, proposals=proposals
            )
            if smoke else cfg.preflight(os.environ, cells=cells)
        )
        _bz_preflight(cfg.validation_wrapper, args.profile, args.cpl_remote)
        dependencies = bz_live_dependencies(
            cfg, cpl_remote=args.cpl_remote, profile=args.profile,
            remote_workspace=args.remote_workspace,
            physical_device=args.physical_device, environ=os.environ,
        )
        if smoke:
            assert proposals is not None
            wave = SmokeWave(
                cfg,
                LiveComposition(
                    cfg, dependencies, proposals=proposals,
                    lineage_prefix="smoke",
                ).execute,
                cells=cells,
                execution_profile=args.profile,
                knowledge_identity=local_preflight["knowledge_identity"],
                proposals=proposals,
            )
        else:
            wave = Phase1Wave(
                cfg,
                LiveComposition(cfg, dependencies).execute,
                cells=cells,
                execution_profile=args.profile,
            )
        value = wave.run() if args.command in ("run", "smoke") else wave.resume()
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
