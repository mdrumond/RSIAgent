"""Foundation CLI; execution commands are supplied by later workflow modules."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

from .configuration import list_folders, set_configuration, show_configuration
from .contracts import CycleSpec, ProjectSpec
from .instances import create_instance, inspect_instance
from .job_cli import register as register_jobs, execute as execute_jobs
from .cli_evidence import register as register_evidence
from .cli_skills import register as register_skills, execute as execute_skills
from .runtime_cli import register as register_runtime


def register_configuration_commands(commands):
    """Keep input editing registration separate from future execution commands."""
    config = commands.add_parser("config", help="read or revise instance inputs")
    actions = config.add_subparsers(dest="action", required=True)
    actions.add_parser("show")
    update = actions.add_parser("set")
    update.add_argument("--file", type=Path, required=True)
    update.add_argument("--expect-revision")
    folders = commands.add_parser("folders", help="browse allowed node folders")
    actions = folders.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("list")
    listing.add_argument("path", nargs="?", type=Path)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rsi", description="RSI programming workflows")
    parser.add_argument("--instance", type=Path, help="independent RSI instance directory")
    commands = parser.add_subparsers(dest="command", required=True)
    register_configuration_commands(commands)
    register_jobs(commands)
    register_evidence(commands)
    register_skills(commands)
    commands.add_parser("capabilities", help="discover supported instance interfaces")
    validate = commands.add_parser("validate-spec", help="validate and normalise a JSON specification")
    validate.add_argument("path", type=Path)
    validate.add_argument("--kind", choices=("cycle", "project"), default="cycle")
    instance = commands.add_parser("instance", help="create or inspect an independent RSI instance")
    actions = instance.add_subparsers(dest="action", required=True)
    create = actions.add_parser(
        "create", description="Snapshot committed HEAD or the installed release. "
        "Requires Linux renameat2 for atomic publication; does not start a dashboard.")
    create.add_argument("path", type=Path)
    for name in ("baseline-kdb", "baseline-skills", "skill-map"):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--repository")
    create.add_argument("--repository-revision")
    inspect = actions.add_parser("inspect")
    inspect.add_argument("path", type=Path)
    register_runtime(commands, actions)
    args = parser.parse_args(argv)
    try:
        if args.instance is not None:
            from .pinned_cli import dispatch
            result = dispatch(args.instance, list(sys.argv[1:] if argv is None else argv))
            if result is not None:
                return result
        if hasattr(args, "runtime_handler"):
            print(json.dumps(args.runtime_handler(args), indent=2, allow_nan=False))
            return 0
        if args.command == "capabilities":
            print(json.dumps(dict(manual_tuning=True, skill_content_editing=True,
                                  skill_map_editing=True, job_accounting=True,
                                  project_waves=True, wave_reconciliation=True, harness_recovery=True)))
            return 0
        if args.command == "skills":
            print(json.dumps(execute_skills(args), indent=2, allow_nan=False))
            return 0
        if hasattr(args, "evidence_handler"):
            print(json.dumps(args.evidence_handler(args), indent=2, allow_nan=False))
            return 0
        if args.command in ("phase1", "phase2", "inference", "job"):
            print(json.dumps(execute_jobs(args), indent=2, allow_nan=False))
            return 0
        if args.command in ("config", "folders"):
            if args.command == "folders":
                result = list_folders(args.instance, args.path)
            elif args.action == "show":
                result = show_configuration(args.instance)
            else:
                result = set_configuration(
                    args.instance, json.loads(args.file.read_text(encoding="utf-8")),
                    expect_revision=args.expect_revision)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        if args.command == "instance":
            if args.action == "create":
                result = create_instance(
                    args.path, baseline_kdb=args.baseline_kdb,
                    baseline_skills=args.baseline_skills, skill_map=args.skill_map,
                    repository=args.repository, repository_revision=args.repository_revision)
            else:
                result = inspect_instance(args.path)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        model = CycleSpec if args.kind == "cycle" else ProjectSpec
        spec = model.from_dict(json.loads(args.path.read_text(encoding="utf-8")))
        print(json.dumps(spec.to_dict(), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"rsi: {exc}", file=sys.stderr)
        return 2
    return 0
