"""Foundation CLI; execution commands are supplied by later workflow modules."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

from .contracts import CycleSpec, ProjectSpec
from .instances import create_instance, inspect_instance


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rsi", description="RSI programming workflows")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-spec", help="validate and normalise a JSON specification")
    validate.add_argument("path", type=Path)
    validate.add_argument("--kind", choices=("cycle", "project"), default="cycle")
    instance = commands.add_parser("instance", help="create or inspect an independent RSI instance")
    actions = instance.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create")
    create.add_argument("path", type=Path)
    for name in ("baseline-kdb", "baseline-skills", "skill-map"):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--repository")
    create.add_argument("--repository-revision")
    inspect = actions.add_parser("inspect")
    inspect.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
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
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"rsi: {exc}", file=sys.stderr)
        return 2
    return 0
