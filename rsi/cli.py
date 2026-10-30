"""Foundation CLI; execution commands are supplied by later workflow modules."""
import argparse
import json
from pathlib import Path
import sys

from .contracts import CycleSpec, ProjectSpec


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rsi", description="RSI programming workflows")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-spec", help="validate and normalise a JSON specification")
    validate.add_argument("path", type=Path)
    validate.add_argument("--kind", choices=("cycle", "project"), default="cycle")
    args = parser.parse_args(argv)
    try:
        model = CycleSpec if args.kind == "cycle" else ProjectSpec
        spec = model.from_dict(json.loads(args.path.read_text(encoding="utf-8")))
        print(json.dumps(spec.to_dict(), indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        print(f"rsi: {exc}", file=sys.stderr)
        return 2
    return 0
