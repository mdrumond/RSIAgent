"""Stage one native attention case or host-grade its retained runtime stdout.

The source is agent-provided. Neither subcommand launches remote work or builds
a toolchain. Grade output separately; never put the reference in a device bundle.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from benchmarks.ascend.attention import (
    ATTENTION_CASES, AttentionReceipt, _grade, attention_reference, build_execution,
)
from .attention import build_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument("--stage", type=Path, required=True)
    stage.add_argument("--target", required=True)
    stage.add_argument("--arch", required=True)
    stage.add_argument("--profile", choices=("bz-a3-1", "bz-a3-2", "gz-a3"), default="bz-a3-1")
    grade = commands.add_parser("grade")
    grade.add_argument("--stdout", type=Path, required=True)
    grade.add_argument("--exit-code", type=int, default=0)
    for command in (stage, grade):
        command.add_argument("--case", choices=tuple(c.case_id for c in ATTENTION_CASES), required=True)
    args = parser.parse_args(argv)
    case = next(c for c in ATTENTION_CASES if c.case_id == args.case)
    execution = build_execution(case)
    if args.command == "stage":
        plan = build_plan(args.source.read_text(), execution, request_id="attention-smoke",
                          attempt_id=args.case, execution_profile=args.profile,
                          target=args.target, arch=args.arch)
        args.stage.mkdir(parents=True, exist_ok=False)
        for item in plan.files:
            (args.stage / item.relative_path).write_text(item.content)
        (args.stage / "input.json").write_text(json.dumps(plan.input_payload))
        print(json.dumps({"execution_id": plan.execution_id, "case_id": case.case_id}))
        return 0
    stdout = args.stdout.read_text()
    rows = [line.removeprefix("A3KERNEL_OUTPUT=") for line in stdout.splitlines()
            if line.startswith("A3KERNEL_OUTPUT=")]
    output = tuple(json.loads(rows[0])) if len(rows) == 1 else ()
    evidence = _grade(case, AttentionReceipt(args.exit_code, output, stdout=stdout),
                      attention_reference(execution))
    print(json.dumps(asdict(evidence)))
    return 0 if evidence.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
