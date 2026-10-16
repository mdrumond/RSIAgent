"""Validate a saved or published candidate on A2 without making any model calls."""

import argparse
import hashlib
import json
from pathlib import Path
import uuid

from .experiment import LOCAL
from .remote import Validator
from .tasks import contract_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run", help="existing run directory name")
    inputs.add_argument("--source", type=Path, help="published kernel to validate without provider calls")
    parser.add_argument("--task", choices=("matmul-add", "vector-add"), default="matmul-add", help="task for --source; saved runs retain their recorded task")
    parser.add_argument("--expected-sha256", help="optional digest assertion for --source")
    parser.add_argument("--connection", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.source:
        if args.source.is_symlink() or not args.source.is_file() or args.source.stat().st_size > 65536:
            parser.error("source must be a regular file of at most 64 KiB")
        source = args.source.read_bytes()
        if args.expected_sha256 and hashlib.sha256(source).hexdigest() != args.expected_sha256:
            parser.error("source digest differs from expected SHA-256")
        task = args.task
        destination = LOCAL / "validations" / uuid.uuid4().hex[:12]
    else:
        if args.expected_sha256:
            parser.error("expected-sha256 applies only to --source")
        runs = (LOCAL / "runs").resolve()
        run = (runs / args.run).resolve()
        if run.parent != runs or not run.is_dir():
            raise ValueError("run must name an existing smoke run")
        original = json.loads((run / "result.json").read_text())
        task = original.get("task", "vector-add")
        if original.get("task_contract_sha256", contract_digest(task)) != contract_digest(task):
            raise ValueError("saved task contract differs from the registered validator")
        source = (run / "kernel.cpp").read_bytes()
        if hashlib.sha256(source).hexdigest() != original["final_source_sha256"]:
            raise ValueError("candidate has changed since the recorded smoke attempt")
        destination = run / ("board-validation-" + uuid.uuid4().hex[:12])
    destination.mkdir(parents=True)
    (destination / "kernel.cpp").write_bytes(source)
    result = Validator(destination, json.loads(args.connection.read_text()), task=task).validate()
    print(json.dumps({"run": args.run, "status": result["status"], "passed": result["passed"],
                      "source_sha256": result["source_sha256"]}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
