"""Generic native replay; preserve raw msprof units without A5 interpretation."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from a3_profile_driver import _new_report_directory, _raw_rows, _report_digest


def main(argv=None, *, runner=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("timing", "profile"))
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--warm-up", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--metric", choices=("ArithmeticUtilization", "PipeUtilization"))
    args = parser.parse_args(argv)
    root = args.candidate_dir.resolve()
    if not args.candidate_dir.is_absolute() or args.warm_up < 0 or args.launch_count < 1:
        parser.error("absolute candidate directory and valid replay dimensions required")
    build = json.loads((root / "build.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if build["kernel_name"] != args.kernel or manifest["execution_id"] != args.execution_id or manifest["source_fingerprint"] != args.source_fingerprint:
        raise ValueError("measurement identity differs from candidate manifest")
    if hashlib.sha256((root / "a3_candidate.so").read_bytes()).hexdigest() != args.library_sha256:
        raise ValueError("measurement binary differs from compiled library")
    command = (sys.executable, str(root / "host_driver.py"), "--mode",
               "benchmark" if args.mode == "timing" else "replay",
               "--warm-up", str(args.warm_up), "--launch-count", str(args.launch_count), "input.json")
    report = None
    if args.mode == "profile":
        if args.metric is None:
            parser.error("profile requires a supported metric")
        report = _new_report_directory(root, args.metric)
        command = ("msprof", f"--output={report}", f"--aic-metrics={args.metric}", *command)
    completed = runner(command, cwd=root, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout or "measurement failed")
    print(completed.stdout)
    if report is not None:
        kernels, values, timeline, columns = _raw_rows(report, args.metric, args.kernel)
        print("A3PROFILE_COMPACT=" + json.dumps(dict(
            exported_kernels=kernels, metric_values=values, timeline=timeline,
            selected_columns=columns, report_sha256=_report_digest(report))))
    print("A3PROFILE_META=" + json.dumps(dict(mode=args.mode, kernel_name=args.kernel,
        target=build["target"], language="ascend-c", runtime="native-ascend-c",
        logical_device=0, remote_report=str(report) if report else None)))


if __name__ == "__main__":
    main()
