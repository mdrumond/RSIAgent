"""Checked A3 timing and raw-msprof driver staged with each candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Callable


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
ReportDirectoryFactory = Callable[[Path, str], Path]
_OUTPUT = "A3KERNEL_OUTPUT="
_KERNEL_COLUMNS = {"op name", "op_name", "opname", "kernel name", "kernel_name"}
_TIMELINE_WORDS = ("duration", "start", "end", "timestamp")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("timing", "profile", "replay"):
        command = commands.add_parser(name)
        command.add_argument("--candidate-dir", required=True, type=Path)
        command.add_argument("--logical-device", required=True, type=int)
        command.add_argument("--length", required=True, type=int)
        command.add_argument("--block-count", required=True, type=int)
        command.add_argument("--warm-up", type=int, default=0)
        command.add_argument("--launch-count", type=int, default=1)
        if name == "profile":
            command.add_argument("--metric", choices=("Basic", "PipeUtilization"), required=True)
            command.add_argument("--kernel", choices=("vector_add",), required=True)
    return parser


def _validate(args: argparse.Namespace) -> Path:
    root = args.candidate_dir.resolve()
    if not args.candidate_dir.is_absolute() or args.logical_device != 0:
        raise ValueError("candidate-dir must be absolute and logical-device must be 0")
    if not 1 <= args.length <= 4096 or args.block_count != 1:
        raise ValueError("length must be in [1,4096] and block-count must be 1")
    if args.warm_up < 0 or args.launch_count < 1:
        raise ValueError("warm-up must be non-negative and launch-count must be positive")
    for name in ("host_driver.py", "input.json", "a3_candidate.so"):
        if not (root / name).is_file():
            raise RuntimeError(f"staged candidate is missing {name}")
    payload = json.loads((root / "input.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload.get("input_a", ())) != args.length:
        raise ValueError("staged input length does not match --length")
    return root


def _launch(root: Path, length: int, run: ProcessRunner) -> None:
    argv = (sys.executable, "host_driver.py", "input.json")
    completed = run(argv, cwd=root, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout or "candidate launch failed").strip())
    records = [line.removeprefix(_OUTPUT) for line in completed.stdout.splitlines()
               if line.startswith(_OUTPUT)]
    if len(records) != 1:
        raise RuntimeError("candidate launch must emit one A3KERNEL_OUTPUT marker")
    try:
        values = json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("candidate output is not JSON") from exc
    if (
        not isinstance(values, list) or len(values) != length
        or any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values)
    ):
        raise RuntimeError("candidate output must be a finite numeric array of the staged length")


def _metadata(mode: str, report: Path | None) -> dict[str, object]:
    return {
        "language": "ascend-c", "logical_device": 0, "mode": mode,
        "remote_report": str(report) if report else None,
        "runtime": "native-ascend-c", "target": "Ascend910B4",
    }


def _report_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("msprof produced no retained report files")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _raw_rows(root: Path, metric: str) -> tuple[list[str], list[list[object]], list[list[object]]]:
    kernels: set[str] = set()
    values: list[list[object]] = []
    timeline: list[list[object]] = []
    metric_key = "".join(character for character in metric.lower() if character.isalnum())
    tables = sorted(
        path for path in root.rglob("*.csv")
        if metric_key in "".join(character for character in path.name.lower() if character.isalnum())
    )
    if not tables:
        raise RuntimeError(f"msprof report contains no {metric} CSV table")
    for path in tables:
        relative = path.relative_to(root).as_posix()
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row_number, row in enumerate(csv.DictReader(stream)):
                for column, raw in row.items():
                    if column.strip().lower() in _KERNEL_COLUMNS and raw:
                        kernels.add(raw.strip())
                    try:
                        number = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(number):
                        continue
                    item = [f"{relative}:{column}:{row_number}", number]
                    (timeline if any(word in column.lower() for word in _TIMELINE_WORDS)
                     else values).append(item)
    return sorted(kernels), values, timeline


def _replay_argv(root: Path, args: argparse.Namespace) -> tuple[str, ...]:
    return (
        sys.executable, str((root / "a3_profile_driver.py").resolve()), "replay",
        "--candidate-dir", str(root), "--logical-device", "0",
        "--length", str(args.length), "--block-count", "1",
        "--warm-up", "0", "--launch-count", str(args.launch_count),
    )


def _new_report_directory(root: Path, metric: str) -> Path:
    parent = root / ".a3-msprof"
    parent.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{metric.lower()}-", dir=parent))


def main(
    argv: list[str] | None = None, *, process_runner: ProcessRunner = subprocess.run,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    report_directory_factory: ReportDirectoryFactory = _new_report_directory,
) -> int:
    args = _parser().parse_args(argv)
    root = _validate(args)
    if args.mode == "replay":
        for _ in range(args.launch_count):
            _launch(root, args.length, process_runner)
        return 0

    for _ in range(args.warm_up):
        _launch(root, args.length, process_runner)
    if args.mode == "timing":
        for _ in range(args.launch_count):
            start = clock_ns()
            _launch(root, args.length, process_runner)
            elapsed = (clock_ns() - start) / 1000.0
            if elapsed <= 0:
                raise RuntimeError("timing clock returned a non-positive sample")
            print(f"A3TIMING_US={elapsed:.6f}")
        print("A3PROFILE_META=" + json.dumps(_metadata("timing", None), sort_keys=True))
        return 0

    # Verify once without instrumentation before the separately profiled replay.
    _launch(root, args.length, process_runner)
    report = report_directory_factory(root, args.metric).resolve()
    application = shlex.join(_replay_argv(root, args))
    command = (
        "msprof", f"--output={report}", f"--aic-metrics={args.metric}",
        f"--application={application}",
    )
    completed = process_runner(command, cwd=root, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout or "msprof failed").strip())
    kernels, values, timeline = _raw_rows(report, args.metric)
    if args.kernel not in kernels:
        raise RuntimeError("msprof raw table does not contain the expected kernel")
    compact = {
        "exported_kernels": kernels, "metric_values": values, "timeline": timeline,
        "report_sha256": _report_digest(report),
    }
    print("A3PROFILE_COMPACT=" + json.dumps(compact, sort_keys=True, separators=(",", ":")))
    print("A3PROFILE_META=" + json.dumps(_metadata("profile", report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
