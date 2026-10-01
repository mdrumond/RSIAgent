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
from typing import Callable


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
ReportDirectoryFactory = Callable[[Path, str], Path]
_OUTPUT = "A3KERNEL_OUTPUT="
_KERNEL_COLUMNS = {"op name", "op_name", "opname", "kernel name", "kernel_name"}
_TIMELINE_WORDS = ("duration", "start", "end", "timestamp")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("timing", "profile"):
        command = commands.add_parser(name)
        command.add_argument("--candidate-dir", required=True, type=Path)
        command.add_argument("--logical-device", required=True, type=int)
        command.add_argument("--length", required=True, type=int)
        command.add_argument("--block-count", required=True, type=int)
        command.add_argument("--warm-up", type=int, default=0)
        command.add_argument("--launch-count", type=int, default=1)
        if name == "profile":
            command.add_argument(
                "--metric",
                choices=("Basic", "ArithmeticUtilization", "PipeUtilization"),
                required=True,
            )
            command.add_argument("--kernel", choices=("vector_add",), required=True)
    return parser


def _validate(args: argparse.Namespace) -> tuple[Path, str]:
    root = args.candidate_dir.resolve()
    if not args.candidate_dir.is_absolute() or args.logical_device != 0:
        raise ValueError("candidate-dir must be absolute and logical-device must be 0")
    if not 1 <= args.length <= 4096 or not 1 <= args.block_count <= 32:
        raise ValueError("length must be in [1,4096] and block-count in [1,32]")
    if args.warm_up < 0 or args.launch_count < 1:
        raise ValueError("warm-up must be non-negative and launch-count must be positive")
    for name in ("host_driver.py", "input.json", "a3_candidate.so"):
        if not (root / name).is_file():
            raise RuntimeError(f"staged candidate is missing {name}")
    payload = json.loads((root / "input.json").read_text(encoding="utf-8"))
    expected = {"input_a", "input_b", "logical_length", "padded_length", "block_count"}
    if (
        not isinstance(payload, dict) or set(payload) != expected
        or payload["logical_length"] != args.length
        or len(payload["input_a"]) != payload["padded_length"]
        or len(payload["input_b"]) != payload["padded_length"]
    ):
        raise ValueError("staged input dimensions do not match --length")
    payload["block_count"] = args.block_count
    profile_input = ".a3-profile-input.json"
    (root / profile_input).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return root, profile_input


def _host_run(
    root: Path, input_name: str, run: ProcessRunner, *, mode: str = "run",
    warm_up: int = 0, launch_count: int = 1,
) -> subprocess.CompletedProcess[str]:
    argv = (
        sys.executable, "host_driver.py", "--mode", mode,
        "--warm-up", str(warm_up), "--launch-count", str(launch_count), input_name,
    )
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
    payload = json.loads((root / input_name).read_text(encoding="utf-8"))
    expected = [a + b for a, b in zip(payload["input_a"], payload["input_b"])]
    if (
        not isinstance(values, list) or len(values) != len(expected)
        or any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values)
    ):
        raise RuntimeError("candidate output must be a finite numeric array of the staged length")
    if any(abs(float(value) - float(want)) > 1e-5 for value, want in zip(values, expected)):
        raise RuntimeError("candidate output failed host verification")
    return completed


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


def _raw_rows(
    root: Path, metric: str
) -> tuple[list[str], list[list[object]], list[list[object]], list[str]]:
    kernels: set[str] = set()
    values: list[list[object]] = []
    timeline: list[list[object]] = []
    selected_columns: set[str] = set()
    for path in sorted(root.rglob("*.csv")):
        relative = path.relative_to(root).as_posix()
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            if not any(column.strip().lower() in _KERNEL_COLUMNS for column in columns):
                continue
            selected_columns.update(f"{relative}:{column}" for column in columns)
            for row_number, row in enumerate(reader):
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
    if not selected_columns:
        raise RuntimeError(
            f"msprof report contains no kernel-schema CSV table for {metric}"
        )
    return sorted(kernels), values, timeline, sorted(selected_columns)


def _replay_argv(root: Path, input_name: str, args: argparse.Namespace) -> tuple[str, ...]:
    return (
        sys.executable, str((root / "host_driver.py").resolve()),
        "--mode", "replay", "--warm-up", "0",
        "--launch-count", str(args.launch_count), input_name,
    )


def _new_report_directory(root: Path, metric: str) -> Path:
    parent = root / ".a3-msprof"
    parent.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{metric.lower()}-", dir=parent))


def main(
    argv: list[str] | None = None, *, process_runner: ProcessRunner = subprocess.run,
    report_directory_factory: ReportDirectoryFactory = _new_report_directory,
) -> int:
    args = _parser().parse_args(argv)
    root, input_name = _validate(args)
    if args.mode == "timing":
        completed = _host_run(
            root, input_name, process_runner, mode="benchmark",
            warm_up=args.warm_up, launch_count=args.launch_count,
        )
        raw_samples = [
            line.removeprefix("A3INNER_TIMING_US=") for line in completed.stdout.splitlines()
            if line.startswith("A3INNER_TIMING_US=")
        ]
        if len(raw_samples) != args.launch_count:
            raise RuntimeError("benchmark host emitted the wrong timing sample count")
        try:
            samples = [float(sample) for sample in raw_samples]
        except ValueError as exc:
            raise RuntimeError("benchmark host emitted a non-numeric timing sample") from exc
        if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
            raise RuntimeError("benchmark host emitted a non-positive timing sample")
        for sample in samples:
            print(f"A3TIMING_US={sample:.6f}")
        print("A3PROFILE_META=" + json.dumps(_metadata("timing", None), sort_keys=True))
        return 0

    # Verify once without instrumentation before the separately profiled replay.
    _host_run(
        root, input_name, process_runner, mode="replay",
        warm_up=args.warm_up, launch_count=1,
    )
    report = report_directory_factory(root, args.metric).resolve()
    application = shlex.join(_replay_argv(root, input_name, args))
    command = (
        "msprof", f"--output={report}", f"--aic-metrics={args.metric}",
        f"--application={application}",
    )
    completed = process_runner(command, cwd=root, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout or "msprof failed").strip())
    kernels, values, timeline, selected_columns = _raw_rows(report, args.metric)
    if args.kernel not in kernels:
        raise RuntimeError("msprof raw table does not contain the expected kernel")
    compact = {
        "exported_kernels": kernels, "metric_values": values, "timeline": timeline,
        "selected_columns": selected_columns,
        "report_sha256": _report_digest(report),
    }
    print("A3PROFILE_COMPACT=" + json.dumps(compact, sort_keys=True, separators=(",", ":")))
    print("A3PROFILE_META=" + json.dumps(_metadata("profile", report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
