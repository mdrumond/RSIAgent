"""Checked A3 timing and raw-msprof driver staged with each candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
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
        command.add_argument("--execution-id", required=True)
        command.add_argument("--source-fingerprint", required=True)
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


def _validate_candidate_identity(
    root: Path, execution_id: str, source_fingerprint: str
) -> None:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("staged candidate manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("execution_id") != execution_id:
        raise ValueError("staged candidate has a different execution identity")
    if manifest.get("source_fingerprint") != source_fingerprint:
        raise ValueError("staged candidate has a different source identity")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("staged candidate manifest has no file digests")
    for name, expected in files.items():
        path = root / name if isinstance(name, str) else root
        if (
            not isinstance(expected, str) or len(expected) != 64
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("staged file digest does not match candidate manifest")


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
    _validate_candidate_identity(root, args.execution_id, args.source_fingerprint)
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
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    digest = hashlib.sha256(data.encode()).hexdigest()
    profile_input = f".a3-profile-input-{digest}.json"
    destination = root / profile_input
    with tempfile.NamedTemporaryFile(
        "w", dir=root, prefix=".a3-profile-input-", suffix=".tmp", delete=False,
        encoding="utf-8",
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if destination.read_text(encoding="utf-8") != data:
            raise RuntimeError("profile input digest conflicts with retained content") from None
    finally:
        temporary.unlink(missing_ok=True)
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
    expected = [
        a + b for a, b in zip(payload["input_a"], payload["input_b"])
    ][:payload["logical_length"]]
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
    root: Path, metric: str, expected_kernel: str
) -> tuple[list[str], list[list[object]], list[list[object]], list[str]]:
    kernels: set[str] = set()
    values: list[list[object]] = []
    timeline: list[list[object]] = []
    selected_columns: set[str] = set()
    metric_tables = 0
    normalized_metric = "".join(character.lower() for character in metric if character.isalnum())
    for path in sorted(root.rglob("*.csv")):
        relative = path.relative_to(root).as_posix()
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            kernel_columns = tuple(
                column for column in columns
                if column.strip().lower() in _KERNEL_COLUMNS
            )
            is_metric_table = (
                not kernel_columns
                and normalized_metric in "".join(
                    character.lower() for character in relative if character.isalnum()
                )
            )
            if not kernel_columns and not is_metric_table:
                continue
            if is_metric_table:
                metric_tables += 1
            for row_number, row in enumerate(reader):
                row_kernels = {
                    row[column].strip() for column in kernel_columns
                    if row.get(column) and row[column].strip()
                }
                kernels.update(row_kernels)
                if kernel_columns and expected_kernel not in row_kernels:
                    continue
                selected_columns.update(f"{relative}:{column}" for column in columns)
                for column, raw in row.items():
                    if column in kernel_columns:
                        continue
                    try:
                        number = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(number):
                        continue
                    item = [f"{relative}:{column}:{row_number}", number]
                    (timeline if any(word in column.lower() for word in _TIMELINE_WORDS)
                     else values).append(item)
    if expected_kernel not in kernels:
        raise RuntimeError(
            "msprof report does not bind the expected kernel"
        )
    if metric != "Basic" and metric_tables == 0:
        raise RuntimeError(f"msprof report contains no {metric} metric table")
    return sorted(kernels), values, timeline, sorted(selected_columns)


def _replay_argv(root: Path, input_name: str, args: argparse.Namespace) -> tuple[str, ...]:
    interpreter = Path(sys.executable)
    if not interpreter.is_absolute():
        raise RuntimeError("active Python interpreter path must be absolute")
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise RuntimeError("active Python interpreter must be an executable file")
    return (
        str(interpreter), str((root / "host_driver.py").resolve()),
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
    command = (
        "msprof", f"--output={report}", f"--aic-metrics={args.metric}",
        *_replay_argv(root, input_name, args),
    )
    completed = process_runner(command, cwd=root, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout or "msprof failed").strip())
    kernels, values, timeline, selected_columns = _raw_rows(
        report, args.metric, args.kernel
    )
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
