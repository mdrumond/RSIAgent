"""Concrete BZ-A5 execution for host-owned profiling campaigns."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from typing import Callable

from benchmarks.a5kernels.profiling import (
    CaptureCommand,
    EvidenceArchive,
    EvidenceEntry,
    ProfileCapture,
    ProfileMetric,
    ProfileRequest,
    TimingCommand,
    TimingResult,
)


_TIMING_MARKER = "A5KERNEL_TIMING_US="
_REMOTE_MARKER = "MSPROF_PROFILE_REMOTE_DIR="


class BZProfileBackend:
    """Run Catlass timing and profiler replays through checked-in wrappers."""

    def __init__(
        self,
        *,
        validation_wrapper: str,
        collection_wrapper: str,
        catlass_source: str,
        evidence_directory: str,
        process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout: int = 600,
    ) -> None:
        if Path(validation_wrapper).name != "catlass-validation.sh":
            raise ValueError("validation_wrapper must identify catlass-validation.sh")
        if Path(collection_wrapper).name != "collect_profile.sh":
            raise ValueError("collection_wrapper must identify collect_profile.sh")
        if not PurePosixPath(catlass_source).is_absolute():
            raise ValueError("catlass_source must be an absolute retained BZ path")
        if timeout < 1:
            raise ValueError("timeout must be positive")
        self._validation = validation_wrapper
        self._collector = collection_wrapper
        self._catlass_source = catlass_source
        self._evidence_directory = Path(evidence_directory)
        self._run = process_runner
        self._timeout = timeout

    def time(self, command: TimingCommand) -> TimingResult:
        completed = self._call(
            self._adapter_argv(command.replay_id, "run", command.request)
        )
        values = _marked_values(completed.stdout, _TIMING_MARKER)
        if completed.returncode or len(values) != 1:
            raise RuntimeError(
                "canonical timing replay failed or returned no unique timing"
            )
        try:
            duration = float(values[0])
        except ValueError as exc:
            raise RuntimeError(
                "canonical timing replay returned an invalid duration"
            ) from exc
        return TimingResult(
            duration,
            command.request.source_fingerprint,
            command.request.execution_id,
            command.replay_id,
        )

    def capture(self, command: CaptureCommand) -> ProfileCapture:
        request = command.request
        argv = list(self._adapter_argv(command.replay_id, "profile", request))
        separator = argv.index("--")
        options = [
            "--device",
            str(request.device),
            "--metric",
            command.metric.value,
            "--warm-up",
            str(request.warm_up),
            "--launch-count",
            str(request.launch_count),
            "--experiment",
            command.replay_id,
        ]
        if command.kernel_name is not None:
            options += ["--kernel-name", command.kernel_name]
        argv[separator:separator] = options
        completed = self._call(tuple(argv))
        remote_values = _marked_values(completed.stdout, _REMOTE_MARKER)
        if completed.returncode or len(remote_values) != 1:
            raise RuntimeError(f"{command.metric.value} profiling replay failed")
        remote_tree = remote_values[0]
        if not PurePosixPath(remote_tree).is_absolute():
            raise RuntimeError("profiler returned a non-absolute retained tree")
        archive = self._collect(command, remote_tree)
        kernels, summary = _read_metric_archive(
            self._evidence_directory / command.replay_id / archive.archive_name,
            command.metric,
        )
        if command.metric is ProfileMetric.PIPE_UTILIZATION:
            # The wrapper has already required exact-name OpBasicInfo evidence for
            # this replay. Pipe rows themselves are block/sub-block keyed.
            assert command.kernel_name is not None
            kernels = (command.kernel_name,)
        return ProfileCapture(
            command.metric,
            request.source_fingerprint,
            request.execution_id,
            command.replay_id,
            kernels,
            summary,
            archive,
        )

    def _adapter_argv(
        self, replay_id: str, action: str, request: ProfileRequest
    ) -> tuple[str, ...]:
        workload = self._bound_workload_argv(request)
        if action == "run":
            workload = (
                "env",
                f"BZ_A5_PROFILE_PHYSICAL_DEVICE={request.device}",
                "A5KERNEL_EMIT_TIMING=1",
                *workload,
            )
        return (
            self._validation,
            "--profile",
            "bz-a5",
            "--operation",
            replay_id,
            action,
            "--catlass-src",
            self._catlass_source,
            "--timeout",
            str(self._timeout),
            "--",
            *workload,
        )

    def _bound_workload_argv(self, request: ProfileRequest) -> tuple[str, ...]:
        if request.plan.language != "catlass-dsl":
            raise ValueError("BZProfileBackend requires a catlass-dsl execution plan")
        if request.implementation not in {"catlass", "catlass-dsl", "dsl"}:
            raise ValueError("BZProfileBackend supports only Catlass DSL requests")
        provenance = dict(request.plan.runtime_provenance)
        required = (
            "catlass_revision",
            "catlass_source",
            "manifest_sha256",
            "bridge_sha256",
            "ascendnpu_ir_gitlink",
            "ascendnpu_ir_install_commit",
            "cann_version",
        )
        if any(not provenance.get(name) for name in required):
            raise ValueError("profiling request lacks complete Catlass runtime provenance")
        if provenance["catlass_source"] != self._catlass_source:
            raise ValueError("profiling request targets a different Catlass source")
        remote_directory = (
            f".a5kernels/{request.execution_id}/{request.attempt_id}"
        )
        staged_paths = {source.relative_path for source in request.plan.files}
        # Correctness execution stages input.json alongside the declared plan files.
        staged_paths.add("input.json")
        workload = tuple(
            f"{remote_directory}/{argument}"
            if argument in staged_paths
            else argument
            for argument in request.workload_argv
        )
        return (
            *workload,
            provenance["catlass_revision"],
            provenance["manifest_sha256"],
            provenance["bridge_sha256"],
            provenance["ascendnpu_ir_gitlink"],
            provenance["ascendnpu_ir_install_commit"],
            provenance["cann_version"],
        )

    def _collect(
        self, command: CaptureCommand, remote_tree: str
    ) -> EvidenceArchive:
        destination = self._evidence_directory / command.replay_id
        destination.mkdir(parents=True, exist_ok=True)
        request = command.request
        implementation = "dsl"
        argv = [
            self._collector,
            "--implementation",
            implementation,
            "--remote-root",
            remote_tree,
            "--output",
            str(destination),
            "--mode",
            "summary",
            "--operation",
            f"{command.replay_id}-collect",
        ]
        # BasicInfo rows carry an operator-name column and can be curated by exact
        # name. PipeUtilization rows are keyed by block/sub-block instead; the
        # profiler replay itself is already exact-name filtered, so asking the
        # curator to filter those rows would discard the requested metric export.
        if command.metric is ProfileMetric.BASIC_INFO:
            argv += ["--kernel-name", request.expected_kernel]
        if implementation == "dsl":
            argv += ["--catlass-src", self._catlass_source]
        completed = self._call(tuple(argv))
        archive_path = destination / "ascend-profile-summary.tar.gz"
        if completed.returncode or not archive_path.is_file():
            raise RuntimeError("compact profile evidence collection failed")
        data = archive_path.read_bytes()
        return EvidenceArchive(
            archive_path.name,
            hashlib.sha256(data).hexdigest(),
            len(data),
            remote_tree,
            True,
            _archive_entries(data),
        )

    def _call(self, argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        return self._run(argv, text=True, capture_output=True, check=False)


def _marked_values(output: str, marker: str) -> list[str]:
    return [
        line.removeprefix(marker)
        for line in output.splitlines()
        if line.startswith(marker)
    ]


def _archive_entries(data: bytes) -> tuple[EvidenceEntry, ...]:
    entries = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != member.name:
                raise ValueError("profile archive contains a non-normalized path")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("profile archive contains an unreadable entry")
            contents = stream.read()
            entries.append(
                EvidenceEntry(
                    member.name,
                    hashlib.sha256(contents).hexdigest(),
                    len(contents),
                )
            )
    return tuple(entries)


def _read_metric_archive(
    path: Path, metric: ProfileMetric
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    kernels: list[str] = []
    summary: list[tuple[str, str]] = []
    expected_report = (
        "OpBasicInfo" if metric is ProfileMetric.BASIC_INFO else metric.value
    )
    report_seen = False
    with tarfile.open(path, mode="r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile() or not member.name.endswith(".csv.gz"):
                continue
            if not Path(member.name).name.startswith(expected_report):
                continue
            report_seen = True
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("profile archive contains an unreadable metric")
            with gzip.GzipFile(fileobj=stream) as compressed:
                rows = csv.DictReader(
                    io.TextIOWrapper(compressed, encoding="utf-8-sig")
                )
                for row in rows:
                    name = next(
                        (
                            row.get(key)
                            for key in (
                                "Op Name",
                                "OpName",
                                "Kernel Name",
                                "kernel_name",
                            )
                            if row.get(key)
                        ),
                        None,
                    )
                    if name and name not in kernels:
                        kernels.append(name)
                    summary.extend(
                        (f"{Path(member.name).name}:{key}", value)
                        for key, value in sorted(row.items())
                        if value is not None and value.strip()
                    )
    if not report_seen:
        raise ValueError(f"compact evidence is missing {expected_report} metrics")
    return tuple(kernels), tuple(summary)
