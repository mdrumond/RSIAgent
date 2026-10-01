"""Timing and compact raw-msprof evidence through neutral BZ-A3 profiles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Callable, Mapping

from .phase1_protocol import VerifiedResult
from .profiling_gz import GZA3ProfilingBackend


_PROFILES = frozenset(("bz-a3-1", "bz-a3-2"))


@dataclass(frozen=True)
class BZA3RunEvidence:
    """Local provenance for one retained BZ-A3 timing or profiling operation."""

    replay_id: str
    request_id: str
    mode: str
    profile: str
    handle: str
    status: str
    physical_device: int
    logical_device: int
    remote_report: str | None
    stdout_sha256: str
    stderr_sha256: str


class BZA3ProfilingBackend(GZA3ProfilingBackend):
    """Run verified candidates through one explicit native BZ-A3 profile.

    The candidate-owned driver emits raw timing and compact msprof values.  No
    A5 event interpretation, bandwidth formula, or product ceiling is applied.
    The full vendor report stays at the absolute remote path recorded here.
    """

    def __init__(
        self,
        *,
        validation_wrapper: str,
        cpl_remote: str,
        profile: str,
        remote_candidate_directory: str,
        evidence_directory: Path,
        physical_device: int,
        verified_results: Mapping[str, VerifiedResult],
        process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout: int = 600,
    ) -> None:
        if profile not in _PROFILES:
            raise ValueError("profile must be bz-a3-1 or bz-a3-2")
        if Path(cpl_remote).name != "cpl-remote":
            raise ValueError("cpl_remote must identify the user-wide cpl-remote executable")
        super().__init__(
            validation_wrapper=validation_wrapper,
            remote_candidate_directory=remote_candidate_directory,
            evidence_directory=evidence_directory,
            physical_device=physical_device,
            verified_results=verified_results,
            process_runner=process_runner,
            timeout=timeout,
        )
        self._profile = profile
        self._label = "BZ-A3"
        self._cpl_remote = cpl_remote

    def _call(self, argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "CPL_REMOTE": self._cpl_remote,
        }
        try:
            return self._run(
                argv, text=True, capture_output=True, check=False,
                timeout=self._timeout, env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("BZ-A3 profiling wrapper timed out") from exc

    def _evidence(
        self, replay: str, request_id: str, mode: str, handle: str, status: str,
        meta: dict[str, object], completed: subprocess.CompletedProcess[str],
    ) -> BZA3RunEvidence:
        return BZA3RunEvidence(
            replay_id=replay,
            request_id=request_id,
            mode=mode,
            profile=self._profile,
            handle=handle,
            status=status,
            physical_device=self._device,
            logical_device=0,
            remote_report=(
                meta.get("remote_report")
                if isinstance(meta.get("remote_report"), str) else None
            ),
            stdout_sha256=hashlib.sha256(completed.stdout.encode()).hexdigest(),
            stderr_sha256=hashlib.sha256(completed.stderr.encode()).hexdigest(),
        )

    def _evidence_from_dict(self, value: dict[str, object]) -> BZA3RunEvidence:
        evidence = BZA3RunEvidence(**value)
        if evidence.profile != self._profile:
            raise RuntimeError("retained BZ-A3 evidence uses a foreign profile")
        return evidence


__all__ = ["BZA3ProfilingBackend", "BZA3RunEvidence"]
