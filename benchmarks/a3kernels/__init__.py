"""Deterministic Ascend C hello runtime for A3."""

from benchmarks.a3kernels.protocol import RunRequest, VerifiedResult
from benchmarks.a3kernels.runner import A3KernelRunner
from benchmarks.a3kernels.profiling_bz import BZA3ProfilingBackend, BZA3RunEvidence

__all__ = [
    "A3KernelRunner", "BZA3ProfilingBackend", "BZA3RunEvidence",
    "RunRequest", "VerifiedResult",
]
