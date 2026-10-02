"""Deterministic Ascend C hello runtime for A3."""

from benchmarks.a3kernels.protocol import RunRequest, VerifiedResult
from benchmarks.a3kernels.runner import A3KernelRunner

__all__ = [
    "A3KernelRunner", "RunRequest", "VerifiedResult",
]
