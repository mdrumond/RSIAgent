"""Host-owned execution contracts for registered A5 Phase-1 projects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib

from benchmarks.a5kernels.candidate import CatlassCandidateBackend
from benchmarks.a5kernels.phase1_registry import CurriculumProposal, ProjectFamily
from benchmarks.a5kernels.runner import ExecutionBackend


class RecoveryEvidence(str, Enum):
    """Host observation required before a recovery project can complete."""

    COMPILE_FAILURE = "compile-failure"
    HOST_VERIFICATION_FAILURE = "host-verification-failure"


@dataclass(frozen=True)
class RecoveryStarter:
    source: str
    required_evidence: RecoveryEvidence
    fault_count: int

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def _resolve_runtime(
    proposal: CurriculumProposal,
) -> tuple[int, int, int, RecoveryStarter | None]:
    """Derive the only runtime fields admitted for a registered proposal."""

    if not isinstance(proposal, CurriculumProposal):
        raise ValueError("proposal must be a validated CurriculumProposal")
    parameters = dict(proposal.parameters)
    family = proposal.family
    logical_length = int(parameters.get("length", 32))
    block_count = int(parameters.get("block_count", 1))
    recovery = None
    if family is ProjectFamily.COMPILE_RECOVERY:
        recovery = RecoveryStarter(
            COMPILE_FAILURE_STARTER, RecoveryEvidence.COMPILE_FAILURE,
            int(parameters["faults"]),
        )
    elif family is ProjectFamily.RUNTIME_RECOVERY:
        recovery = RecoveryStarter(
            HOST_FAILURE_STARTER, RecoveryEvidence.HOST_VERIFICATION_FAILURE,
            int(parameters["faults"]),
        )
    return (
        logical_length, ((logical_length + 63) // 64) * 64,
        block_count, recovery,
    )


@dataclass(frozen=True)
class Phase1ProjectRuntime:
    """Resolved, command-free runtime policy for one validated proposal."""

    proposal: CurriculumProposal
    logical_length: int
    padded_length: int
    block_count: int
    recovery: RecoveryStarter | None = None

    def __post_init__(self) -> None:
        expected = _resolve_runtime(self.proposal)
        actual = (
            self.logical_length, self.padded_length, self.block_count, self.recovery,
        )
        if actual != expected:
            raise ValueError("runtime fields must exactly match the registered proposal")

    @classmethod
    def from_proposal(cls, proposal: CurriculumProposal) -> "Phase1ProjectRuntime":
        """Resolve all execution values from the immutable registry selection."""

        return cls(proposal, *_resolve_runtime(proposal))

    def backend(
        self, execution_backend: ExecutionBackend, *, seed: int = 0, device: int = 0
    ) -> CatlassCandidateBackend:
        return CatlassCandidateBackend(
            execution_backend, length=self.logical_length,
            padded_length=self.padded_length, block_count=self.block_count,
            seed=seed, device=device,
        )

    def as_dict(self) -> dict[str, object]:
        recovery = None if self.recovery is None else {
            "fault_count": self.recovery.fault_count,
            "required_evidence": self.recovery.required_evidence.value,
            "starter_sha256": self.recovery.source_sha256,
        }
        return {
            "family": self.proposal.family.value,
            "logical_length": self.logical_length,
            "padded_length": self.padded_length,
            "block_count": self.block_count,
            "recovery": recovery,
        }


_SOURCE_HEAD = """\
import catlass.tla as tla

@tla.kernel
def vector_add(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    with tla.vector():
"""

# Both starters satisfy the public source ABI. The first reaches the compiler
# with an undefined symbol. The second can compile but host comparison catches
# that it copied one input instead of adding both inputs.
COMPILE_FAILURE_STARTER = _SOURCE_HEAD + "        tla.copy(gm_c, missing_input)\n"
HOST_FAILURE_STARTER = _SOURCE_HEAD + "        tla.copy(gm_c, gm_a)\n"
