"""Host-owned execution contracts for registered A5 Phase-1 projects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path

from benchmarks.a5kernels.candidate import CatlassCandidateBackend
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.fixtures import Language
from benchmarks.a5kernels.matrix import Workload
from benchmarks.a5kernels.phase1_registry import CurriculumProposal, ProjectFamily
from benchmarks.a5kernels.profiling import (
    AccessClass,
    PaddingClass,
    ParallelismClass,
    ShapeClass,
    StudyDimensions,
    StudyVariant,
)
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
            _recovery_starter(family, int(parameters["faults"])),
            RecoveryEvidence.COMPILE_FAILURE, int(parameters["faults"]),
        )
    elif family is ProjectFamily.RUNTIME_RECOVERY:
        recovery = RecoveryStarter(
            _recovery_starter(family, int(parameters["faults"])),
            RecoveryEvidence.HOST_VERIFICATION_FAILURE, int(parameters["faults"]),
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

    @property
    def study_dimensions(self) -> StudyDimensions:
        """Derive the typed study point from this registered runtime."""

        if self.proposal.family not in {
            ProjectFamily.LENGTH_KNEE,
            ProjectFamily.CROSS_LAYER_LAUNCH,
            ProjectFamily.MSPROF_PIPE,
        }:
            raise ValueError("project family does not register a performance study")
        padding = (
            PaddingClass.NONE
            if self.padded_length == self.logical_length
            else PaddingClass.ALIGN_64
        )
        return StudyDimensions(
            ShapeClass(f"n{self.logical_length}"),
            padding,
            AccessClass.CONTIGUOUS,
            ParallelismClass(self.block_count),
        )

    def run_study(
        self,
        execution_backend: ExecutionBackend,
        workspace: Path,
        attempt_id: str,
        ledger: EvidenceLedger,
        *,
        seed: int = 0,
        device: int = 0,
    ) -> StudyVariant:
        """Run registered correctness and return its exact profiling variant."""

        return self.backend(
            execution_backend, seed=seed, device=device
        ).run_study(
            workspace,
            Language.CATLASS_DSL.value,
            Workload.SMOKE_VECTOR_ADD,
            attempt_id,
            ledger,
            self.study_dimensions,
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


_COMPILE_SOURCE_HEAD = """\
import catlass.tla as tla

@tla.kernel
def vector_add(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    with tla.vector():
"""

_COMPILE_FAULTS = (
    "        tla.copy(gm_c, missing_input)\n",
    "        tla.copy(gm_c, missing_input_second)\n",
)
_RUNTIME_SOURCE_HEAD = """\
import catlass.tla as tla

PADDED_VECTOR_ELE = 448
VL_ELE = 64
FAULT_ELE = 16

@tla.kernel
def vector_add(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    n_ele = gm_a.origin_shape[0]
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    ptr_a = tla.allocate(PADDED_VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ptr_b = tla.allocate(PADDED_VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ptr_c = tla.allocate(PADDED_VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_a = tla.make_tensor_like(ptr_a, gm_a, tla.arch.RowMajor)
    ub_b = tla.make_tensor_like(ptr_b, gm_b, tla.arch.RowMajor)
    ub_c = tla.make_tensor_like(ptr_c, gm_c, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(ub_a, gm_a)
        tla.copy(ub_b, gm_b)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                tile_a = tla.tile_view(ub_a, tla.make_shape(VL_ELE), tla.make_coord(i))
                tile_b = tla.tile_view(ub_b, tla.make_shape(VL_ELE), tla.make_coord(i))
                tile_c = tla.tile_view(ub_c, tla.make_shape(VL_ELE), tla.make_coord(i))
                tile_c.store(tla.add(tile_a.load(), tile_b.load()))
"""

_RUNTIME_FAULTS = (
    """\
            fault_a_0 = tla.tile_view(ub_a, tla.make_shape(FAULT_ELE), tla.make_coord(0))
            fault_b_0 = tla.tile_view(ub_b, tla.make_shape(FAULT_ELE), tla.make_coord(0))
            fault_c_0 = tla.tile_view(ub_c, tla.make_shape(FAULT_ELE), tla.make_coord(0))
            fault_c_0.store(fault_a_0.load())
""",
    """\
            fault_a_1 = tla.tile_view(ub_a, tla.make_shape(FAULT_ELE), tla.make_coord(1))
            fault_b_1 = tla.tile_view(ub_b, tla.make_shape(FAULT_ELE), tla.make_coord(1))
            fault_c_1 = tla.tile_view(ub_c, tla.make_shape(FAULT_ELE), tla.make_coord(1))
            fault_c_1.store(fault_b_1.load())
""",
)

_RUNTIME_SOURCE_TAIL = """\
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_c, ub_c)
        tla.pipe_barrier(tla.pipes.ALL)
"""


def _recovery_starter(family: ProjectFamily, fault_count: int) -> str:
    if fault_count not in (1, 2):  # pragma: no cover - registry invariant
        raise ValueError("recovery fault count must be 1 or 2")
    if family is ProjectFamily.COMPILE_RECOVERY:
        return _COMPILE_SOURCE_HEAD + "".join(_COMPILE_FAULTS[:fault_count])
    elif family is ProjectFamily.RUNTIME_RECOVERY:
        return (
            _RUNTIME_SOURCE_HEAD
            + "".join(_RUNTIME_FAULTS[:fault_count])
            + _RUNTIME_SOURCE_TAIL
        )
    else:  # pragma: no cover - internal call invariant
        raise ValueError("recovery starter requires a recovery family")


# Stable public one-fault fixtures retained for callers and default proposals.
COMPILE_FAILURE_STARTER = _recovery_starter(ProjectFamily.COMPILE_RECOVERY, 1)
HOST_FAILURE_STARTER = _recovery_starter(ProjectFamily.RUNTIME_RECOVERY, 1)
