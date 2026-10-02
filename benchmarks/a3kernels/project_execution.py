"""Host-owned runtime and performance dispatch for registered A3 projects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Protocol

from benchmarks.a3kernels.phase1_memory import (
    AgentInterpretation,
    HostFact,
    ProjectMemory,
)
from benchmarks.a3kernels.phase1_protocol import FailedEvidence, VerifiedResult
from benchmarks.a3kernels.phase1_registry import (
    CurriculumProposal,
    EvidencePreset,
    ProjectFamily,
)
from benchmarks.a3kernels.profiling import (
    A3ProfilingSession,
    CandidateBinding,
    CompactProfileResult,
    ProfileMetric,
    ProfileRequest,
    StudyDimensions,
    TimingResult,
)


class RecoveryEvidence(str, Enum):
    COMPILE_FAILURE = "compile-failure"
    HOST_VERIFICATION_FAILURE = "host-verification-failure"


class PerformancePreset(str, Enum):
    NONE = "none"
    TIMING = "timing"
    PIPE = "msprof-pipe"


@dataclass(frozen=True)
class RecoveryStarter:
    source: str
    required_evidence: RecoveryEvidence
    fault_count: int

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


_SIGNATURE = """\
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
    uint32_t count, uint32_t buffer_bytes) {
"""
_COMPILE_FAULTS = (
    "    missing_compile_symbol_0(input_a);\n",
    "    missing_compile_symbol_1(input_b);\n",
)
_RUNTIME_FAULTS = (
    """\
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECIN> input_buffer;
    AscendC::TBuf<AscendC::QuePosition::VECOUT> output_buffer;
    pipe.InitBuffer(input_buffer, buffer_bytes);
    pipe.InitBuffer(output_buffer, buffer_bytes);
    AscendC::GlobalTensor<float> input_global;
    AscendC::GlobalTensor<float> output_global;
    input_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(input_a), count);
    output_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(output), count);
    auto input_local = input_buffer.Get<float>();
    auto output_local = output_buffer.Get<float>();
    AscendC::DataCopyExtParams copy{};
    copy.blockCount = 1;
    copy.blockLen = count * sizeof(float);
    AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    AscendC::DataCopyPad(input_local, input_global, copy, padding);
    auto loaded = pipe.FetchEventID<AscendC::HardEvent::MTE2_V>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loaded);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loaded);
    AscendC::Add(output_local, input_local, input_local, count);
    auto computed = pipe.FetchEventID<AscendC::HardEvent::V_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(computed);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(computed);
    AscendC::DataCopyPad(output_global, output_local, copy);
""",
    """\
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECIN> input_buffer;
    AscendC::TBuf<AscendC::QuePosition::VECOUT> output_buffer;
    pipe.InitBuffer(input_buffer, buffer_bytes);
    pipe.InitBuffer(output_buffer, buffer_bytes);
    AscendC::GlobalTensor<float> input_global;
    AscendC::GlobalTensor<float> output_global;
    input_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(input_b), count);
    output_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(output), count);
    auto input_local = input_buffer.Get<float>();
    auto output_local = output_buffer.Get<float>();
    AscendC::DataCopyExtParams copy{};
    copy.blockCount = 1;
    copy.blockLen = count * sizeof(float);
    AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    AscendC::DataCopyPad(input_local, input_global, copy, padding);
    auto loaded = pipe.FetchEventID<AscendC::HardEvent::MTE2_V>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loaded);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loaded);
    AscendC::Add(output_local, input_local, input_local, count);
    auto computed = pipe.FetchEventID<AscendC::HardEvent::V_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(computed);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(computed);
    AscendC::DataCopyPad(output_global, output_local, copy);
""",
)


def _recovery_starter(family: ProjectFamily, faults: int) -> RecoveryStarter:
    if type(faults) is not int or faults not in (1, 2):
        raise ValueError("recovery fault count must be 1 or 2")
    if family is ProjectFamily.COMPILE_RECOVERY:
        source = _SIGNATURE + "".join(_COMPILE_FAULTS[:faults]) + "}\n"
        evidence = RecoveryEvidence.COMPILE_FAILURE
    elif family is ProjectFamily.RUNTIME_RECOVERY:
        source = (
            _SIGNATURE
            + "    // Valid Ascend C starter with deliberately incorrect output semantics.\n"
            + _RUNTIME_FAULTS[faults - 1]
            + "}\n"
        )
        evidence = RecoveryEvidence.HOST_VERIFICATION_FAILURE
    else:
        raise ValueError("recovery starter requires a recovery project")
    return RecoveryStarter(source, evidence, faults)


_PRESET_BY_EVIDENCE = {
    EvidencePreset.CORRECTNESS: PerformancePreset.NONE,
    EvidencePreset.RECOVERY: PerformancePreset.NONE,
    EvidencePreset.CORRECTNESS_TIMING: PerformancePreset.TIMING,
    EvidencePreset.MSPROF: PerformancePreset.PIPE,
}


def _resolve(
    proposal: CurriculumProposal,
) -> tuple[int, int, int, PerformancePreset, RecoveryStarter | None]:
    if not isinstance(proposal, CurriculumProposal):
        raise ValueError("proposal must be a validated CurriculumProposal")
    parameters = dict(proposal.parameters)
    length = int(parameters.get("length", 32))
    blocks = int(parameters.get("block_count", 1))
    recovery = None
    if proposal.family in {
        ProjectFamily.COMPILE_RECOVERY,
        ProjectFamily.RUNTIME_RECOVERY,
    }:
        recovery = _recovery_starter(proposal.family, int(parameters["faults"]))
    return (
        length,
        ((length + 63) // 64) * 64,
        blocks,
        _PRESET_BY_EVIDENCE[proposal.evidence_preset],
        recovery,
    )


@dataclass(frozen=True)
class ProjectRuntimePolicy:
    proposal: CurriculumProposal
    logical_length: int
    padded_length: int
    block_count: int
    performance_preset: PerformancePreset
    recovery: RecoveryStarter | None
    target: str = "Ascend910B4"
    language: str = "ascend-c"

    def __post_init__(self) -> None:
        if self.target != "Ascend910B4" or self.language != "ascend-c":
            raise ValueError("project runtime must target A3 Ascend C")
        if (
            self.logical_length,
            self.padded_length,
            self.block_count,
            self.performance_preset,
            self.recovery,
        ) != _resolve(self.proposal):
            raise ValueError("runtime fields must exactly match the registered proposal")

    @classmethod
    def from_proposal(cls, proposal: CurriculumProposal) -> "ProjectRuntimePolicy":
        return cls(proposal, *_resolve(proposal))

    @property
    def study_dimensions(self) -> StudyDimensions:
        if self.performance_preset is PerformancePreset.NONE:
            raise ValueError("project does not register a performance study")
        return StudyDimensions(self.logical_length, self.block_count, 5, 20)

    def as_dict(self) -> dict[str, object]:
        recovery = None
        if self.recovery is not None:
            recovery = {
                "required_evidence": self.recovery.required_evidence.value,
                "fault_count": self.recovery.fault_count,
                "starter_sha256": self.recovery.source_sha256,
            }
        return {
            "project_id": self.proposal.project_id,
            "family": self.proposal.family.value,
            "target": self.target,
            "language": self.language,
            "logical_length": self.logical_length,
            "padded_length": self.padded_length,
            "block_count": self.block_count,
            "performance_preset": self.performance_preset.value,
            "recovery": recovery,
        }


class CandidateBackend(Protocol):
    def run(self, source: str, workdir: Path, **options: object) -> VerifiedResult | FailedEvidence: ...


@dataclass(frozen=True)
class ProjectOutcome:
    policy: ProjectRuntimePolicy
    verified: VerifiedResult | None = None
    failure: FailedEvidence | None = None
    timing: TimingResult | None = None
    profile: CompactProfileResult | None = None

    def __post_init__(self) -> None:
        if (self.verified is None) == (self.failure is None):
            raise ValueError("outcome requires exactly one terminal candidate result")
        if self.failure is not None and (self.timing is not None or self.profile is not None):
            raise ValueError("failed candidates cannot carry performance evidence")

    def host_facts(self) -> tuple[HostFact, ...]:
        if self.failure is not None:
            category = "compile" if self.failure.stage == "compile" else "runtime"
            return (
                HostFact(
                    category,
                    f"{self.failure.stage} failed with {self.failure.error_type}",
                    self.failure.attestation_sha256,
                ),
            )
        facts = [
            HostFact(
                "host-verification",
                "host verification passed",
                self.verified.evidence_sha256,
            )
        ]
        if self.timing is not None:
            facts.append(
                HostFact(
                    "profiling",
                    f"host timing median was {self.timing.median_us:g} us",
                    self.timing.evidence_sha256,
                )
            )
        if self.profile is not None:
            facts.append(
                HostFact(
                    "profiling",
                    f"host retained {self.profile.metric.value} evidence",
                    self.profile.evidence_sha256,
                )
            )
        return tuple(facts)

    def to_memory(
        self,
        *,
        cell_id: str,
        lineage_id: str,
        actions: tuple[str, ...],
        interpretations: tuple[AgentInterpretation, ...] = (),
    ) -> ProjectMemory:
        source = (
            self.verified.source_fingerprint
            if self.verified is not None
            else self.failure.source_fingerprint
        )
        return ProjectMemory(
            self.policy.proposal,
            self.policy.proposal.project_id,
            cell_id,
            lineage_id,
            source,
            actions,
            self.host_facts(),
            interpretations,
        )


class ProjectDispatcher:
    def __init__(
        self,
        candidate_backend: CandidateBackend,
        profiling: A3ProfilingSession | None = None,
    ) -> None:
        self._candidate = candidate_backend
        self._profiling = profiling

    def execute(
        self,
        proposal: CurriculumProposal,
        source: str | None,
        workdir: Path,
        *,
        request_id: str,
        attempt_id: str,
        seed: int = 0,
        replay_id: str | None = None,
    ) -> ProjectOutcome:
        policy = ProjectRuntimePolicy.from_proposal(proposal)
        selected_source = source
        if selected_source is None:
            if policy.recovery is None:
                raise ValueError("a non-recovery project requires candidate source")
            selected_source = policy.recovery.source
        terminal = self._candidate.run(
            selected_source,
            workdir,
            request_id=request_id,
            attempt_id=attempt_id,
            length=policy.logical_length,
            padded_length=policy.padded_length,
            block_count=policy.block_count,
            seed=seed,
            project_id=proposal.project_id,
        )
        if isinstance(terminal, FailedEvidence):
            return ProjectOutcome(policy, failure=terminal)
        if not isinstance(terminal, VerifiedResult) or not terminal.passed:
            raise ValueError("candidate backend must return host-verified success or failure")
        if policy.performance_preset is PerformancePreset.NONE:
            return ProjectOutcome(policy, verified=terminal)
        if self._profiling is None:
            raise RuntimeError("registered performance project requires profiling")
        binding = CandidateBinding(terminal.execution_id, terminal.source_fingerprint)
        selected_replay = replay_id or (
            f"a3-project-{proposal.project_id[:8]}-{terminal.execution_id[:8]}"
        )
        if policy.performance_preset is PerformancePreset.TIMING:
            timing = self._profiling.time(
                binding, policy.study_dimensions, replay_id=selected_replay
            )
            return ProjectOutcome(policy, verified=terminal, timing=timing)
        request = ProfileRequest(
            binding, policy.study_dimensions, ProfileMetric.PIPE_UTILIZATION
        )
        profile = self._profiling.profile(request, replay_id=selected_replay)
        if profile is None:
            raise ValueError("registered msprof project requires profiling evidence")
        return ProjectOutcome(policy, verified=terminal, profile=profile)


__all__ = [
    "PerformancePreset", "ProjectDispatcher", "ProjectOutcome",
    "ProjectRuntimePolicy", "RecoveryEvidence", "RecoveryStarter",
]
