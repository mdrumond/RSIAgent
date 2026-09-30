"""Host-owned performance execution for registered Phase 1 evidence presets."""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks.a5kernels.phase1_registry import (
    CurriculumProposal,
    EvidencePreset,
)
from benchmarks.a5kernels.phase1_runtime import Phase1ProjectRuntime
from benchmarks.a5kernels.profiling import (
    AccessClass,
    PaddingClass,
    ParallelismClass,
    Phase1PerformanceStudy,
    Phase1StudyBackend,
    Phase1StudyResult,
    ShapeClass,
    StudyPreset,
    StudyVariant,
)


_STUDY_BY_EVIDENCE = {
    EvidencePreset.CORRECTNESS_TIMING: StudyPreset.TIMING_STUDY,
    EvidencePreset.MSPROF: StudyPreset.PIPE_UTILIZATION,
}


class Phase1PerformanceExecution:
    """Dispatch only the performance work allowed by the curriculum registry."""

    def __init__(
        self,
        backend: Phase1StudyBackend,
        *,
        max_archive_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self._study = Phase1PerformanceStudy(
            backend, max_archive_bytes=max_archive_bytes
        )

    def run(
        self,
        proposal: CurriculumProposal,
        variants: Sequence[StudyVariant],
    ) -> Phase1StudyResult | None:
        """Run the study registered by ``proposal`` for its exact variants.

        Correctness and recovery evidence are produced by their project runtimes;
        they deliberately do not cause timing or profiler dispatch here.
        """

        if not isinstance(proposal, CurriculumProposal):
            raise ValueError("proposal must be a validated CurriculumProposal")
        preset = _STUDY_BY_EVIDENCE.get(proposal.evidence_preset)
        if preset is None:
            return None
        variants = tuple(variants)
        if not variants or any(not isinstance(item, StudyVariant) for item in variants):
            raise ValueError("proposal execution requires StudyVariant values")
        self._validate_dimensions(proposal, variants)
        return self._study.run(preset, variants)

    @staticmethod
    def _validate_dimensions(
        proposal: CurriculumProposal, variants: Sequence[StudyVariant]
    ) -> None:
        runtime = Phase1ProjectRuntime.from_proposal(proposal)
        expected_padding = (
            PaddingClass.NONE
            if runtime.padded_length == runtime.logical_length
            else PaddingClass.ALIGN_64
        )
        expected = {
            "shape": ShapeClass(f"n{runtime.logical_length}"),
            "padding": expected_padding,
            "access": AccessClass.CONTIGUOUS,
            "parallelism": ParallelismClass(runtime.block_count),
        }
        for field, value in expected.items():
            if any(getattr(item.dimensions, field) is not value for item in variants):
                raise ValueError(
                    f"study {field} does not match the proposal runtime"
                )
