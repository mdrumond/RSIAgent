"""Host-owned performance execution for registered Phase 1 evidence presets."""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks.a5kernels.phase1_registry import (
    CurriculumProposal,
    EvidencePreset,
    ProjectFamily,
)
from benchmarks.a5kernels.profiling import (
    Phase1PerformanceStudy,
    Phase1StudyBackend,
    Phase1StudyResult,
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
        variants = tuple(variants)
        if not variants or any(not isinstance(item, StudyVariant) for item in variants):
            raise ValueError("proposal execution requires StudyVariant values")
        self._validate_dimensions(proposal, variants)
        preset = _STUDY_BY_EVIDENCE.get(proposal.evidence_preset)
        if preset is None:
            return None
        return self._study.run(preset, variants)

    @staticmethod
    def _validate_dimensions(
        proposal: CurriculumProposal, variants: Sequence[StudyVariant]
    ) -> None:
        parameters = dict(proposal.parameters)
        if proposal.family in {
            ProjectFamily.VECTOR_ADD_BASELINE,
            ProjectFamily.PADDED_MULTITILE,
            ProjectFamily.LENGTH_KNEE,
        }:
            expected = parameters["length"]
            if any(
                int(item.dimensions.shape.value.removeprefix("n")) != expected
                for item in variants
            ):
                raise ValueError("study shape does not match proposal length")
        elif proposal.family is ProjectFamily.CROSS_LAYER_LAUNCH:
            expected = parameters["block_count"]
            if any(item.dimensions.parallelism.value != expected for item in variants):
                raise ValueError(
                    "study parallelism does not match proposal block_count"
                )
