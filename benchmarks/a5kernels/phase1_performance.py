"""Host-owned performance execution for registered Phase 1 evidence presets."""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks.a5kernels.phase1_registry import EvidencePreset
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
        evidence_preset: EvidencePreset,
        variants: Sequence[StudyVariant],
    ) -> Phase1StudyResult | None:
        """Run the fixed study for ``evidence_preset``, if it requests one.

        Correctness and recovery evidence are produced by their project runtimes;
        they deliberately do not cause timing or profiler dispatch here.
        """

        if not isinstance(evidence_preset, EvidencePreset):
            raise ValueError("evidence_preset must be a host-owned EvidencePreset")
        preset = _STUDY_BY_EVIDENCE.get(evidence_preset)
        if preset is None:
            return None
        return self._study.run(preset, variants)
