import json

from benchmarks.a5kernels.phase1_experiments import (
    A5Phase1Plan, ProgrammingGuideIdentity, ProfilingGuidance,
    build_a5_phase1_plan, build_cells,
)
from benchmarks.a5kernels.profiling_guidance import ProfilingGuidanceIdentity


def test_canonical_builder_admits_new_guide_and_preserves_it_in_json(tmp_path):
    guide = tmp_path / "programming.md"
    guide.write_text("Frozen programming instructions.")
    profiling = ProfilingGuidanceIdentity("a5-new-profiling-guidance-v1", "a" * 64, "c" * 40)
    plan = build_a5_phase1_plan(guide, cpl_skills_revision="c" * 40, profiling_guide=profiling)
    assert plan.profiling_guide == profiling
    assert len(plan.runnable_cells) == 12
    value = json.loads(plan.to_json())
    assert value["profiling_guide"] == profiling.as_dict()
    recovered_guide = ProgrammingGuideIdentity(**value["guide"])
    recovered_profiling = ProfilingGuidanceIdentity(**value["profiling_guide"])
    recovered = A5Phase1Plan(build_cells(recovered_guide, profiling_guide=recovered_profiling),
                             recovered_guide, profiling_guide=recovered_profiling)
    assert recovered.to_json() == plan.to_json()
    for cell in plan.cells:
        assert cell.profiling_guide == (profiling if cell.profiling is ProfilingGuidance.NEW else None)


def test_legacy_plan_serialization_remains_identical(tmp_path):
    guide = tmp_path / "programming.md"
    guide.write_text("Frozen programming instructions.")
    plan = build_a5_phase1_plan(guide, cpl_skills_revision="c" * 40)
    expected = {
        "schema": plan.schema, "target": "a5", "language": "catlass-dsl",
        "guide": plan.guide.as_dict(), "cells": [cell.as_dict() for cell in build_cells(plan.guide)],
    }
    assert plan.to_json() == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    assert "profiling_guide" not in plan.as_dict()
    assert len(plan.runnable_cells) == 8 and len(plan.pending_cells) == 4
