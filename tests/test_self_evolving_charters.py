"""Input propagation and rejection at charter boundaries."""

import pytest

from explore.charter import (
    self_evolving_actor_charter,
    self_evolving_actor_memory_distillation_msg,
    self_evolving_curriculum_charter,
    self_evolving_verifier_charter,
    self_evolving_target_actor_charter,
)


TARGET = "Arrange three supplied clips to reproduce the disclosed title sequence."
PROJECT = "Create an editable project at /home/user/work/project.mlt."


def test_curriculum_is_persistent_meta_search_not_solver_or_grader():
    prompt = self_evolving_curriculum_charter(
        TARGET,
        project_history="project-a: PASS; project-b: FAIL",
        latest_outcome="VERDICT: FAIL\nThe second title was absent.",
        curriculum_notes="Current hypothesis: transfer is brittle.",
    )

    assert TARGET in prompt


def test_curriculum_requires_an_exact_target_seed():
    with pytest.raises(ValueError, match="target_direction"):
        self_evolving_curriculum_charter("  ")


def test_phase1_curriculum_searches_distribution_and_uses_saturation():
    distribution = (
        "Allowed training distribution: office documents, media editing, and "
        "CAD tasks in disposable desktop environments."
    )
    prompt = self_evolving_curriculum_charter(
        distribution,
        project_history="project 1: PASS; project 2: FAIL",
        latest_outcome="project 2 exposed brittle cross-application transfer",
        phase1_exploration=True,
    )

    assert distribution in prompt


def test_curriculum_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        self_evolving_curriculum_charter(
            TARGET, target_gated=True, phase1_exploration=True)


def test_fresh_actor_gets_project_and_memory_but_not_a_grade():
    prompt = self_evolving_actor_charter(
        PROJECT,
        memory_listing="video.md 2034 bytes",
        handoff_path="~/handoffs/actor.md",
    )

    assert PROJECT in prompt
    assert "video.md 2034 bytes" in prompt
    assert "~/handoffs/actor.md" in prompt


def test_verifier_owns_grade_and_free_form_evidence():
    execution_path = "/home/user/.test_actor_execution"
    prompt = self_evolving_verifier_charter(
        PROJECT,
        report_path="~/handoffs/verifier.md",
        actor_execution_path=execution_path,
    )

    assert "~/handoffs/verifier.md" in prompt
    assert execution_path in prompt


def test_verifier_does_not_invent_gui_only_provenance():
    project = (
        "Create and save an editable project in the named editor application."
    )
    actor_claim = (
        "STATUS: SUBMIT\nI used a submitted program to automate the named editor "
        "application and save its native editable project."
    )
    prompt = self_evolving_verifier_charter(project)

    assert project in prompt
    assert actor_claim not in prompt


def test_practice_verifier_receives_immutable_original_fixture_view():
    original = "/home/user/.practice_original_project"
    prompt = self_evolving_verifier_charter(
        "Transform the supplied input without replacing it.",
        original_fixtures_path=original,
    )

    assert original in prompt


def test_verifier_does_not_accept_backend_compatibility_as_native_compliance():
    project = (
        "Use the named editor application. Export its timeline as "
        "/home/user/work/final.mp4 and save the editable project."
    )
    actor_claim = (
        "STATUS: SUBMIT\nThe MP4 is visually identical, but I hand-authored the "
        "editor's interchange file and rendered it with a bundled command-line "
        "backend rather than using the named editor application."
    )
    prompt = self_evolving_verifier_charter(project)

    assert project in prompt
    assert actor_claim not in prompt


def test_verifier_keeps_an_explicit_gui_workflow_binding():
    project = (
        "Using only the named editor's visible GUI controls, save the editable "
        "project and export the final video; do not edit the project externally."
    )
    actor_claim = (
        "STATUS: SUBMIT\nI generated the project externally and opened it later.")
    prompt = self_evolving_verifier_charter(project)

    assert project in prompt
    assert actor_claim not in prompt


def test_failed_memory_never_claims_success_and_invalid_outcomes_fail_closed():
    with pytest.raises(ValueError, match="PASS or FAIL"):
        self_evolving_actor_memory_distillation_msg("YIELD", "report")


def test_target_actor_receives_target_and_memory_listing():
    listing = "lesson.md 100B"
    actor = self_evolving_target_actor_charter(TARGET, memory_listing=listing)
    assert TARGET in actor
    assert listing in actor
