from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from benchmarks.a5kernels.phase1_experiments import (
    A5BackendModel,
    A5Language,
    A5ModelIdentity,
    A5Phase1Plan,
    A5Target,
    GUIDE_SCHEMA,
    KnowledgeMode,
    MODEL_IDENTITIES,
    PINNED_CATLASS_REVISION,
    PLAN_SCHEMA,
    PendingTreatmentError,
    ProfilingGuidance,
    ProgrammingGuideIdentity,
    TreatmentStatus,
    admit_runnable_cell,
    admit_programming_guide,
    build_a5_phase1_plan,
    build_cells,
)


CPL_REVISION = "1" * 40


def write_guide(path, content="# Catlass DSL\nUse host verification.\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def admitted(tmp_path, content="# Catlass DSL\nUse host verification.\n"):
    return admit_programming_guide(
        write_guide(tmp_path / "agent-guide.md", content),
        cpl_skills_revision=CPL_REVISION,
    )


def test_guide_admission_records_content_and_revisions_not_path(tmp_path):
    guide_path = write_guide(tmp_path / "host-specific" / "agent-guide.md")
    identity = admit_programming_guide(
        guide_path, cpl_skills_revision=CPL_REVISION
    )

    assert identity.as_dict() == {
        "schema": GUIDE_SCHEMA,
        "guide_sha256": hashlib.sha256(guide_path.read_bytes()).hexdigest(),
        "cpl_skills_revision": CPL_REVISION,
        "catlass_revision": PINNED_CATLASS_REVISION,
    }
    assert str(tmp_path) not in json.dumps(identity.as_dict())
    assert not hasattr(identity, "path")


def test_identical_guide_bytes_at_different_paths_have_same_identity(tmp_path):
    first = write_guide(tmp_path / "one" / "guide.md")
    second = write_guide(tmp_path / "two" / "renamed.md")

    assert admit_programming_guide(
        first, cpl_skills_revision=CPL_REVISION
    ) == admit_programming_guide(second, cpl_skills_revision=CPL_REVISION)


@pytest.mark.parametrize("revision", ["", "1" * 39, "A" * 40, "g" * 40, None])
def test_guide_admission_requires_full_lowercase_cpl_revision(tmp_path, revision):
    with pytest.raises(ValueError, match="cpl_skills_revision"):
        admit_programming_guide(
            write_guide(tmp_path / "guide.md"), cpl_skills_revision=revision
        )


def test_guide_admission_rejects_catlass_revision_drift(tmp_path):
    with pytest.raises(ValueError, match="Catlass revision"):
        admit_programming_guide(
            write_guide(tmp_path / "guide.md"),
            cpl_skills_revision=CPL_REVISION,
            catlass_revision="2" * 40,
        )


@pytest.mark.parametrize("condition", ["missing", "empty", "binary"])
def test_guide_admission_rejects_unusable_content(tmp_path, condition):
    path = tmp_path / "guide.md"
    if condition == "empty":
        path.write_bytes(b"")
    elif condition == "binary":
        path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="guide"):
        admit_programming_guide(path, cpl_skills_revision=CPL_REVISION)


def test_registered_profiles_are_exact_and_deepseek_is_direct():
    assert [profile.as_dict() for profile in MODEL_IDENTITIES] == [
        {
            "backend_model": "openai/gpt-5.6-sol",
            "profile_id": "openai-gpt-5.6-sol-v1",
            "provider": "OpenAI",
            "route": "direct:https://api.openai.com/v1",
            "credential_env": "OPENAI_API_KEY",
        },
        {
            "backend_model": "deepseek-flash",
            "profile_id": "deepseek-flash-native-v1",
            "provider": "DeepSeek",
            "route": "direct:https://api.deepseek.com",
            "credential_env": "DEEPSEEK_API_KEY",
        },
    ]
    with pytest.raises(ValueError, match="registered A5 profile"):
        A5ModelIdentity(
            A5BackendModel.DEEPSEEK_FLASH,
            "deepseek-flash-native-v1",
            "DeepSeek",
            "openrouter:DeepSeek",
            "DEEPSEEK_API_KEY",
        )

    serialized = json.dumps([item.as_dict() for item in MODEL_IDENTITIES])
    assert "OPENROUTER_API_KEY" not in serialized
    assert "openrouter:" not in serialized


def test_plan_is_exact_catlass_only_twelve_cell_cross_product(tmp_path):
    plan = build_a5_phase1_plan(
        write_guide(tmp_path / "guide.md"), cpl_skills_revision=CPL_REVISION
    )

    assert plan.schema == PLAN_SCHEMA
    assert plan.target is A5Target.A5
    assert plan.language is A5Language.CATLASS_DSL
    assert len(plan.cells) == 12
    assert {
        (cell.model.backend_model, cell.knowledge, cell.profiling)
        for cell in plan.cells
    } == {
        (model, knowledge, profiling)
        for model in A5BackendModel
        for knowledge in KnowledgeMode
        for profiling in ProfilingGuidance
    }
    assert {cell.target for cell in plan.cells} == {A5Target.A5}
    assert {cell.language for cell in plan.cells} == {A5Language.CATLASS_DSL}
    assert len(plan.runnable_cells) == 8
    assert len(plan.pending_cells) == 4
    assert {
        (cell.profiling, cell.status, cell.runnable) for cell in plan.cells
    } == {
        (ProfilingGuidance.WITHOUT_GUIDANCE, TreatmentStatus.READY, True),
        (ProfilingGuidance.WITH_GUIDANCE, TreatmentStatus.READY, True),
        (ProfilingGuidance.NEW, TreatmentStatus.PENDING, False),
    }


def test_one_admitted_guide_identity_binds_every_cell(tmp_path):
    plan = build_a5_phase1_plan(
        write_guide(tmp_path / "guide.md"), cpl_skills_revision=CPL_REVISION
    )

    assert all(cell.guide is plan.guide for cell in plan.cells)
    assert {json.dumps(cell.guide.as_dict(), sort_keys=True) for cell in plan.cells} == {
        json.dumps(plan.guide.as_dict(), sort_keys=True)
    }


def test_serialization_is_deterministic_and_contains_no_admission_path(tmp_path):
    guide_path = write_guide(tmp_path / "private-location" / "guide.md")
    first = build_a5_phase1_plan(
        guide_path, cpl_skills_revision=CPL_REVISION
    )
    second = build_a5_phase1_plan(
        guide_path, cpl_skills_revision=CPL_REVISION
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.to_json().endswith("\n")
    assert first.to_json() == json.dumps(
        first.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n"
    assert str(guide_path) not in first.to_json()


def test_cell_ids_are_stable_and_change_with_guide_identity(tmp_path):
    first = build_cells(admitted(tmp_path / "first", "first guide\n"))
    repeated = build_cells(admitted(tmp_path / "repeat", "first guide\n"))
    changed = build_cells(admitted(tmp_path / "changed", "second guide\n"))

    assert [cell.cell_id for cell in first] == [cell.cell_id for cell in repeated]
    assert {cell.cell_id for cell in first}.isdisjoint(
        {cell.cell_id for cell in changed}
    )
    assert all(cell.cell_id.startswith("a5-cell-") for cell in first)


def test_new_treatment_is_pending_and_fails_before_dispatch(tmp_path):
    cell = next(
        cell
        for cell in build_cells(admitted(tmp_path))
        if cell.profiling is ProfilingGuidance.NEW
    )
    dispatches = []

    def dispatch(candidate):
        dispatches.append(admit_runnable_cell(candidate))

    with pytest.raises(PendingTreatmentError, match="pending until the new profiler"):
        dispatch(cell)

    assert dispatches == []
    payload = cell.as_dict()
    assert payload["profiling"] == "new"
    assert payload["status"] == "pending"


def test_ready_treatments_keep_existing_identity_and_pass_admission(tmp_path):
    cells = build_cells(admitted(tmp_path))
    ready = [cell for cell in cells if cell.profiling is not ProfilingGuidance.NEW]

    assert len(ready) == 8
    assert all(admit_runnable_cell(cell) is cell for cell in ready)
    assert all(cell.as_dict()["status"] == "ready" for cell in ready)
    assert [cell.cell_id for cell in ready] == [
        "a5-cell-33eb4f8487bbc477",
        "a5-cell-95e087ce908ee6b6",
        "a5-cell-b0c27c98a9f7d197",
        "a5-cell-f41ab3e2d72235fc",
        "a5-cell-8ccdb70257d91f84",
        "a5-cell-c8e7a4fac1408e3e",
        "a5-cell-94158efe1d6f2200",
        "a5-cell-faf4080210856be3",
    ]


def test_treatment_pairs_differ_only_in_declared_dimension(tmp_path):
    cells = build_cells(admitted(tmp_path))
    indexed = {
        (cell.model.backend_model, cell.knowledge, cell.profiling): cell
        for cell in cells
    }

    for model in A5BackendModel:
        base = indexed[
            model, KnowledgeMode.WITHOUT_KDB, ProfilingGuidance.WITHOUT_GUIDANCE
        ].identity_values()
        with_kdb = indexed[
            model, KnowledgeMode.WITH_KDB, ProfilingGuidance.WITHOUT_GUIDANCE
        ].identity_values()
        with_profile = indexed[
            model, KnowledgeMode.WITHOUT_KDB, ProfilingGuidance.WITH_GUIDANCE
        ].identity_values()
        assert {
            key for key in base if base[key] != with_kdb[key]
        } == {"knowledge"}
        assert {
            key for key in base if base[key] != with_profile[key]
        } == {"profiling"}


def test_model_pairs_differ_only_in_profile_identity(tmp_path):
    cells = build_cells(admitted(tmp_path))
    for knowledge in KnowledgeMode:
        for profiling in ProfilingGuidance:
            selected = [
                cell.identity_values()
                for cell in cells
                if cell.knowledge is knowledge and cell.profiling is profiling
            ]
            assert len(selected) == 2
            assert {
                key for key in selected[0] if selected[0][key] != selected[1][key]
            } == {"model"}


def test_plan_rejects_missing_duplicate_or_foreign_cells(tmp_path):
    guide = admitted(tmp_path)
    cells = build_cells(guide)
    with pytest.raises(ValueError, match="twelve unique"):
        A5Phase1Plan(cells[:-1], guide)
    with pytest.raises(ValueError, match="twelve unique"):
        A5Phase1Plan((*cells[:-1], cells[0]), guide)

    other = admitted(tmp_path / "other", "different guide\n")
    with pytest.raises(ValueError, match="bind the admitted guide"):
        A5Phase1Plan((*cells[:-1], build_cells(other)[-1]), guide)

    with pytest.raises(ValueError, match="A5Phase1Cell"):
        A5Phase1Plan((*cells[:-1], object()), guide)


def test_cell_and_guide_are_immutable_and_fail_closed(tmp_path):
    guide = admitted(tmp_path)
    cell = build_cells(guide)[0]
    with pytest.raises(FrozenInstanceError):
        guide.guide_sha256 = "0" * 64
    with pytest.raises(ValueError, match="cell_id"):
        replace(cell, cell_id="a5-cell-forged")
    with pytest.raises(ValueError, match="registered A5 types"):
        replace(cell, knowledge="with-kdb")


def test_guide_identity_requires_exact_schema_and_digest():
    values = {
        "schema": GUIDE_SCHEMA,
        "guide_sha256": "0" * 64,
        "cpl_skills_revision": CPL_REVISION,
        "catlass_revision": PINNED_CATLASS_REVISION,
    }
    with pytest.raises(ValueError, match="guide schema"):
        ProgrammingGuideIdentity(**{**values, "schema": "future-v2"})
    with pytest.raises(ValueError, match="guide_sha256"):
        ProgrammingGuideIdentity(**{**values, "guide_sha256": "ABC"})
