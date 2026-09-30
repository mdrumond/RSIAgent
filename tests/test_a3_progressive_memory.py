import json

import pytest

from benchmarks.a3kernels.phase1_memory import (
    AgentInterpretation,
    HostFact,
    Phase1LearningJournal,
    ProjectMemory,
)
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS


CELL = "a3-cell-0123456789abcdef"
OTHER_CELL = "a3-cell-fedcba9876543210"
LINEAGE = "pilot-lineage-1"
EVIDENCE = "a" * 64


def memory(ordinal, *, cell_id=CELL, lineage_id=LINEAGE):
    proposal = DEFAULT_PROPOSALS[ordinal - 1]
    return ProjectMemory(
        proposal=proposal,
        project_id=proposal.project_id,
        cell_id=cell_id,
        lineage_id=lineage_id,
        source_revision="b" * 64,
        actions=("compile candidate", "run host verifier"),
        host_facts=(
            HostFact(
                "host-verification",
                f"verified output for project {ordinal}",
                EVIDENCE,
            ),
        ),
        interpretations=(
            AgentInterpretation("The result suggests a reusable lesson.", (EVIDENCE,)),
        ),
    )


def journal(path, *, cell_id=CELL, lineage_id=LINEAGE):
    return Phase1LearningJournal(
        path, DEFAULT_PROPOSALS, cell_id=cell_id, lineage_id=lineage_id
    )


def test_append_reopen_and_context_keep_host_authority_explicit(tmp_path):
    active = journal(tmp_path / "learning.jsonl")
    first = active.append(memory(1))
    second = active.append(memory(2))

    reopened = journal(active.path)
    context = json.loads(reopened.project_context())
    assert first["previous_sha256"] == "0" * 64
    assert second["previous_sha256"] == first["entry_sha256"]
    assert context["target"] == "Ascend910B4"
    assert context["language"] == "ascend-c"
    assert context["cell_id"] == CELL
    assert context["lineage_id"] == LINEAGE
    assert context["projects"][0]["host_facts"][0]["authority"] == "host"
    assert context["projects"][0]["agent_interpretations"][0]["supports"] == [EVIDENCE]


def test_context_is_byte_identical_after_checkpoint_and_resume(tmp_path):
    active = journal(tmp_path / "learning.jsonl")
    active.append(memory(1))
    active.append(memory(2))
    before = active.project_context()

    reopened = journal(active.path)
    assert reopened.project_context() == before
    state = reopened.resume_state()
    assert state.completed_projects == 2
    assert state.next_ordinal == 3
    assert state.latest_checkpoint == 0
    assert state.cell_id == CELL and state.lineage_id == LINEAGE
    assert state.head_sha256 == active.resume_state().head_sha256


def test_lineages_and_cells_cannot_reopen_each_others_memory(tmp_path):
    path = tmp_path / "learning.jsonl"
    journal(path).append(memory(1))
    with pytest.raises(ValueError, match="resume validation"):
        journal(path, cell_id=OTHER_CELL).read()
    with pytest.raises(ValueError, match="resume validation"):
        journal(path, lineage_id="other-lineage").read()


def test_append_rejects_memory_from_other_binding_or_project(tmp_path):
    active = journal(tmp_path / "learning.jsonl")
    with pytest.raises(ValueError, match="cell and lineage"):
        active.append(memory(1, cell_id=OTHER_CELL))
    wrong = memory(2)
    with pytest.raises(ValueError, match="next registered project"):
        active.append(wrong)


def test_interrupted_final_append_is_discarded_then_recovered(tmp_path):
    active = journal(tmp_path / "learning.jsonl")
    first = active.append(memory(1))
    with active.path.open("ab") as stream:
        stream.write(b'{"schema":"a3-phase1-memory-v1","sequence":2')

    resumed = journal(active.path)
    assert resumed.resume_state().head_sha256 == first["entry_sha256"]
    resumed.append(memory(2))
    assert [entry["sequence"] for entry in resumed.read()] == [1, 2]
    assert len(active.path.read_text().splitlines()) == 2


def test_committed_corruption_and_tampering_fail_closed(tmp_path):
    path = tmp_path / "learning.jsonl"
    active = journal(path)
    active.append(memory(1))
    entry = json.loads(path.read_text())
    entry["memory"]["source_revision"] = "c" * 64
    path.write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValueError, match="resume validation"):
        journal(path).read()

    path.write_text('{"sequence":2\n')
    with pytest.raises(ValueError, match="journal JSON"):
        journal(path).read()


@pytest.mark.parametrize(
    "fact, message",
    [
        (("other", "fact", EVIDENCE), "category"),
        (("compile", "", EVIDENCE), "statement"),
        (("compile", "fact", "not-a-hash"), "SHA-256"),
        (("compile", "fact", EVIDENCE, "agent"), "host authority"),
    ],
)
def test_host_fact_requires_host_owned_verified_evidence(fact, message):
    with pytest.raises(ValueError, match=message):
        HostFact(*fact)


def test_interpretation_can_only_cite_same_project_host_evidence():
    proposal = DEFAULT_PROPOSALS[0]
    with pytest.raises(ValueError, match="only this project's host evidence"):
        ProjectMemory(
            proposal,
            proposal.project_id,
            CELL,
            LINEAGE,
            "b" * 64,
            ("compile",),
            (HostFact("compile", "compiled", EVIDENCE),),
            (AgentInterpretation("unsupported inference", ("c" * 64,)),),
        )


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("project_id", "f" * 64, "project identity"),
        ("cell_id", "a5-cell-0123456789abcdef", "A3 cell"),
        ("cell_id", "catlass-cell", "A3 cell"),
        ("source_revision", "not-a-digest", "source revision"),
    ],
)
def test_memory_rejects_foreign_or_unbound_identity(field, value, message):
    values = memory(1).__dict__.copy()
    values[field] = value
    with pytest.raises(ValueError, match=message):
        ProjectMemory(**values)


def test_resume_exposes_four_and_eight_project_checkpoints(tmp_path):
    active = journal(tmp_path / "learning.jsonl")
    for ordinal in range(1, 5):
        active.append(memory(ordinal))
    halfway = active.resume_state()
    assert (halfway.completed_projects, halfway.latest_checkpoint) == (4, 4)
    assert halfway.checkpoint_status == "CONTINUE"

    for ordinal in range(5, 9):
        active.append(memory(ordinal))
    final = active.resume_state()
    assert (final.completed_projects, final.next_ordinal) == (8, None)
    assert final.latest_checkpoint == 8
    assert final.checkpoint_status == "SATURATED"
