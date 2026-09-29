from __future__ import annotations

import json

import pytest

from benchmarks.a5kernels.knowledge_agent import Citation
from benchmarks.a5kernels.phase1_memory import (
    AgentInterpretation,
    HostFact,
    Phase1LearningJournal,
    Phase1ProjectMemory,
)
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS


EVIDENCE = "a" * 64


def memory(ordinal, *, interpretation=True):
    return Phase1ProjectMemory(
        proposal=DEFAULT_PROPOSALS[ordinal - 1],
        source_revision=f"source-{ordinal}",
        actions=("compile registered candidate", "run host verifier"),
        host_facts=(
            HostFact("host-verification", f"host result {ordinal}", EVIDENCE),
        ),
        interpretations=(
            (AgentInterpretation("The result suggests a reusable lesson.", (EVIDENCE,)),)
            if interpretation else ()
        ),
        kdb_citations=(
            Citation("catlass", "guide.py", 1, 8, "b" * 64),
        ),
    )


def test_journal_is_append_only_and_context_keeps_evidence_labels(tmp_path):
    journal = Phase1LearningJournal(tmp_path / "learning.jsonl", DEFAULT_PROPOSALS)
    first = journal.append(memory(1))
    second = journal.append(memory(2, interpretation=False))

    assert first["previous_sha256"] == "0" * 64
    assert second["previous_sha256"] == first["entry_sha256"]
    context = json.loads(journal.project_context())
    assert context["completed_projects"] == 2
    assert context["projects"][0]["host_facts"] == [{
        "category": "host-verification",
        "statement": "host result 1",
        "evidence_sha256": EVIDENCE,
    }]
    assert context["projects"][0]["agent_interpretations"][0]["statement"].startswith(
        "The result suggests"
    )
    assert context["projects"][1]["agent_interpretations"] == []
    assert context["projects"][0]["kdb_citations"][0]["path"] == "guide.py"


def test_context_is_byte_stable_across_reopen(tmp_path):
    path = tmp_path / "learning.jsonl"
    journal = Phase1LearningJournal(path, tuple(reversed(DEFAULT_PROPOSALS)))
    journal.append(memory(1))
    expected = journal.project_context()

    reopened = Phase1LearningJournal(path, DEFAULT_PROPOSALS)
    assert reopened.project_context() == expected
    assert reopened.resume_state().head_sha256 == journal.resume_state().head_sha256


def test_resume_state_exposes_registered_checkpoints_four_and_eight(tmp_path):
    journal = Phase1LearningJournal(tmp_path / "learning.jsonl", DEFAULT_PROPOSALS)
    assert journal.resume_state().latest_checkpoint == 0
    assert journal.resume_state().checkpoint_status == "CONTINUE"
    for ordinal in range(1, 5):
        journal.append(memory(ordinal))

    halfway = journal.resume_state()
    assert (halfway.completed_projects, halfway.next_ordinal) == (4, 5)
    assert halfway.latest_checkpoint == 4
    assert halfway.checkpoint_status == "CONTINUE"

    for ordinal in range(5, 9):
        journal.append(memory(ordinal))
    final = journal.resume_state()
    assert (final.completed_projects, final.next_ordinal) == (8, None)
    assert final.latest_checkpoint == 8
    assert final.checkpoint_status == "SATURATED"
    with pytest.raises(ValueError, match="already complete"):
        journal.append(memory(8))


def test_append_rejects_skipped_or_reordered_project(tmp_path):
    journal = Phase1LearningJournal(tmp_path / "learning.jsonl", DEFAULT_PROPOSALS)
    with pytest.raises(ValueError, match="next registered proposal"):
        journal.append(memory(2))
    journal.append(memory(1))
    with pytest.raises(ValueError, match="next registered proposal"):
        journal.append(memory(1))


@pytest.mark.parametrize("field", ["memory", "previous_sha256", "plan_fingerprint"])
def test_resume_rejects_tampered_entries(tmp_path, field):
    path = tmp_path / "learning.jsonl"
    journal = Phase1LearningJournal(path, DEFAULT_PROPOSALS)
    journal.append(memory(1))
    entry = json.loads(path.read_text())
    if field == "memory":
        entry[field]["source_revision"] = "edited"
    else:
        entry[field] = "f" * 64
    path.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ValueError, match="failed resume validation"):
        journal.resume_state()


def test_resume_rejects_other_plan_even_with_shared_first_project(tmp_path):
    path = tmp_path / "learning.jsonl"
    Phase1LearningJournal(path, DEFAULT_PROPOSALS).append(memory(1))
    different_plan = DEFAULT_PROPOSALS[:-1]

    with pytest.raises(ValueError, match="failed resume validation"):
        Phase1LearningJournal(path, different_plan).read()


@pytest.mark.parametrize(
    "fact,error",
    [
        (("other", "fact", EVIDENCE), "category"),
        (("compile", "", EVIDENCE), "statement"),
        (("compile", "fact", "not-a-hash"), "SHA-256"),
    ],
)
def test_host_fact_requires_named_host_evidence(fact, error):
    with pytest.raises(ValueError, match=error):
        HostFact(*fact)


def test_project_memory_cannot_exist_without_host_fact():
    with pytest.raises(ValueError, match="host fact"):
        Phase1ProjectMemory(DEFAULT_PROPOSALS[0], "source", (), ())
