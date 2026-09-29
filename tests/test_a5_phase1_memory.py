from __future__ import annotations

import json

import pytest

import benchmarks.a5kernels.phase1_memory as phase1_memory
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


def test_resume_rejects_derived_plan_drift_with_same_proposals(tmp_path, monkeypatch):
    path = tmp_path / "learning.jsonl"
    Phase1LearningJournal(path, DEFAULT_PROPOSALS).append(memory(1))
    original = phase1_memory.dry_run_plan

    def changed_plan(proposals):
        plan = original(proposals)
        plan["projects"][0]["coverage"] = ["derived-plan-drift"]
        return plan

    monkeypatch.setattr(phase1_memory, "dry_run_plan", changed_plan)
    with pytest.raises(ValueError, match="failed resume validation"):
        Phase1LearningJournal(path, DEFAULT_PROPOSALS).read()


def test_resume_ignores_torn_tail_and_next_append_recovers_framing(tmp_path):
    path = tmp_path / "learning.jsonl"
    journal = Phase1LearningJournal(path, DEFAULT_PROPOSALS)
    first = journal.append(memory(1))
    with path.open("ab") as stream:
        stream.write(b'{"schema":"a5-phase1-project-memory-v1","sequence":2')

    resumed = Phase1LearningJournal(path, DEFAULT_PROPOSALS)
    assert resumed.resume_state().completed_projects == 1
    assert resumed.resume_state().head_sha256 == first["entry_sha256"]
    resumed.append(memory(2))

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert [entry["sequence"] for entry in resumed.read()] == [1, 2]
    assert json.loads(lines[1])["memory"]["source_revision"] == "source-2"


def test_resume_rejects_newline_terminated_incomplete_record(tmp_path):
    path = tmp_path / "learning.jsonl"
    journal = Phase1LearningJournal(path, DEFAULT_PROPOSALS)
    journal.append(memory(1))
    with path.open("ab") as stream:
        stream.write(b'{"sequence":2\n')

    with pytest.raises(ValueError, match="invalid Phase 1 learning journal JSON"):
        journal.resume_state()


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


@pytest.mark.parametrize("actions", ["compile", {"compile": True}, ["compile"]])
def test_project_actions_requires_exact_tuple(actions):
    with pytest.raises(TypeError, match="actions must be a tuple"):
        Phase1ProjectMemory(
            DEFAULT_PROPOSALS[0], "source", actions,
            (HostFact("compile", "compiled", EVIDENCE),),
        )


@pytest.mark.parametrize("field", ["collection", "path"])
def test_citation_identity_requires_nonempty_strings(field):
    citation = {
        "collection": "catlass", "path": "guide.py", "start_line": 1,
        "end_line": 2, "chunk_hash": "b" * 64,
    }
    citation[field] = 1
    with pytest.raises(ValueError, match="exact validated location"):
        Phase1ProjectMemory(
            DEFAULT_PROPOSALS[0], "source", (),
            (HostFact("host-verification", "passed", EVIDENCE),),
            kdb_citations=(Citation(**citation),),
        )


@pytest.mark.parametrize("shape", ["list", "generator", "mapping"])
@pytest.mark.parametrize("field", ["host_facts", "interpretations", "kdb_citations"])
def test_project_tuple_collections_reject_iterable_impostors(field, shape):
    values = {
        "host_facts": (HostFact("compile", "compiled", EVIDENCE),),
        "interpretations": (AgentInterpretation("Observed", (EVIDENCE,)),),
        "kdb_citations": (Citation("catlass", "guide.py", 1, 2, "b" * 64),),
    }
    original = values[field]
    if shape == "list":
        values[field] = list(original)
    elif shape == "generator":
        values[field] = (item for item in original)
    else:
        values[field] = {index: item for index, item in enumerate(original)}

    with pytest.raises(TypeError, match=f"{field} must be a tuple"):
        Phase1ProjectMemory(
            DEFAULT_PROPOSALS[0], "source", ("compile",), **values,
        )


def test_valid_tuple_collections_persist_and_reload(tmp_path):
    path = tmp_path / "learning.jsonl"
    journal = Phase1LearningJournal(path, DEFAULT_PROPOSALS)
    journal.append(memory(1))

    reloaded = journal.read()[0]["memory"]
    assert reloaded["actions"] == ["compile registered candidate", "run host verifier"]
    assert reloaded["host_facts"][0]["category"] == "host-verification"
    assert reloaded["agent_interpretations"][0]["supports"] == [EVIDENCE]
    assert reloaded["kdb_citations"][0]["collection"] == "catlass"


@pytest.mark.parametrize("sequence", [True, 1.0])
def test_durable_sequence_requires_exact_integer(tmp_path, sequence):
    path = tmp_path / "learning.jsonl"
    journal = Phase1LearningJournal(path, DEFAULT_PROPOSALS)
    journal.append(memory(1))
    entry = json.loads(path.read_text())
    entry["sequence"] = sequence
    payload = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = phase1_memory._digest(payload)
    path.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ValueError, match="failed resume validation"):
        journal.read()


def test_durable_sequence_accepts_ordinary_integer(tmp_path):
    journal = Phase1LearningJournal(tmp_path / "learning.jsonl", DEFAULT_PROPOSALS)
    journal.append(memory(1))

    assert journal.read()[0]["sequence"] == 1
    assert type(journal.read()[0]["sequence"]) is int


@pytest.mark.parametrize("supports", [[EVIDENCE], EVIDENCE, {EVIDENCE: True}])
def test_interpretation_supports_requires_exact_tuple(supports):
    with pytest.raises(TypeError, match="must be a tuple"):
        AgentInterpretation("Interpretation", supports)


@pytest.mark.parametrize("supports", [EVIDENCE, {EVIDENCE: True}])
def test_durable_interpretation_supports_requires_json_array(tmp_path, supports):
    path = tmp_path / "learning.jsonl"
    journal = Phase1LearningJournal(path, DEFAULT_PROPOSALS)
    journal.append(memory(1))
    entry = json.loads(path.read_text())
    entry["memory"]["agent_interpretations"][0]["supports"] = supports
    payload = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = phase1_memory._digest(payload)
    path.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ValueError, match="invalid Phase 1 project memory payload"):
        journal.read()


@pytest.mark.parametrize("actions", ["compile", {"compile": True}])
def test_durable_actions_reject_string_and_mapping(tmp_path, actions):
    path = tmp_path / "learning.jsonl"
    journal = Phase1LearningJournal(path, DEFAULT_PROPOSALS)
    journal.append(memory(1))
    entry = json.loads(path.read_text())
    entry["memory"]["actions"] = actions
    payload = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = phase1_memory._digest(payload)
    path.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ValueError, match="invalid Phase 1 project memory payload"):
        journal.project_context()


@pytest.mark.parametrize("field", ["collection", "path"])
def test_durable_citation_identity_rejects_numeric_fields(tmp_path, field):
    path = tmp_path / "learning.jsonl"
    journal = Phase1LearningJournal(path, DEFAULT_PROPOSALS)
    journal.append(memory(1))
    entry = json.loads(path.read_text())
    entry["memory"]["kdb_citations"][0][field] = 7
    payload = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = phase1_memory._digest(payload)
    path.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ValueError, match="invalid Phase 1 project memory payload"):
        journal.project_context()
