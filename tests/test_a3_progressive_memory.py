import json

import pytest

from benchmarks.a3kernels.phase1_memory import (
    AgentInterpretation,
    AuthoritativeEvidence,
    HostFact,
    Phase1LearningJournal,
    ProjectMemory,
)
from benchmarks.a3kernels.phase1_registry import (
    DEFAULT_PROPOSALS,
    EvidencePreset,
    ProjectFamily,
)


CELL = "a3-cell-0123456789abcdef"
OTHER_CELL = "a3-cell-fedcba9876543210"
LINEAGE = "pilot-lineage-1"
EVIDENCE = f"{10:064x}"


KIND_OFFSET = {
    "host-verification": 0,
    "compile": 1,
    "runtime": 2,
    "profiling": 3,
}


def evidence(ordinal, kind="host-verification", *, failed=False):
    value = ordinal * 10 + KIND_OFFSET[kind] + (1000 if failed else 0)
    return f"{value:064x}"


def fact_statement(ordinal, kind):
    return f"{kind} result for project {ordinal}"


def completion_kinds(proposal):
    kinds = ["host-verification"]
    if proposal.evidence_preset is EvidencePreset.CORRECTNESS_TIMING:
        kinds.append("runtime")
    elif proposal.evidence_preset is EvidencePreset.RECOVERY:
        kinds.append(
            "compile"
            if proposal.family is ProjectFamily.COMPILE_RECOVERY
            else "runtime"
        )
    elif proposal.evidence_preset is EvidencePreset.MSPROF:
        kinds.append("profiling")
    return tuple(kinds)


class Resolver:
    def __init__(self, records=None):
        self.records = records if records is not None else {
            evidence(ordinal, kind): AuthoritativeEvidence(
                kind,
                evidence(ordinal, kind),
                proposal.project_id,
                "b" * 64,
                fact_statement(ordinal, kind),
                True,
            )
            for ordinal, proposal in enumerate(DEFAULT_PROPOSALS, 1)
            for kind in completion_kinds(proposal)
        }

    def resolve(self, digest):
        return self.records.get(digest)


def memory(ordinal, *, cell_id=CELL, lineage_id=LINEAGE):
    proposal = DEFAULT_PROPOSALS[ordinal - 1]
    return ProjectMemory(
        proposal=proposal,
        project_id=proposal.project_id,
        cell_id=cell_id,
        lineage_id=lineage_id,
        source_revision="b" * 64,
        actions=("compile candidate", "run host verifier"),
        host_facts=tuple(
            HostFact(kind, fact_statement(ordinal, kind), evidence(ordinal, kind))
            for kind in completion_kinds(proposal)
        ),
        interpretations=(
            AgentInterpretation(
                "The result suggests a reusable lesson.", (evidence(ordinal),)
            ),
        ),
    )


def journal(path, *, cell_id=CELL, lineage_id=LINEAGE, resolver=None):
    return Phase1LearningJournal(
        path,
        DEFAULT_PROPOSALS,
        cell_id=cell_id,
        lineage_id=lineage_id,
        evidence_resolver=resolver or Resolver(),
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


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("kind", "profiling", "kind"),
        ("evidence_sha256", "f" * 64, "digest"),
        ("project_id", "f" * 64, "project"),
        ("candidate_sha256", "f" * 64, "candidate"),
        ("success", False, "success"),
    ],
)
def test_append_requires_matching_authoritative_host_result(
    tmp_path, field, value, message
):
    record = AuthoritativeEvidence(
        "host-verification",
        EVIDENCE,
        DEFAULT_PROPOSALS[0].project_id,
        "b" * 64,
        fact_statement(1, "host-verification"),
        True,
    )
    values = record.__dict__.copy()
    values[field] = value
    resolver = Resolver({EVIDENCE: AuthoritativeEvidence(**values)})

    with pytest.raises(ValueError, match=message):
        journal(tmp_path / "unverified.jsonl", resolver=resolver).append(memory(1))


def test_append_rejects_fabricated_statement_with_real_evidence(tmp_path):
    legitimate = memory(1)
    fabricated = ProjectMemory(
        **{
            **legitimate.__dict__,
            "host_facts": (
                HostFact("host-verification", "fabricated timing: 1 ns", EVIDENCE),
            ),
        }
    )

    with pytest.raises(ValueError, match="statement"):
        journal(tmp_path / "fabricated.jsonl").append(fabricated)


@pytest.mark.parametrize("ordinal", [1, 3, 4, 5, 8])
def test_completion_accepts_successful_preset_evidence(tmp_path, ordinal):
    proposal = DEFAULT_PROPOSALS[ordinal - 1]
    active = Phase1LearningJournal(
        tmp_path / f"preset-{ordinal}.jsonl",
        (proposal,),
        cell_id=CELL,
        lineage_id=LINEAGE,
        evidence_resolver=Resolver(),
    )

    active.append(memory(ordinal))

    state = active.resume_state()
    assert state.completed_projects == 1
    assert state.next_ordinal is None


@pytest.mark.parametrize(
    ("ordinal", "missing_kind"),
    [
        (3, "compile"),
        (4, "runtime"),
        (5, "runtime"),
        (8, "profiling"),
    ],
)
def test_supplementary_facts_do_not_advance_without_preset_evidence(
    tmp_path, ordinal, missing_kind
):
    proposal = DEFAULT_PROPOSALS[ordinal - 1]
    complete = memory(ordinal)
    supplementary = ProjectMemory(
        **{
            **complete.__dict__,
            "host_facts": tuple(
                fact for fact in complete.host_facts if fact.category != missing_kind
            ),
        }
    )
    active = Phase1LearningJournal(
        tmp_path / f"supplementary-{ordinal}.jsonl",
        (proposal,),
        cell_id=CELL,
        lineage_id=LINEAGE,
        evidence_resolver=Resolver(),
    )

    active.append(supplementary)
    pending = active.resume_state()
    assert pending.completed_projects == 0
    assert pending.next_ordinal == 1
    assert len(active.read()) == 1

    active.append(complete)
    assert active.resume_state().completed_projects == 1
    assert len(active.read()) == 2


def test_failed_verification_is_retained_but_does_not_complete_project(tmp_path):
    proposal = DEFAULT_PROPOSALS[0]
    failed_digest = evidence(1, failed=True)
    failed_statement = "host verification failed for project 1"
    records = dict(Resolver().records)
    records[failed_digest] = AuthoritativeEvidence(
        "host-verification",
        failed_digest,
        proposal.project_id,
        "b" * 64,
        failed_statement,
        False,
    )
    failed = ProjectMemory(
        **{
            **memory(1).__dict__,
            "host_facts": (
                HostFact(
                    "host-verification", failed_statement, failed_digest, success=False
                ),
            ),
            "interpretations": (),
        }
    )
    active = Phase1LearningJournal(
        tmp_path / "failed.jsonl",
        (proposal,),
        cell_id=CELL,
        lineage_id=LINEAGE,
        evidence_resolver=Resolver(records),
    )

    active.append(failed)
    assert active.resume_state().completed_projects == 0
    assert len(active.read()) == 1

    active.append(memory(1))
    assert active.resume_state().completed_projects == 1
    assert len(active.read()) == 2


def test_compile_fact_is_supplementary_for_correctness_project(tmp_path):
    proposal = DEFAULT_PROPOSALS[0]
    compile_digest = evidence(1, "compile")
    records = dict(Resolver().records)
    records[compile_digest] = AuthoritativeEvidence(
        "compile",
        compile_digest,
        proposal.project_id,
        "b" * 64,
        fact_statement(1, "compile"),
        True,
    )
    compile_only = ProjectMemory(
        **{
            **memory(1).__dict__,
            "host_facts": (
                HostFact("compile", fact_statement(1, "compile"), compile_digest),
            ),
            "interpretations": (),
        }
    )
    active = Phase1LearningJournal(
        tmp_path / "compile-only.jsonl",
        (proposal,),
        cell_id=CELL,
        lineage_id=LINEAGE,
        evidence_resolver=Resolver(records),
    )

    active.append(compile_only)
    assert active.resume_state().completed_projects == 0

    active.append(memory(1))
    assert active.resume_state().completed_projects == 1


def test_supplementary_entry_does_not_advance_checkpoint_or_saturation(tmp_path):
    active = journal(tmp_path / "checkpoint.jsonl")
    for ordinal in range(1, 5):
        active.append(memory(ordinal))
    fifth = memory(5)
    active.append(
        ProjectMemory(
            **{
                **fifth.__dict__,
                "host_facts": tuple(
                    fact
                    for fact in fifth.host_facts
                    if fact.category == "host-verification"
                ),
            }
        )
    )

    state = active.resume_state()
    assert state.completed_projects == 4
    assert state.next_ordinal == 5
    assert state.latest_checkpoint == 4
    assert state.checkpoint_status == "CONTINUE"


def test_append_rejects_unresolved_host_fact_without_writing(tmp_path):
    active = journal(tmp_path / "missing.jsonl", resolver=Resolver({}))

    with pytest.raises(ValueError, match="resolve"):
        active.append(memory(1))

    assert not active.path.exists()


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
