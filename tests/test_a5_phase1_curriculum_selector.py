from __future__ import annotations

import json

import pytest

from benchmarks.a5kernels.phase1_curriculum import Capability
from benchmarks.a5kernels.phase1_curriculum_selector import (
    CurriculumAgent,
    CurriculumContext,
    CurriculumSession,
    PriorOutcome,
    TerminalDecision,
    WaveDecision,
    project_information_sha256,
)


MEMORY = "a" * 64


def project(project_id="movement", capabilities=None, **updates):
    value = {
        "project_id": project_id,
        "capabilities": capabilities or ["gm-ub-movement"],
        "shape": [257],
        "dtype": "float32",
        "block_count": 1,
        "padding_multiple": 64,
        "failure_mode": "none",
        "timing_lengths": [],
        "profile_metric": None,
        "hypothesis": "Host evidence will test one reusable Catlass behavior.",
    }
    value.update(updates)
    return value


def wave(projects, *, coverage=(), memory=MEMORY, wave_id="wave-001"):
    return "DECISION: WAVE\n" + json.dumps({
        "wave_id": wave_id,
        "pre_wave_memory_sha256": memory,
        "pre_wave_coverage": list(coverage),
        "projects": [{"spec": item, "requires": []} for item in projects],
    })


def context(**updates):
    values = {
        "objective": "Learn reusable Catlass DSL kernel programming from host evidence.",
        "wave_index": 1,
        "pre_wave_memory_sha256": MEMORY,
        "pre_wave_coverage": (),
        "prior_outcomes": (),
    }
    values.update(updates)
    return CurriculumContext(**values)


def test_prompt_is_model_neutral_and_contains_objective_coverage_outcomes_and_grammar():
    prior_spec = project("prior")
    prior = PriorOutcome(
        "prior", project_information_sha256(prior_spec), True,
        "Movement passed host verification.",
    )
    calls = []

    def complete(messages):
        calls.append(messages)
        return "DECISION: SATURATED\n" + json.dumps({"reason": "coverage complete"})

    decision, _ = CurriculumAgent(complete).select(context(
        pre_wave_coverage=(Capability.GM_UB_MOVEMENT,), prior_outcomes=(prior,)
    ))
    prompt = calls[0][-1]["content"]
    assert isinstance(decision, TerminalDecision)
    assert all(term in prompt for term in (
        "Learn reusable Catlass", "gm-ub-movement", "Movement passed",
        "timing_lengths", "DECISION: WAVE", "DECISION: SATURATED",
    ))
    assert all(term not in prompt.lower() for term in (
        "openai", "deepseek", "temperature", "api_key",
    ))


def test_strict_parser_accepts_wave_and_preserves_context_across_turns():
    replies = [
        wave([project()]),
        "DECISION: SATURATED\n" + json.dumps({"reason": "all capabilities covered"}),
    ]
    calls = []

    def complete(messages):
        calls.append(tuple(messages))
        return replies.pop(0)

    agent = CurriculumAgent(complete)
    first, session = agent.select(context())
    assert isinstance(first, WaveDecision)
    assert first.wave.projects[0].spec.project_id == "movement"
    next_context = context(
        wave_index=2,
        pre_wave_coverage=(Capability.GM_UB_MOVEMENT,),
        prior_outcomes=(PriorOutcome(
            "movement", project_information_sha256(project()), True, "verified",
        ),),
    )
    second, session = agent.select(next_context, session)
    assert isinstance(second, TerminalDecision)
    assert session.turns == 2
    assert calls[1][:3] == calls[0] + ({"role": "assistant", "content": wave([project()])},)


@pytest.mark.parametrize("reply", [
    "DECISION: PROJECT\n{}",
    "prefix\nDECISION: SATURATED\n{\"reason\":\"done\"}",
    "DECISION: SATURATED\n{\"reason\":\"done\",\"extra\":1}",
    "DECISION: WAVE\n{}\ntrailing",
])
def test_decisions_are_strict(reply):
    with pytest.raises(ValueError):
        CurriculumAgent(lambda _: reply).select(context())


@pytest.mark.parametrize("bad_project", [
    project("gdn-study"),
    project("bsa-study"),
    project("matmul-study"),
    {**project(), "source": "kernel code"},
    {**project(), "command": "python run.py"},
])
def test_held_out_names_and_unsupported_payloads_are_rejected(bad_project):
    with pytest.raises(ValueError):
        CurriculumAgent(lambda _: wave([bad_project])).select(context())


def test_rejects_repeats_no_information_and_sibling_dependencies():
    item = project()
    prior = PriorOutcome("old", project_information_sha256(item), True, "verified")
    with pytest.raises(ValueError, match="repeats prior information"):
        CurriculumAgent(lambda _: wave([item])).select(context(prior_outcomes=(prior,)))

    with pytest.raises(ValueError, match="new coverage"):
        CurriculumAgent(lambda _: wave([project(
            capabilities=["gm-ub-movement"], shape=[513]
        )], coverage=["gm-ub-movement"])).select(context(
            pre_wave_coverage=(Capability.GM_UB_MOVEMENT,)
        ))

    response = json.loads(wave([project()]).split("\n", 1)[1])
    response["projects"].append({
        "spec": project("consumer", ["simt-mask"]),
        "requires": ["gm-ub-movement"],
    })
    with pytest.raises(ValueError, match="pre-wave coverage"):
        CurriculumAgent(lambda _: "DECISION: WAVE\n" + json.dumps(response)).select(context())


def test_host_private_held_out_signature_is_rejected_without_entering_prompt():
    item = project("generic-cube", ["cube-tiling"], shape=[64, 64])
    signature = project_information_sha256(item)
    seen = []

    def complete(messages):
        seen.append(messages[-1]["content"])
        return wave([item])

    with pytest.raises(ValueError, match="held-out contract"):
        CurriculumAgent(complete, held_out_information_sha256=(signature,)).select(context())
    assert signature not in seen[0]


def test_scripted_multi_wave_proof_grows_coverage_then_saturates():
    replies = iter([
        wave([project()]),
        wave([project("tail", ["padded-tail"], shape=[259])],
             coverage=["gm-ub-movement"], wave_id="wave-002"),
        "DECISION: SATURATED\n" + json.dumps({"reason": "requested coverage achieved"}),
    ])
    agent = CurriculumAgent(lambda _: next(replies))
    ctx = context()
    session = CurriculumSession()
    observed = []
    for _ in range(3):
        decision, session = agent.select(ctx, session)
        if isinstance(decision, TerminalDecision):
            assert decision.status == "SATURATED"
            break
        observed.append(decision.wave.pre_wave_coverage)
        learned = decision.wave.projects[0].spec.capabilities[-1]
        ctx = context(
            wave_index=ctx.wave_index + 1,
            pre_wave_coverage=ctx.pre_wave_coverage + (learned,),
            prior_outcomes=ctx.prior_outcomes + (PriorOutcome(
                decision.wave.projects[0].spec.project_id,
                project_information_sha256(decision.wave.projects[0].spec),
                True, "host-verified synthetic proof outcome",
            ),),
        )
    assert observed == [(), (Capability.GM_UB_MOVEMENT,)]
    assert ctx.pre_wave_coverage == (
        Capability.GM_UB_MOVEMENT, Capability.PADDED_TAIL,
    )
    assert session.turns == 3
