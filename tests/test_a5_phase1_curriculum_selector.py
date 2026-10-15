from __future__ import annotations

import json

import pytest

from benchmarks.a5kernels.phase1_curriculum import Capability
from benchmarks.a5kernels.phase1_curriculum_selector import (
    CurriculumAgent,
    CurriculumContext,
    CurriculumSession,
    ProjectInformation,
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
        "graph": {
            "inputs": [{"name": "x", "shape": [257]}],
            "nodes": [
                {"name": "y", "op": "copy", "operands": ["x"], "axis": None},
            ],
            "output": "y",
        },
    }
    value.update(updates)
    return value


def with_shape(item, shape):
    item = dict(item)
    item["shape"] = list(shape)
    item["graph"] = {
        "inputs": [{"name": "x", "shape": list(shape)}],
        "nodes": [{"name": "y", "op": "copy", "operands": ["x"], "axis": None}],
        "output": "y",
    }
    return item


def outcome(item, project_id="old"):
    return PriorOutcome(project_id, ProjectInformation.from_project(item), True, "verified")


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
        "prior", ProjectInformation.from_project(prior_spec), True,
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
        "timing_lengths", '"graph"', '"copy"',
        "DECISION: WAVE", "DECISION: SATURATED",
    ))
    assert all(term not in prompt.lower() for term in (
        "openai", "deepseek", "temperature", "api_key",
    ))
    grammar = json.loads(prompt.splitlines()[-1])["bounded_project_grammar"]
    assert grammar["padding_multiples"] == [1, 16, 32, 64, 128, 256]
    assert grammar["block_count_bounds"] == [1, 32]
    assert "[1, 4096]" in grammar["shape"]
    assert "65536" in grammar["shape"]
    assert "[1, 4096]" in grammar["timing_lengths"]
    assert "one-dimensional" in grammar["timing_lengths"]
    low_precision = grammar["low_precision_rules"]
    fp8 = low_precision["float8_e4m3fn/float8_e5m2"]
    mx = low_precision["mxfp8_e4m3fn/mxfp8_e5m2"]
    assert "exactly one matmul" in fp8 and "no padded-tail" in fp8
    assert "copy-only graphs preserve FP8" in fp8
    assert "every non-copy FP8 graph computes all intermediates and output in FP32" in fp8
    assert "only one matmul" in mx and "inputs in order" in mx
    assert "no padded-tail" in mx and "FP32 output" in mx
    assert "per 32 K" in mx and "gm_scale0 and gm_scale1 before gm_output" in mx
    rules = grammar["capability_operation_rules"]
    assert "if and only if" in rules["rule"]
    assert rules["simd-vector"] == ["add", "multiply", "relu"]
    assert rules["simt-reduction"] == ["reduce-sum"]
    assert rules["simt-mask"] == ["mask-positive"]
    assert rules["cube-tiling"] == ["matmul"]
    assert "copy alone does not require" in rules["gm-ub-movement"]


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
        prior_outcomes=(outcome(project(), "movement"),),
    )
    second, session = agent.select(next_context, session)
    assert isinstance(second, TerminalDecision)
    assert session.turns == 2
    assert calls[1][:3] == calls[0] + ({"role": "assistant", "content": wave([project()])},)


@pytest.mark.parametrize("invalid", [
    "DECISION: WAVE\n{broken",
    wave([project(capabilities=["gm-ub-movement", "simd-vector"])]),
])
def test_invalid_selection_is_corrected_with_full_session_history(invalid):
    calls = []
    valid = wave([project()])

    def complete(messages):
        calls.append(messages)
        return invalid if len(calls) == 1 else valid

    decision, session = CurriculumAgent(complete).select(context())
    assert isinstance(decision, WaveDecision)
    assert len(calls) == 2
    assert calls[1][:-1] == calls[0] + ({"role": "assistant", "content": invalid},)
    assert "validation" in calls[1][-1]["content"]
    assert "unchanged" in calls[1][-1]["content"]
    assert session.turns == 2
    assert len(session.messages) == 5
    assert session.messages[-1]["content"] == valid
    CurriculumSession(session.messages, session.turns)
    _, next_session = CurriculumAgent(lambda _: valid).select(context(), session)
    assert next_session.messages[:5] == session.messages
    assert next_session.turns == 3


def test_selection_validation_exhaustion_is_bounded_and_diagnostic():
    calls = []

    def complete(messages):
        calls.append(messages)
        return wave([project(capabilities=["simd-vector"])])

    with pytest.raises(ValueError, match="3 attempts.*simd-vector"):
        CurriculumAgent(complete).select(context())
    assert len(calls) == 3
    assert [len(messages) for messages in calls] == [2, 4, 6]


def test_provider_value_error_is_not_retried_as_proposal_validation():
    calls = []
    failure = ValueError("provider refused request")

    def complete(messages):
        calls.append(messages)
        raise failure

    with pytest.raises(ValueError) as caught:
        CurriculumAgent(complete).select(context())
    assert caught.value is failure
    assert len(calls) == 1


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
    {**project(), "source": "kernel code"},
    {**project(), "command": "python run.py"},
])
def test_held_out_names_and_unsupported_payloads_are_rejected(bad_project):
    with pytest.raises(ValueError):
        CurriculumAgent(lambda _: wave([bad_project])).select(context())


def test_rejects_repeats_no_information_and_sibling_dependencies():
    item = project()
    prior = outcome(item)
    with pytest.raises(ValueError, match="repeats prior information"):
        CurriculumAgent(lambda _: wave([item])).select(context(prior_outcomes=(prior,)))

    no_information = project("retuned", block_count=2)
    with pytest.raises(ValueError, match="repeats prior information"):
        CurriculumAgent(lambda _: wave([no_information], coverage=[
            "gm-ub-movement"
        ])).select(context(
            pre_wave_coverage=(Capability.GM_UB_MOVEMENT,), prior_outcomes=(prior,)
        ))

    response = json.loads(wave([project()]).split("\n", 1)[1])
    response["projects"].append({
        "spec": project("consumer", ["simt-mask"], graph={
            "inputs": [{"name": "x", "shape": [257]}],
            "nodes": [{
                "name": "y", "op": "mask-positive", "operands": ["x"], "axis": None,
            }],
            "output": "y",
        }),
        "requires": ["gm-ub-movement"],
    })
    with pytest.raises(ValueError, match="pre-wave coverage"):
        CurriculumAgent(lambda _: "DECISION: WAVE\n" + json.dumps(response)).select(context())


@pytest.mark.parametrize("item", [
    with_shape(project("new-shape"), [1025]),
    project("new-dtype", dtype="float16"),
    project(
        "new-operation", ["simd-vector"],
        graph={
            "inputs": [{"name": "x", "shape": [257]}],
            "nodes": [{"name": "y", "op": "relu", "operands": ["x"], "axis": None}],
            "output": "y",
        },
    ),
    project(
        "new-failure", ["gm-ub-movement", "failure-repair"],
        failure_mode="compile",
    ),
    project(
        "new-profile", ["gm-ub-movement", "profiling"],
        profile_metric="timeline",
    ),
    project(
        "new-timing-shapes", ["gm-ub-movement", "timing-sweep"],
        timing_lengths=[64, 128],
    ),
])
def test_information_gain_uses_executable_dimensions_not_capability_names(item):
    prior = outcome(project())
    decision, _ = CurriculumAgent(lambda _: wave([item], coverage=[
        "gm-ub-movement"
    ])).select(context(
        pre_wave_coverage=(Capability.GM_UB_MOVEMENT,), prior_outcomes=(prior,)
    ))
    assert isinstance(decision, WaveDecision)


@pytest.mark.parametrize(
    ("item", "capability"),
    [
        (
            project(
                "timing", ["gm-ub-movement", "timing-sweep"],
                timing_lengths=[64, 128],
            ),
            Capability.TIMING_SWEEP,
        ),
        (
            project(
                "profile", ["gm-ub-movement", "profiling"],
                profile_metric="timeline",
            ),
            Capability.PROFILING,
        ),
    ],
)
def test_unavailable_factory_capabilities_are_prompted_and_rejected(item, capability):
    prompts = []

    def complete(messages):
        prompts.append(messages[-1]["content"])
        return wave([item])

    with pytest.raises(ValueError, match="unavailable"):
        CurriculumAgent(
            complete, unavailable_capabilities=(capability,),
        ).select(context())

    assert '"unavailable_capabilities"' in prompts[0]
    assert capability.value in prompts[0]


def test_missing_or_invalid_graph_is_rejected_without_compatibility_default():
    missing = project()
    missing.pop("graph")
    invalid = project(graph={
        "inputs": [{"name": "x", "shape": [257]}],
        "nodes": [{"name": "y", "op": "unknown", "operands": ["x"], "axis": None}],
        "output": "y",
    })
    for item in (missing, invalid):
        with pytest.raises(ValueError):
            CurriculumAgent(lambda _, item=item: wave([item])).select(context())


def test_host_private_held_out_signature_is_rejected_without_entering_prompt():
    item = project(
        "generic-cube", ["cube-tiling"], shape=[64, 64, 64],
        graph={
            "inputs": [
                {"name": "a", "shape": [64, 64]},
                {"name": "b", "shape": [64, 64]},
            ],
            "nodes": [
                {"name": "c", "op": "matmul", "operands": ["a", "b"], "axis": None},
            ],
            "output": "c",
        },
    )
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
        wave([with_shape(project("new-shape"), [4096])],
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
            pre_wave_coverage=tuple(dict.fromkeys(ctx.pre_wave_coverage + (learned,))),
            prior_outcomes=ctx.prior_outcomes + (PriorOutcome(
                decision.wave.projects[0].spec.project_id,
                ProjectInformation.from_project(decision.wave.projects[0].spec),
                True, "host-verified synthetic proof outcome",
            ),),
        )
    assert observed == [(), (Capability.GM_UB_MOVEMENT,)]
    assert ctx.pre_wave_coverage == (
        Capability.GM_UB_MOVEMENT,
    )
    assert session.turns == 3


def test_terminal_decisions_require_host_observations_and_hide_held_out_names():
    for reply in (
        "DECISION: SATURATED\n" + json.dumps({"reason": "looks done"}),
        "DECISION: STALLED\n" + json.dumps({"reason": "nothing to do"}),
        "DECISION: SATURATED\n" + json.dumps({"reason": "GDN is covered"}),
    ):
        with pytest.raises(ValueError):
            CurriculumAgent(lambda _, reply=reply: reply).select(context())


def test_unverified_prior_can_be_retried_and_does_not_count_as_coverage():
    item = project()
    failed = PriorOutcome(
        "movement", ProjectInformation.from_project(item), False,
        "candidate failed host verification",
    )
    decision, _ = CurriculumAgent(lambda _: wave([item])).select(
        context(prior_outcomes=(failed,))
    )
    assert isinstance(decision, WaveDecision)


def test_siblings_need_marginal_semantic_gain_not_exact_shape_variation():
    first = project("first")
    second = with_shape(project("second"), [259])
    with pytest.raises(ValueError, match="marginal"):
        CurriculumAgent(lambda _: wave([first, second])).select(context())


def test_held_out_names_are_rejected_in_graph_identifiers():
    item = project(graph={
        "inputs": [{"name": "gdn", "shape": [257]}],
        "nodes": [{
            "name": "y", "op": "copy", "operands": ["gdn"], "axis": None,
        }],
        "output": "y",
    })
    with pytest.raises(ValueError, match="held-out"):
        CurriculumAgent(lambda _: wave([item])).select(context())


def test_long_explanations_and_distinct_learning_focus_are_admitted():
    first = project("matmul-study", hypothesis=("Question: matrix multiplication. " * 80).strip(),
                    learning_focus=["sync-correctness"])
    second = project("performance", learning_focus=["sync-performance"])
    decision, _ = CurriculumAgent(lambda _: wave([first, second])).select(context())
    assert len(decision.wave.projects) == 2
    assert decision.wave.projects[0].spec.hypothesis == first["hypothesis"].strip()


def test_wave_cannot_overshoot_remaining_target():
    reply = wave([project(), project("second", learning_focus=["debug-print"])])
    with pytest.raises(ValueError, match="remaining project target"):
        CurriculumAgent(lambda _: reply).select(context(remaining_projects=1))


def test_documented_mxfp8_example_matches_published_grammar():
    from pathlib import Path
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    text = Path("docs/ASCEND_CURRICULUM_EXPERIMENTS.md").read_text()
    example = json.loads(text.split("```json\n", 1)[1].split("```", 1)[0])
    spec = ProjectSpec.from_mapping(example)
    assert spec.dtype.value == "mxfp8_e4m3fn"
    decision, _ = CurriculumAgent(lambda _: wave([example])).select(context())
    assert decision.wave.projects[0].spec == spec


def test_small_adaptive_waves_are_bounded_by_remaining_target():
    from benchmarks.a5kernels.phase1_curriculum_selector import curriculum_prompt
    for remaining, maximum in ((3, 3), (64, 8), (128, 8)):
        prompt = curriculum_prompt(context(remaining_projects=remaining))
        payload = json.loads(prompt.rsplit("\n", 1)[-1])
        assert payload["max_wave_projects"] == maximum
        projects = [with_shape(project(f"item-{i}"), (i + 1,))
                    for i in range(maximum + 1)]
        with pytest.raises(ValueError, match="wave limit or remaining"):
            CurriculumAgent(lambda _: wave(projects)).select(
                context(remaining_projects=remaining))


def test_prompt_defines_dependencies_axes_masks_and_output_dtype():
    from benchmarks.a5kernels.phase1_curriculum_selector import curriculum_prompt
    grammar = json.loads(curriculum_prompt(context()).rsplit("\n", 1)[-1])["bounded_project_grammar"]
    assert "never project IDs" in grammar["requires"]
    assert "[] for the initial wave" in grammar["requires"]
    assert "zero-based axis" in grammar["graph"]["axis"]
    assert "null for every non-reduction" in grammar["graph"]["axis"]
    assert "1 if x > 0 else 0" in grammar["graph"]["mask-positive"]
    assert "input dtype by default" in grammar["output_dtype"]
    assert "matmul outputs float32" in grammar["output_dtype"]
    assert "copy-only graphs preserve FP8" in grammar["output_dtype"]
    assert "every non-copy FP8 graph computes intermediates and output in float32" in grammar["output_dtype"]


def test_learning_focus_cannot_bypass_held_out_executable_signature():
    signature = project_information_sha256(project())
    for focus in (["tool-msprof"], ["sync-performance", "pipe-mte2"], ["pipe-mte2", "sync-performance"]):
        proposal = project(learning_focus=focus)
        with pytest.raises(ValueError, match="held-out contract"):
            CurriculumAgent(lambda _: wave([proposal]), held_out_information_sha256=[signature]).select(context())
