"""Generated metadata is repaired; graph semantics and retained records are strict."""
from copy import deepcopy
from itertools import product
import json

import pytest

from benchmarks.a5kernels.phase1_curriculum import Capability, ProjectSpec, WaveSpec
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent, curriculum_prompt
from benchmarks.a5kernels.phase1_project import ProjectGraph
from benchmarks.ascend.targets import a3_curriculum_policy
from test_a5_phase1_curriculum_selector import context, project, wave


@pytest.mark.parametrize("output", [{"name": "y"}, ["y"], None, 1, "x", "missing"])
def test_output_is_validation_error_and_enters_bounded_correction(output):
    invalid = project()
    invalid["graph"]["output"] = output
    with pytest.raises(ValueError, match="string naming a computed node"):
        ProjectGraph.from_mapping(invalid["graph"])
    replies = [wave([invalid]), wave([project()])]
    calls = []

    def complete(messages):
        calls.append(messages)
        return replies[len(calls) - 1]

    decision, session = CurriculumAgent(complete).select(context())
    assert len(calls) == 2
    assert "string naming a computed node" in calls[1][-1]["content"]
    assert session.messages[2]["content"] == replies[0]
    assert decision.wave.projects[0].spec.graph.output == "y"


@pytest.mark.parametrize("target", ["a5", "a3"])
@pytest.mark.parametrize("flags", list(product([False, True], repeat=4)))
def test_operation_combinations_use_every_node_and_target_labels(target, flags):
    arithmetic, reduction, mask, cube = flags
    policy = a3_curriculum_policy() if target == "a3" else None
    item = project(capabilities=["gm-ub-movement", "padded-tail", "simt-mask", "vector-reduction"])
    graph = {"inputs": [{"name": "x", "shape": [2, 2]}], "nodes": [], "output": "y"}
    previous = "x"
    operations = (["matmul"] if cube else []) + (["add", "multiply", "relu"] if arithmetic else [])
    operations += (["mask-positive"] if mask else []) + (["reduce-sum"] if reduction else []) + ["copy"]
    for index, op in enumerate(operations):
        name = f"n{index}"
        graph["nodes"].append({"name": name, "op": op, "operands":
            [previous, previous] if op in {"matmul", "add", "multiply"} else [previous],
            "axis": 0 if op == "reduce-sum" else None})
        previous = name
    graph["output"] = previous
    item["graph"] = graph
    proposal = json.loads(wave([item]).split("\n", 1)[1])
    original = deepcopy(proposal)
    normalized = CurriculumAgent(lambda _: "", target_policy=policy)._normalize_generated_wave(proposal)
    labels = normalized["projects"][0]["spec"]["capabilities"]
    expected = ["gm-ub-movement", "padded-tail"]
    if arithmetic:
        expected.append("simd-vector")
    if reduction:
        expected.append("vector-reduction" if target == "a3" else "simt-reduction")
    if mask:
        expected.append("vector-mask" if target == "a3" else "simt-mask")
    if cube:
        expected.append("cube-tiling")
    assert labels == expected
    assert proposal == original
    assert normalized["projects"][0]["requires"] == []
    assert normalized["projects"][0]["spec"]["graph"] == graph


@pytest.mark.parametrize("target", ["a5", "a3"])
@pytest.mark.parametrize("op", ["copy", "add", "multiply", "relu", "reduce-sum", "mask-positive", "matmul"])
def test_generated_wave_persists_derived_spec_without_rewriting_response(target, op):
    policy = a3_curriculum_policy() if target == "a3" else None
    item = project(capabilities=["gm-ub-movement", "simd-vector", "simt-mask"])
    shape = [2, 2] if op == "matmul" else [257]
    item["shape"] = [2, 2, 2] if op == "matmul" else shape
    item["graph"]["inputs"][0]["shape"] = shape
    item["graph"]["nodes"][0].update(op=op,
        operands=["x", "x"] if op in {"add", "multiply", "matmul"} else ["x"],
        axis=0 if op == "reduce-sum" else None)
    if op == "matmul":
        item["graph"]["inputs"].append({"name": "z", "shape": shape})
        item["graph"]["nodes"][0]["operands"] = ["x", "z"]
    raw = wave([item])
    decision, session = CurriculumAgent(lambda _: raw, target_policy=policy).select(context())
    assert session.messages[-1]["content"] == raw
    retained = decision.wave.as_dict()
    assert WaveSpec.from_mapping(retained) == decision.wave
    expected = {"copy": None, "add": "simd-vector", "multiply": "simd-vector", "relu": "simd-vector",
        "reduce-sum": "vector-reduction" if target == "a3" else "simt-reduction",
        "mask-positive": "vector-mask" if target == "a3" else "simt-mask", "matmul": "cube-tiling"}[op]
    assert [c.value for c in decision.wave.projects[0].spec.capabilities] == ["gm-ub-movement"] + ([expected] if expected else [])
    # Strict retained-data loading must never derive or remove labels.
    with pytest.raises(ValueError, match="capability must agree"):
        ProjectSpec.from_mapping(item)
    with pytest.raises(ValueError, match="capability must agree"):
        WaveSpec.from_mapping(json.loads(raw.split("\n", 1)[1]))


@pytest.mark.parametrize("change, diagnostic", [
    (lambda p: p.update(capabilities=["gm-ub-movement", "invented"]), "unregistered capability"),
    (lambda p: p.pop("capabilities"), "project fields"),
    (lambda p: p.update(capabilities=["gm-ub-movement", "gm-ub-movement"]), "unique"),
    (lambda p: p["graph"]["nodes"][0].update(operands=["absent"]), "previously defined"),
    (lambda p: p.update(shape=[256]), "shape"),
    (lambda p: p.update(failure_mode="compile"), "failure-repair"),
    (lambda p: p.update(profile_metric="pipe-utilization"), "profiling"),
    (lambda p: p["graph"]["nodes"][0].update(axis=0), "no axis"),
])
def test_normalization_does_not_hide_invalid_proposals(change, diagnostic):
    item = project()
    change(item)
    calls = []

    def complete(messages):
        calls.append(messages)
        return wave([item])

    with pytest.raises(ValueError, match=f"3 attempts.*{diagnostic}"):
        CurriculumAgent(complete).select(context())
    assert len(calls) == 3


def test_dependencies_are_not_derived_from_project_operations():
    raw = json.loads(wave([project()]).split("\n", 1)[1])
    raw["projects"][0]["requires"] = ["simd-vector"]
    with pytest.raises(ValueError, match="unsupported requires"):
        CurriculumAgent(lambda _: "DECISION: WAVE\n" + json.dumps(raw)).select(context())


@pytest.mark.parametrize("policy", [None, a3_curriculum_policy()])
def test_grammar_gives_valid_typed_graph_and_explains_derivation(policy):
    grammar = json.loads(curriculum_prompt(context(), target_policy=policy).splitlines()[-1])["bounded_project_grammar"]
    assert ProjectGraph.from_mapping(grammar["graph"]["example"]).output == "y"
    assert "string naming a computed node" in grammar["graph"]["types"]["output"]
    assert "derived automatically" in grammar["capability_operation_rules"]["rule"]
    assert "capabilities" in grammar["project_fields_exactly"]
