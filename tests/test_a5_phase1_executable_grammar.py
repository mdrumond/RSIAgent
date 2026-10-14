from __future__ import annotations

import pytest

from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_project import (
    EvidenceClassification,
    ProjectGraph,
    classify_execution,
    evaluate_reference,
    materialize_inputs,
)


def graph(**overrides: object) -> ProjectGraph:
    value: dict[str, object] = {
        "inputs": [
            {"name": "lhs", "shape": [2, 3]},
            {"name": "rhs", "shape": [2, 3]},
        ],
        "nodes": [
            {"name": "sum", "op": "add", "operands": ["lhs", "rhs"], "axis": None},
            {"name": "mask", "op": "mask-positive", "operands": ["sum"], "axis": None},
            {"name": "total", "op": "reduce-sum", "operands": ["mask"], "axis": 1},
        ],
        "output": "total",
    }
    value.update(overrides)
    return ProjectGraph.from_mapping(value)


def project(project_graph: ProjectGraph, **overrides: object) -> ProjectSpec:
    value: dict[str, object] = {
        "project_id": "generic-learning-project",
        "capabilities": [
            "gm-ub-movement", "simd-vector", "simt-mask", "simt-reduction",
        ],
        "shape": [2, 3],
        "dtype": "float32",
        "block_count": 1,
        "padding_multiple": 1,
        "failure_mode": "none",
        "timing_lengths": [],
        "profile_metric": None,
        "hypothesis": "Host output, not source inspection, decides correctness.",
        "graph": project_graph.as_dict(),
    }
    value.update(overrides)
    return ProjectSpec.from_mapping(value)


def test_graph_is_bounded_topological_and_round_trips():
    spec = graph()
    assert spec.output_shape == (2,)
    assert ProjectGraph.from_mapping(spec.as_dict()) == spec

    bad = spec.as_dict()
    bad["nodes"][0]["operands"] = ["future", "rhs"]
    with pytest.raises(ValueError, match="previously defined"):
        ProjectGraph.from_mapping(bad)

    with pytest.raises(ValueError, match="at most 12"):
        graph(nodes=[
            {"name": f"n{i}", "op": "copy", "operands": ["lhs"], "axis": None}
            for i in range(13)
        ], output="n12")


@pytest.mark.parametrize(
    "nodes, message",
    [
        ([{"name": "x", "op": "add", "operands": ["lhs"], "axis": None}], "two operands"),
        ([{"name": "x", "op": "reduce-sum", "operands": ["lhs"], "axis": 2}], "axis"),
        ([{"name": "x", "op": "matmul", "operands": ["lhs", "rhs"], "axis": None}], "matmul"),
    ],
)
def test_graph_rejects_invalid_operator_contracts(nodes, message):
    with pytest.raises(ValueError, match=message):
        graph(nodes=nodes, output="x")


def test_graph_covers_cube_without_a_benchmark_contract():
    cube = ProjectGraph.from_mapping({
        "inputs": [
            {"name": "a", "shape": [3, 4]},
            {"name": "b", "shape": [4, 2]},
        ],
        "nodes": [
            {"name": "c", "op": "matmul", "operands": ["a", "b"], "axis": None},
        ],
        "output": "c",
    })
    assert cube.output_shape == (3, 2)
    assert "benchmark" not in repr(cube).lower()


def test_host_inputs_are_deterministic_and_padding_is_not_logical_data():
    spec = project(
        ProjectGraph.from_mapping({
            "inputs": [{"name": "x", "shape": [2, 3]}],
            "nodes": [{"name": "y", "op": "copy", "operands": ["x"], "axis": None}],
            "output": "y",
        }),
        capabilities=["gm-ub-movement", "padded-tail"],
        padding_multiple=16,
    )
    first = materialize_inputs(spec)
    second = materialize_inputs(spec)

    assert first == second
    assert first[0].shape == (2, 3)
    assert first[0].physical_shape == (2, 16)
    assert len(first[0].logical_values) == 6
    assert len(first[0].storage) == 32
    expected = evaluate_reference(spec, first)
    assert expected.values == first[0].logical_values
    assert expected.physical_shape == (2, 16)
    assert len(expected.storage) == 32
    assert classify_execution(
        expected, compile_succeeded=True, actual=expected.values
    ).classification is EvidenceClassification.HOST_MISMATCH
    assert classify_execution(
        expected, compile_succeeded=True, actual=expected.storage
    ).classification is EvidenceClassification.HOST_VERIFIED


def test_pure_reference_evaluates_vector_mask_reduction_and_cube():
    vector_spec = project(graph())
    inputs = materialize_inputs(vector_spec)
    expected_rows = []
    for row in range(2):
        values = [
            inputs[0].logical_values[row * 3 + col]
            + inputs[1].logical_values[row * 3 + col]
            for col in range(3)
        ]
        expected_rows.append(sum(value > 0.0 for value in values))
    assert evaluate_reference(vector_spec, inputs).values == tuple(expected_rows)

    cube = ProjectGraph.from_mapping({
        "inputs": [{"name": "a", "shape": [2, 3]}, {"name": "b", "shape": [3, 2]}],
        "nodes": [{"name": "c", "op": "matmul", "operands": ["a", "b"], "axis": None}],
        "output": "c",
    })
    cube_spec = project(cube, capabilities=["cube-tiling"], shape=[2, 3, 2])
    result = evaluate_reference(cube_spec, materialize_inputs(cube_spec))
    assert result.shape == (2, 2)
    assert len(result.values) == 4


def test_evidence_never_claims_correctness_without_host_comparison():
    spec = project(graph())
    expected = evaluate_reference(spec, materialize_inputs(spec))

    assert classify_execution(expected, compile_succeeded=False).classification is EvidenceClassification.COMPILE_FAILURE
    assert classify_execution(expected, compile_succeeded=True).classification is EvidenceClassification.RUNTIME_FAILURE
    assert classify_execution(
        expected, compile_succeeded=True, runtime_error="device launch failed"
    ).classification is EvidenceClassification.RUNTIME_FAILURE
    assert classify_execution(
        expected, compile_succeeded=True, actual=[0.0, 0.0]
    ).classification is EvidenceClassification.HOST_MISMATCH
    verified = classify_execution(
        expected, compile_succeeded=True, actual=expected.values
    )
    assert verified.classification is EvidenceClassification.HOST_VERIFIED
    assert verified.verified is True


def test_project_graph_is_part_of_canonical_project_contract():
    spec = project(graph())
    assert spec.graph.output_shape == (2,)
    assert ProjectSpec.from_mapping(spec.as_dict()) == spec
    with pytest.raises(ValueError, match="fields must be exactly"):
        mapping = spec.as_dict()
        mapping.pop("graph")
        ProjectSpec.from_mapping(mapping)


def test_capability_claims_must_agree_with_executable_operations():
    with pytest.raises(ValueError, match="cube-tiling.*agree"):
        project(graph(), capabilities=["cube-tiling"], shape=[2, 3, 2])
    cube = ProjectGraph.from_mapping({
        "inputs": [{"name": "a", "shape": [2, 3]}, {"name": "b", "shape": [3, 2]}],
        "nodes": [{"name": "c", "op": "matmul", "operands": ["a", "b"], "axis": None}],
        "output": "c",
    })
    with pytest.raises(ValueError, match="cube-tiling.*agree"):
        project(cube)
