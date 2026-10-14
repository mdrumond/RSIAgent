from __future__ import annotations

import pytest

from benchmarks.a5kernels.phase1_curriculum import (
    Capability,
    FailureMode,
    ProjectOutcome,
    ProjectSpec,
    WaveProject,
    WaveSpec,
    consolidate_wave,
)


MEMORY = "a" * 64


def project(project_id: str, **overrides: object) -> ProjectSpec:
    value: dict[str, object] = {
        "project_id": project_id,
        "capabilities": ["gm-ub-movement", "simd-vector"],
        "shape": [257],
        "dtype": "float32",
        "block_count": 1,
        "padding_multiple": 64,
        "failure_mode": "none",
        "timing_lengths": [],
        "profile_metric": None,
        "hypothesis": "Host verification will distinguish valid movement from guesses.",
    }
    value.update(overrides)
    capabilities = set(value["capabilities"])
    op = "relu" if "simd-vector" in capabilities else "copy"
    nodes = [{"name": "y", "op": op, "operands": ["x"], "axis": None}]
    inputs = [{"name": "x", "shape": [4]}]
    if "simt-mask" in capabilities:
        nodes = [{"name": "mask", "op": "mask-positive", "operands": ["x"], "axis": None}]
    if "simt-reduction" in capabilities:
        prior = nodes[-1]["name"]
        nodes.append({"name": "total", "op": "reduce-sum", "operands": [prior], "axis": 0})
    if "cube-tiling" in capabilities:
        inputs = [{"name": "x", "shape": [2, 3]}, {"name": "z", "shape": [3, 2]}]
        nodes = [{"name": "y", "op": "matmul", "operands": ["x", "z"], "axis": None}]
    value.setdefault("graph", {"inputs": inputs, "nodes": nodes, "output": nodes[-1]["name"]})
    return ProjectSpec.from_mapping(value)


def test_project_grammar_composes_generic_catlass_capabilities():
    tail = project(
        "movement-tail",
        capabilities=["gm-ub-movement", "simd-vector", "padded-tail"],
    )
    simt = project(
        "simt-mask-reduce",
        capabilities=["simt-mask", "simt-reduction"],
        shape=[9, 127],
        padding_multiple=1,
    )
    cube = project(
        "cube-tile",
        capabilities=["cube-tiling", "gm-ub-movement"],
        shape=[64, 96, 128],
        dtype="float16",
        padding_multiple=16,
        block_count=4,
    )
    recovery = project(
        "compile-repair",
        capabilities=["failure-repair"],
        failure_mode="compile",
        shape=[64],
        padding_multiple=1,
    )
    performance = project(
        "timing-profile",
        capabilities=["timing-sweep", "profiling"],
        timing_lengths=[32, 64, 128],
        profile_metric="pipe-utilization",
        shape=[128],
        padding_multiple=1,
    )

    assert tail.capabilities[-1] is Capability.PADDED_TAIL
    assert simt.shape == (9, 127)
    assert cube.block_count == 4
    assert recovery.failure_mode is FailureMode.COMPILE
    assert performance.timing_lengths == (32, 64, 128)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"capabilities": ["failure-repair"]}, "failure-repair"),
        ({"failure_mode": "compile"}, "failure mode"),
        ({"capabilities": ["timing-sweep"]}, "timing-sweep"),
        ({"timing_lengths": [32, 64]}, "timing lengths"),
        ({"capabilities": ["profiling"]}, "profiling"),
        ({"profile_metric": "pipe-utilization"}, "profile metric"),
        ({"capabilities": ["cube-tiling"], "shape": [64]}, "cube-tiling"),
        ({"capabilities": ["padded-tail"], "shape": [64]}, "padded-tail"),
    ],
)
def test_project_grammar_rejects_incoherent_compositions(overrides, message):
    with pytest.raises(ValueError, match=message):
        project("invalid", **overrides)


@pytest.mark.parametrize("field", ["command", "argv", "benchmark", "source"])
def test_project_grammar_rejects_commands_and_benchmark_specific_escape_hatches(field):
    with pytest.raises(ValueError, match="fields must be exactly"):
        project("invalid", **{field: "gdn"})


def test_wave_preserves_curriculum_order_and_one_pre_wave_memory_identity():
    spec = WaveSpec(
        wave_id="wave-001",
        pre_wave_memory_sha256=MEMORY,
        pre_wave_coverage=(Capability.GM_UB_MOVEMENT,),
        projects=(
            WaveProject(project("first")),
            WaveProject(
                project("second", capabilities=["simt-mask"]),
                requires=(Capability.GM_UB_MOVEMENT,),
            ),
            WaveProject(project("third", capabilities=["simt-reduction"])),
        ),
    )

    assert [item.spec.project_id for item in spec.projects] == [
        "first", "second", "third",
    ]
    assert {item.pre_wave_memory_sha256 for item in spec.assignments()} == {MEMORY}
    assert [len(batch) for batch in spec.execution_batches(max_parallel=2)] == [2, 1]
    assert spec.semantic_width == 3


def test_wave_dependencies_must_exist_before_wave_and_cannot_name_siblings():
    with pytest.raises(ValueError, match="pre-wave coverage"):
        WaveSpec(
            "wave-001", MEMORY, (),
            (
                WaveProject(project("producer")),
                WaveProject(
                    project("consumer", capabilities=["simt-mask"]),
                    requires=(Capability.GM_UB_MOVEMENT,),
                ),
            ),
        )


def test_consolidation_is_all_or_nothing_and_authored_sequential():
    wave = WaveSpec(
        "wave-001", MEMORY, (),
        (WaveProject(project("alpha")), WaveProject(project("beta"))),
    )
    assignments = {item.project_id: item for item in wave.assignments()}

    def outcome(project_id, digest, *, verified=True, fingerprint=None, ordinal=None):
        assignment = assignments[project_id]
        return ProjectOutcome(
            project_id, digest,
            fingerprint or assignment.wave_fingerprint,
            ordinal or assignment.authored_ordinal,
            verified=verified,
        )

    with pytest.raises(ValueError, match="complete wave"):
        consolidate_wave(wave, (outcome("alpha", "1" * 64),))
    with pytest.raises(ValueError, match="verified"):
        consolidate_wave(wave, (
            outcome("alpha", "1" * 64),
            outcome("beta", "2" * 64, verified=False),
        ))
    with pytest.raises(ValueError, match="current wave assignments"):
        consolidate_wave(wave, (
            outcome("alpha", "1" * 64, fingerprint="f" * 64),
            outcome("beta", "2" * 64),
        ))

    metadata = consolidate_wave(wave, (
        outcome("beta", "2" * 64),
        outcome("alpha", "1" * 64),
    ))
    assert metadata.pre_wave_memory_sha256 == MEMORY
    assert [item.project_id for item in metadata.commits] == ["alpha", "beta"]
    assert [item.commit_ordinal for item in metadata.commits] == [1, 2]


def test_contract_round_trip_is_deterministic():
    wave = WaveSpec(
        "wave-001", MEMORY, (Capability.GM_UB_MOVEMENT,),
        (WaveProject(project("tail"), (Capability.GM_UB_MOVEMENT,)),),
    )
    assert WaveSpec.from_mapping(wave.as_dict()) == wave
    assert wave.fingerprint == WaveSpec.from_mapping(wave.as_dict()).fingerprint


def test_wave_requires_immutable_project_tuple_and_integer_padding():
    with pytest.raises(ValueError, match="immutable tuple"):
        WaveSpec("wave-001", MEMORY, (), [WaveProject(project("alpha"))])
    with pytest.raises(ValueError, match="registered alignment"):
        project("invalid", padding_multiple=True)
