"""References establish correctness feasibility; curriculum selects focused practice."""
from dataclasses import replace
import json

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.ascend.qualified_targets import TargetContract, qualify_target
from benchmarks.ascend.targets import a3_curriculum_policy
from test_a5_phase1_project_executor import FakeBackend, _compile, _runtime, _spec

RUNTIME = dict(target="a3", runtime="registered-native", product="observed-product",
               compiler="observed-compiler-revision", toolchain="observed-cann", device="0")


def qualified(tmp_path, *, wrong=False, timings=()):
    store = ContentAddressedArtifacts(tmp_path / "artifacts")
    spec = _spec("movement")
    backend = FakeBackend(_compile())
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "events.jsonl"),
                               runtime_identity=dict(RUNTIME))
    expected = executor.prepare(spec).expected.storage
    backend.runtime_result = _runtime([999.] * len(expected) if wrong else expected)
    target = qualify_target(spec, "private working reference", RUNTIME, executor, store,
                            policy=a3_curriculum_policy(), timings=timings)
    return target, store, backend


def test_qualification_executes_reference_and_rechecks_numeric_output(tmp_path):
    target, store, backend = qualified(tmp_path)
    assert len(backend.compile_requests) == len(backend.run_requests) == 1
    target.verify(store, RUNTIME, a3_curriculum_policy())
    assert TargetContract.from_mapping(target.as_dict()).target_id == target.target_id
    with pytest.raises(ValueError, match="host oracle"):
        qualified(tmp_path / "wrong", wrong=True)
    reference = store.read_json(target.reference_sha256)
    observed = store.read_json(reference["artifacts"]["observed"])
    observed["output"][0] += 999
    reference["artifacts"]["observed"] = store.put_json(observed)
    # A saved successful classification cannot override incorrect actual output.
    forged = replace(target, reference_sha256=store.put_json(reference))
    with pytest.raises(ValueError, match="not host-verified"):
        forged.verify(store, RUNTIME)


@pytest.mark.parametrize("key", list(RUNTIME))
def test_target_rejects_changed_runtime(tmp_path, key):
    target, store, _ = qualified(tmp_path)
    with pytest.raises(ValueError, match="runtime changed"):
        target.verify(store, {**RUNTIME, key: "changed"})


def test_missing_reference_changed_spec_and_unsupported_policy(tmp_path):
    target, store, _ = qualified(tmp_path)
    with pytest.raises(FileNotFoundError):
        replace(target, reference_sha256="a" * 64).verify(store, RUNTIME)
    changed = replace(target.spec, project_id="another-goal")
    with pytest.raises(ValueError, match="does not match"):
        replace(target, spec=changed).verify(store, RUNTIME)
    with pytest.raises(ValueError, match="dtype unavailable"):
        target.verify(store, RUNTIME, a3_curriculum_policy(dtypes=("float16",)))
    with pytest.raises(ValueError, match="runtime identity"):
        replace(target, runtime={"target": "a3"})
    with pytest.raises(TypeError):
        target.runtime["compiler"] = "changed"


def test_optional_timing_requires_units_conditions_and_retained_evidence(tmp_path):
    target, store, _ = qualified(tmp_path)
    reference = store.read_json(target.reference_sha256)
    reference["timings"] = [{"value": 1.5, "unit": "us", "conditions": "device events; warm; no diagnostics",
                             "evidence_sha256": store.put_bytes(b"actual device measurement")}]
    replace(target, reference_sha256=store.put_json(reference)).verify(store, RUNTIME)
    reference["timings"][0]["unit"] = "seconds including compilation"
    with pytest.raises(ValueError, match="device duration"):
        replace(target, reference_sha256=store.put_json(reference)).verify(store, RUNTIME)


def context(target, verified=False):
    return TargetCurriculumContext(target, {"verified": verified,
        "diagnostic": "host mismatch on padded output", "device_kernel_us": None},
        ({"practice": "prior repair failed"},), 2, 2, ("msprof",))


def practice_wire():
    project = replace(_spec("movement"), project_id="investigate-tail")
    return "DECISION: PRACTICE\n" + json.dumps({"reason": "Investigate observed mismatch",
        "project": project.as_dict(), "weakness": "padded output differs",
        "control": "compare tail ownership with one fixed launch",
        "evidence": "host compare full physical output", "retry_plan": "apply verified tail bounds"})








def test_compile_success_without_numerical_runtime_is_not_feasibility(tmp_path):
    store = ContentAddressedArtifacts(tmp_path / "objects")
    backend = FakeBackend(_compile(), _runtime(passed=False))
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "events.jsonl"),
                               runtime_identity=dict(RUNTIME))
    with pytest.raises(ValueError, match="execute and pass"):
        qualify_target(_spec("movement"), "reference", RUNTIME, executor, store)
    backend.compile_result = _compile(passed=False)
    with pytest.raises(ValueError, match="execute and pass"):
        qualify_target(_spec("movement"), "reference", RUNTIME, executor, store)


def test_reference_input_changes_are_rejected_even_when_saved_output_matches(tmp_path):
    target, store, _ = qualified(tmp_path)
    reference = store.read_json(target.reference_sha256)
    inputs = store.read_json(reference["artifacts"]["inputs"])
    inputs["tensors"][0]["storage"][0] += 1
    reference["artifacts"]["inputs"] = store.put_json(inputs)
    with pytest.raises(ValueError, match="inputs or oracle changed"):
        replace(target, reference_sha256=store.put_json(reference)).verify(store, RUNTIME)
















@pytest.mark.parametrize("changed", [None, *RUNTIME])
def test_reference_runtime_must_match_observed_executor_before_backend_work(tmp_path, changed):
    store = ContentAddressedArtifacts(tmp_path / "objects")
    backend = FakeBackend(_compile())
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "events.jsonl"),
        runtime_identity={**RUNTIME, changed: "different-observation"} if changed is not None else None)
    with pytest.raises(ValueError, match="reference executor.*runtime"):
        qualify_target(_spec("movement"), "reference", RUNTIME, executor, store)
    assert not backend.compile_requests and not backend.run_requests




@pytest.mark.parametrize("stage", ["qualify", "reverify"])
@pytest.mark.parametrize("payload", [b"", b"retained device timing observation"])
def test_timing_requires_nonempty_retained_evidence_at_qualification_and_verification(tmp_path, stage, payload):
    store = ContentAddressedArtifacts(tmp_path / "artifacts")
    timing = {"value": 1.5, "unit": "us", "conditions": "device events; warm; no diagnostics",
              "evidence_sha256": store.put_bytes(payload)}
    def validate():
        if stage == "qualify":
            qualified(tmp_path, timings=(timing,))
        else:
            target, _, _ = qualified(tmp_path)
            reference = store.read_json(target.reference_sha256)
            reference["timings"] = [timing]
            replace(target, reference_sha256=store.put_json(reference)).verify(store, RUNTIME)
    if payload:
        validate()
    else:
        with pytest.raises(ValueError, match="timing evidence is empty"):
            validate()



def test_runtime_observer_is_lazy_and_rechecked_before_each_qualification(tmp_path):
    observed, observations = dict(RUNTIME), []
    def observe():
        observations.append(dict(observed))
        return dict(observed)
    backend = FakeBackend(_compile())
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "events.jsonl"), runtime_identity=observe)
    assert not observations
    spec = _spec("movement")
    backend.runtime_result = _runtime(executor.prepare(spec).expected.storage)
    store = ContentAddressedArtifacts(tmp_path / "objects")
    qualify_target(spec, "reference", RUNTIME, executor, store)
    assert observations == [RUNTIME]
    assert len(backend.compile_requests) == len(backend.run_requests) == 1
    observed["device"] = "different-device"
    with pytest.raises(ValueError, match="reference executor runtime differs"):
        qualify_target(spec, "reference", RUNTIME, executor, store)
    assert len(observations) == 2
    assert len(backend.compile_requests) == len(backend.run_requests) == 1
