"""References establish correctness feasibility; curriculum selects focused practice."""
from dataclasses import replace
import json

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.ascend.phase2_targets import TargetContract, qualify_target
from benchmarks.ascend.phase2_target_selector import (
    TargetCurriculumAgent, TargetCurriculumContext, target_curriculum_prompt,
)
from benchmarks.ascend.targets import a3_curriculum_policy
from test_a5_phase1_project_executor import FakeBackend, _compile, _runtime, _spec

RUNTIME = dict(target="a3", runtime="registered-native", product="observed-product",
               compiler="observed-compiler-revision", toolchain="observed-cann", device="0")


def qualified(tmp_path, *, wrong=False, timings=()):
    store = ContentAddressedArtifacts(tmp_path / "artifacts")
    spec = _spec("movement")
    backend = FakeBackend(_compile())
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "events.jsonl"))
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


def test_phase2_prompt_is_target_specific_and_keeps_reference_private(tmp_path):
    target, _, _ = qualified(tmp_path)
    prompt = target_curriculum_prompt(context(target), a3_curriculum_policy())
    assert "private working reference" not in prompt
    assert "reference_sha256" not in prompt
    assert "Phase-1" not in prompt
    payload = json.loads(prompt.rsplit("\n", 1)[-1])
    assert payload["target"]["target_id"] == target.target_id
    assert payload["target"]["correctness"]["atol_and_rtol"] == 1e-5
    assert payload["latest_target_attempt"]["diagnostic"] == "host mismatch on padded output"
    assert "[a-z][a-z0-9]{0,15}" in prompt
    assert payload["project_grammar"]["dtypes"] == ["float32"]


def test_practice_requires_concrete_fields_budget_and_fixed_target(tmp_path):
    target, _, _ = qualified(tmp_path)
    agent = TargetCurriculumAgent(lambda _: practice_wire(), target_policy=a3_curriculum_policy())
    decision, session = agent.select(context(target))
    assert decision.action == "practice" and session.turns == 1
    assert decision.project.project_id != target.spec.project_id
    assert decision.as_dict()["retry_plan"] == "apply verified tail bounds"
    with pytest.raises(ValueError, match="practice budget"):
        agent.parse(practice_wire(), replace(context(target), remaining_practice_attempts=0))
    with pytest.raises(ValueError, match="retry budget"):
        agent.parse(practice_wire(), replace(context(target), remaining_target_attempts=0))
    value = json.loads(practice_wire().split("\n", 1)[1])
    value["project"] = target.spec.as_dict()
    with pytest.raises(ValueError, match="replace the target"):
        agent.parse("DECISION: PRACTICE\n" + json.dumps(value), context(target))


def test_failed_target_cannot_be_ready_and_invalid_proposal_is_corrected(tmp_path):
    target, _, _ = qualified(tmp_path)
    replies = iter(['DECISION: READY\n{"reason":"practice passed"}',
                    'DECISION: RETRY_TARGET\n{"reason":"test learned repair"}'])
    calls = []
    agent = TargetCurriculumAgent(lambda messages: (calls.append(messages), next(replies))[1])
    decision, session = agent.select(context(target))
    assert decision.action == "retry-target" and session.turns == 2
    assert "host-verified target" in calls[1][-1]["content"]
    assert agent.parse('DECISION: READY\n{"reason":"target passed; no useful practice"}',
                       context(target, True)).action == "ready"
    with pytest.raises(ValueError, match="fields"):
        agent.parse('DECISION: READY\n{"reason":"done","target":{}}', context(target, True))


def test_compile_success_without_numerical_runtime_is_not_feasibility(tmp_path):
    store = ContentAddressedArtifacts(tmp_path / "objects")
    backend = FakeBackend(_compile(), _runtime(passed=False))
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "events.jsonl"))
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
