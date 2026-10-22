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
        agent.parse(practice_wire(), replace(context(target), practice_seconds_remaining=0))
    with pytest.raises(ValueError, match="retry budget"):
        agent.parse(practice_wire(), replace(context(target), remaining_seconds=0))
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


@pytest.mark.parametrize("field", ["project_id", "hypothesis", "graph-input", "graph-node"])
def test_practice_rejects_held_out_names(tmp_path, field):
    target, _, _ = qualified(tmp_path)
    value = json.loads(practice_wire().split("\n", 1)[1])
    project = value["project"]
    if field == "project_id":
        project["project_id"] = "practice-bsa"
    elif field == "hypothesis":
        project["hypothesis"] = "Reconstruct the held-out GDN kernel from this target failure."
    elif field == "graph-input":
        old = project["graph"]["inputs"][0]["name"]
        project["graph"]["inputs"][0]["name"] = "bsa"
        for node in project["graph"]["nodes"]:
            node["operands"] = ["bsa" if name == old else name for name in node["operands"]]
    else:
        project["graph"]["nodes"][-1]["name"] = "gdn"
        project["graph"]["output"] = "gdn"
    with pytest.raises(ValueError, match="held-out benchmark name"):
        TargetCurriculumAgent(lambda _: "").parse(
            "DECISION: PRACTICE\n" + json.dumps(value), context(target))


def test_configured_held_out_signatures_apply_only_to_practice(tmp_path):
    from benchmarks.a5kernels.phase1_curriculum_selector import project_information_sha256
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    target, _, _ = qualified(tmp_path)
    value = json.loads(practice_wire().split("\n", 1)[1])
    digest = project_information_sha256(ProjectSpec.from_mapping(value["project"]))
    agent = TargetCurriculumAgent(lambda _: "", held_out_information_sha256=(digest,))
    # Learning focus and new IDs cannot evade executable held-out matching.
    value["project"]["learning_focus"] = ["pipe-vector"]
    with pytest.raises(ValueError, match="held-out contract"):
        agent.parse("DECISION: PRACTICE\n" + json.dumps(value), context(target))
    assert agent.parse('DECISION: RETRY_TARGET\n{"reason":"retry unchanged goal"}',
                       context(target)).action == "retry-target"
    with pytest.raises(ValueError, match="SHA-256"):
        TargetCurriculumAgent(lambda _: "", held_out_information_sha256=("invalid",))


@pytest.mark.parametrize("focus,tool", [("tool-msprof", "request-profile"),
    ("tool-msdebug", "request-debug"), ("tool-mssanitizer", "request-sanitizer"),
    ("debug-print", "debug-print")])
def test_practice_tool_focus_requires_available_route(tmp_path, focus, tool):
    target, _, _ = qualified(tmp_path)
    value = json.loads(practice_wire().split("\n", 1)[1])
    value["project"]["learning_focus"] = [focus]
    wire = "DECISION: PRACTICE\n" + json.dumps(value)
    agent = TargetCurriculumAgent(lambda _: wire)
    absent = replace(context(target), available_tools=())
    with pytest.raises(ValueError, match="unavailable tools"):
        agent.parse(wire, absent)
    assert agent.parse(wire, replace(absent, available_tools=(tool,))).action == "practice"
    grammar = json.loads(target_curriculum_prompt(absent).rsplit("\n", 1)[-1])["project_grammar"]
    assert focus not in grammar["optional_project_fields"]["learning_focus"]


@pytest.mark.parametrize("capability,tool", [("profiling", "request-profile"),
                                           ("timing-sweep", "request-timing")])
def test_tool_dependent_capabilities_are_rejected_and_model_can_repair(tmp_path, capability, tool):
    target, _, _ = qualified(tmp_path)
    value = json.loads(practice_wire().split("\n", 1)[1])
    project = value["project"]
    project["capabilities"].append(capability)
    if capability == "profiling":
        project["profile_metric"] = "pipe-utilization"
    else:
        # Timing sweeps require unpadded, one-dimensional graph inputs.
        project["capabilities"].remove("padded-tail")
        project["shape"] = [16]
        project["padding_multiple"] = 1
        project["timing_lengths"] = [16, 32]
        for item in project["graph"]["inputs"]:
            item["shape"] = [16]
    wire = "DECISION: PRACTICE\n" + json.dumps(value)
    absent = replace(context(target), available_tools=())
    agent = TargetCurriculumAgent(lambda _: wire)
    with pytest.raises(ValueError, match="unavailable tools"):
        agent.parse(wire, absent)
    assert agent.parse(wire, replace(absent, available_tools=(tool,))).action == "practice"
    replies = iter([wire, practice_wire()])
    decision, session = TargetCurriculumAgent(lambda _: next(replies)).select(absent)
    assert decision.action == "practice" and session.turns == 2
    grammar = json.loads(target_curriculum_prompt(absent).rsplit("\n", 1)[-1])["project_grammar"]
    assert capability not in grammar["capabilities"]
    assert grammar["profile_metrics"] == []


@pytest.mark.parametrize("field", ["reason", "weakness", "control", "evidence", "retry_plan"])
def test_practice_rationale_rejects_held_out_names_and_can_be_corrected(tmp_path, field):
    target, _, _ = qualified(tmp_path)
    value = json.loads(practice_wire().split("\n", 1)[1])
    value[field] = "Reconstruct the held-out BSA benchmark with this practice."
    invalid = "DECISION: PRACTICE\n" + json.dumps(value)
    replies = iter([invalid, practice_wire()])
    calls = []
    agent = TargetCurriculumAgent(lambda messages: (calls.append(messages), next(replies))[1])
    with pytest.raises(ValueError, match="held-out benchmark name"):
        agent.parse(invalid, context(target))
    decision, session = agent.select(context(target))
    assert decision.action == "practice" and session.turns == 2
    assert "held-out benchmark name" in calls[1][-1]["content"]


@pytest.mark.parametrize("runtime_target,policy_target", [("a3", "a5"), ("a5", "a3")])
def test_curriculum_rejects_wrong_target_policy_before_completion_and_direct_parse(
        tmp_path, runtime_target, policy_target):
    target, _, _ = qualified(tmp_path)
    target = replace(target, runtime={**target.runtime, "target": runtime_target})
    policy = replace(a3_curriculum_policy(), target=policy_target)
    calls = []
    agent = TargetCurriculumAgent(lambda messages: calls.append(messages), target_policy=policy)
    with pytest.raises(ValueError, match="target policy and runtime differ"):
        target_curriculum_prompt(context(target), policy)
    with pytest.raises(ValueError, match="target policy and runtime differ"):
        agent.select(context(target))
    assert calls == []
    with pytest.raises(ValueError, match="target policy and runtime differ"):
        agent.parse(practice_wire(), context(target))
    with pytest.raises(ValueError, match="target policy and runtime differ"):
        agent.parse('DECISION: RETRY_TARGET\n{"reason":"retry unchanged goal"}', context(target))


@pytest.mark.parametrize("capability,op,axis", [
    ("vector-reduction", "reduce-sum", 1), ("vector-mask", "mask-positive", None)])
def test_native_vector_practice_requires_explicit_native_policy(tmp_path, capability, op, axis):
    target, _, _ = qualified(tmp_path)
    value = json.loads(practice_wire().split("\n", 1)[1])
    value["project"]["capabilities"] = ["gm-ub-movement", capability]
    value["project"]["graph"]["nodes"][-1].update(op=op, axis=axis)
    wire = "DECISION: PRACTICE\n" + json.dumps(value)
    with pytest.raises(ValueError, match="native vector capability requires an explicit target policy"):
        TargetCurriculumAgent(lambda _: "").parse(wire, context(target))
    agent = TargetCurriculumAgent(lambda _: wire, target_policy=a3_curriculum_policy())
    decision, session = agent.select(context(target))
    assert decision.action == "practice" and session.turns == 1
    assert capability in {item.value for item in decision.project.capabilities}


@pytest.mark.parametrize("changed", [None, *RUNTIME])
def test_reference_runtime_must_match_observed_executor_before_backend_work(tmp_path, changed):
    store = ContentAddressedArtifacts(tmp_path / "objects")
    backend = FakeBackend(_compile())
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "events.jsonl"),
        runtime_identity={**RUNTIME, changed: "different-observation"} if changed is not None else None)
    with pytest.raises(ValueError, match="reference executor.*runtime"):
        qualify_target(_spec("movement"), "reference", RUNTIME, executor, store)
    assert not backend.compile_requests and not backend.run_requests


def test_selector_retains_previous_practice_rationale_with_latest_result(tmp_path):
    target, _, _ = qualified(tmp_path)
    calls = []
    invalid = json.loads(practice_wire().split("\n", 1)[1])
    invalid["reason"] = "Reconstruct held-out BSA"
    replies = iter(["DECISION: PRACTICE\n" + json.dumps(invalid), practice_wire(),
                    'DECISION: RETRY_TARGET\n{"reason":"apply learned tail ownership"}'])
    agent = TargetCurriculumAgent(lambda messages: (calls.append(messages), next(replies))[1])
    decision, session = agent.select(context(target))
    latest = replace(context(target), lessons=({"project_id": decision.project.project_id,
        "verified": True, "host_feedback": "padded output now matches"},))
    followup, updated = agent.select(latest, session)
    assert followup.action == "retry-target" and updated.turns == 3
    assert len(calls[-1]) == 2
    prompt = calls[-1][-1]["content"]
    assert practice_wire() in prompt
    assert "compare tail ownership with one fixed launch" in prompt
    assert "padded output now matches" in prompt
    assert "BSA" not in prompt and "Invalid proposal" not in prompt
    assert "BSA" in json.dumps(session.messages)


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


def test_native_phase2_curriculum_has_shared_metric_sweep_rule(tmp_path):
    from benchmarks.ascend.phase2_target_selector import SYSTEM
    from benchmarks.ascend.targets import METRIC_SWEEP_GUIDANCE
    target, _, _ = qualified(tmp_path)
    prompt = target_curriculum_prompt(context(target), a3_curriculum_policy())
    assert METRIC_SWEEP_GUIDANCE in SYSTEM
    grammar = json.loads(prompt.rsplit('\n', 1)[-1])['project_grammar']
    assert 'timing_lengths' in grammar
    assert 'Do not invent sweep fields' in SYSTEM
