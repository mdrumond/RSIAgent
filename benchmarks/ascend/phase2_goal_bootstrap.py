"""Resumable curriculum-authored goals, qualified outside learner memory."""
from copy import copy
import base64
import json
from pathlib import Path
from types import SimpleNamespace

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_adaptive import AdaptiveProjectRequest
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_curriculum_selector import _grammar, _HELD_OUT_WORDS
from benchmarks.a5kernels.phase1_project import materialize_inputs, evaluate_reference
from benchmarks.a5kernels.phase1_wave_runtime import SemanticProjectFailure
from .phase2_targets import TargetContract, qualify_target
from .agent_presentation import MemoryArtifactReader, memory_presentation, compact_observation, artifact_page_presentation


def executable_identity(spec):
    """Ignore cosmetic graph names, hypotheses and learning labels."""
    names = {item.name: index for index, item in enumerate(spec.graph.inputs)}
    nodes = []
    for node in spec.graph.nodes:
        nodes.append((node.op.value, [names[name] for name in node.operands], node.axis))
        names[node.name] = len(names)
    return canonical_digest(dict(dtype=spec.dtype.value, shape=spec.shape,
        padding=spec.padding_multiple, inputs=[item.shape for item in spec.graph.inputs],
        nodes=nodes, output=names[spec.graph.output]))


def bootstrap_goal(coordinator, *, root, private_root, objective, available_tools):
    """Return only an authenticated contract/store; never publish reference lessons.

    Provider and transport exceptions escape without closing the current proposal.
    Its private Actor branch and native handles survive for explicit resume.
    """
    from .campaign_cli import _read, _write
    root, private_root = Path(root), Path(private_root)
    private_root.mkdir(parents=True, exist_ok=True)
    actor = coordinator.execute_project
    from benchmarks.a5kernels.phase1_adaptive_actor import (
        disabled_knowledge_action, disabled_profile_action, disabled_diagnostic_action,
    )
    knowledge_enabled = actor.query_knowledge is not disabled_knowledge_action
    runtime = actor.executor.runtime_identity
    if runtime is None:
        raise ValueError("autonomous goals require observed runtime identity")
    policy = coordinator.curriculum.target_policy
    snapshot = coordinator.snapshot
    entries = list(snapshot.memory_entries())
    initial_seed = canonical_bytes(dict(schema="ascend-phase2-curriculum-seed-v1",
                                       snapshot_id=snapshot.snapshot_id, phase1_memory=entries))
    seed = canonical_bytes(dict(schema="a5-phase1-adaptive-memory-v1",
        initial_memory_base64=base64.b64encode(initial_seed).decode(), project_memory_base64=[]))
    identity = dict(snapshot_id=snapshot.snapshot_id, runtime=dict(runtime), objective=objective,
                    available_tools=list(available_tools))
    state_path = private_root / "selection.json"
    state = (_read(state_path) if state_path.exists() else
             dict(identity=identity, stage="selection", proposal_count=0, candidate=None,
                  diagnostic=None, history=[]))
    if state["identity"] != identity:
        raise ValueError("autonomous goal seed, runtime or objective changed")
    store = ContentAddressedArtifacts(private_root / "artifacts")

    def save():
        _write(state_path, state)
        _write(root / "goal-selection.json", {key: state.get(key) for key in (
            "stage", "proposal_count", "candidate", "diagnostic", "qualified_target_id")})

    save()
    if state["stage"] == "qualified":
        target = TargetContract.from_mapping(state["target"])
        target.verify(store, runtime, policy)
        return target, store
    prior_specs = [ProjectSpec.from_mapping(row["project_spec"]) for row in entries]
    seen = {executable_identity(spec) for spec in prior_specs}
    seen_ids = {spec.project_id for spec in prior_specs}
    grammar = policy.grammar(_grammar()) if policy else _grammar()
    for key in ("wave_project_fields_exactly", "wave_fields_exactly", "requires"):
        grammar.pop(key, None)
    from .phase2_target_selector import _TOOL_FIELDS
    for values in (grammar["capabilities"], grammar["optional_project_fields"]["learning_focus"]):
        values[:] = [name for name in values if name not in _TOOL_FIELDS
                     or _TOOL_FIELDS[name] in available_tools]
    if "request-profile" not in available_tools:
        grammar["profile_metrics"] = []
    if "request-timing" not in available_tools:
        grammar["timing_lengths"] = "[]; timing is unavailable"
    # Reference and learner get separate conversations, artifacts, evidence and ledger.
    reference_actor = copy(actor)
    reference_actor.trace_recorder_factory = None
    reference_actor.request_profile = reference_actor.request_timing = disabled_profile_action
    reference_actor.request_diagnostic = disabled_diagnostic_action
    reference_actor.diagnostic_instructions = (
        "Private reference construction establishes numerical feasibility only. "
        "Timing, profiling, debugger and sanitizer actions are disabled here; "
        "do not request them. Write, compile, run, inspect host errors, then submit.\n")
    reference_actor.executor = copy(actor.executor)
    from benchmarks.a5kernels.evidence import EvidenceLedger
    reference_actor.executor._ledger = EvidenceLedger(private_root / "reference-evidence.jsonl")
    backend = copy(actor.executor._backend)
    if hasattr(backend, "root"):
        backend.root = private_root / "native-evidence"
    if hasattr(backend, "_root"):
        backend._root = private_root / "catlass-evidence"
    reference_actor.executor._backend = backend
    system = {"role": "system", "content": (
        "You are the Ascend Phase-2 goal Curriculum Agent. Author ONE new, bounded "
        "kernel task building on Phase1 lessons, inspired by KDB code when available. "
        "Not a renamed Phase1 task. Do not name held-out benchmarks. Reference source "
        "will be independently constructed and host-qualified. Return JSON only: "
        '{"query":"specific kernel code search"} or '
        '{"read_artifact":{"sha256":"hash","offset":0,"limit":8192}} or '
        '{"project":{ProjectSpec},"rationale":"why this is useful and new"}. '
        "At most six tool reads including three code searches per proposal. "
        "Memory index is presentation only; original memories and observations are hash-addressed. "
        "When KDB is available, search it before proposing the goal.")}
    memory = memory_presentation(seed, store)
    historical_reader = None
    while state["proposal_count"] < 3 or state["stage"] == "qualification":
        if state["stage"] == "selection":
            messages = [system, {"role": "user", "content": json.dumps(dict(
                objective=objective, runtime=dict(runtime), available_tools=available_tools,
                kdb_available=knowledge_enabled,
                project_grammar=grammar, phase1_memory=memory, previous=state["history"],
                feedback=state["diagnostic"]), sort_keys=True)}]
            citations = []
            # Persist each completed model turn/query before another provider request.
            turns = state.setdefault("selection_turns", [])
            query_count = 0
            for query_index in range(7):
                if query_index < len(turns):
                    response = turns[query_index]
                else:
                    response = coordinator.curriculum._complete(messages)
                    turns.append(response)
                    save()
                messages.append({"role": "assistant", "content": response})
                try:
                    value = json.loads(response)
                    if isinstance(value, dict) and set(value) == {"read_artifact"} and query_index < 6:
                        try:
                            result = artifact_page_presentation(value["read_artifact"], store)
                        except FileNotFoundError:
                            if historical_reader is None:
                                historical_reader = MemoryArtifactReader(SimpleNamespace(
                                    pre_wave_memory=seed, memory_root=coordinator.root,
                                    branch_root=private_root), store)
                            try:
                                result = artifact_page_presentation(value["read_artifact"], historical_reader)
                            except FileNotFoundError as error:
                                raise ValueError("requested memory artifact was not retained") from error
                        messages.append({"role": "user", "content": json.dumps(result)})
                        continue
                    if isinstance(value, dict) and set(value) == {"query"} and query_index < 6 and query_count < 3:
                        query_count += 1
                        if not isinstance(value["query"], str) or not value["query"].strip():
                            raise ValueError("knowledge query must be nonempty text")
                        query_path = private_root / f"query-{state['proposal_count']}-{query_index}.json"
                        if query_path.exists():
                            result = _read(query_path)
                        else:
                            context = SimpleNamespace(request=SimpleNamespace(branch_root=private_root), artifacts=store)
                            result = dict(actor.query_knowledge({"query": value["query"], "limit": 3}, context).observation)
                            _write(query_path, result)
                        citations.extend(row["citation"] for row in result.get("results", []))
                        messages.append({"role": "user", "content": json.dumps(compact_observation(result, store))})
                        continue
                    if not isinstance(value, dict) or set(value) != {"project", "rationale"}:
                        raise ValueError("expected project and rationale after at most three searches")
                    if knowledge_enabled and query_count == 0:
                        raise ValueError("KDB-enabled goal selection must query kernel code first")
                    spec = ProjectSpec.from_mapping(value["project"])
                    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
                        raise ValueError("goal requires a rationale")
                    if _HELD_OUT_WORDS.search(json.dumps(value)):
                        raise ValueError("goal names a held-out benchmark")
                    if spec.project_id in seen_ids or executable_identity(spec) in seen:
                        raise ValueError("goal duplicates a Phase1 task or only renames it")
                    if policy:
                        policy.admit(spec)
                    # Reuse practice admission for tool availability and held-out rules.
                    from .phase2_target_selector import TargetCurriculumAgent, TargetCurriculumContext
                    context = TargetCurriculumContext(TargetContract(prior_specs[0], runtime, "0" * 64),
                        {"verified": False}, (), 1, 1, tuple(available_tools), tuple(seen_ids))
                    TargetCurriculumAgent(lambda _: None, target_policy=policy,
                        held_out_information_sha256=coordinator.curriculum._held_out).parse(
                        'DECISION: PRACTICE\n' + json.dumps(dict(project=spec.as_dict(), reason=value["rationale"],
                        weakness="new goal", control="fixed task", evidence="host qualification", retry_plan="fixed target")), context)
                except (ValueError, TypeError, KeyError) as error:
                    state.update(proposal_count=state["proposal_count"] + 1,
                                 diagnostic=str(error), selection_turns=[])
                    state["history"].append(dict(response=response, diagnostic=str(error)))
                    save()
                    break
                state.update(stage="qualification", proposal_count=state["proposal_count"] + 1,
                             candidate=dict(spec=spec.as_dict(), rationale=value["rationale"], citations=citations),
                             selection_turns=[])
                save()
                break
        if state["stage"] != "qualification":
            continue
        spec = ProjectSpec.from_mapping(state["candidate"]["spec"])
        branch = private_root / f"reference-{state['proposal_count']}"
        branch.mkdir(parents=True, exist_ok=True)
        try:
            source_path = branch / "qualified-source.json"
            if source_path.exists():
                source = store.read_bytes(_read(source_path)["source_sha256"]).decode()
            else:
                inputs = materialize_inputs(spec)
                result = reference_actor(AdaptiveProjectRequest(spec, 0, seed, branch,
                    inputs, evaluate_reference(spec, inputs), memory_root=coordinator.root))
                if not result.evidence.verified:
                    raise ValueError("private reference Actor did not pass host verification: " + result.summary)
                source_sha = json.loads(result.evidence_payload)["source_sha256"]
                source = ContentAddressedArtifacts(branch / "artifacts").read_bytes(source_sha).decode()
                _write(source_path, {"source_sha256": store.put_bytes(source.encode())})
            target = qualify_target(spec, source, runtime, reference_actor.executor, store, policy=policy)
        except (SemanticProjectFailure, ValueError) as error:
            state["history"].append(dict(candidate=state["candidate"], diagnostic=str(error)))
            state.update(stage="selection", diagnostic=str(error), candidate=None)
            save()
            continue
        state.update(stage="qualified", target=target.as_dict(), qualified_target_id=target.target_id,
                     diagnostic=None)
        save()
        return target, store
    state.update(stage="failed", diagnostic=state["diagnostic"] or "Three goal proposals exhausted")
    save()
    raise ValueError("Autonomous Phase2 bootstrap failed: " + state["diagnostic"])
