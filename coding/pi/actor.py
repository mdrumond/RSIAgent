"""Pi action driver, retaining the existing noncoding terminal distillation."""
import json
import hashlib
import re

from benchmarks.a5kernels.phase1_provider import (
    DirectActorDriver, _ActorContinuation, _ProviderEvidenceRecorder,
)
from benchmarks.a5kernels.trial import ActorOutcome
from benchmarks.ascend.agent_context import WorkingContext
from .rpc import PiDeadlineError
from .adapters import A5_SCHEMAS
from .session import PiCodingSession, ENGINE_IDENTITY, ActionError, runtime_preflight



def observation_context(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value



def task_contract(executor):
    """Expose layout and ABI metadata without oracle values or tensor storage."""
    prepared = getattr(executor, "prepared", None)
    if prepared is None:
        return None
    spec, abi = prepared.spec, prepared.abi
    return {
        "shape": list(spec.shape), "dtype": spec.dtype.value,
        "padding_multiple": spec.padding_multiple, "block_count": spec.block_count,
        "abi": {"kernel_name": abi.kernel_name, "input_arguments": list(abi.input_arguments),
                "output_argument": abi.output_argument},
        "graph": spec.graph.as_dict(),
        "inputs": [{"argument": name, "logical_shape": list(tensor.shape),
                    "physical_shape": list(tensor.physical_shape)}
                   for name, tensor in zip(abi.input_arguments, prepared.inputs, strict=True)],
        "output": {"argument": abi.output_argument, "logical_shape": list(prepared.expected.shape),
                   "physical_shape": list(prepared.expected.physical_shape),
                   "dtype": prepared.expected.dtype},
    }


def typed_contract(prompt):
    """Replace the registered JSON wire protocol without changing legacy callers."""
    return re.sub(r"Respond with exactly one JSON object:.*?(?=You cannot choose paths)",
        "Select one declared typed host tool. Read artifacts with read-artifact using its "
        "sha256 and returned next_offset; pages are UTF-8 text, at most 8192 bytes. "
        "Diagnostic observations never establish correctness or timing.\n", prompt, flags=re.S)


class PiActorDriver(DirectActorDriver):
    session_factory = PiCodingSession

    def __init__(self, *args, max_seconds=3600, **kwargs):
        runtime_preflight()
        kwargs.pop("max_turns", None)  # Legacy count limits no longer bound execution.
        super().__init__(*args, max_seconds=max_seconds, **kwargs)

    def with_trace(self, trace_event):
        return type(self)(self.provider, max_seconds=self.max_seconds, clock=self.clock,
                          trace_event=trace_event, system_prompt=self.system_prompt)

    def _new_session(self, profile):
        return self.session_factory(model=profile.identity.backend_model.value, schemas=A5_SCHEMAS,
            environ=self.provider.coding_environment(), deadline=self.clock() + self.max_seconds)

    def __call__(self, instruction, *, profile, workspace, context, turn_parser, action_executor):
        # The legacy pilot uses a closed OpenRouter profile. It must be migrated
        # explicitly by its composition, rather than silently changing provider.
        recorder = _ProviderEvidenceRecorder(workspace)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": instruction}]
        working = WorkingContext(messages)
        for message in messages:
            self._trace("message", message)
        tokens = 0
        turn = 0
        deadline = min(self.clock() + self.max_seconds,
                       getattr(action_executor, "deadline", float("inf")))
        contract = task_contract(action_executor)
        latest = None
        history = []
        source = None
        allowed = set(A5_SCHEMAS)
        knowledge = getattr(action_executor, "knowledge", None)
        query = getattr(action_executor, "query_knowledge", None)
        if ((knowledge is not None and not knowledge.enabled)
                or getattr(query, "__name__", "") == "disabled_knowledge_action"):
            allowed.discard("query-knowledge")
        if getattr(action_executor, "profiling_guidance", None) is False:
            allowed.discard("request-profile")
        profile_action = getattr(action_executor, "request_profile", None)
        if getattr(profile_action, "__name__", "") == "disabled_profile_action":
            allowed.discard("request-profile")
        if getattr(getattr(action_executor, "request_timing", None), "__name__", "") == "disabled_profile_action":
            allowed.discard("request-timing")
        if getattr(getattr(action_executor, "request_diagnostic", None), "__name__", "") == "disabled_diagnostic_action":
            allowed -= {"request-debug", "request-sanitizer"}
        with self._new_session(profile) as session:
            session.deadline = min(getattr(session, "deadline", float("inf")), deadline)
            gateway = getattr(session, "gateway", None)
            if gateway is not None:
                gateway.deadline = min(gateway.deadline, deadline)
            while self.clock() < deadline:
                turn += 1
                before = session.completion_tokens
                def record(kind, payload):
                    self._trace("tool", {"name": kind, "turn": turn, **payload})
                    if recorder.enabled:
                        digest = recorder.artifacts.put_json(payload)
                        recorder._append(kind, digest, turn=turn)
                try:
                    name, args = session.next_action({
                        "instruction": instruction, "contract": typed_contract(self.system_prompt),
                        "task_contract": contract,
                        "latest_host_observation": observation_context(latest), "source": source,
                        "recent_host_observations": history[-6:],
                        "remaining_seconds": max(0, deadline-self.clock()),
                        "coding": ENGINE_IDENTITY,
                    }, allowed, record=record, source=source,
                       write_action=lambda s: ("write", {"slot": "kernel", "content": s}))
                    text = json.dumps({name: args})
                except PiDeadlineError:
                    tokens += session.completion_tokens-before
                    break
                except ActionError as error:
                    text = "{}"
                    latest = json.dumps({"host": {"event": "invalid-action", "reason": str(error),
                        "allowed_actions": sorted(allowed)}})
                tokens += session.completion_tokens-before
                recorder.response(text, stage="action", turn=turn)
                self._trace("message", {"role": "assistant", "content": text})
                action = turn_parser(text)
                if action is None:
                    if text != "{}" or latest is None:
                        latest = json.dumps({"host": {"event": "invalid-action",
                            "reason": "Selected action does not match the host protocol",
                            "allowed_actions": sorted(allowed)}})
                else:
                    outcome = action_executor(action)
                    latest = outcome.observation
                    working.append({"role": "user", "content": latest})
                    if action.kind == "write":
                        actual = getattr(action_executor, "source", None)
                        if isinstance(actual, str):
                            source = actual
                        else:
                            host = json.loads(latest).get("host", {})
                            if host.get("event") == "written":
                                source = args["content"]
                    self._trace("tool", {"name": action.kind, "status": "observed", "observation": latest})
                    if outcome.terminal:
                        # Distillation receives bounded terminal evidence and the
                        # original contract, not a second coding continuation.
                        retained = messages + [{"role": "user", "content": latest}]
                        ceiling = self.provider.context_budget(profile).target // 4
                        if len(instruction.encode()) > ceiling:
                            retained[1] = {"role": "user", "content":
                                "Task SHA256: " + hashlib.sha256(instruction.encode()).hexdigest()
                                + "\nTask preview (terminal distillation uses the admitted host evidence):\n"
                                + instruction.encode()[:ceiling//2].decode(errors="ignore")}
                            working.fixed = retained[:2]
                        working.recent = retained[2:]
                        self._trace("terminal", {"status": outcome.status,
                                                "admitted_artifacts": dict(outcome.admitted_artifacts)})
                        return ActorOutcome(outcome.status, turn, tokens,
                            admitted_artifacts=dict(outcome.admitted_artifacts),
                            continuation=_ActorContinuation(profile, tuple(retained), recorder,
                                                            working, turn, deadline))
                recorder.observation(latest, turn=turn)
                history.append(latest)
                self._trace("message", {"role": "user", "content": latest})
        return ActorOutcome("time-budget-exhausted", turn, tokens)
