"""Pi action driver, retaining the existing noncoding terminal distillation."""
import json
import hashlib
from contextlib import contextmanager

from benchmarks.a5kernels.phase1_provider import (
    DirectActorDriver, _ActorSelection,
)
from benchmarks.a5kernels.trial import actor_system_prompt, trial_action
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
    input_names = abi.input_arguments[:len(prepared.inputs)]
    auxiliary = [{"argument": name, "kind": "auxiliary"}
                 for name in abi.input_arguments[len(prepared.inputs):]]
    if spec.dtype.value in {"mxfp8_e4m3fn", "mxfp8_e5m2"}:
        for index, item in enumerate(auxiliary):
            if index >= len(prepared.inputs):
                break
            rows, cols = prepared.inputs[index].shape
            k, outer = (cols, rows) if index == 0 else (rows, cols)
            groups = ((k + 31) // 32 + 1) // 2 * 2
            item.update(kind="mx-scale", dtype="E8M0", group_size=32,
                logical_shape=[outer, groups] if index == 0 else [groups, outer],
                physical_shape=[outer, groups],
                layout="RowMajorMxScaleA" if index == 0 else "ColMajorMxScaleB")
    return {
        "shape": list(spec.shape), "dtype": spec.dtype.value,
        "padding_multiple": spec.padding_multiple, "block_count": spec.block_count,
        "abi": {"kernel_name": abi.kernel_name, "input_arguments": list(abi.input_arguments),
                "output_argument": abi.output_argument},
        "graph": spec.graph.as_dict(),
        "inputs": [{"argument": name, "logical_shape": list(tensor.shape),
                    "physical_shape": list(tensor.physical_shape)}
                   for name, tensor in zip(input_names, prepared.inputs, strict=True)],
        "auxiliary_inputs": auxiliary,
        "output": {"argument": abi.output_argument, "logical_shape": list(prepared.expected.shape),
                   "physical_shape": list(prepared.expected.physical_shape),
                   "dtype": prepared.expected.dtype},
    }


class PiActorDriver(DirectActorDriver):
    session_factory = PiCodingSession

    def __init__(self, *args, max_seconds=3600, system_prompt=None, **kwargs):
        runtime_preflight()
        kwargs.pop("max_turns", None)
        super().__init__(*args, max_seconds=max_seconds,
                         system_prompt=system_prompt or actor_system_prompt(typed=True), **kwargs)

    def _new_session(self, profile):
        return self.session_factory(model=profile.identity.backend_model.value, schemas=A5_SCHEMAS,
            environ=self.provider.coding_environment(), deadline=self.clock() + self.max_seconds)

    @contextmanager
    def _action_session(self, profile, action_executor, deadline):
        with self._new_session(profile) as session:
            session.deadline = min(getattr(session, "deadline", float("inf")), deadline)
            gateway = getattr(session, "gateway", None)
            if gateway is not None:
                gateway.deadline = min(gateway.deadline, deadline)
            session.host_source = getattr(action_executor, "source", None)
            yield session

    @staticmethod
    def _allowed_actions(executor):
        allowed = set(A5_SCHEMAS)
        knowledge = getattr(executor, "knowledge", None)
        query = getattr(executor, "query_knowledge", None)
        if ((knowledge is not None and not knowledge.enabled)
                or getattr(query, "__name__", "") == "disabled_knowledge_action"):
            allowed.discard("query-knowledge")
        if getattr(executor, "profiling_guidance", None) is False:
            allowed.discard("request-profile")
        for attribute, actions, disabled in (
            ("request_profile", {"request-profile"}, "disabled_profile_action"),
            ("request_timing", {"request-timing"}, "disabled_profile_action"),
            ("request_diagnostic", {"request-debug", "request-sanitizer"}, "disabled_diagnostic_action"),
        ):
            if getattr(getattr(executor, attribute, None), "__name__", "") == disabled:
                allowed -= actions
        return allowed

    def _select_action(self, session, *, working, profile, recorder, turn,
                       feedback, action_executor, turn_parser, deadline):
        allowed = self._allowed_actions(action_executor)
        def record(kind, payload):
            self._trace("tool", {"name": kind, "turn": turn, **payload})
            if recorder.enabled:
                digest = recorder.artifacts.put_json(payload)
                recorder._append(kind, digest, turn=turn)
        before = session.completion_tokens
        error = None
        recent = working.recent[-6:]
        latest = next((row["content"] for row in reversed(recent) if row["role"] == "user"), None)
        try:
            name, args = session.next_action({
                "instruction": working.fixed[1]["content"], "contract": self.system_prompt,
                "task_contract": task_contract(action_executor),
                "latest_host_observation": observation_context(latest),
                "source": session.host_source,
                "recent_exchanges": recent[:-1] if recent and recent[-1]["role"] == "user" else recent,
                "remaining_seconds": max(0, deadline-self.clock()), "coding": ENGINE_IDENTITY,
                "user_feedback": feedback, "retained_work_log": working.work_log,
            }, allowed, record=record, source=session.host_source,
                write_action=lambda source: ("write", {"slot": "kernel", "content": source}))
            action = trial_action(name, args) if name in allowed else None
            text = json.dumps({name: args})  # Evidence representation, never reparsed for dispatch.
        except PiDeadlineError:
            return _ActorSelection(None, "", session.completion_tokens-before, exhausted=True)
        except ActionError as exc:
            error, action, text = str(exc)[:500], None, "{}"
        invalid = None
        if action is None:
            host = {"event": "invalid-action",
                "description": "No accepted action was selected. No action was executed.",
                "next_action": "Use one of the advertised allowed tools with arguments matching its schema.",
                "allowed_actions": sorted(getattr(session, "allowed", allowed)),
                "reason": error or "Selected action does not match the host protocol"}
            if error:
                host["error"] = error
            invalid = json.dumps({"host": host})
        return _ActorSelection(action, text, session.completion_tokens-before, invalid)

    def _observed_action(self, session, action, observation, action_executor):
        if action.kind == "write":
            actual = getattr(action_executor, "source", None)
            host = observation_context(observation)
            if isinstance(actual, str):
                session.host_source = actual
            elif isinstance(host, dict) and host.get("host", {}).get("event") == "written":
                session.host_source = action.payload["content"]

    def _terminal_context(self, working, profile):
        # Distill bounded terminal evidence rather than another coding continuation.
        retained = [*working.fixed, working.recent[-1]]
        instruction = retained[1]["content"]
        ceiling = self.provider.context_budget(profile).target // 4
        if len(instruction.encode()) > ceiling:
            retained[1] = {"role": "user", "content":
                "Task SHA256: " + hashlib.sha256(instruction.encode()).hexdigest()
                + "\nTask preview (terminal distillation uses the admitted host evidence):\n"
                + instruction.encode()[:ceiling//2].decode(errors="ignore")}
        working.fixed, working.recent = retained[:2], retained[2:]
        return retained
