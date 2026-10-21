"""Target-driven Phase 2 curriculum, independent of broad Phase 1 selection."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping

from benchmarks.a5kernels.phase1_curriculum import Capability, ProjectSpec
from benchmarks.a5kernels.phase1_curriculum_selector import (
    CurriculumSession, _grammar, _HELD_OUT_WORDS, _digest, project_information_sha256,
)
from benchmarks.ascend.phase2_targets import TargetContract
from benchmarks.ascend.targets import CurriculumPolicy


SYSTEM = """You are the target-driven Ascend Phase-2 Curriculum Agent.
A host-qualified target is fixed. Diagnose its latest actual attempt, choose one
small controlled practice experiment, learn from its result, then retry that same
target. Practice is useful only when it addresses a specific observed weakness.
Difficulty or broad coverage alone is not a reason to select practice. A passing
target may still benefit from practice, but every practice requires a fresh target
attempt. Never relax correctness, change the target, invent measurements or impose
an unmeasured speedup threshold. Never mention or reconstruct held-out benchmarks
in practice projects. Reference source is intentionally private.
Return one exact decision wire form and JSON, without Markdown or extra text."""


@dataclass(frozen=True)
class TargetCurriculumContext:
    target: TargetContract
    latest_attempt: Mapping[str, object]
    lessons: tuple[Mapping[str, object], ...]
    remaining_seconds: float
    practice_seconds_remaining: float
    available_tools: tuple[str, ...] = ()
    used_project_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.target, TargetContract):
            raise TypeError("target must be TargetContract")
        if type(self.latest_attempt.get("verified")) is not bool:
            raise ValueError("latest target attempt requires host verified verdict")
        if any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in (
            self.remaining_seconds, self.practice_seconds_remaining)):
            raise ValueError("remaining time budgets must be finite nonnegative seconds")


@dataclass(frozen=True)
class TargetDecision:
    action: str
    reason: str
    project: ProjectSpec | None = None
    weakness: str | None = None
    control: str | None = None
    evidence: str | None = None
    retry_plan: str | None = None

    def as_dict(self):
        result = {"action": self.action, "reason": self.reason}
        if self.project is not None:
            result.update(project=self.project.as_dict(), weakness=self.weakness,
                          control=self.control, evidence=self.evidence, retry_plan=self.retry_plan)
        return result


_TOOL_ALIASES = {
    "request-profile": {"request-profile", "msprof"},
    "request-debug": {"request-debug", "msdebug"},
    "request-sanitizer": {"request-sanitizer", "mssanitizer"},
    "request-timing": {"request-timing"},
    "debug-print": {"debug-print", "tla.print"},
}
_TOOL_FIELDS = {
    "profiling": "request-profile", "tool-msprof": "request-profile",
    "tool-msdebug": "request-debug", "tool-mssanitizer": "request-sanitizer",
    "debug-print": "debug-print", "timing-sweep": "request-timing",
}


def _available_tools(context):
    actual = set(context.available_tools)
    return {name for name, aliases in _TOOL_ALIASES.items() if actual.intersection(aliases)}


def target_curriculum_prompt(context: TargetCurriculumContext,
                             policy: CurriculumPolicy | None = None) -> str:
    grammar = _grammar()
    if policy is not None:
        if policy.target != context.target.runtime["target"]:
            raise ValueError("target policy and runtime differ")
        grammar = policy.grammar(grammar)
    available = _available_tools(context)
    grammar["capabilities"] = [name for name in grammar["capabilities"]
                               if _TOOL_FIELDS.get(name, "") in available or name not in _TOOL_FIELDS]
    grammar["optional_project_fields"]["learning_focus"] = [
        name for name in grammar["optional_project_fields"]["learning_focus"]
        if _TOOL_FIELDS.get(name, "") in available or name not in _TOOL_FIELDS]
    if "request-profile" not in available:
        grammar["profile_metrics"] = []
        grammar["profile_metric_rule"] = "null; profiling tool is unavailable"
    if "request-timing" not in available:
        grammar["timing_lengths"] = "[]; timing tool is unavailable"
    for key in ("wave_project_fields_exactly", "wave_fields_exactly", "requires"):
        grammar.pop(key, None)
    grammar["names"] = "graph input/node names match [a-z][a-z0-9]{0,15}; graph.output is one name string, never an object"
    grammar["work_bound"] = "each matmul work m*k*n at most 2000000"
    return ("Choose PRACTICE with exactly one ProjectSpec, RETRY_TARGET to apply lessons, "
        "READY only after a verified target and no further useful practice, or STALLED "
        "when useful work is blocked. STALLED is not success. Do not return READY for "
        "practice success. Each practice project_id must be fresh: never reuse any "
        "used_project_ids or the fixed target ID. Use only available tools; missing tools are unavailable, "
        "never zero. Separate debug/print/sanitizer runs from device timing. Sync "
        "practice must test correctness and compare performance only for correct variants.\n"
        "Wire forms:\nDECISION: PRACTICE\n"
        '{"reason":"...","project":{ProjectSpec},"weakness":"observed target failure or bottleneck",'
        '"control":"one controlled change","evidence":"host checks and relevant measurements",'
        '"retry_plan":"how the result changes the unchanged target"}\n'
        'DECISION: RETRY_TARGET\n{"reason":"..."}\n'
        'DECISION: READY\n{"reason":"..."}\n'
        'DECISION: STALLED\n{"reason":"..."}\n'
        + json.dumps({"target": context.target.prompt_dict(),
            "latest_target_attempt": dict(context.latest_attempt), "lessons": list(context.lessons),
            "remaining_seconds": context.remaining_seconds,
            "practice_seconds_remaining": context.practice_seconds_remaining,
            "available_tools": list(context.available_tools),
            "used_project_ids": list(context.used_project_ids), "project_grammar": grammar}, sort_keys=True))


class TargetCurriculumAgent:
    def __init__(self, complete, *, target_policy=None, held_out_information_sha256=()):
        if not callable(complete):
            raise TypeError("complete must be callable")
        self.complete = complete
        self.target_policy = target_policy
        self._held_out = frozenset(_digest(item, "held-out information identity")
                                   for item in held_out_information_sha256)

    def select(self, context, session=None):
        session = CurriculumSession() if session is None else session
        system = {"role": "system", "content": SYSTEM}
        history = list(session.messages) if session.messages else [system]
        prompt = target_curriculum_prompt(context, self.target_policy)
        if session.messages:
            prompt = ("Previous accepted curriculum decision for interpreting practice results; "
                      "the current context below owns the goal, budgets and host verdicts:\n"
                      + session.messages[-1]["content"] + "\n" + prompt)
        turn = [{"role": "user", "content": prompt}]
        for attempt in range(1, 4):
            response = self.complete([system, *turn])
            turn.append({"role": "assistant", "content": response})
            try:
                decision = self.parse(response, context)
            except (ValueError, TypeError, KeyError) as exc:
                if attempt == 3:
                    raise ValueError(f"Phase 2 selection failed after 3 attempts: {exc}") from exc
                turn.append({"role": "user", "content": f"Invalid proposal: {exc}. Correct only this decision; target and budgets are unchanged."})
                continue
            return decision, CurriculumSession(tuple([*history, *turn]), session.turns + attempt)
        raise AssertionError("unreachable")

    def parse(self, text, context):
        if self.target_policy is not None and self.target_policy.target != context.target.runtime["target"]:
            raise ValueError("target policy and runtime differ")
        if not isinstance(text, str) or "\n" not in text:
            raise ValueError("expected exact decision and JSON")
        header, raw = text.split("\n", 1)
        actions = {"DECISION: PRACTICE": "practice", "DECISION: RETRY_TARGET": "retry-target",
                   "DECISION: READY": "ready", "DECISION: STALLED": "stalled"}
        if header not in actions:
            raise ValueError("unknown Phase 2 decision")
        action, value = actions[header], json.loads(raw)
        fields = {"reason", "project", "weakness", "control", "evidence", "retry_plan"} if action == "practice" else {"reason"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("decision fields differ from wire form")
        for key in fields - {"project"}:
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError(f"{key} must be nonempty text")
        if action == "ready" and context.latest_attempt["verified"] is not True:
            raise ValueError("readiness requires a host-verified target")
        if action in {"practice", "retry-target"} and context.remaining_seconds == 0:
            raise ValueError("no target retry budget remains")
        if action == "practice":
            if context.practice_seconds_remaining == 0:
                raise ValueError("no practice budget remains")
            project = ProjectSpec.from_mapping(value["project"])
            if project.project_id == context.target.spec.project_id:
                raise ValueError("practice must not replace the target")
            if project.project_id in context.used_project_ids:
                raise ValueError("practice requires a fresh project ID; this ID was already used")
            names = (*(value[key] for key in fields - {"project"}),
                     project.project_id, project.hypothesis, project.graph.output,
                     *(item.name for item in project.graph.inputs),
                     *(item.name for item in project.graph.nodes))
            if any(_HELD_OUT_WORDS.search(name) for name in names):
                raise ValueError("practice exposes a held-out benchmark name")
            if project_information_sha256(project) in self._held_out:
                raise ValueError("practice matches a held-out contract or shape")
            requested = {item.value for item in project.capabilities} | set(project.learning_focus)
            missing = {_TOOL_FIELDS[name] for name in requested if name in _TOOL_FIELDS} - _available_tools(context)
            if missing:
                raise ValueError("practice requires unavailable tools: " + ", ".join(sorted(missing)))
            if self.target_policy is not None:
                self.target_policy.admit(project)
            elif any(item in {Capability.VECTOR_REDUCTION, Capability.VECTOR_MASK}
                     for item in project.capabilities):
                raise ValueError("native vector capability requires an explicit target policy")
            value["project"] = project
        return TargetDecision(action=action, **value)
