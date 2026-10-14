"""Provider-neutral Curriculum Agent protocol for A5 Phase-1 waves.

This module selects and validates semantic work.  It intentionally knows
nothing about provider routes, model parameters, live project execution, or
memory publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable, Mapping, Sequence

from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.phase1_curriculum import (
    Capability,
    DType,
    FailureMode,
    ProfileMetric,
    ProjectSpec,
    WaveSpec,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DECISION = re.compile(r"DECISION: (WAVE|SATURATED|STALLED)\n(.+)", re.DOTALL)
_HELD_OUT_WORDS = re.compile(
    r"(?i)(?:\bgdn\b|gated[ -]delta|\bbsa\b|block[ -]sparse[ -]attention|"
    r"\bmatmul\b|matrix[ -]multiplication)"
)


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return value


def project_information_sha256(value: ProjectSpec | Mapping[str, object]) -> str:
    """Identify executable information while ignoring labels and prose."""

    spec = value if isinstance(value, ProjectSpec) else ProjectSpec.from_mapping(value)
    body = spec.as_dict()
    body.pop("project_id")
    body.pop("hypothesis")
    return canonical_digest(body)


@dataclass(frozen=True)
class PriorOutcome:
    project_id: str
    information_sha256: str
    verified: bool
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, str) or not self.project_id:
            raise ValueError("prior project_id must be non-empty")
        _digest(self.information_sha256, "prior information identity")
        if type(self.verified) is not bool:
            raise ValueError("prior verified must be boolean")
        if (
            not isinstance(self.summary, str)
            or not self.summary.strip()
            or self.summary != self.summary.strip()
            or len(self.summary) > 500
        ):
            raise ValueError("prior summary must be a trimmed 1-500 character string")

    def prompt_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "verified": self.verified,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class CurriculumContext:
    objective: str
    wave_index: int
    pre_wave_memory_sha256: str
    pre_wave_coverage: tuple[Capability, ...]
    prior_outcomes: tuple[PriorOutcome, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.objective, str)
            or not self.objective.strip()
            or self.objective != self.objective.strip()
        ):
            raise ValueError("objective must be a non-empty trimmed string")
        if type(self.wave_index) is not int or self.wave_index < 1:
            raise ValueError("wave_index must be a positive integer")
        _digest(self.pre_wave_memory_sha256, "pre-wave memory identity")
        if any(not isinstance(item, Capability) for item in self.pre_wave_coverage):
            raise ValueError("coverage must contain registered capabilities")
        if len(set(self.pre_wave_coverage)) != len(self.pre_wave_coverage):
            raise ValueError("coverage must be unique")
        if any(not isinstance(item, PriorOutcome) for item in self.prior_outcomes):
            raise ValueError("prior outcomes must be registered records")


@dataclass(frozen=True)
class CurriculumSession:
    messages: tuple[Mapping[str, str], ...] = ()
    turns: int = 0

    def __post_init__(self) -> None:
        if type(self.turns) is not int or self.turns < 0:
            raise ValueError("session turns must be nonnegative")
        if self.turns * 2 != max(0, len(self.messages) - 1):
            raise ValueError("session history does not match completed turns")


@dataclass(frozen=True)
class WaveDecision:
    wave: WaveSpec


@dataclass(frozen=True)
class TerminalDecision:
    status: str
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"SATURATED", "STALLED"}:
            raise ValueError("invalid terminal curriculum status")
        if (
            not isinstance(self.reason, str)
            or not self.reason.strip()
            or self.reason != self.reason.strip()
            or len(self.reason) > 500
        ):
            raise ValueError("terminal reason must be a trimmed 1-500 character string")


CurriculumDecision = WaveDecision | TerminalDecision
Completion = Callable[[Sequence[Mapping[str, str]]], str]


_SYSTEM = """You are the provider-neutral A5 Catlass Phase-1 Curriculum Agent.
Select generic, host-verifiable projects that maximize reusable information.
Never mention or reconstruct held-out benchmarks. Projects in one wave are
siblings: they share one pre-wave memory and may depend only on capabilities
already present before the wave. Return exactly one allowed decision and JSON.
No Markdown fences or text may precede or follow it."""


def _grammar() -> dict[str, object]:
    return {
        "project_fields_exactly": [
            "project_id", "capabilities", "shape", "dtype", "block_count",
            "padding_multiple", "failure_mode", "timing_lengths",
            "profile_metric", "hypothesis",
        ],
        "capabilities": [item.value for item in Capability],
        "dtypes": [item.value for item in DType],
        "failure_modes": [item.value for item in FailureMode],
        "profile_metrics": [item.value for item in ProfileMetric],
        "wave_project_fields_exactly": ["spec", "requires"],
        "wave_fields_exactly": [
            "wave_id", "pre_wave_memory_sha256", "pre_wave_coverage", "projects",
        ],
    }


def curriculum_prompt(context: CurriculumContext) -> str:
    payload = {
        "objective": context.objective,
        "next_wave_id": f"wave-{context.wave_index:03d}",
        "pre_wave_memory_sha256": context.pre_wave_memory_sha256,
        "pre_wave_coverage": [item.value for item in context.pre_wave_coverage],
        "prior_outcomes": [item.prompt_dict() for item in context.prior_outcomes],
        "bounded_project_grammar": _grammar(),
    }
    instructions = (
        "Choose DECISION: WAVE only when it adds at least one uncovered capability "
        "and no project repeats prior executable information. Copy the supplied "
        "wave ID, memory identity, and coverage exactly. Otherwise choose SATURATED "
        "when the objective is covered, or STALLED when useful work is blocked.\n"
        "Allowed wire forms:\n"
        "DECISION: WAVE\n{one strict WaveSpec JSON object}\n"
        "DECISION: SATURATED\n{\"reason\":\"...\"}\n"
        "DECISION: STALLED\n{\"reason\":\"...\"}\n"
    )
    return instructions + json.dumps(payload, sort_keys=True, separators=(",", ":"))


class CurriculumAgent:
    """Maintain curriculum conversation state and admit semantic decisions."""

    def __init__(
        self,
        complete: Completion,
        *,
        held_out_information_sha256: Sequence[str] = (),
    ) -> None:
        if not callable(complete):
            raise TypeError("complete must be callable")
        self._complete = complete
        self._held_out = frozenset(
            _digest(item, "held-out information identity")
            for item in held_out_information_sha256
        )

    def select(
        self,
        context: CurriculumContext,
        session: CurriculumSession | None = None,
    ) -> tuple[CurriculumDecision, CurriculumSession]:
        if not isinstance(context, CurriculumContext):
            raise TypeError("context must be CurriculumContext")
        current = CurriculumSession() if session is None else session
        if not isinstance(current, CurriculumSession):
            raise TypeError("session must be CurriculumSession")
        history = list(current.messages) if current.messages else [
            {"role": "system", "content": _SYSTEM}
        ]
        history.append({"role": "user", "content": curriculum_prompt(context)})
        completion = self._complete(tuple(dict(item) for item in history))
        if not isinstance(completion, str):
            raise TypeError("curriculum completion must be text")
        decision = self._parse_and_validate(completion, context)
        history.append({"role": "assistant", "content": completion})
        return decision, CurriculumSession(tuple(history), current.turns + 1)

    def _parse_and_validate(
        self, text: str, context: CurriculumContext,
    ) -> CurriculumDecision:
        match = _DECISION.fullmatch(text)
        if match is None:
            raise ValueError("curriculum response must use one exact decision wire form")
        status, payload = match.groups()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("curriculum decision payload must be one JSON object") from exc
        if not isinstance(value, Mapping):
            raise ValueError("curriculum decision payload must be one JSON object")
        if status != "WAVE":
            if set(value) != {"reason"}:
                raise ValueError("terminal decision fields must be exactly reason")
            return TerminalDecision(status, value["reason"])
        if _HELD_OUT_WORDS.search(text):
            raise ValueError("wave exposes a held-out benchmark")
        wave = WaveSpec.from_mapping(value)
        expected_id = f"wave-{context.wave_index:03d}"
        if wave.wave_id != expected_id:
            raise ValueError("wave ID does not match curriculum context")
        if wave.pre_wave_memory_sha256 != context.pre_wave_memory_sha256:
            raise ValueError("wave memory does not match curriculum context")
        if wave.pre_wave_coverage != context.pre_wave_coverage:
            raise ValueError("wave coverage does not match curriculum context")
        prior_ids = {item.project_id for item in context.prior_outcomes}
        prior_information = {item.information_sha256 for item in context.prior_outcomes}
        signatures = tuple(
            project_information_sha256(item.spec) for item in wave.projects
        )
        if any(item.spec.project_id in prior_ids for item in wave.projects):
            raise ValueError("wave repeats a prior project ID")
        if len(set(signatures)) != len(signatures):
            raise ValueError("wave repeats executable information between siblings")
        if prior_information.intersection(signatures):
            raise ValueError("wave repeats prior information")
        if self._held_out.intersection(signatures):
            raise ValueError("wave matches a held-out contract or shape")
        selected = {
            capability for item in wave.projects for capability in item.spec.capabilities
        }
        if selected <= set(context.pre_wave_coverage):
            raise ValueError("wave must add new coverage")
        return WaveDecision(wave)


__all__ = [
    "CurriculumAgent", "CurriculumContext", "CurriculumDecision",
    "CurriculumSession", "PriorOutcome", "TerminalDecision", "WaveDecision",
    "curriculum_prompt", "project_information_sha256",
]
