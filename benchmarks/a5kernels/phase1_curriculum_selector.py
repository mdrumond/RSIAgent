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
from benchmarks.a5kernels.phase1_project import ProjectOp


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DECISION = re.compile(r"DECISION: (WAVE|SATURATED|STALLED)\n(.+)", re.DOTALL)
_HELD_OUT_WORDS = re.compile(
    r"(?i)(?:\bgdn\b|gated[ -]delta|\bbsa\b|block[ -]sparse[ -]attention|"
    r"\bmatmul\b|matrix[ -]multiplication)"
)


def _shape_class(shape: tuple[int, ...]) -> str:
    """Map exact extents to reusable rank/scale/alignment information."""

    largest = max(shape)
    scale = next(
        label for bound, label in (
            (64, "tiny"), (256, "small"), (1024, "medium"),
            (4096, "large"), (16384, "xlarge"),
        ) if largest <= bound
    )
    alignment = "aligned-256" if all(item % 256 == 0 for item in shape) else (
        "aligned-16" if all(item % 16 == 0 for item in shape) else "tail"
    )
    return f"rank-{len(shape)}:{scale}:{alignment}"


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return value


def project_information_sha256(value: ProjectSpec | Mapping[str, object]) -> str:
    """Identify the admitted information dimensions, not labels or tuning."""

    return canonical_digest(ProjectInformation.from_project(value).as_dict())


@dataclass(frozen=True)
class ProjectInformation:
    """Dimensions on which a project can add reusable curriculum coverage."""

    operations: tuple[ProjectOp, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtype: DType
    failure_mode: FailureMode
    profile_metric: ProfileMetric | None

    @classmethod
    def from_project(
        cls, value: ProjectSpec | Mapping[str, object],
    ) -> "ProjectInformation":
        spec = value if isinstance(value, ProjectSpec) else ProjectSpec.from_mapping(value)
        return cls(
            tuple(sorted({node.op for node in spec.graph.nodes}, key=lambda item: item.value)),
            tuple(sorted({
                spec.shape,
                *(item.shape for item in spec.graph.inputs),
                spec.graph.output_shape,
                *((length,) for length in spec.timing_lengths),
            })),
            spec.dtype,
            spec.failure_mode,
            spec.profile_metric,
        )

    def __post_init__(self) -> None:
        if not self.operations or any(
            not isinstance(item, ProjectOp) for item in self.operations
        ) or tuple(sorted(set(self.operations), key=lambda item: item.value)) != self.operations:
            raise ValueError("information operations must be unique and canonical")
        if not self.shapes or tuple(sorted(set(self.shapes))) != self.shapes:
            raise ValueError("information shapes must be unique and canonical")
        if not isinstance(self.dtype, DType) or not isinstance(
            self.failure_mode, FailureMode
        ) or self.profile_metric is not None and not isinstance(
            self.profile_metric, ProfileMetric
        ):
            raise ValueError("information dimensions must use registered values")

    @property
    def tokens(self) -> frozenset[tuple[str, str]]:
        rows = {
            *(('operation', item.value) for item in self.operations),
            *(('shape-class', _shape_class(item)) for item in self.shapes),
            ('dtype', self.dtype.value),
            ('failure', self.failure_mode.value),
            ('profile', 'none' if self.profile_metric is None else self.profile_metric.value),
        }
        return frozenset(rows)

    def as_dict(self) -> dict[str, object]:
        return {
            "operations": [item.value for item in self.operations],
            "shapes": [list(item) for item in self.shapes],
            "dtype": self.dtype.value,
            "failure_mode": self.failure_mode.value,
            "profile_metric": (
                None if self.profile_metric is None else self.profile_metric.value
            ),
        }


@dataclass(frozen=True)
class PriorOutcome:
    project_id: str
    information: ProjectInformation
    verified: bool
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, str) or not self.project_id:
            raise ValueError("prior project_id must be non-empty")
        if not isinstance(self.information, ProjectInformation):
            raise ValueError("prior information must contain admitted dimensions")
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
            "information": self.information.as_dict(),
        }

    @property
    def information_sha256(self) -> str:
        return canonical_digest(self.information.as_dict())


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
            "profile_metric", "hypothesis", "graph",
        ],
        "capabilities": [item.value for item in Capability],
        "dtypes": [item.value for item in DType],
        "failure_modes": [item.value for item in FailureMode],
        "profile_metrics": [item.value for item in ProfileMetric],
        "graph": {
            "fields_exactly": ["inputs", "nodes", "output"],
            "input_fields_exactly": ["name", "shape"],
            "node_fields_exactly": ["name", "op", "operands", "axis"],
            "operations": [item.value for item in ProjectOp],
            "bounds": "1-3 inputs, 1-12 topological nodes, 1-3 dimensions",
        },
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
        "Choose DECISION: WAVE only when every project adds operation, shape, dtype, "
        "failure, or profile information not covered by prior outcomes. Capability "
        "labels alone are not information gain. Copy the supplied "
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
            terminal = TerminalDecision(status, value["reason"])
            if _HELD_OUT_WORDS.search(terminal.reason):
                raise ValueError("terminal decision exposes a held-out benchmark name")
            if status == "SATURATED" and not any(
                item.verified for item in context.prior_outcomes
            ):
                raise ValueError("saturation requires host-verified prior evidence")
            if status == "STALLED" and not context.prior_outcomes:
                raise ValueError("stall requires an observed prior attempt")
            return terminal
        wave = WaveSpec.from_mapping(value)
        if any(
            _HELD_OUT_WORDS.search(item.spec.project_id)
            or _HELD_OUT_WORDS.search(item.spec.hypothesis)
            or any(
                _HELD_OUT_WORDS.search(name)
                for name in (
                    *(graph_input.name for graph_input in item.spec.graph.inputs),
                    *(node.name for node in item.spec.graph.nodes),
                    item.spec.graph.output,
                )
            )
            for item in wave.projects
        ):
            raise ValueError("wave exposes a held-out benchmark name")
        expected_id = f"wave-{context.wave_index:03d}"
        if wave.wave_id != expected_id:
            raise ValueError("wave ID does not match curriculum context")
        if wave.pre_wave_memory_sha256 != context.pre_wave_memory_sha256:
            raise ValueError("wave memory does not match curriculum context")
        if wave.pre_wave_coverage != context.pre_wave_coverage:
            raise ValueError("wave coverage does not match curriculum context")
        verified_outcomes = tuple(
            item for item in context.prior_outcomes if item.verified
        )
        prior_ids = {item.project_id for item in verified_outcomes}
        prior_information = {item.information_sha256 for item in verified_outcomes}
        covered_dimensions = frozenset(
            token
            for item in verified_outcomes
            for token in item.information.tokens
        )
        information = tuple(
            ProjectInformation.from_project(item.spec) for item in wave.projects
        )
        signatures = tuple(
            canonical_digest(item.as_dict()) for item in information
        )
        if any(item.spec.project_id in prior_ids for item in wave.projects):
            raise ValueError("wave repeats a prior project ID")
        if len(set(signatures)) != len(signatures):
            raise ValueError("wave repeats executable information between siblings")
        if prior_information.intersection(signatures):
            raise ValueError("wave repeats prior information")
        if self._held_out.intersection(signatures):
            raise ValueError("wave matches a held-out contract or shape")
        marginal_coverage = set(covered_dimensions)
        for item in information:
            if not item.tokens - marginal_coverage:
                raise ValueError(
                    "every project must add marginal operation/shape-class/dtype/"
                    "failure/profile information"
                )
            marginal_coverage.update(item.tokens)
        return WaveDecision(wave)


__all__ = [
    "CurriculumAgent", "CurriculumContext", "CurriculumDecision",
    "CurriculumSession", "PriorOutcome", "ProjectInformation",
    "TerminalDecision", "WaveDecision",
    "curriculum_prompt", "project_information_sha256",
]
