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

from benchmarks.ascend.targets import CurriculumPolicy

from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.phase1_curriculum import (
    Capability,
    LEARNING_FOCI,
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
    r"(?i)(?:\bgdn\b|gated[ -]delta|\bbsa\b|block[ -]sparse[ -]attention"
    r")"
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
    learning_focus: tuple[str, ...] = ()

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
            spec.learning_focus,
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
        """Coarse advisory coverage; token novelty is not an admission rule."""
        rows = {
            *(('operation', item.value) for item in self.operations),
            *(('shape-class', _shape_class(item)) for item in self.shapes),
            *(('learning-focus', item) for item in self.learning_focus),
            *((('experiment', canonical_digest({
                'focus': sorted(self.learning_focus), 'dtype': self.dtype.value,
                'operations': [item.value for item in self.operations],
            })),) if self.learning_focus else ()),
            ('dtype', self.dtype.value),
            ('failure', self.failure_mode.value),
            ('profile', 'none' if self.profile_metric is None else self.profile_metric.value),
        }
        return frozenset(rows)

    def as_dict(self) -> dict[str, object]:
        return {
            **({"learning_focus": list(self.learning_focus)} if self.learning_focus else {}),
            "operations": [item.value for item in self.operations],
            "shapes": [list(item) for item in self.shapes],
            "dtype": self.dtype.value,
            "failure_mode": self.failure_mode.value,
            "profile_metric": (
                None if self.profile_metric is None else self.profile_metric.value
            ),
        }

    @property
    def comparison_sha256(self) -> str:
        """Order-independent admission key, separate from retained identities.

        Keep serialization unchanged so historical specs, prompts, and evidence
        retain their original hashes when a campaign resumes.
        """
        value = self.as_dict()
        if self.learning_focus:
            value["learning_focus"] = sorted(self.learning_focus)
        return canonical_digest(value)


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
        ):
            raise ValueError("prior summary must be a non-empty trimmed string")

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
    remaining_projects: int = 64
    curriculum_memory: tuple[Mapping[str, object], ...] = ()

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
        if type(self.remaining_projects) is not int or self.remaining_projects < 0:
            raise ValueError("remaining_projects must be nonnegative")
        if any(not isinstance(item, Capability) for item in self.pre_wave_coverage):
            raise ValueError("coverage must contain registered capabilities")
        if len(set(self.pre_wave_coverage)) != len(self.pre_wave_coverage):
            raise ValueError("coverage must be unique")
        if any(not isinstance(item, PriorOutcome) for item in self.prior_outcomes):
            raise ValueError("prior outcomes must be registered records")


@dataclass(frozen=True)
class CurriculumSession:
    """Complete selection chronology, not a verbatim provider wire transcript."""

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
        ):
            raise ValueError("terminal reason must be a non-empty trimmed string")


CurriculumDecision = WaveDecision | TerminalDecision
Completion = Callable[[Sequence[Mapping[str, str]]], str]


_SYSTEM = """You are the provider-neutral A5 Catlass Phase-1 Curriculum Agent.
Select generic, host-verifiable projects that maximize reusable information.
Never mention or reconstruct held-out benchmarks. Projects in one wave are
siblings: they share one pre-wave memory and may depend only on capabilities
already present before the wave. Return exactly one allowed decision and JSON.
No Markdown fences or text may precede or follow it."""


ASCEND_EXPERIMENT_GUIDANCE = """
Build an adaptive Ascend A5 curriculum, starting with 64 projects and extending
later. Cover SCALAR address/control work; VECTOR arithmetic; CUBE matrix work;
MTE2 GM->UB/L1; MTE1 L1->L0 operands; MTE3 UB->GM; FIX Cube result conversion
and output. Exercise these through legal copy/vector/matmul graphs, not invented
pipe opcodes. Cover float16, bfloat16, float32 and supported 8-bit FP8/MXFP8.
MXFP8 uses quantized operands plus E8M0 scales for each 32 K elements; compare
against the quantized/scaled host reference, not original unquantized values.

For each hypothesis write a concrete experiment with five labeled parts:
Question; Controlled change; Tools; Measurements; Expected lesson. There is no
500-character limit. Use optional learning_focus to distinguish different
experiments on the same graph; a new ID or rewritten hypothesis is not novelty.
Plan small runnable projects with explicit source changes, not conditional menus.

Examples:
* Copy: Question: does redundant MTE2->VECTOR->MTE3 ordering stall movement?
  Controlled change: hold shape/dtype/launch fixed and compare two correct
  sources, minimal ordering vs extra legal barriers. Tools: msprof timeline.
  Measurements: verify both against host, retain source hashes and device kernel
  duration for both; extract explicit wait intervals when the trace exposes them.
  Expected lesson: identify redundant ordering cost. Focus inter-pipe-sync and
  sync-performance; do not infer wait time from blank timeline space.
* Reduction: Question: which shared-UB SIMT barrier is required? Controlled change:
  diagnose an ordinary wrong-result case and repair with uniform sync_threads.
  Tools: mssanitizer memcheck then racecheck/synccheck, bounded msdebug inspection,
  and tla.print selected values. Measurements: sanitizer findings and host errors
  before/after repair. Expected lesson: correct producer/consumer ownership.
  Do not time broken sources or deliberately hang a device.
* Cube/Vector: compare correct cross-core producer/consumer flag ordering and a
  correct over-synchronized variant. Use a supported cross_flag set/wait protocol;
  test SIMD local-memory ordering and SIMT barriers in separate focused projects.
  Verify all variants; profile Cube, operand movement, FIX and cross-core waits.

Use msprof via request-profile for pipe-utilization, memory-throughput, or
common-clock timeline. request-timing measures the current verified shape when
there is no 1-D sweep; timing_lengths is only for 1-D sweeps. Keep launch plus
synchronize latency separate from msprof diagnostic device kernel duration.
Instrumented duration is not uninstrumented kernel runtime. Report explicit wait
events/endpoints, not gaps; missing counters are unavailable, never zero.
Use request-debug (msdebug) to inspect supported bounded kernel state, not as
a timer. Use request-sanitizer (mssanitizer) for memory/race/init/sync correctness;
its pass never replaces the host oracle. Respect actual route availability and
record unsupported tool/device combinations rather than inventing evidence.
Use tla.print inside the kernel for selected scalars/tensor prefixes in a debug
source only. FP8/BF16/E8M0 need conversion to printable FP32 or storage bytes.
GM/UB printing supports rank 1/2; L1/L0C have stricter layout/prefix rules: consult
the installed Catlass programming API. Remove printing before timing. Never mix
sanitizer, debugger, debug-print and performance runs. Retain measurements for
both correct source variants, and reverify/remeasure the final source.
"""


def _grammar() -> dict[str, object]:
    return {
        "project_fields_exactly": [
            "project_id", "capabilities", "shape", "dtype", "block_count",
            "padding_multiple", "failure_mode", "timing_lengths",
            "profile_metric", "hypothesis", "graph",
        ],
        "optional_project_fields": {"learning_focus": list(LEARNING_FOCI)},
        "capabilities": [item.value for item in Capability if item not in {
            Capability.VECTOR_REDUCTION, Capability.VECTOR_MASK,
        }],
        "dtypes": [item.value for item in DType],
        "low_precision_rules": {
            "float8_e4m3fn/float8_e5m2": "movement/vector graphs allowed; Cube requires exactly one matmul node with operands equal to graph inputs in order, no padded-tail capability; copy-only graphs preserve FP8 output; every non-copy FP8 graph computes all intermediates and output in FP32",
            "mxfp8_e4m3fn/mxfp8_e5m2": "only one matmul node with operands equal to the two graph inputs in order; no padded-tail capability; shape [m,k,n], inputs [m,k] and [k,n], FP32 output; host supplies E8M0 scales per 32 K elements (even-padded group count) as gm_scale0 and gm_scale1 before gm_output",
        },
        "failure_modes": [item.value for item in FailureMode],
        "profile_metrics": [item.value for item in ProfileMetric],
        "output_dtype": "input dtype by default; FP8 copy-only graphs preserve FP8; every non-copy FP8 graph computes intermediates and output in float32; MXFP8 matmul outputs float32",
        "padding_multiples": [1, 16, 32, 64, 128, 256],
        "block_count_bounds": [1, 32],
        "shape": "1-3 dimensions, each integer in [1, 4096]; graph inputs must match shape except matmul shape [m,k,n] with inputs [m,k] and [k,n]; each graph tensor may contain at most 65536 elements",
        "timing_lengths": "0-8 unique ascending integers in [1, 4096]; timing-sweep requires at least two lengths and one-dimensional inputs",
        "composition": "capabilities must agree with graph operations; failure-repair requires a non-none failure_mode; profiling requires profile_metric; padded-tail requires non-aligned output extent and physical tensors no larger than 65536 elements",
        "capability_operation_rules": {
            "rule": "Each listed capability must be present if and only if the graph contains at least one of its listed operations; inspect every node, not just the output. These are RSI metadata rules, not Catlass API restrictions.",
            "simd-vector": ["add", "multiply", "relu"],
            "simt-reduction": ["reduce-sum"],
            "simt-mask": ["mask-positive"],
            "cube-tiling": ["matmul"],
            "gm-ub-movement": "No operation equivalence is enforced; copy alone does not require simd-vector.",
        },
        "graph": {
            "fields_exactly": ["inputs", "nodes", "output"],
            "input_fields_exactly": ["name", "shape"],
            "node_fields_exactly": ["name", "op", "operands", "axis"],
            "operations": [item.value for item in ProjectOp],
            "axis": "null for every non-reduction operation; reduce-sum requires an integer zero-based axis within operand rank",
            "mask-positive": "elementwise 1 if x > 0 else 0, converted to output dtype",
            "bounds": "1-3 inputs, 1-12 topological nodes, 1-3 dimensions",
        },
        "wave_project_fields_exactly": ["spec", "requires"],
        "requires": "list of Capability values already in pre_wave_coverage, never project IDs; [] for the initial wave",
        "wave_fields_exactly": [
            "wave_id", "pre_wave_memory_sha256", "pre_wave_coverage", "projects",
        ],
    }


def curriculum_prompt(
    context: CurriculumContext,
    unavailable_capabilities: tuple[Capability, ...] = (),
    *, target_policy: "CurriculumPolicy | None" = None,
) -> str:
    payload = {
        "objective": context.objective,
        "remaining_projects": context.remaining_projects,
        "max_wave_projects": min(8, context.remaining_projects),
        "next_wave_id": f"wave-{context.wave_index:03d}",
        "pre_wave_memory_sha256": context.pre_wave_memory_sha256,
        "pre_wave_coverage": [item.value for item in context.pre_wave_coverage],
        "prior_outcomes": [item.prompt_dict() for item in context.prior_outcomes],
        "curriculum_memory": list(context.curriculum_memory),
        "bounded_project_grammar": (
            _grammar() if target_policy is None else target_policy.grammar(_grammar())
        ),
    }
    if unavailable_capabilities:
        payload["unavailable_capabilities"] = [
            item.value for item in unavailable_capabilities
        ]
    instructions = (
        "Choose DECISION: WAVE for useful experiments informed by prior outcomes. "
        "Use curriculum_memory to avoid repeating low-yield or failed strategies. "
        "A failed attempt is not kernel knowledge or a permanent operation ban: "
        "a smaller, changed experiment with a fresh ID may still be useful. "
        "no_new_knowledge means verified execution yielded no reusable lesson. "
        "Coverage dimensions are advisory: new exact shapes within a covered shape "
        "class and new combinations of covered dimensions are allowed. Avoid exact "
        "repeats of verified prior or sibling project information; changing only "
        "an ID, hypothesis, or launch tuning does not distinguish information. "
        "Select no more than max_wave_projects (at most eight), then adapt the next wave to observed results. Capability "
        "labels alone are not information gain. Copy the supplied "
        "wave ID, memory identity, and coverage exactly. Otherwise choose SATURATED "
        "when the objective is covered, or STALLED when useful work is blocked.\n"
        "Allowed wire forms:\n"
        "DECISION: WAVE\n{one strict WaveSpec JSON object}\n"
        "DECISION: SATURATED\n{\"reason\":\"...\"}\n"
        "DECISION: STALLED\n{\"reason\":\"...\"}\n"
    )
    if unavailable_capabilities:
        instructions += (
            "Do not select unavailable capabilities or their associated project "
            "fields; this execution factory cannot collect their required evidence.\n"
        )
    guidance = ASCEND_EXPERIMENT_GUIDANCE if target_policy is None else target_policy.guidance
    return instructions + guidance + "\n" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


class CurriculumAgent:
    """Maintain curriculum conversation state and admit semantic decisions."""

    def __init__(
        self,
        complete: Completion,
        *,
        held_out_information_sha256: Sequence[str] = (),
        unavailable_capabilities: Sequence[Capability] = (),
        target_policy: "CurriculumPolicy | None" = None,
        profiling_guidance: str | None = None,
        require_fresh_project_ids: bool = False,
    ) -> None:
        if not callable(complete):
            raise TypeError("complete must be callable")
        self._complete = complete
        self.target_policy = target_policy
        self.profiling_guidance = profiling_guidance
        self.require_fresh_project_ids = require_fresh_project_ids
        self._held_out = frozenset(
            _digest(item, "held-out information identity")
            for item in held_out_information_sha256
        )
        self.unavailable_capabilities = tuple(unavailable_capabilities)
        if (
            any(not isinstance(item, Capability) for item in self.unavailable_capabilities)
            or len(set(self.unavailable_capabilities)) != len(self.unavailable_capabilities)
        ):
            raise ValueError("unavailable capabilities must be registered and unique")

    def select(
        self,
        context: CurriculumContext,
        session: CurriculumSession | None = None,
        *,
        remaining_projects: int | None = None,
    ) -> tuple[CurriculumDecision, CurriculumSession]:
        if not isinstance(context, CurriculumContext):
            raise TypeError("context must be CurriculumContext")
        current = CurriculumSession() if session is None else session
        if not isinstance(current, CurriculumSession):
            raise TypeError("session must be CurriculumSession")
        system = {"role": "system", "content": (
            _SYSTEM if self.target_policy is None else self.target_policy.system
        )}
        history = list(current.messages) if current.messages else [system]
        selection_start = len(history)
        history.append({
            "role": "user",
            "content": (
                "Profiling guidance:\n" + self.profiling_guidance + "\n"
                if self.profiling_guidance is not None else ""
            ) + curriculum_prompt(context, self.unavailable_capabilities,
                                  target_policy=self.target_policy),
        })
        if remaining_projects is not None:
            history[-1]["content"] = (
                f"Project budget: select no more than {remaining_projects} projects in this wave.\n"
                + history[-1]["content"]
            )
        for attempt in range(1, 4):
            # Provider failures are not proposal-validation failures. Let the
            # caller's existing infrastructure handling observe them unchanged.
            # The current host context contains all authoritative prior outcomes.
            # Keep old selections for audit/resume, but do not resend redundant
            # historical contexts. Include all corrections within this selection.
            wire = [system, *history[selection_start:]]
            completion = self._complete(tuple(dict(item) for item in wire))
            if not isinstance(completion, str):
                raise TypeError("curriculum completion must be text")
            history.append({"role": "assistant", "content": completion})
            try:
                decision = self._parse_and_validate(completion, context)
                if isinstance(decision, WaveDecision) and remaining_projects is not None and len(decision.wave.projects) > remaining_projects:
                    raise ValueError("wave exceeds remaining project budget; select a smaller wave")
            except ValueError as exc:
                if attempt == 3:
                    raise ValueError(
                        f"curriculum selection failed validation after 3 attempts: {exc}"
                    ) from exc
                history.append({
                    "role": "user",
                    "content": (
                        f"Proposal validation failed: {exc}. Correct your proposal "
                        "and return one exact decision wire form. The curriculum "
                        "context, wave identity, memory, coverage, and grammar "
                        "are unchanged; no project has been admitted."
                    ),
                })
                continue
            return decision, CurriculumSession(tuple(history), current.turns + attempt)
        raise AssertionError("unreachable selection attempt")

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
        for project in wave.projects:
            if self.target_policy is not None:
                self.target_policy.admit(project.spec)
            elif any(item in {Capability.VECTOR_REDUCTION, Capability.VECTOR_MASK}
                     for item in project.spec.capabilities):
                raise ValueError("native vector capability requires an explicit target policy")
        if len(wave.projects) > min(8, context.remaining_projects):
            raise ValueError("wave exceeds eight-project wave limit or remaining project target")
        unavailable = set(self.unavailable_capabilities)
        if any(unavailable.intersection(item.spec.capabilities) for item in wave.projects):
            raise ValueError("wave selects a capability unavailable in this factory")
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
        prior_ids = {item.project_id for item in (
            context.prior_outcomes if self.require_fresh_project_ids else verified_outcomes)}
        prior_information = {
            item.information.comparison_sha256 for item in verified_outcomes
        }
        information = tuple(
            ProjectInformation.from_project(item.spec) for item in wave.projects
        )
        signatures = tuple(
            item.comparison_sha256 for item in information
        )
        if any(item.spec.project_id in prior_ids for item in wave.projects):
            raise ValueError("wave repeats a prior project ID")
        if len(set(signatures)) != len(signatures):
            raise ValueError("wave repeats executable information between siblings")
        if prior_information.intersection(signatures):
            raise ValueError("wave repeats prior information")
        executable_signatures = {
            canonical_digest({key: value for key, value in item.as_dict().items()
                              if key != "learning_focus"})
            for item in information
        }
        if self._held_out.intersection(executable_signatures):
            raise ValueError("wave matches a held-out contract or shape")
        return WaveDecision(wave)


__all__ = [
    "CurriculumAgent", "CurriculumContext", "CurriculumDecision",
    "CurriculumSession", "PriorOutcome", "ProjectInformation",
    "TerminalDecision", "WaveDecision",
    "curriculum_prompt", "project_information_sha256",
]
