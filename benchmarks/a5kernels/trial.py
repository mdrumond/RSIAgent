"""Restricted, host-owned lifecycle for model-driven A5 kernel trials."""

from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, field as dataclass_field, replace
import difflib
import hashlib
import json
import time
import math
from pathlib import Path
import re
from typing import Callable, Mapping, Protocol

from benchmarks.a5kernels.evidence import EvidenceKind, EvidenceLedger
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, KnowledgeQuery
from benchmarks.a5kernels.matrix import (
    CapabilityUnavailableError,
    ExperimentCell,
    ExperimentOrchestrator,
    ProfilingMode,
    RuntimeCapabilities,
    Workload,
    validate_path_component,
)
from benchmarks.a5kernels.model_profile import A5ModelProfile, load_a5_model_profile
from benchmarks.a5kernels.protocol import ExecutionPlan, VerifiedResult


_EMPTY_ACTIONS = {"compile", "run", "request-profile", "request-timing", "request-debug", "submit"}
_SOURCE_PATHS = {
    "catlass-dsl": "kernel.py",
    "ascend-c": "kernel.cpp",
    "triton-ascend": "kernel.py",
}


@dataclass(frozen=True)
class TrialAction:
    kind: str
    payload: Mapping[str, object]
    dup: int = 1


@dataclass(frozen=True)
class ActionOutcome:
    """Dependency-light result adapted to core's domain result at the seam."""

    observation: str
    terminal: bool = False
    status: str = "done"
    admitted_artifacts: Mapping[str, str] = dataclass_field(default_factory=dict)
    budget_exempt: bool = False


def parse_trial_action(text: str) -> TrialAction | None:
    """Accept exactly one JSON action and no agent-selected commands or scores."""

    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or len(value) != 1:
        return None
    kind, payload = next(iter(value.items()))
    if kind in _EMPTY_ACTIONS:
        return TrialAction(kind, {}) if payload == {} else None
    if not isinstance(payload, dict):
        return None
    if kind == "read-artifact" and set(payload) <= {"sha256", "offset", "limit"}:
        digest = payload.get("sha256")
        offset, limit = payload.get("offset", 0), payload.get("limit", 8192)
        if (isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                and type(offset) is int and offset >= 0
                and type(limit) is int and 1 <= limit <= 8192):
            return TrialAction(kind, {"sha256": digest, "offset": offset, "limit": limit})
        return None
    if kind == "request-sanitizer" and set(payload) == {"tool"}:
        if payload["tool"] in ("memcheck", "racecheck", "initcheck", "synccheck"):
            return TrialAction(kind, payload)
        return None
    if kind == "propose" and set(payload) == {"text"}:
        return TrialAction(kind, payload) if isinstance(payload["text"], str) else None
    if kind == "write" and set(payload) == {"slot", "content"}:
        if payload["slot"] == "kernel" and isinstance(payload["content"], str):
            return TrialAction(kind, payload)
        return None
    if kind == "query-knowledge" and set(payload) <= {"query", "limit"}:
        query, limit = payload.get("query"), payload.get("limit", 5)
        if isinstance(query, str) and query.strip() and isinstance(limit, int) \
                and not isinstance(limit, bool) and 1 <= limit <= 20:
            return TrialAction(kind, {"query": query.strip(), "limit": limit})
    return None


class KernelTrialBackend(Protocol):
    """Concrete runtimes choose all commands and return host-attested results."""

    def compile(self, workspace: Path, language: str,
                attempt_id: str) -> Mapping[str, object]: ...

    def run(
        self,
        workspace: Path,
        language: str,
        workload: Workload,
        attempt_id: str,
        ledger: EvidenceLedger,
    ) -> "CandidateRun": ...


@dataclass(frozen=True)
class CandidateRun:
    """A matching host plan/result; failed runs may lack a discovered kernel name."""

    plan: ExecutionPlan
    verified: VerifiedResult
    kernel_name: str | None
    workload: Workload
    # Hash the authored kernel slot bytes before composing any host runtime.
    candidate_source_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.candidate_source_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.candidate_source_sha256) is None):
            raise ValueError("candidate_source_sha256 must hash the exact authored source bytes")
        if self.kernel_name is None:
            if self.verified.passed:
                raise ValueError("passed candidates require an exact kernel_name")
        elif (not isinstance(self.kernel_name, str) or not self.kernel_name
              or self.kernel_name != self.kernel_name.strip()):
            raise ValueError("kernel_name must be an exact non-empty name")
        if not isinstance(self.workload, Workload):
            raise ValueError("workload must be a Workload")
        expected = {
            "request_id": self.plan.request_id,
            "execution_id": self.plan.execution_id,
            "attempt_id": self.plan.attempt_id,
            "language": self.plan.language,
            "runtime_provenance": self.plan.runtime_provenance,
            "source_fingerprint": self.plan.source_fingerprint,
        }
        for field, value in expected.items():
            if getattr(self.verified, field) != value:
                raise ValueError(f"verified result does not match plan {field}")


class ProfileEvaluation(Protocol):
    """Treatment-aware feedback plus treatment-independent final evaluation."""

    def intermediate(self, run: CandidateRun) -> Mapping[str, object]: ...

    def final(self, run: CandidateRun) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class ActorOutcome:
    status: str
    iterations: int
    # Core currently logs provider usage but does not expose an authoritative
    # per-attempt total.  Keep that absence explicit rather than recording a
    # misleading zero in experiment results.
    tokens: int | None = None
    wall_time_s: float = 0.0
    terminal_response: str | None = None
    distillation: object | None = None
    admitted_artifacts: Mapping[str, str] = dataclass_field(default_factory=dict)
    continuation: object | None = None


class ActorDriver(Protocol):
    def __call__(
        self,
        instruction: str,
        *,
        profile: A5ModelProfile,
        workspace: Path,
        context: Path,
        turn_parser: Callable[[str], TrialAction | None],
        action_executor: Callable[[TrialAction], ActionOutcome],
    ) -> ActorOutcome: ...


class CoreAttemptDriver:
    """Bind a restricted A5 role to ``core.loop.run_attempt``'s opt-in seam."""

    def __init__(self, vm, cfg, sink_factory: Callable[[Path], object]):
        self.vm, self.cfg, self.sink_factory = vm, cfg, sink_factory

    def __call__(self, instruction: str, *, profile: A5ModelProfile,
                 workspace: Path, context: Path, turn_parser, action_executor) -> ActorOutcome:
        from core.loop import DomainActionResult, run_attempt

        cfg = copy(self.cfg)
        cfg.model = profile.model
        cfg.max_tokens = profile.generation.max_tokens
        cfg.temperature = profile.generation.temperature
        cfg.primary_temperature = profile.generation.temperature
        cfg.retry_temperature = profile.generation.temperature
        cfg.top_p = profile.generation.top_p
        cfg.reasoning_effort = profile.generation.reasoning_effort
        cfg.allow_truncation_retry = profile.generation.allow_truncation_retry
        cfg.provider_order = (profile.provider,)
        cfg.provider_allow_fallbacks = profile.allow_fallbacks
        cfg.provider_require_parameters = profile.require_parameters
        cfg.independent_verify = False
        # Core's optional summarizer has its own generation policy. Keep the
        # complete history so every A5 completion uses the fixed actor route.
        cfg.history_keep_pairs = 0
        def execute(action):
            outcome = action_executor(action)
            return DomainActionResult(
                outcome.observation,
                outcome.terminal,
                outcome.status,
                budget_exempt=outcome.budget_exempt,
            )

        result, _ = run_attempt(
            instruction, self.vm, cfg, self.sink_factory(context),
            turn_parser=turn_parser, action_executor=execute,
            system_prompt=_SYSTEM_PROMPT, instruction_is_complete_opening=True,
            action_nudge=_ACTION_NUDGE, strict_action_nudge=_ACTION_NUDGE,
            surface_baseline="",
        )
        return ActorOutcome(
            result.status,
            result.turns,
            wall_time_s=result.wall_secs,
        )


_SYSTEM_PROMPT = """You develop one A5 kernel. Respond with exactly one JSON object:
{"propose":{"text":"..."}}, {"write":{"slot":"kernel","content":"..."}},
{"compile":{}}, {"run":{}}, {"query-knowledge":{"query":"...","limit":5}},
{"request-profile":{}}, {"request-timing":{}}, {"request-debug":{}},
{"request-sanitizer":{"tool":"memcheck"}}, or {"submit":{}}.
Adaptive projects also support {"read-artifact":{"sha256":"<hash>","offset":0,"limit":8192}}.
Read full retained evidence in byte pages (maximum 8192 bytes); follow next_offset.
UTF-8 pages contain readable text, not base64; binary pages return metadata only.
Diagnostic actions are bound by adaptive projects; sanitizer tool can also be racecheck,
initcheck, or synccheck. Diagnostic observations never establish correctness or timing.
You cannot choose paths or commands and cannot report a score or authoritative result.
Compile checks source with the host compiler only; it never launches the kernel or
establishes runtime or numerical correctness."""

_ACTION_NUDGE = """Respond with exactly one supported JSON action: propose, write,
compile, run, query-knowledge, request-profile, request-timing, request-debug,
request-sanitizer, read-artifact (adaptive projects), or submit. Use the object shape shown
in the system instructions and no surrounding prose."""


_TRIAL_CONTRACTS = {
    ("catlass-dsl", Workload.SMOKE_VECTOR_ADD): {
        "entry_point": "@tla.kernel\ndef vector_add(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:",
        "arguments": "gm_a and gm_b are inputs; gm_c is the output. All are contiguous one-dimensional float32 tla.Tensor values with equal shapes.",
        "shape": "This production smoke case has logical length N = 32. The host zero-pads inputs to P = 64 elements. Read P from gm_a.origin_shape[0]; handle every element of that padded shape. This trial does not establish correctness for other logical lengths.",
        "output": "Write gm_c[i] = gm_a[i] + gm_b[i] for 0 <= i < P. Do not mutate inputs. Return no value; the first N elements carry logical data, and the host evaluates all P output elements including the zero-padded tail.",
        "correctness": "The host compares against its original input sums. Every output must be finite, have the expected length, and have maximum absolute error <= 1e-5.",
        "source": "Import only 'import catlass.tla as tla'. Define exactly the single synchronous vector_add kernel above; use no helper functions, additional decorators, or top-level execution. Optional module constants must be numeric literals.",
        "runtime": "The host owns allocation, compilation for A5 (3510), launch with block_num=1, synchronization, and verification. Supply only the kernel module, not a run function or host driver.",
    },
}


def _trial_instruction(
    language: str,
    workload: Workload,
    *,
    logical_length: int = 32,
    padded_length: int = 64,
    block_count: int = 1,
) -> str:
    contract = _TRIAL_CONTRACTS.get((language, workload))
    if contract is None:
        raise CapabilityUnavailableError(
            f"no executable trial contract for {language}/{workload.value}; "
            "this combination is preregistered for future implementation")
    if (
        type(logical_length) is not int
        or type(padded_length) is not int
        or type(block_count) is not int
        or logical_length < 1
        or padded_length < logical_length
        or block_count < 1
    ):
        raise ValueError("invalid host-owned executable contract dimensions")
    contract = dict(contract)
    contract["shape"] = (
        f"This production case has logical length N = {logical_length}. "
        f"The host zero-pads inputs to P = {padded_length} elements. Read P from "
        "gm_a.origin_shape[0]; handle every element of that padded shape. This "
        "trial does not establish correctness for other logical lengths."
    )
    contract["output"] = (
        "Write gm_c[i] = gm_a[i] + gm_b[i] for 0 <= i < P. Do not mutate "
        f"inputs. Return no value; the first {logical_length} elements carry "
        f"logical data, and the host evaluates all {padded_length} output "
        "elements including the zero-padded tail."
    )
    contract["runtime"] = (
        "The host owns allocation, compilation for A5 (3510), launch with "
        f"block_num={block_count}, synchronization, and verification. Supply "
        "only the kernel module, not a run function or host driver."
    )
    return (
        f"Develop a {language} kernel for the {workload.value} workload. "
        f"Write the complete source to the kernel slot ({_SOURCE_PATHS[language]}). "
        "Use compile and run observations to revise it, then submit the final source.\n"
        "Executable contract:\n" + json.dumps(contract, sort_keys=True, indent=2)
    )


class TrialActionExecutor:
    def __init__(self, *, workspace: Path, language: str, workload: Workload,
                 backend: KernelTrialBackend, ledger: EvidenceLedger,
                 knowledge: KnowledgeAgent, profiling: ProfileEvaluation,
                 profiling_guidance: bool,
                 max_actions: int = 40, max_source_bytes: int = 128_000,
                 max_seconds: float = 3600,
                 unmetered_compile_probes: int | None = None,
                 artifacts: ContentAddressedArtifacts | None = None):
        self.workspace, self.language, self.workload = workspace, language, workload
        self.backend, self.ledger = backend, ledger
        self.knowledge, self.profiling = knowledge, profiling
        self.profiling_guidance = profiling_guidance
        from benchmarks.ascend.time_budget import deadline_after
        self.deadline = deadline_after(max_seconds)
        self.max_actions, self.max_source_bytes = max_actions, max_source_bytes
        self.unmetered_compile_probes = unmetered_compile_probes
        self.artifacts = artifacts
        # The same source can recur across cells and determinism repeats. BZ
        # sessions use attempt IDs, so local workspace isolation must reach them.
        self._attempt_namespace = hashlib.sha256(
            str(workspace.resolve()).encode("utf-8")).hexdigest()[:24]
        self.actions = 0
        self.total_actions = 0
        self.compile_probes = 0
        self.latest_run: CandidateRun | None = None
        self.final_run: CandidateRun | None = None
        self.final_profile: Mapping[str, object] | None = None
        self.admitted_artifacts: dict[str, str] = {}

    def __call__(self, action: TrialAction) -> ActionOutcome:
        if not isinstance(action, TrialAction):
            raise TypeError("A5 trials accept only TrialAction values")
        self.total_actions += 1
        compile_exempt = action.kind == "compile"
        if compile_exempt:
            self.compile_probes += 1
        else:
            self.actions += 1
        exhausted = time.monotonic() >= self.deadline
        action_evidence = {
            "ordinal": self.total_actions,
            "action": action.kind,
            "payload_sha256": hashlib.sha256(
                json.dumps(action.payload, sort_keys=True).encode()).hexdigest(),
            "budget": {"remaining_seconds": max(0, self.deadline-time.monotonic())},
        }
        if exhausted:
            self.ledger.append(EvidenceKind.ACTION, action_evidence)
            return ActionOutcome(
                json.dumps({"error": "time budget exhausted"}),
                True,
                "safety_ceiling",
                budget_exempt=compile_exempt,
            )
        handler = getattr(self, "_" + action.kind.replace("-", "_"))
        action_value = {"kind": action.kind, "payload": action.payload}
        action_ref = self.artifacts.put_json(action_value) if self.artifacts else None
        if action_ref is not None:
            action_evidence["payload_artifact_sha256"] = action_ref
        self.ledger.append(EvidenceKind.ACTION, action_evidence)
        if action_ref is not None:
            self._artifact("action-payload", action_ref, ordinal=self.total_actions)
            self.admitted_artifacts[action_ref] = "action"
        outcome = handler(action.payload)
        if self.artifacts is not None:
            observation_ref = self.artifacts.put_bytes(outcome.observation.encode("utf-8"))
            self._artifact("host-observation", observation_ref, ordinal=self.total_actions)
            evidence_kind = action.kind
            try:
                host = json.loads(outcome.observation).get("host", {})
            except (AttributeError, json.JSONDecodeError):
                host = {}
            if action.kind in {"run", "submit"} and host.get("passed") is True:
                evidence_kind = "host-verification"
            elif action.kind == "request-profile":
                evidence_kind = "profiling"
            self.admitted_artifacts[observation_ref] = evidence_kind
        return replace(outcome, budget_exempt=compile_exempt,
                       admitted_artifacts=dict(self.admitted_artifacts))

    def _propose(self, payload) -> ActionOutcome:
        return self._observation("proposal-recorded", chars=len(payload["text"]))

    def _write(self, payload) -> ActionOutcome:
        content = payload["content"]
        if not content or len(content.encode()) > self.max_source_bytes:
            return self._observation("write-rejected", reason="source size")
        path = self.workspace / _SOURCE_PATHS[self.language]
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(content, encoding="utf-8")
        if self.artifacts is not None:
            source_ref = self.artifacts.put_bytes(content.encode("utf-8"))
            difference = "".join(difflib.unified_diff(
                previous.splitlines(keepends=True), content.splitlines(keepends=True),
                fromfile="previous/kernel", tofile="candidate/kernel",
            ))
            diff_ref = self.artifacts.put_bytes(difference.encode("utf-8"))
            self._artifact("candidate-source", source_ref, ordinal=self.total_actions)
            self._artifact("candidate-diff", diff_ref, ordinal=self.total_actions,
                           source_sha256=source_ref)
            self.admitted_artifacts[source_ref] = "source"
            self.admitted_artifacts[diff_ref] = "diff"
        self.latest_run = None
        return self._observation("written", slot="kernel", sha256=hashlib.sha256(
            content.encode()).hexdigest())

    def _compile(self, _payload) -> ActionOutcome:
        if not self._has_source():
            return self._observation("compile", error="write kernel source first")
        try:
            result = dict(self.backend.compile(
                self.workspace, self.language,
                f"{self._attempt_namespace}-compile-{self.total_actions}"))
        except ValueError as exc:
            return self._observation("compile", status="source-validation-error",
                                     error=str(exc))
        return self._observation("compile", result=result)

    def _run(self, _payload) -> ActionOutcome:
        if not self._has_source():
            return self._observation("run", error="write kernel source first")
        self.latest_run = None
        attempt_id = f"{self._attempt_namespace}-candidate-{self.total_actions}"
        try:
            candidate = self.backend.run(
                self.workspace, self.language, self.workload,
                attempt_id, self.ledger)
            self._validate_candidate(candidate, attempt_id)
        except ValueError as exc:
            return self._observation("run", status="source-validation-error",
                                     error=str(exc))
        self.latest_run = candidate
        result = self.latest_run.verified
        finite_error = math.isfinite(result.max_abs_error)
        status = ("runtime-error" if result.exit_code != 0 else
                  "passed" if result.passed else "incorrect-output")
        return self._observation(
            "run", passed=result.passed, exit_code=result.exit_code, status=status,
            max_abs_error=result.max_abs_error if finite_error else None,
            error_status=None if finite_error else "non-finite-max-abs-error")

    def _query_knowledge(self, payload) -> ActionOutcome:
        results = self.knowledge.query(KnowledgeQuery(**payload))
        if not self.knowledge.enabled:
            return ActionOutcome(KnowledgeAgent.DISABLED_OBSERVATION)
        response = {"knowledge_query": {"available": True, "results": [
            {"citation": asdict(result.citation), "text": result.text}
            for result in results
        ]}}
        return ActionOutcome(json.dumps(response, sort_keys=True, separators=(",", ":")))

    def _request_debug(self, _payload) -> ActionOutcome:
        return self._observation(
            "diagnostic", available=False,
            reason="debugger and sanitizer execution is supported by adaptive Phase 1 only",
        )

    def _request_sanitizer(self, payload) -> ActionOutcome:
        return self._request_debug(payload)

    def _request_timing(self, _payload) -> ActionOutcome:
        return self._observation(
            "timing", available=False,
            reason="timing-sweep measurements are supported by adaptive Phase 1 only",
        )

    def _request_profile(self, _payload) -> ActionOutcome:
        if not self.profiling_guidance:
            return self._observation("profile", available=False)
        if self.latest_run is None or not self.latest_run.verified.passed:
            return self._observation("profile", available=True,
                                     error="run a correct candidate first")
        return self._observation("profile", available=True,
                                 result=dict(self.profiling.intermediate(self.latest_run)))

    def _submit(self, _payload) -> ActionOutcome:
        if not self._has_source():
            return self._observation("submit", error="write kernel source first")
        self.final_run = None
        self.final_profile = None
        attempt_id = f"{self._attempt_namespace}-final"
        try:
            candidate = self.backend.run(
                self.workspace, self.language, self.workload,
                attempt_id, self.ledger)
            self._validate_candidate(candidate, attempt_id)
        except ValueError as exc:
            return self._observation("submit", status="source-validation-error",
                                     error=str(exc))
        self.final_run = candidate
        if self.final_run.verified.passed:
            self.final_profile = self.profiling.final(self.final_run)
        else:
            self.final_profile = {
                "available": False,
                "reason": "correctness-failed",
            }
        return self._observation("submitted", terminal=True,
                                 passed=self.final_run.verified.passed,
                                 attestation_sha256=self.final_run.verified.attestation_sha256)

    def _has_source(self) -> bool:
        return (self.workspace / _SOURCE_PATHS[self.language]).is_file()

    def _validate_candidate(self, candidate: CandidateRun, attempt_id: str) -> None:
        if not isinstance(candidate, CandidateRun):
            raise TypeError("kernel backend must return a CandidateRun")
        if candidate.plan.attempt_id != attempt_id:
            raise ValueError("candidate attempt_id does not match requested attempt")
        if candidate.plan.language != self.language:
            raise ValueError("candidate language does not match scheduled language")
        if candidate.workload is not self.workload:
            raise ValueError("candidate workload does not match scheduled workload")
        current_source = (self.workspace / _SOURCE_PATHS[self.language]).read_bytes()
        if candidate.candidate_source_sha256 != hashlib.sha256(current_source).hexdigest():
            raise ValueError("candidate source does not match current workspace kernel")

    def _artifact(self, role: str, digest: str, **fields: object) -> None:
        self.ledger.append(EvidenceKind.ARTIFACT, {
            "role": role, "sha256": digest, **fields,
        })

    @staticmethod
    def _observation(event: str, terminal: bool = False, **fields) -> ActionOutcome:
        return ActionOutcome(json.dumps({"host": {"event": event, **fields}},
                                        sort_keys=True, separators=(",", ":")),
                             terminal=terminal)


@dataclass(frozen=True)
class TrialResult:
    outcome: ActorOutcome
    run: CandidateRun
    final_profile: Mapping[str, object]
    workspace: Path
    memory: Path
    context: Path
    evidence_sha256: str
    artifacts: Path | None = None

    @property
    def verified(self) -> VerifiedResult:
        return self.run.verified


def _retain_final_artifacts(
    *, workspace: Path, language: str, executor: TrialActionExecutor,
    artifacts: ContentAddressedArtifacts, ledger: EvidenceLedger,
    outcome: ActorOutcome,
) -> None:
    """Bind final host-verified state into the trial's content-addressed store."""

    final_source = (workspace / _SOURCE_PATHS[language]).read_bytes()
    source_ref = artifacts.put_bytes(final_source)
    if source_ref != executor.final_run.candidate_source_sha256:
        raise RuntimeError("retained final source does not match verified candidate")
    ledger.append(EvidenceKind.ARTIFACT, {
        "role": "final-source", "sha256": source_ref,
    })
    profile_ref = artifacts.put_json(executor.final_profile)
    ledger.append(EvidenceKind.ARTIFACT, {
        "role": "final-profile", "sha256": profile_ref,
    })
    if outcome.terminal_response is not None:
        response_ref = artifacts.put_bytes(outcome.terminal_response.encode("utf-8"))
        ledger.append(EvidenceKind.ARTIFACT, {
            "role": "post-terminal-response", "sha256": response_ref,
        })


class TrialOrchestrator:
    def __init__(self, root: Path, capabilities: RuntimeCapabilities,
                 backend: KernelTrialBackend, actor: ActorDriver,
                 *, knowledge_factory: Callable[[Path, ExperimentCell], KnowledgeAgent],
                 profiling_factory: Callable[[ExperimentCell], ProfileEvaluation]):
        self.root, self.scheduler = Path(root), ExperimentOrchestrator(capabilities)
        self.backend, self.actor = backend, actor
        self.knowledge_factory, self.profiling_factory = knowledge_factory, profiling_factory

    def run(self, cell: ExperimentCell, workload: Workload,
            *, run_identity: str | None = None) -> TrialResult:
        self.scheduler.schedule(cell, workload)
        instruction = _trial_instruction(cell.language.value, workload)
        profile = load_a5_model_profile()
        if cell.model.model_id != profile.model:
            raise ValueError("experiment cell does not use the fixed A5 model profile")
        workspace_id = validate_path_component(cell.workspace_id, "workspace_id")
        trial_root = self.root / workspace_id / workload.value
        if run_identity is not None:
            try:
                trial_root /= validate_path_component(run_identity, "run_identity")
            except ValueError as exc:
                raise ValueError(
                    "run_identity must be a safe host-owned identifier"
                ) from exc
        trial_root.mkdir(parents=True, exist_ok=False)
        workspace, memory, context = (trial_root / name for name in
                                      ("workspace", "memory", "context"))
        for path in (workspace, memory, context):
            path.mkdir()
        ledger = EvidenceLedger(trial_root / "evidence.jsonl")
        artifacts = ContentAddressedArtifacts(trial_root / "artifacts")
        instruction_ref = artifacts.put_bytes(instruction.encode("utf-8"))
        system_ref = artifacts.put_bytes(_SYSTEM_PROMPT.encode("utf-8"))
        ledger.append(EvidenceKind.REQUEST, {
            "cell": asdict(cell), "workload": workload.value,
            "model_profile_sha256": profile.fingerprint,
            "instruction_artifact_sha256": instruction_ref,
        })
        ledger.append(EvidenceKind.ARTIFACT, {
            "role": "actor-instruction", "sha256": instruction_ref,
        })
        ledger.append(EvidenceKind.ARTIFACT, {
            "role": "actor-system-prompt", "sha256": system_ref,
        })
        knowledge = self.knowledge_factory(memory, cell)
        try:
            if knowledge.enabled != (cell.knowledge.value == "with-kdb"):
                raise ValueError("Knowledge Agent treatment does not match experiment cell")
            profiling = self.profiling_factory(cell)
            executor = TrialActionExecutor(
                workspace=workspace, language=cell.language.value, workload=workload,
                backend=self.backend, ledger=ledger, knowledge=knowledge,
                profiling=profiling,
                profiling_guidance=cell.profiling is ProfilingMode.WITH_GUIDANCE,
                artifacts=artifacts)
            outcome = self.actor(
                instruction,
                profile=profile, workspace=workspace, context=context,
                turn_parser=parse_trial_action, action_executor=executor)
            if executor.final_run is None or executor.final_profile is None:
                raise RuntimeError("Actor ended without a host-verified submit action")
            _retain_final_artifacts(
                workspace=workspace, language=cell.language.value,
                executor=executor, artifacts=artifacts, ledger=ledger,
                outcome=outcome,
            )
            return TrialResult(outcome, executor.final_run, executor.final_profile,
                               workspace, memory, context, ledger.head_sha256,
                               artifacts.root)
        finally:
            knowledge.close()

    @staticmethod
    def run_bound(
        *,
        trial_root: Path,
        request_payload: Mapping[str, object],
        language: str,
        workload: Workload,
        instruction: str,
        profile,
        backend: KernelTrialBackend,
        actor: ActorDriver,
        knowledge: KnowledgeAgent,
        profiling: ProfileEvaluation,
        profiling_guidance: bool,
        ledger: EvidenceLedger,
        unmetered_compile_probes: int | None = None,
        max_seconds: float = 3600,
    ) -> TrialResult:
        """Run one host-bound project without the legacy pilot matrix scheduler.

        The Phase 1 cell registry has its own exact model and treatment identities.
        This entry point preserves the established restricted action executor while
        leaving cell selection and durable resume policy to the outer composition.
        """

        trial_root = Path(trial_root)
        trial_root.mkdir(parents=True, exist_ok=False)
        workspace, memory, context = (
            trial_root / name for name in ("workspace", "memory", "context")
        )
        for path in (workspace, memory, context):
            path.mkdir()
        request = dict(request_payload)
        request["trial_budget"] = {"max_seconds": max_seconds, "policy": "elapsed-time-v1"}
        actor_instruction = instruction + (
            "\n\nCompile and repair within the remaining time. Compilation does not launch "
            "the kernel or establish correctness. Use run and submit for host verification."
        )
        artifacts = ContentAddressedArtifacts(trial_root / "artifacts")
        instruction_ref = artifacts.put_bytes(actor_instruction.encode("utf-8"))
        system_ref = artifacts.put_bytes(_SYSTEM_PROMPT.encode("utf-8"))
        ledger.append(EvidenceKind.REQUEST, {
            **request, "instruction_artifact_sha256": instruction_ref,
        })
        ledger.append(EvidenceKind.ARTIFACT, {
            "role": "actor-instruction", "sha256": instruction_ref,
        })
        ledger.append(EvidenceKind.ARTIFACT, {
            "role": "actor-system-prompt", "sha256": system_ref,
        })
        executor = TrialActionExecutor(
            workspace=workspace,
            language=language,
            workload=workload,
            backend=backend,
            ledger=ledger,
            knowledge=knowledge,
            profiling=profiling,
            profiling_guidance=profiling_guidance,
            unmetered_compile_probes=unmetered_compile_probes,
            max_seconds=max_seconds,
            artifacts=artifacts,
        )
        outcome = actor(
            actor_instruction,
            profile=profile,
            workspace=workspace,
            context=context,
            turn_parser=parse_trial_action,
            action_executor=executor,
        )
        if executor.final_run is None or executor.final_profile is None:
            raise RuntimeError("Actor ended without a host-verified submit action")
        final_source = (workspace / _SOURCE_PATHS[language]).read_bytes()
        source_ref = artifacts.put_bytes(final_source)
        if source_ref != executor.final_run.candidate_source_sha256:
            raise RuntimeError("retained final source does not match verified candidate")
        ledger.append(EvidenceKind.ARTIFACT, {
            "role": "final-source", "sha256": source_ref,
        })
        profile_ref = artifacts.put_json(executor.final_profile)
        ledger.append(EvidenceKind.ARTIFACT, {
            "role": "final-profile", "sha256": profile_ref,
        })
        admitted = dict(outcome.admitted_artifacts)
        admitted[source_ref] = "source"
        admitted[profile_ref] = "profiling"
        outcome = replace(outcome, admitted_artifacts=admitted)
        return TrialResult(
            outcome,
            executor.final_run,
            executor.final_profile,
            workspace,
            memory,
            context,
            ledger.head_sha256,
            artifacts.root,
        )
