"""Bounded Actor bridge for adaptive, host-verified Phase-1 projects."""

from __future__ import annotations

import base64
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping, Protocol

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_adaptive import (
    AdaptiveProjectRequest,
    AdaptiveProjectResult,
)
from benchmarks.a5kernels.phase1_memory import (
    AgentDistillation,
    HostFact,
    Phase1ProjectMemory,
)
from benchmarks.a5kernels.phase1_project_executor import (
    PreparedProject,
    ProjectExecutionResult,
    ProjectExecutor,
)
from benchmarks.a5kernels.trial import (
    ActionOutcome,
    ActorOutcome,
    TrialAction,
    parse_trial_action,
)


class AdaptiveActorDriver(Protocol):
    """The action and terminal-distillation subset used by this adapter."""

    def __call__(
        self, instruction: str, *, profile: object, workspace: Path,
        context: Path, turn_parser: object, action_executor: object,
    ) -> ActorOutcome: ...

    def distill(
        self, outcome: ActorOutcome, *,
        admitted_artifacts: Mapping[str, str] | None = None,
        host_summaries: tuple[Mapping[str, object], ...] = (),
    ) -> ActorOutcome: ...


def _same_prepared(left: PreparedProject, right: PreparedProject) -> bool:
    return (
        left.spec == right.spec
        and left.inputs == right.inputs
        and left.expected == right.expected
        and left.family == right.family
        and left.abi == right.abi
        and left.instruction == right.instruction
    )


class _ProjectActions:
    def __init__(
        self,
        executor: ProjectExecutor,
        request: AdaptiveProjectRequest,
        prepared: PreparedProject,
        artifacts: ContentAddressedArtifacts,
        *,
        max_actions: int,
        max_source_bytes: int,
    ) -> None:
        self.executor = executor
        self.request = request
        self.prepared = prepared
        self.artifacts = artifacts
        self.max_actions = max_actions
        self.max_source_bytes = max_source_bytes
        self.actions: list[str] = []
        self.source: str | None = None
        self.source_sha256: str | None = None
        self.latest: ProjectExecutionResult | None = None
        self.latest_source_sha256: str | None = None
        self.admitted: dict[str, str] = {}

    def __call__(self, action: TrialAction) -> ActionOutcome:
        if not isinstance(action, TrialAction):
            raise TypeError("adaptive projects accept only TrialAction values")
        self.actions.append(action.kind)
        if len(self.actions) > self.max_actions:
            return self._observation({"error": "action budget exhausted"}, True, "budget")
        if action.kind == "propose":
            return self._observation({"event": "proposal-recorded"})
        if action.kind == "write":
            return self._write(action)
        if action.kind in {"compile", "run"}:
            return self._evaluate(action.kind)
        if action.kind == "submit":
            return self._submit()
        return self._observation({
            "error": f"{action.kind} is not bound for this project",
            "available": ["propose", "write", "compile", "run", "submit"],
        })

    def _write(self, action: TrialAction) -> ActionOutcome:
        source = action.payload["content"]
        if not isinstance(source, str) or not source.strip():
            return self._observation({"error": "candidate source must be non-empty"})
        encoded = source.encode("utf-8")
        if len(encoded) > self.max_source_bytes:
            return self._observation({"error": "candidate source exceeds byte limit"})
        self.source = source
        self.source_sha256 = self.artifacts.put_bytes(encoded)
        self.admitted[self.source_sha256] = "candidate-source"
        self.latest = None
        self.latest_source_sha256 = None
        return self._observation({
            "event": "candidate-written", "source_sha256": self.source_sha256,
        })

    def _evaluate(self, action: str) -> ActionOutcome:
        if self.source is None:
            return self._observation({"error": "write candidate source first"})
        result = self.executor.execute(self.request.spec, self.source)
        if not _same_prepared(result.prepared, self.prepared):
            raise ValueError("execution rematerialized different project inputs or oracle")
        self.latest = result
        self.latest_source_sha256 = self.source_sha256
        retained = {}
        for name, payload in result.artifacts.items():
            digest = self.artifacts.put_json(payload)
            if digest != result.artifact_sha256[name]:
                raise ValueError("execution artifact hash does not match its payload")
            role = (
                "host-verification"
                if name == "classification" and result.evidence.verified
                else f"execution-{name}"
            )
            self.admitted[digest] = role
            retained[name] = {"sha256": digest, "payload": payload}
        return self._observation({
            "event": "host-execution", "requested_action": action,
            "source_sha256": self.source_sha256,
            "classification": result.evidence.classification.value,
            "detail": result.evidence.detail,
            "artifacts": retained,
        })

    def _submit(self) -> ActionOutcome:
        if (
            self.latest is None
            or self.latest_source_sha256 != self.source_sha256
            or not self.latest.evidence.verified
        ):
            return self._observation({
                "error": "submit requires HOST_VERIFIED evidence for current source"
            })
        return self._observation({
            "event": "host-verified-submit",
            "classification": self.latest.evidence.classification.value,
            "source_sha256": self.source_sha256,
            "classification_sha256": self.latest.artifact_sha256["classification"],
        }, True, "host-verified")

    @staticmethod
    def _observation(
        payload: Mapping[str, object], terminal: bool = False, status: str = "continue",
    ) -> ActionOutcome:
        return ActionOutcome(
            canonical_bytes({"host": payload}).decode("ascii"), terminal, status,
        )


class AdaptiveActorProjectExecutor:
    """Drive bounded candidate repair, then emit only distilled verified memory."""

    def __init__(
        self, executor: ProjectExecutor, actor: AdaptiveActorDriver, profile: object,
        *, max_actions: int = 40, max_source_bytes: int = 128_000,
    ) -> None:
        if not callable(actor):
            raise TypeError("actor must be callable")
        if type(max_actions) is not int or not 1 <= max_actions <= 40:
            raise ValueError("max_actions must be in [1, 40]")
        if type(max_source_bytes) is not int or max_source_bytes < 1:
            raise ValueError("max_source_bytes must be positive")
        self.executor = executor
        self.actor = actor
        self.profile = profile
        self.max_actions = max_actions
        self.max_source_bytes = max_source_bytes

    def __call__(self, request: AdaptiveProjectRequest) -> AdaptiveProjectResult:
        if not isinstance(request, AdaptiveProjectRequest):
            raise TypeError("request must be AdaptiveProjectRequest")
        prepared = self.executor.prepare(request.spec)
        if (
            prepared.spec != request.spec
            or prepared.inputs != request.inputs
            or prepared.expected != request.oracle
        ):
            raise ValueError("request spec, inputs, or oracle differ from rematerialized project")
        root = Path(request.branch_root)
        artifacts = ContentAddressedArtifacts(root / "artifacts")
        memory_sha256 = artifacts.put_bytes(request.pre_wave_memory)
        actions = _ProjectActions(
            self.executor, request, prepared, artifacts,
            max_actions=self.max_actions, max_source_bytes=self.max_source_bytes,
        )
        opening = prepared.instruction + "\nPre-wave memory (immutable): " + json.dumps({
            "sha256": memory_sha256,
            "base64": base64.b64encode(request.pre_wave_memory).decode("ascii"),
        }, sort_keys=True, separators=(",", ":"))
        outcome = self.actor(
            opening, profile=self.profile, workspace=root / "actor",
            context=artifacts.objects / memory_sha256[:2] / memory_sha256[2:],
            turn_parser=parse_trial_action, action_executor=actions,
        )
        result = actions.latest
        if (
            not isinstance(outcome, ActorOutcome)
            or outcome.status != "host-verified"
            or result is None
            or not result.evidence.verified
            or actions.latest_source_sha256 != actions.source_sha256
        ):
            raise RuntimeError("Actor ended without host-verified current source")
        facts = (HostFact(
            "host-verification", result.evidence.detail,
            result.artifact_sha256["classification"],
        ),)
        learned = self.actor.distill(
            outcome, admitted_artifacts=actions.admitted,
            host_summaries=tuple(asdict(fact) for fact in facts),
        )
        if not isinstance(learned.distillation, AgentDistillation):
            raise RuntimeError("terminal project omitted structured distillation")
        if learned.terminal_response is None:
            raise RuntimeError("terminal project omitted exact distillation response")
        distillation = learned.distillation
        artifacts.put_bytes(learned.terminal_response.encode("utf-8"))
        assert actions.source_sha256 is not None
        memory = Phase1ProjectMemory(
            request.spec, actions.source_sha256, tuple(actions.actions), facts,
            lessons=distillation.lessons,
            no_new_knowledge=distillation.no_new_knowledge,
            admitted_artifact_sha256=distillation.admitted_artifact_sha256,
            artifact_store="artifacts",
        )
        evidence_payload = canonical_bytes({
            "classification": result.evidence.classification.value,
            "detail": result.evidence.detail,
            "source_sha256": actions.source_sha256,
            "artifacts": dict(result.artifact_sha256),
        })
        return AdaptiveProjectResult(
            memory, result.evidence, evidence_payload,
            f"{request.spec.project_id} passed host verification",
        )


__all__ = ["AdaptiveActorDriver", "AdaptiveActorProjectExecutor"]
