"""Durable, barriered execution for one agent-authored A5 Phase-1 wave."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_curriculum import (
    ConsolidationCommit,
    ProjectAssignment,
    ProjectOutcome,
    WaveConsolidation,
    WaveSpec,
    consolidate_wave,
)
from benchmarks.a5kernels.phase1_project import (
    ExecutionEvidence,
    HostTensor,
    ReferenceOutput,
    evaluate_reference,
    materialize_inputs,
)


PLAN_SCHEMA = "a5-phase1-live-wave-plan-v1"
OUTCOME_SCHEMA = "a5-phase1-live-wave-outcome-v1"
PUBLICATION_SCHEMA = "a5-phase1-live-wave-publication-v1"


class SemanticProjectFailure(RuntimeError):
    """A normally completed Actor attempt without host-verified current source.

    Provider, context-budget, transport and host infrastructure exceptions must
    propagate instead; this is not a catch-all exception conversion.
    """

    def __init__(self, summary: str, evidence: dict):
        super().__init__(summary)
        self.summary, self.evidence = summary, evidence


@dataclass(frozen=True)
class WaveProjectRequest:
    """One isolated sibling invocation from an immutable memory snapshot."""

    assignment: ProjectAssignment
    pre_wave_memory: bytes
    branch_root: Path
    inputs: tuple[HostTensor, ...]
    oracle: ReferenceOutput


@dataclass(frozen=True)
class WaveProjectResult:
    """Host-verifiable output returned by a project executor."""

    memory_delta: bytes
    evidence: ExecutionEvidence
    evidence_payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.memory_delta, bytes):
            raise TypeError("memory_delta must be immutable bytes")
        if not isinstance(self.evidence, ExecutionEvidence):
            raise TypeError("evidence must be a host execution classification")
        if not isinstance(self.evidence_payload, bytes) or not self.evidence_payload:
            raise TypeError("evidence_payload must be non-empty immutable bytes")

    @property
    def verified(self) -> bool:
        return self.evidence.verified


WaveProjectExecutor = Callable[[WaveProjectRequest], WaveProjectResult]


class LiveWaveCoordinator:
    """Wait for all siblings; optionally skip completed semantic failures.

    Only verified deltas become commits. The default retains the original
    strict all-verified barrier for standalone callers.
    """

    def __init__(
        self,
        root: Path,
        wave: WaveSpec,
        pre_wave_memory: bytes,
        execute: WaveProjectExecutor,
        max_parallel: int,
        *, allow_semantic_failures: bool = False,
    ) -> None:
        if not isinstance(wave, WaveSpec):
            raise TypeError("wave must be a WaveSpec")
        if not isinstance(pre_wave_memory, bytes):
            raise TypeError("pre-wave memory must be immutable bytes")
        if hashlib.sha256(pre_wave_memory).hexdigest() != wave.pre_wave_memory_sha256:
            raise ValueError("pre-wave memory identity does not match WaveSpec")
        if not callable(execute):
            raise TypeError("project executor must be callable")
        if type(max_parallel) is not int or max_parallel < 1:
            raise ValueError("max_parallel must be a positive integer")
        self.root = Path(root)
        self.wave = wave
        self.pre_wave_memory = pre_wave_memory
        self.execute = execute
        self.max_parallel = max_parallel
        self.allow_semantic_failures = allow_semantic_failures

    def run(self) -> WaveConsolidation:
        self._ensure_plan()
        published = self._load_publication()
        if published is not None:
            return published

        assignments = self.wave.assignments()
        outcomes: dict[str, ProjectOutcome] = {}
        pending = []
        for assignment in assignments:
            outcome = self._load_outcome(assignment)
            if outcome is not None and (outcome.verified or self.allow_semantic_failures):
                outcomes[assignment.project_id] = outcome
            else:
                pending.append(assignment)

        failures: list[tuple[str, Exception]] = []
        with ThreadPoolExecutor(
            max_workers=self.max_parallel,
            thread_name_prefix=f"a5-{self.wave.wave_id}",
        ) as pool:
            futures = {
                pool.submit(self._execute_one, assignment): assignment
                for assignment in pending
            }
            for future in as_completed(futures):
                assignment = futures[future]
                try:
                    outcomes[assignment.project_id] = future.result()
                except Exception as exc:  # wait for siblings before surfacing it
                    failures.append((assignment.project_id, exc))

        if failures:
            project_id, failure = failures[0]
            raise RuntimeError(
                f"wave project {project_id!r} failed before verification"
            ) from failure
        try:
            consolidation = consolidate_wave(self.wave, tuple(outcomes.values()),
                                             allow_semantic_failures=self.allow_semantic_failures)
        except ValueError as exc:
            raise RuntimeError(f"wave is not verified: {exc}") from exc
        self._publish(consolidation)
        return consolidation

    def resume(self) -> WaveConsolidation:
        """Reuse retained terminal branches and execute only unfinished siblings."""

        return self.run()

    def _execute_one(self, assignment: ProjectAssignment) -> ProjectOutcome:
        branch_root = self.root / "branches" / assignment.project_id
        branch_root.mkdir(parents=True, exist_ok=True)
        inputs = materialize_inputs(assignment.spec)
        try:
            result = self.execute(WaveProjectRequest(
                assignment=assignment,
                pre_wave_memory=self.pre_wave_memory,
                branch_root=branch_root,
                inputs=inputs,
                oracle=evaluate_reference(assignment.spec, inputs),
            ))
        except SemanticProjectFailure as exc:
            if not self.allow_semantic_failures:
                raise
            # No kernel memory is manufactured for an unverified attempt.
            from benchmarks.a5kernels.phase1_project import EvidenceClassification
            payload = canonical_bytes({"classification": "actor-unverified",
                                       "summary": exc.summary, **exc.evidence})
            result = WaveProjectResult(payload, ExecutionEvidence(
                EvidenceClassification.ACTOR_UNVERIFIED, exc.summary), payload)
        if not isinstance(result, WaveProjectResult):
            raise TypeError("project executor must return WaveProjectResult")
        delta_sha256 = hashlib.sha256(result.memory_delta).hexdigest()
        evidence_sha256 = hashlib.sha256(result.evidence_payload).hexdigest()
        # Invalidate the previous generation first.  A crash after this point is
        # an unfinished retry, not a mismatched old outcome/new delta pair.
        (branch_root / "outcome.json").unlink(missing_ok=True)
        self._atomic_bytes(branch_root / "memory-delta.bin", result.memory_delta)
        self._atomic_bytes(branch_root / "execution-evidence.bin", result.evidence_payload)
        body = {
            "schema": OUTCOME_SCHEMA,
            "wave_fingerprint": self.wave.fingerprint,
            "project_id": assignment.project_id,
            "authored_ordinal": assignment.authored_ordinal,
            "pre_wave_memory_sha256": assignment.pre_wave_memory_sha256,
            "memory_delta_sha256": delta_sha256,
            "evidence_sha256": evidence_sha256,
            "classification": result.evidence.classification.value,
            "verified": result.verified,
        }
        self._atomic_json(branch_root / "outcome.json", body)
        return ProjectOutcome(
            assignment.project_id,
            delta_sha256,
            assignment.wave_fingerprint,
            assignment.authored_ordinal,
            result.verified,
        )

    def _load_outcome(self, assignment: ProjectAssignment) -> ProjectOutcome | None:
        branch_root = self.root / "branches" / assignment.project_id
        path = branch_root / "outcome.json"
        if not path.exists():
            return None
        value = self._read_json(path)
        expected = {
            "schema", "wave_fingerprint", "project_id", "authored_ordinal",
            "pre_wave_memory_sha256", "memory_delta_sha256", "evidence_sha256",
            "classification", "verified",
        }
        if set(value) != expected or any((
            value["schema"] != OUTCOME_SCHEMA,
            value["wave_fingerprint"] != self.wave.fingerprint,
            value["project_id"] != assignment.project_id,
            value["authored_ordinal"] != assignment.authored_ordinal,
            value["pre_wave_memory_sha256"] != self.wave.pre_wave_memory_sha256,
        )):
            raise ValueError("retained branch outcome does not match wave plan")
        delta_path = branch_root / "memory-delta.bin"
        try:
            delta_sha256 = hashlib.sha256(delta_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError("retained branch memory delta is unavailable") from exc
        if delta_sha256 != value["memory_delta_sha256"]:
            raise ValueError("retained branch memory delta does not match outcome")
        evidence_path = branch_root / "execution-evidence.bin"
        try:
            evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError("retained execution evidence is unavailable") from exc
        if evidence_sha256 != value["evidence_sha256"]:
            raise ValueError("retained execution evidence does not match outcome")
        if value["verified"] != (value["classification"] == "host-verified"):
            raise ValueError("retained verification is not derived from classification")
        return ProjectOutcome(
            assignment.project_id,
            value["memory_delta_sha256"],
            assignment.wave_fingerprint,
            assignment.authored_ordinal,
            value["verified"],
        )

    def _ensure_plan(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        expected = {
            "schema": PLAN_SCHEMA,
            "wave": self.wave.as_dict(),
            "wave_fingerprint": self.wave.fingerprint,
        }
        path = self.root / "plan.json"
        if path.exists():
            if self._read_json(path) != expected:
                raise ValueError("existing wave plan does not match requested plan")
            return
        self._atomic_json(path, expected)

    def _publish(self, consolidation: WaveConsolidation) -> None:
        memory = canonical_bytes({
            "schema": "a5-phase1-consolidated-memory-v1",
            "pre_wave_memory_sha256": self.wave.pre_wave_memory_sha256,
            "deltas": [
                {
                    "project_id": commit.project_id,
                    "memory_delta_hex": (
                        self.root / "branches" / commit.project_id / "memory-delta.bin"
                    ).read_bytes().hex(),
                }
                for commit in consolidation.commits
            ],
        }) + b"\n"
        memory_sha256 = hashlib.sha256(memory).hexdigest()
        self._atomic_bytes(self.root / "consolidated-memory.json", memory)
        value = {
            "schema": PUBLICATION_SCHEMA,
            "wave_id": consolidation.wave_id,
            "wave_fingerprint": self.wave.fingerprint,
            "pre_wave_memory_sha256": consolidation.pre_wave_memory_sha256,
            "consolidated_memory_sha256": memory_sha256,
            "commits": [
                {
                    "project_id": item.project_id,
                    "authored_ordinal": item.authored_ordinal,
                    "commit_ordinal": item.commit_ordinal,
                    "memory_delta_sha256": item.memory_delta_sha256,
                }
                for item in consolidation.commits
            ],
        }
        skipped = [assignment.project_id for assignment in self.wave.assignments()
                   if not self._load_outcome(assignment).verified]
        if skipped:
            value["skipped_projects"] = skipped
        self._atomic_json(self.root / "published.json", value)

    def _load_publication(self) -> WaveConsolidation | None:
        path = self.root / "published.json"
        if not path.exists():
            return None
        value = self._read_json(path)
        if (
            set(value) - {"skipped_projects"} != {
                "schema", "wave_id", "wave_fingerprint",
                "pre_wave_memory_sha256", "commits",
                "consolidated_memory_sha256",
            }
            or value["schema"] != PUBLICATION_SCHEMA
            or value["wave_id"] != self.wave.wave_id
            or value["wave_fingerprint"] != self.wave.fingerprint
            or value["pre_wave_memory_sha256"] != self.wave.pre_wave_memory_sha256
            or not isinstance(value["commits"], list)
        ):
            raise ValueError("published wave does not match requested plan")
        commits = tuple(ConsolidationCommit(**item) for item in value["commits"])
        expected = consolidate_wave(
            self.wave,
            tuple(
                outcome
                for assignment in self.wave.assignments()
                if (outcome := self._load_outcome(assignment)) is not None
            ),
            allow_semantic_failures=self.allow_semantic_failures,
        )
        skipped = [assignment.project_id for assignment in self.wave.assignments()
                   if not self._load_outcome(assignment).verified]
        if value.get("skipped_projects", []) != skipped:
            raise ValueError("published skipped projects differ from branch outcomes")
        loaded = WaveConsolidation(
            value["wave_id"], value["pre_wave_memory_sha256"], commits,
        )
        if loaded != expected:
            raise ValueError("published wave does not match retained branch outcomes")
        try:
            memory = (self.root / "consolidated-memory.json").read_bytes()
        except OSError as exc:
            raise ValueError("published consolidated memory is unavailable") from exc
        if hashlib.sha256(memory).hexdigest() != value["consolidated_memory_sha256"]:
            raise ValueError("published consolidated memory does not match publication")
        return loaded

    def read_published_memory(self) -> bytes:
        """Return the exact next-wave memory snapshot after verified publication."""

        if self._load_publication() is None:
            raise ValueError("wave has not been published")
        return (self.root / "consolidated-memory.json").read_bytes()

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to read retained wave state: {path.name}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"retained wave state is not an object: {path.name}")
        return value

    @staticmethod
    def _atomic_bytes(path: Path, value: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(value)
        os.replace(temporary, path)

    @classmethod
    def _atomic_json(cls, path: Path, value: dict[str, object]) -> None:
        cls._atomic_bytes(path, canonical_bytes(value) + b"\n")


__all__ = [
    "LiveWaveCoordinator", "WaveProjectExecutor", "WaveProjectRequest",
    "WaveProjectResult", "SemanticProjectFailure",
]
