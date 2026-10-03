"""Append-only progressive memory for isolated A3 Phase 1 lineages."""

from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from benchmarks.a3kernels.phase1_evidence import canonical_bytes, canonical_digest
from benchmarks.a3kernels.phase1_registry import (
    CHECKPOINTS,
    CurriculumProposal,
    dry_run_plan,
    saturation_status,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_CELL_ID = re.compile(r"a3-cell-[0-9a-f]{16}")
_LINEAGE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_GENESIS = "0" * 64
_ENTRY_SCHEMA = "a3-phase1-memory-v1"
_CONTEXT_SCHEMA = "a3-phase1-context-v1"


@dataclass(frozen=True)
class HostFact:
    category: str
    statement: str
    evidence_sha256: str
    authority: str = "host"
    success: bool = True

    def __post_init__(self) -> None:
        if self.category not in {"compile", "runtime", "host-verification", "profiling"}:
            raise ValueError("host fact category is not recognized")
        if type(self.statement) is not str or not self.statement.strip():
            raise ValueError("host fact statement must be non-empty")
        if type(self.evidence_sha256) is not str or not _SHA256.fullmatch(
            self.evidence_sha256
        ):
            raise ValueError("host fact evidence must be a lowercase SHA-256")
        if self.authority != "host":
            raise ValueError("measured facts require host authority")
        if type(self.success) is not bool:
            raise ValueError("host fact success must be a boolean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "HostFact":
        expected = {
            "category", "statement", "evidence_sha256", "authority", "success"
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("invalid host fact schema")
        return cls(**value)


@dataclass(frozen=True)
class AuthoritativeEvidence:
    """One host result addressable by its retained evidence digest."""

    kind: str
    evidence_sha256: str
    project_id: str
    candidate_sha256: str
    success: bool

    def __post_init__(self) -> None:
        if self.kind not in {"compile", "runtime", "host-verification", "profiling"}:
            raise ValueError("authoritative evidence kind is not recognized")
        for value, label in (
            (self.evidence_sha256, "evidence digest"),
            (self.project_id, "project identity"),
            (self.candidate_sha256, "candidate identity"),
        ):
            if type(value) is not str or not _SHA256.fullmatch(value):
                raise ValueError(f"authoritative {label} must be a lowercase SHA-256")
        if type(self.success) is not bool:
            raise ValueError("authoritative success must be a boolean")


class EvidenceResolver(Protocol):
    def resolve(self, evidence_sha256: str) -> AuthoritativeEvidence | None: ...


@dataclass(frozen=True)
class AgentInterpretation:
    statement: str
    supports: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.statement) is not str or not self.statement.strip():
            raise ValueError("agent interpretation must be non-empty")
        if type(self.supports) is not tuple:
            raise TypeError("interpretation supports must be a tuple")
        if any(type(item) is not str or not _SHA256.fullmatch(item) for item in self.supports):
            raise ValueError("interpretation supports must be evidence SHA-256 values")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AgentInterpretation":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"statement", "supports"}
            or not isinstance(value["supports"], list)
        ):
            raise ValueError("invalid agent interpretation schema")
        return cls(value["statement"], tuple(value["supports"]))


@dataclass(frozen=True)
class ProjectMemory:
    proposal: CurriculumProposal
    project_id: str
    cell_id: str
    lineage_id: str
    source_revision: str
    actions: tuple[str, ...]
    host_facts: tuple[HostFact, ...]
    interpretations: tuple[AgentInterpretation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, CurriculumProposal):
            raise TypeError("proposal must be a CurriculumProposal")
        if self.project_id != self.proposal.project_id:
            raise ValueError("project identity must match the registered proposal")
        if type(self.cell_id) is not str or not _CELL_ID.fullmatch(self.cell_id):
            raise ValueError("memory requires a registered A3 cell identity")
        if type(self.lineage_id) is not str or not _LINEAGE_ID.fullmatch(self.lineage_id):
            raise ValueError("memory requires a safe lineage identity")
        if type(self.source_revision) is not str or not _SHA256.fullmatch(
            self.source_revision
        ):
            raise ValueError("source revision must be a lowercase SHA-256")
        if type(self.actions) is not tuple or any(
            type(item) is not str or not item.strip() for item in self.actions
        ):
            raise TypeError("actions must be a tuple of non-empty strings")
        if type(self.host_facts) is not tuple or not self.host_facts:
            raise ValueError("memory requires a tuple with at least one host fact")
        if any(not isinstance(item, HostFact) for item in self.host_facts):
            raise TypeError("host_facts must contain HostFact values")
        if type(self.interpretations) is not tuple or any(
            not isinstance(item, AgentInterpretation) for item in self.interpretations
        ):
            raise TypeError("interpretations must be a tuple of AgentInterpretation values")
        evidence = {item.evidence_sha256 for item in self.host_facts}
        if any(set(item.supports) - evidence for item in self.interpretations):
            raise ValueError("interpretations may cite only this project's host evidence")

    def as_dict(self) -> dict[str, object]:
        return {
            "target": "Ascend910B4",
            "language": "ascend-c",
            "proposal": self.proposal.as_dict(),
            "project_id": self.project_id,
            "cell_id": self.cell_id,
            "lineage_id": self.lineage_id,
            "source_revision": self.source_revision,
            "actions": list(self.actions),
            "host_facts": [asdict(item) for item in self.host_facts],
            "agent_interpretations": [asdict(item) for item in self.interpretations],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProjectMemory":
        expected = {
            "target", "language", "proposal", "project_id", "cell_id", "lineage_id",
            "source_revision", "actions", "host_facts", "agent_interpretations",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("invalid A3 project memory schema")
        if value["target"] != "Ascend910B4" or value["language"] != "ascend-c":
            raise ValueError("project memory must target A3 Ascend C")
        try:
            actions = value["actions"]
            facts = value["host_facts"]
            interpretations = value["agent_interpretations"]
            if not all(isinstance(items, list) for items in (actions, facts, interpretations)):
                raise TypeError("memory collections must be JSON arrays")
            return cls(
                CurriculumProposal.from_mapping(value["proposal"]),
                value["project_id"],
                value["cell_id"],
                value["lineage_id"],
                value["source_revision"],
                tuple(actions),
                tuple(HostFact.from_mapping(item) for item in facts),
                tuple(AgentInterpretation.from_mapping(item) for item in interpretations),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid A3 project memory payload") from exc


@dataclass(frozen=True)
class ResumeState:
    completed_projects: int
    next_ordinal: int | None
    latest_checkpoint: int
    checkpoint_status: str
    plan_fingerprint: str
    head_sha256: str
    cell_id: str
    lineage_id: str


class Phase1LearningJournal:
    def __init__(
        self,
        path: str | Path,
        proposals: Sequence[CurriculumProposal],
        *,
        cell_id: str,
        lineage_id: str,
        evidence_resolver: EvidenceResolver,
        trial_protocol_sha256: str,
    ) -> None:
        if type(cell_id) is not str or not _CELL_ID.fullmatch(cell_id):
            raise ValueError("journal requires a registered A3 cell identity")
        if type(lineage_id) is not str or not _LINEAGE_ID.fullmatch(lineage_id):
            raise ValueError("journal requires a safe lineage identity")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cell_id = cell_id
        self.lineage_id = lineage_id
        if (
            type(trial_protocol_sha256) is not str
            or not _SHA256.fullmatch(trial_protocol_sha256)
        ):
            raise ValueError("journal requires a trial protocol identity")
        self.trial_protocol_sha256 = trial_protocol_sha256
        if not callable(getattr(evidence_resolver, "resolve", None)):
            raise TypeError("evidence_resolver must provide resolve(digest)")
        self._evidence_resolver = evidence_resolver
        plan = dry_run_plan(proposals)
        self.proposals = tuple(
            CurriculumProposal.from_mapping(
                {
                    key: row[key]
                    for key in (
                        "target", "language", "family", "parameters", "hypothesis",
                        "evidence_preset",
                    )
                }
            )
            for row in plan["projects"]
        )
        self.plan_fingerprint = canonical_digest({
            "plan": plan,
            "trial_protocol_sha256": self.trial_protocol_sha256,
        })

    def append(self, memory: ProjectMemory) -> Mapping[str, object]:
        if not isinstance(memory, ProjectMemory):
            raise TypeError("memory must be a ProjectMemory")
        if memory.cell_id != self.cell_id or memory.lineage_id != self.lineage_id:
            raise ValueError("memory cell and lineage must match the journal")
        self._validate_host_facts(memory)
        with self.path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            committed, torn = self._committed_content(stream.read())
            entries = self._decode(committed)
            ordinal = len(entries) + 1
            if ordinal > len(self.proposals):
                raise ValueError("the Phase 1 plan is already complete")
            if memory.proposal != self.proposals[ordinal - 1]:
                raise ValueError("memory does not match the next registered project")
            body = {
                "schema": _ENTRY_SCHEMA,
                "sequence": ordinal,
                "plan_fingerprint": self.plan_fingerprint,
                "cell_id": self.cell_id,
                "lineage_id": self.lineage_id,
                "previous_sha256": entries[-1]["entry_sha256"] if entries else _GENESIS,
                "memory": memory.as_dict(),
            }
            entry = {**body, "entry_sha256": canonical_digest(body)}
            if torn:
                os.ftruncate(stream.fileno(), len(committed))
            elif committed and not committed.endswith(b"\n"):
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
            stream.seek(0, os.SEEK_END)
            stream.write(canonical_bytes(entry) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            return entry

    def read(self) -> tuple[Mapping[str, object], ...]:
        with self.path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            stream.seek(0)
            return tuple(self._decode(stream.read()))

    def resume_state(self) -> ResumeState:
        entries = self.read()
        completed = len(entries)
        checkpoint = max(point for point in CHECKPOINTS if point <= completed)
        return ResumeState(
            completed,
            completed + 1 if completed < len(self.proposals) else None,
            checkpoint,
            saturation_status(self.proposals[:completed]),
            self.plan_fingerprint,
            entries[-1]["entry_sha256"] if entries else _GENESIS,
            self.cell_id,
            self.lineage_id,
        )

    def project_context(self) -> str:
        entries = self.read()
        return canonical_bytes(
            {
                "schema": _CONTEXT_SCHEMA,
                "target": "Ascend910B4",
                "language": "ascend-c",
                "plan_fingerprint": self.plan_fingerprint,
                "cell_id": self.cell_id,
                "lineage_id": self.lineage_id,
                "completed_projects": len(entries),
                "projects": [entry["memory"] for entry in entries],
            }
        ).decode("utf-8")

    def _decode(self, value: bytes) -> list[dict[str, Any]]:
        committed, _ = self._committed_content(value)
        entries = []
        previous = _GENESIS
        for sequence, raw in enumerate(committed.splitlines(), 1):
            try:
                entry = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid A3 learning journal JSON") from exc
            required = {
                "schema", "sequence", "plan_fingerprint", "cell_id", "lineage_id",
                "previous_sha256", "memory", "entry_sha256",
            }
            if set(entry) != required or entry["schema"] != _ENTRY_SCHEMA:
                raise ValueError("invalid A3 learning journal entry")
            digest = entry.pop("entry_sha256")
            valid = (
                type(entry["sequence"]) is int
                and entry["sequence"] == sequence
                and entry["plan_fingerprint"] == self.plan_fingerprint
                and entry["cell_id"] == self.cell_id
                and entry["lineage_id"] == self.lineage_id
                and entry["previous_sha256"] == previous
                and digest == canonical_digest(entry)
            )
            entry["entry_sha256"] = digest
            if not valid:
                raise ValueError("A3 learning journal failed resume validation")
            memory = ProjectMemory.from_mapping(entry["memory"])
            self._validate_host_facts(memory)
            if (
                sequence > len(self.proposals)
                or memory.proposal != self.proposals[sequence - 1]
                or memory.cell_id != self.cell_id
                or memory.lineage_id != self.lineage_id
            ):
                raise ValueError("A3 journal does not match the configured plan")
            previous = digest
            entries.append(entry)
        return entries

    def _validate_host_facts(self, memory: ProjectMemory) -> None:
        for fact in memory.host_facts:
            resolved = self._evidence_resolver.resolve(fact.evidence_sha256)
            if not isinstance(resolved, AuthoritativeEvidence):
                raise ValueError("host fact must resolve to authoritative evidence")
            checks = (
                ("kind", resolved.kind, fact.category),
                ("digest", resolved.evidence_sha256, fact.evidence_sha256),
                ("project", resolved.project_id, memory.project_id),
                ("candidate", resolved.candidate_sha256, memory.source_revision),
                ("success", resolved.success, fact.success),
            )
            for label, actual, expected in checks:
                if actual != expected:
                    raise ValueError(
                        f"host fact {label} does not match authoritative evidence"
                    )

    @staticmethod
    def _committed_content(value: bytes) -> tuple[bytes, bool]:
        if not value or value.endswith(b"\n"):
            return value, False
        offset = value.rfind(b"\n") + 1
        try:
            json.loads(value[offset:].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return value[:offset], True
        return value, False
