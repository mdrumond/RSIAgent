"""Host-owned, append-only learning memory for the bounded A5 Phase 1 pilot."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .knowledge_agent import Citation
from .phase1_registry import (
    CHECKPOINTS,
    CurriculumProposal,
    dry_run_plan,
    saturation_status,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GENESIS = "0" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class HostFact:
    """A statement backed by retained evidence produced by the host harness."""

    category: str
    statement: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.category not in {"compile", "runtime", "host-verification", "profiling"}:
            raise ValueError("host fact category is not recognized")
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise ValueError("host fact statement must be non-empty")
        if not isinstance(self.evidence_sha256, str) or not _SHA256.fullmatch(
            self.evidence_sha256
        ):
            raise ValueError("host fact evidence_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True)
class AgentInterpretation:
    """Agent-authored inference, explicitly distinct from measured host facts."""

    statement: str
    supports: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise ValueError("agent interpretation must be non-empty")
        if any(not isinstance(item, str) or not _SHA256.fullmatch(item) for item in self.supports):
            raise ValueError("interpretation supports must be evidence SHA-256 values")


@dataclass(frozen=True)
class Phase1ProjectMemory:
    """One completed project's durable, evidence-labelled learning payload."""

    proposal: CurriculumProposal
    source_revision: str
    actions: tuple[str, ...]
    host_facts: tuple[HostFact, ...]
    interpretations: tuple[AgentInterpretation, ...] = ()
    kdb_citations: tuple[Citation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, CurriculumProposal):
            raise TypeError("proposal must be a CurriculumProposal")
        if not isinstance(self.source_revision, str) or not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        if not self.host_facts:
            raise ValueError("a project memory requires at least one host fact")
        if any(not isinstance(item, str) or not item.strip() for item in self.actions):
            raise ValueError("actions must be non-empty strings")
        if not all(isinstance(item, HostFact) for item in self.host_facts):
            raise TypeError("host_facts must contain HostFact values")
        if not all(isinstance(item, AgentInterpretation) for item in self.interpretations):
            raise TypeError("interpretations must contain AgentInterpretation values")
        if not all(isinstance(item, Citation) for item in self.kdb_citations):
            raise TypeError("kdb_citations must contain Citation values")
        for citation in self.kdb_citations:
            if (
                not citation.collection
                or not citation.path
                or type(citation.start_line) is not int
                or type(citation.end_line) is not int
                or citation.start_line < 1
                or citation.end_line < citation.start_line
                or not _SHA256.fullmatch(citation.chunk_hash)
            ):
                raise ValueError("KDB citations must retain an exact validated location")
        evidence = {item.evidence_sha256 for item in self.host_facts}
        if any(set(item.supports) - evidence for item in self.interpretations):
            raise ValueError("interpretations may cite only this project's host evidence")

    def as_dict(self) -> dict[str, object]:
        return {
            "proposal": self.proposal.as_dict(),
            "source_revision": self.source_revision,
            "actions": list(self.actions),
            "host_facts": [asdict(item) for item in self.host_facts],
            "agent_interpretations": [asdict(item) for item in self.interpretations],
            "kdb_citations": [asdict(item) for item in self.kdb_citations],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Phase1ProjectMemory":
        required = {
            "proposal", "source_revision", "actions", "host_facts",
            "agent_interpretations", "kdb_citations",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("invalid Phase 1 project memory fields")
        try:
            actions = value["actions"]
            facts = value["host_facts"]
            interpretations = value["agent_interpretations"]
            citations = value["kdb_citations"]
            if not all(isinstance(items, list) for items in (
                actions, facts, interpretations, citations
            )):
                raise TypeError("project memory collections must be JSON arrays")
            return cls(
                proposal=CurriculumProposal.from_mapping(value["proposal"]),
                source_revision=value["source_revision"],
                actions=tuple(actions),
                host_facts=tuple(HostFact(**item) for item in facts),
                interpretations=tuple(AgentInterpretation(
                    statement=item["statement"], supports=tuple(item["supports"])
                ) for item in interpretations),
                kdb_citations=tuple(Citation(**item) for item in citations),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Phase 1 project memory payload") from exc


@dataclass(frozen=True)
class Phase1ResumeState:
    completed_projects: int
    next_ordinal: int | None
    latest_checkpoint: int
    checkpoint_status: str
    plan_fingerprint: str
    head_sha256: str


class Phase1LearningJournal:
    """Append and validate host-attested project memories against one fixed plan."""

    def __init__(self, path: Path, proposals: Sequence[CurriculumProposal]):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        plan = dry_run_plan(proposals)
        self.proposals = tuple(
            CurriculumProposal.from_mapping({
                key: item[key]
                for key in ("family", "parameters", "hypothesis", "evidence_preset")
            })
            for item in plan["projects"]
        )
        self.plan_fingerprint = _digest(
            [proposal.as_dict() for proposal in self.proposals]
        )

    def append(self, memory: Phase1ProjectMemory) -> Mapping[str, object]:
        if not isinstance(memory, Phase1ProjectMemory):
            raise TypeError("memory must be a Phase1ProjectMemory")
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            entries = self._decode(stream.read())
            ordinal = len(entries) + 1
            if ordinal > len(self.proposals):
                raise ValueError("the Phase 1 plan is already complete")
            if memory.proposal != self.proposals[ordinal - 1]:
                raise ValueError("project memory does not match the next registered proposal")
            payload = {
                "schema": "a5-phase1-project-memory-v1",
                "sequence": ordinal,
                "plan_fingerprint": self.plan_fingerprint,
                "previous_sha256": entries[-1]["entry_sha256"] if entries else _GENESIS,
                "memory": memory.as_dict(),
            }
            entry = {**payload, "entry_sha256": _digest(payload)}
            stream.seek(0, os.SEEK_END)
            stream.write(_canonical(entry).decode("utf-8") + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            return entry

    def read(self) -> tuple[Mapping[str, object], ...]:
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            stream.seek(0)
            return tuple(self._decode(stream.read()))

    def resume_state(self) -> Phase1ResumeState:
        entries = self.read()
        completed = len(entries)
        latest_checkpoint = max(point for point in CHECKPOINTS if point <= completed)
        status = saturation_status(self.proposals[:completed])
        return Phase1ResumeState(
            completed_projects=completed,
            next_ordinal=completed + 1 if completed < len(self.proposals) else None,
            latest_checkpoint=latest_checkpoint,
            checkpoint_status=status,
            plan_fingerprint=self.plan_fingerprint,
            head_sha256=entries[-1]["entry_sha256"] if entries else _GENESIS,
        )

    def project_context(self) -> str:
        """Return stable context for the next Actor without relabelling inference as fact."""
        entries = self.read()
        payload = {
            "schema": "a5-phase1-context-v1",
            "plan_fingerprint": self.plan_fingerprint,
            "completed_projects": len(entries),
            "projects": [entry["memory"] for entry in entries],
        }
        return _canonical(payload).decode("utf-8")

    def _decode(self, value: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        previous = _GENESIS
        for sequence, line in enumerate(value.splitlines(), 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid Phase 1 learning journal JSON") from exc
            required = {
                "schema", "sequence", "plan_fingerprint", "previous_sha256",
                "memory", "entry_sha256",
            }
            if set(entry) != required or entry["schema"] != "a5-phase1-project-memory-v1":
                raise ValueError("invalid Phase 1 learning journal entry")
            digest = entry.pop("entry_sha256")
            valid = (
                entry["sequence"] == sequence
                and entry["plan_fingerprint"] == self.plan_fingerprint
                and entry["previous_sha256"] == previous
                and digest == _digest(entry)
            )
            entry["entry_sha256"] = digest
            if not valid:
                raise ValueError("Phase 1 learning journal failed resume validation")
            memory = Phase1ProjectMemory.from_mapping(entry["memory"])
            proposal = memory.proposal
            if sequence > len(self.proposals) or proposal != self.proposals[sequence - 1]:
                raise ValueError("Phase 1 journal does not match the configured plan")
            previous = digest
            entries.append(entry)
        return entries
