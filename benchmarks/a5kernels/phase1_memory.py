"""Host-owned, append-only learning memory for the bounded A5 Phase 1 pilot."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .knowledge_agent import Citation
from .phase1_curriculum import ProjectSpec
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
        if type(self.supports) is not tuple:
            raise TypeError("interpretation supports must be a tuple")
        if any(not isinstance(item, str) or not _SHA256.fullmatch(item) for item in self.supports):
            raise ValueError("interpretation supports must be evidence SHA-256 values")


_LESSON_CATEGORIES = {
    "api", "shape", "control-flow", "data-movement", "performance",
    "failure-repair", "correctness", "unknown",
}
_CONFIDENCE = {"low", "medium", "high"}
_MAX_LESSON_TEXT = 2_000
_MAX_LESSONS = 8
_MAX_LESSON_EVIDENCE = 8


@dataclass(frozen=True)
class AgentLesson:
    """Compact agent inference whose exact evidence remains in artifact storage."""

    category: str
    claim: str
    applicability: str
    preconditions: str
    limitations: str
    evidence_artifact_sha256: tuple[str, ...]
    failure_signature: str | None
    repair: str | None
    confidence: str

    def __post_init__(self) -> None:
        if self.category not in _LESSON_CATEGORIES:
            raise ValueError("agent lesson category is not recognized")
        for name in ("claim", "applicability", "preconditions", "limitations"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"agent lesson {name} must be non-empty")
            if len(value) > _MAX_LESSON_TEXT:
                raise ValueError(f"agent lesson {name} must be at most 2000 characters")
        if type(self.evidence_artifact_sha256) is not tuple or not self.evidence_artifact_sha256:
            raise TypeError("agent lesson evidence must be a non-empty tuple")
        if len(self.evidence_artifact_sha256) > _MAX_LESSON_EVIDENCE:
            raise ValueError("agent lesson may cite at most 8 artifacts")
        if any(not isinstance(item, str) or not _SHA256.fullmatch(item)
               for item in self.evidence_artifact_sha256):
            raise ValueError("agent lesson evidence must contain artifact SHA-256 values")
        for name in ("failure_signature", "repair"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"agent lesson {name} must be null or non-empty")
            if value is not None and len(value) > _MAX_LESSON_TEXT:
                raise ValueError(f"agent lesson {name} must be at most 2000 characters")
        if self.confidence not in _CONFIDENCE:
            raise ValueError("agent lesson confidence is not recognized")


@dataclass(frozen=True)
class AgentDistillation:
    no_new_knowledge: bool
    lessons: tuple[AgentLesson, ...]
    admitted_artifact_sha256: tuple[str, ...]


def parse_agent_distillation(
    text: str, admitted_artifacts: Mapping[str, str]
) -> AgentDistillation:
    """Parse one exact terminal-learning response against host admissions."""

    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid structured distillation response") from exc
    if not isinstance(value, Mapping) or set(value) != {"distillation"}:
        raise ValueError("invalid structured distillation response")
    body = value["distillation"]
    if not isinstance(body, Mapping) or set(body) != {"no_new_knowledge", "lessons"}:
        raise ValueError("invalid structured distillation response")
    no_new = body["no_new_knowledge"]
    raw_lessons = body["lessons"]
    if (type(no_new) is not bool or not isinstance(raw_lessons, list)
            or len(raw_lessons) > _MAX_LESSONS):
        raise ValueError("invalid structured distillation response")
    fields = {
        "category", "claim", "applicability", "preconditions", "limitations",
        "evidence_artifact_sha256", "failure_signature", "repair", "confidence",
    }
    lessons = []
    for item in raw_lessons:
        if (not isinstance(item, Mapping) or set(item) != fields
                or not isinstance(item["evidence_artifact_sha256"], list)):
            raise ValueError("invalid structured distillation response")
        try:
            lesson = AgentLesson(**{
                **item,
                "evidence_artifact_sha256": tuple(item["evidence_artifact_sha256"]),
            })
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid structured distillation response") from exc
        unsupported = set(lesson.evidence_artifact_sha256) - set(admitted_artifacts)
        if unsupported:
            raise ValueError("agent lesson cites an unadmitted artifact")
        if (lesson.category == "correctness"
                and not any(admitted_artifacts[digest] == "host-verification"
                            for digest in lesson.evidence_artifact_sha256)):
            raise ValueError("correctness requires host-verification evidence")
        if (lesson.category == "performance"
                and not any(admitted_artifacts[digest] == "performance-study"
                            for digest in lesson.evidence_artifact_sha256)):
            raise ValueError("performance requires retained performance-study evidence")
        lessons.append(lesson)
    if no_new != (not lessons):
        raise ValueError("no_new_knowledge must be true exactly when lessons are empty")
    return AgentDistillation(no_new, tuple(lessons), tuple(sorted(admitted_artifacts)))


@dataclass(frozen=True)
class Phase1ProjectMemory:
    """One completed project's durable, evidence-labelled learning payload."""

    proposal: CurriculumProposal | ProjectSpec
    source_revision: str
    actions: tuple[str, ...]
    host_facts: tuple[HostFact, ...]
    interpretations: tuple[AgentInterpretation, ...] = ()
    kdb_citations: tuple[Citation, ...] = ()
    lessons: tuple[AgentLesson, ...] = ()
    no_new_knowledge: bool = True
    admitted_artifact_sha256: tuple[str, ...] = ()
    artifact_store: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, (CurriculumProposal, ProjectSpec)):
            raise TypeError("project identity must be a CurriculumProposal or ProjectSpec")
        if not isinstance(self.source_revision, str) or not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        if type(self.host_facts) is not tuple:
            raise TypeError("host_facts must be a tuple")
        if not self.host_facts:
            raise ValueError("a project memory requires at least one host fact")
        if type(self.actions) is not tuple:
            raise TypeError("actions must be a tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.actions):
            raise ValueError("actions must be non-empty strings")
        if not all(isinstance(item, HostFact) for item in self.host_facts):
            raise TypeError("host_facts must contain HostFact values")
        if type(self.interpretations) is not tuple:
            raise TypeError("interpretations must be a tuple")
        if not all(isinstance(item, AgentInterpretation) for item in self.interpretations):
            raise TypeError("interpretations must contain AgentInterpretation values")
        if type(self.lessons) is not tuple or not all(
            isinstance(item, AgentLesson) for item in self.lessons
        ):
            raise TypeError("lessons must contain AgentLesson values in a tuple")
        if type(self.no_new_knowledge) is not bool or self.no_new_knowledge != (not self.lessons):
            raise ValueError("no_new_knowledge must be explicit and match agent lessons")
        if type(self.admitted_artifact_sha256) is not tuple or any(
            not isinstance(item, str) or not _SHA256.fullmatch(item)
            for item in self.admitted_artifact_sha256
        ):
            raise TypeError("admitted artifacts must be a tuple of SHA-256 values")
        if self.artifact_store is not None:
            if not isinstance(self.artifact_store, str) or not self.artifact_store:
                raise ValueError("artifact_store must be a normalized relative artifact store")
            store = PurePosixPath(self.artifact_store)
            if (
                store.is_absolute()
                or ".." in store.parts
                or store.as_posix() != self.artifact_store
            ):
                raise ValueError("artifact_store must be a normalized relative artifact store")
        admitted = set(self.admitted_artifact_sha256)
        if any(set(item.evidence_artifact_sha256) - admitted for item in self.lessons):
            raise ValueError("agent lessons may cite only admitted project artifacts")
        if type(self.kdb_citations) is not tuple:
            raise TypeError("kdb_citations must be a tuple")
        if not all(isinstance(item, Citation) for item in self.kdb_citations):
            raise TypeError("kdb_citations must contain Citation values")
        for citation in self.kdb_citations:
            if (
                not isinstance(citation.collection, str)
                or not citation.collection
                or not isinstance(citation.path, str)
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

    @property
    def project_identity(self) -> CurriculumProposal | ProjectSpec:
        """Return the exact legacy proposal or adaptive executable project."""

        return self.proposal

    @property
    def legacy_proposal(self) -> CurriculumProposal | None:
        """Expose the old identity without manufacturing one for adaptive projects."""

        if isinstance(self.proposal, CurriculumProposal):
            return self.proposal
        return None

    @property
    def project_spec(self) -> ProjectSpec | None:
        """Expose the executable identity when this is an adaptive project."""

        if isinstance(self.proposal, ProjectSpec):
            return self.proposal
        return None

    def as_dict(self) -> dict[str, object]:
        identity = (
            {"proposal": self.proposal.as_dict()}
            if isinstance(self.proposal, CurriculumProposal)
            else {"project_spec": self.proposal.as_dict()}
        )
        return {
            **identity,
            "source_revision": self.source_revision,
            "actions": list(self.actions),
            "host_facts": [asdict(item) for item in self.host_facts],
            "agent_interpretations": [{
                **asdict(item), "supports": list(item.supports),
            } for item in self.interpretations],
            "kdb_citations": [asdict(item) for item in self.kdb_citations],
            "agent_lessons": [{
                **asdict(item),
                "evidence_artifact_sha256": list(item.evidence_artifact_sha256),
            } for item in self.lessons],
            "no_new_knowledge": self.no_new_knowledge,
            "admitted_artifact_sha256": list(self.admitted_artifact_sha256),
            "artifact_store": self.artifact_store,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Phase1ProjectMemory":
        common = {
            "source_revision", "actions", "host_facts",
            "agent_interpretations", "kdb_citations",
        }
        legacy_layered = {
            "agent_lessons", "no_new_knowledge", "admitted_artifact_sha256",
        }
        layered = legacy_layered | {"artifact_store"}
        identities = ({"proposal"}, {"project_spec"})
        valid_fields = {
            frozenset(common | identity | optional)
            for identity in identities
            for optional in (set(), legacy_layered, layered)
        }
        if not isinstance(value, Mapping) or frozenset(value) not in valid_fields:
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
            if any(
                not isinstance(item, Mapping)
                or set(item) != {"statement", "supports"}
                or not isinstance(item["supports"], list)
                for item in interpretations
            ):
                raise TypeError("interpretation supports must be a JSON array")
            raw_lessons = value.get("agent_lessons", [])
            admitted = value.get("admitted_artifact_sha256", [])
            if not isinstance(raw_lessons, list) or not isinstance(admitted, list):
                raise TypeError("layered project memory collections must be JSON arrays")
            lessons = tuple(AgentLesson(**{
                **item, "evidence_artifact_sha256": tuple(item["evidence_artifact_sha256"]),
            }) for item in raw_lessons)
            project = (
                CurriculumProposal.from_mapping(value["proposal"])
                if "proposal" in value
                else ProjectSpec.from_mapping(value["project_spec"])
            )
            return cls(
                proposal=project,
                source_revision=value["source_revision"],
                actions=tuple(actions),
                host_facts=tuple(HostFact(**item) for item in facts),
                interpretations=tuple(
                    AgentInterpretation(
                        statement=item["statement"],
                        supports=tuple(item["supports"]),
                    )
                    for item in interpretations
                ),
                kdb_citations=tuple(Citation(**item) for item in citations),
                lessons=lessons,
                no_new_knowledge=value.get("no_new_knowledge", True),
                admitted_artifact_sha256=tuple(admitted),
                artifact_store=value.get("artifact_store"),
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
        self.plan_fingerprint = _digest(plan)

    def append(self, memory: Phase1ProjectMemory) -> Mapping[str, object]:
        if not isinstance(memory, Phase1ProjectMemory):
            raise TypeError("memory must be a Phase1ProjectMemory")
        with self.path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            value = stream.read()
            committed, torn_tail = self._committed_content(value)
            entries = self._decode(committed)
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
            if torn_tail:
                # A final unterminated JSON fragment was never committed. Removing
                # only those bytes preserves append-only semantics for every record.
                os.ftruncate(stream.fileno(), len(committed))
            elif committed and not committed.endswith(b"\n"):
                # A complete fsynced record may have lost only its line delimiter.
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
            stream.seek(0, os.SEEK_END)
            stream.write(_canonical(entry) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            return entry

    def read(self) -> tuple[Mapping[str, object], ...]:
        with self.path.open("a+b") as stream:
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

    def _decode(self, value: bytes) -> list[dict[str, Any]]:
        value, _torn_tail = self._committed_content(value)
        entries: list[dict[str, Any]] = []
        previous = _GENESIS
        for sequence, raw_line in enumerate(value.splitlines(), 1):
            try:
                line = raw_line.decode("utf-8")
                entry = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid Phase 1 learning journal JSON") from exc
            required = {
                "schema", "sequence", "plan_fingerprint", "previous_sha256",
                "memory", "entry_sha256",
            }
            if set(entry) != required or entry["schema"] != "a5-phase1-project-memory-v1":
                raise ValueError("invalid Phase 1 learning journal entry")
            digest = entry.pop("entry_sha256")
            valid = (
                type(entry["sequence"]) is int
                and entry["sequence"] == sequence
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

    @staticmethod
    def _committed_content(value: bytes) -> tuple[bytes, bool]:
        """Separate an interrupted final write from newline-framed records.

        Only an unterminated final fragment that is not complete JSON is recoverable.
        Invalid JSON followed by a newline is a committed corrupt record and remains
        a hard error in ``_decode``.
        """
        if not value or value.endswith(b"\n"):
            return value, False
        offset = value.rfind(b"\n") + 1
        tail = value[offset:]
        try:
            json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return value[:offset], True
        return value, False
