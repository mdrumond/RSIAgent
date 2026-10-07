"""Immutable Phase 1 memory handoff for A3 Phase 2 lineages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping

from benchmarks.a3_experiments import (
    A3ExperimentCell,
    ProgrammingLevel,
    build_a3_experiment_plan,
)
from benchmarks.a3kernels.live_composition import infrastructure_retry_summary
from benchmarks.a3kernels.phase1_evidence import (
    GENESIS_HASH,
    EvidenceLedger,
    canonical_bytes,
    canonical_digest,
)
from benchmarks.a3kernels.phase1_memory import (
    AuthoritativeEvidence,
    EvidenceResolver,
    HostFact,
    Phase1LearningJournal,
)
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.phase1_wave import Phase1Wave


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SNAPSHOT_SCHEMA = "a3-phase2-phase1-snapshot-v1"
_LEARNING_SCHEMA = "a3-phase2-learning-v1"
_LEARNING_KINDS = frozenset({
    "target-attempt", "target-verdict", "practice", "curriculum",
})
_AUTHORITATIVE_PROFILES = frozenset({"bz-a3-1", "bz-a3-2"})


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _ReadOnlyAuthorityResolver:
    """Validate a frozen Phase 1 authority ledger without recovering it."""

    def __init__(self, value: bytes) -> None:
        if value and not value.endswith(b"\n"):
            raise ValueError("Phase 1 authority ledger is unterminated")
        records: dict[str, AuthoritativeEvidence] = {}
        for number, line in enumerate(value.splitlines(), 1):
            try:
                record = AuthoritativeEvidence(**json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid Phase 1 authority record at line {number}"
                ) from exc
            if record.evidence_sha256 in records:
                raise ValueError("duplicate Phase 1 authority evidence")
            records[record.evidence_sha256] = record
        self._records = records

    def resolve(self, evidence_sha256: str) -> AuthoritativeEvidence | None:
        return self._records.get(evidence_sha256)


def phase2_cell_pairs(
) -> tuple[tuple[A3ExperimentCell, A3ExperimentCell], ...]:
    """Map each foundation cell to the tiled cell with identical treatments."""
    cells = build_a3_experiment_plan().cells
    tiled = {
        (
            cell.target, cell.language, cell.backend_model,
            cell.knowledge, cell.profiling,
        ): cell
        for cell in cells if cell.programming_level is ProgrammingLevel.TILED
    }
    return tuple(
        (cell, tiled[(
            cell.target, cell.language, cell.backend_model,
            cell.knowledge, cell.profiling,
        )])
        for cell in cells if cell.programming_level is ProgrammingLevel.FOUNDATION
    )


def _cells_by_source() -> dict[str, tuple[A3ExperimentCell, A3ExperimentCell]]:
    return {source.cell_id: (source, destination)
            for source, destination in phase2_cell_pairs()}


def _publish_exact(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != value:
            raise ValueError(f"immutable snapshot conflict: {path.name}") from None
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class Phase1Snapshot:
    root: Path
    manifest: Mapping[str, object]

    @property
    def memory_path(self) -> Path:
        return self.root / "phase1-memory.jsonl"

    @property
    def terminal_path(self) -> Path:
        return self.root / "phase1-terminal.json"

    @property
    def snapshot_id(self) -> str:
        return str(self.manifest["snapshot_id"])

    def verify(self) -> None:
        required = {
            "schema", "target", "language", "phase1_root_sha256",
            "source_cell_id", "destination_cell_id", "model", "knowledge",
            "profiling", "phase1_plan_fingerprint", "trial_protocol_sha256",
            "phase1_evidence_head_sha256", "phase1_terminal_sha256",
            "phase1_memory_sha256", "phase1_memory_entries",
            "phase1_memory_head_sha256", "phase1_lineage_id", "snapshot_id",
        }
        value = dict(self.manifest)
        if set(value) != required or value.pop("snapshot_id") != canonical_digest(value):
            raise ValueError("invalid Phase 1 snapshot manifest")
        if value["schema"] != _SNAPSHOT_SCHEMA:
            raise ValueError("invalid Phase 1 snapshot schema")
        if not self.memory_path.is_file() or not self.terminal_path.is_file():
            raise ValueError("immutable Phase 1 snapshot is incomplete")
        if (
            _sha256(self.memory_path.read_bytes()) != value["phase1_memory_sha256"]
            or _sha256(self.terminal_path.read_bytes())
            != value["phase1_terminal_sha256"]
        ):
            raise ValueError("immutable Phase 1 snapshot bytes changed")

    @classmethod
    def open(cls, root: str | Path) -> "Phase1Snapshot":
        root = Path(root).resolve()
        try:
            manifest = json.loads((root / "snapshot.json").read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid Phase 1 snapshot manifest") from exc
        snapshot = cls(root, manifest)
        snapshot.verify()
        return snapshot

    def memory_entries(self) -> tuple[Mapping[str, object], ...]:
        self.verify()
        return tuple(json.loads(line) for line in self.memory_path.read_bytes().splitlines())


def admit_phase1_snapshot(
    source_cell_root: str | Path,
    destination_root: str | Path,
) -> Phase1Snapshot:
    """Verify and byte-copy one completed Phase 1 lineage into Phase 2."""
    source_root = Path(source_cell_root).resolve()
    memory_path = source_root / "memory.jsonl"
    terminal_path = source_root / "terminal.json"
    authority_path = source_root / "authority.jsonl"
    evidence_path = source_root / "evidence.jsonl"
    try:
        terminal_bytes = terminal_path.read_bytes()
        memory_bytes = memory_path.read_bytes()
        authority_bytes = authority_path.read_bytes()
        evidence_bytes = evidence_path.read_bytes()
        terminal_value = json.loads(terminal_bytes)
        source_cell_id = terminal_value["cell_id"]
        source_cell, destination_cell = _cells_by_source()[source_cell_id]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Phase 1 terminal is missing or invalid") from exc

    terminal = Phase1Wave._read_terminal(source_cell, terminal_path)
    if (
        terminal is None
        or terminal["status"] != "passed"
        or terminal["terminal_reason"] != "completed"
        or terminal["completed_projects"] != len(DEFAULT_PROPOSALS)
    ):
        raise ValueError("Phase 1 terminal is not complete")
    if terminal["execution_profile"] not in _AUTHORITATIVE_PROFILES:
        raise ValueError("Phase 1 source must use an authoritative BZ-A3 profile")
    evidence = EvidenceLedger(evidence_path)
    retries, retry_evidence_sha256 = infrastructure_retry_summary(evidence)
    if (
        terminal["infrastructure_retries_used"] != retries
        or terminal["infrastructure_retry_evidence_sha256"]
        != retry_evidence_sha256
    ):
        raise ValueError("Phase 1 terminal conflicts with its retry evidence")

    try:
        raw_entries = [json.loads(line) for line in memory_bytes.splitlines()]
        lineage_id = raw_entries[0]["lineage_id"]
    except (IndexError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Phase 1 memory journal is invalid") from exc
    journal = Phase1LearningJournal(
        memory_path, DEFAULT_PROPOSALS,
        cell_id=source_cell.cell_id, lineage_id=lineage_id,
        evidence_resolver=_ReadOnlyAuthorityResolver(authority_bytes),
        trial_protocol_sha256=terminal["trial_protocol_sha256"],
    )
    entries = journal.read()
    state = journal.resume_state()
    if len(entries) != len(DEFAULT_PROPOSALS) or state.next_ordinal is not None:
        raise ValueError("Phase 1 memory journal is not complete")
    memory_aggregate = canonical_digest(tuple(
        entry["entry_sha256"] for entry in entries
    ))
    if terminal["evidence_sha256"] != memory_aggregate:
        raise ValueError("Phase 1 terminal conflicts with its memory aggregate")

    phase1_root_sha256 = canonical_digest({
        "schema": "a3-phase2-phase1-root-v1",
        "authority_sha256": _sha256(authority_bytes),
        "evidence_sha256": _sha256(evidence_bytes),
        "memory_sha256": _sha256(memory_bytes),
        "terminal_sha256": _sha256(terminal_bytes),
    })
    body = {
        "schema": _SNAPSHOT_SCHEMA,
        "target": "Ascend910B4",
        "language": "ascend-c",
        "phase1_root_sha256": phase1_root_sha256,
        "source_cell_id": source_cell.cell_id,
        "destination_cell_id": destination_cell.cell_id,
        "model": source_cell.backend_model.value,
        "knowledge": source_cell.knowledge.value,
        "profiling": source_cell.profiling.value,
        "phase1_plan_fingerprint": terminal["plan_fingerprint"],
        "trial_protocol_sha256": terminal["trial_protocol_sha256"],
        "phase1_evidence_head_sha256": evidence.head_sha256,
        "phase1_terminal_sha256": _sha256(terminal_bytes),
        "phase1_memory_sha256": _sha256(memory_bytes),
        "phase1_memory_entries": len(entries),
        "phase1_memory_head_sha256": state.head_sha256,
        "phase1_lineage_id": lineage_id,
    }
    manifest = {**body, "snapshot_id": canonical_digest(body)}
    root = Path(destination_root).resolve()
    _publish_exact(root / "phase1-memory.jsonl", memory_bytes)
    _publish_exact(root / "phase1-terminal.json", terminal_bytes)
    _publish_exact(root / "snapshot.json", canonical_bytes(manifest) + b"\n")
    snapshot = Phase1Snapshot.open(root)
    if dict(snapshot.manifest) != manifest:
        raise ValueError("immutable snapshot manifest conflict")
    return snapshot


@dataclass(frozen=True)
class Phase2Learning:
    kind: str
    statement: str
    project_id: str
    candidate_sha256: str
    host_facts: tuple[HostFact, ...]

    def __post_init__(self) -> None:
        if self.kind not in _LEARNING_KINDS:
            raise ValueError("Phase 2 learning kind is not recognized")
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise ValueError("Phase 2 learning statement must be non-empty")
        if not isinstance(self.project_id, str) or not _SHA256.fullmatch(
            self.project_id
        ):
            raise ValueError("Phase 2 learning project must be a SHA-256")
        if not isinstance(self.candidate_sha256, str) or not _SHA256.fullmatch(
            self.candidate_sha256
        ):
            raise ValueError("Phase 2 learning candidate must be a SHA-256")
        if not isinstance(self.host_facts, tuple) or not self.host_facts:
            raise ValueError("Phase 2 learning requires a nonempty host fact tuple")
        if any(not isinstance(fact, HostFact) for fact in self.host_facts):
            raise TypeError("Phase 2 learning host facts must be HostFact values")


class Phase2LearningJournal:
    def __init__(
        self, path: str | Path, snapshot: Phase1Snapshot,
        evidence_resolver: EvidenceResolver,
    ) -> None:
        if not isinstance(snapshot, Phase1Snapshot):
            raise TypeError("snapshot must be a Phase1Snapshot")
        snapshot.verify()
        if not callable(getattr(evidence_resolver, "resolve", None)):
            raise TypeError("evidence_resolver must provide resolve(digest)")
        self.path = Path(path).resolve()
        self.snapshot = snapshot
        self._evidence_resolver = evidence_resolver

    def append(self, learning: Phase2Learning) -> Mapping[str, object]:
        if not isinstance(learning, Phase2Learning):
            raise TypeError("learning must be Phase2Learning")
        self.snapshot.verify()
        self._validate_host_facts(learning)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            entries = self._decode(stream.read())
            body = {
                "schema": _LEARNING_SCHEMA,
                "sequence": len(entries) + 1,
                "snapshot_id": self.snapshot.snapshot_id,
                "destination_cell_id": self.snapshot.manifest["destination_cell_id"],
                "previous_sha256": (
                    entries[-1]["entry_sha256"] if entries else GENESIS_HASH
                ),
                "learning": asdict(learning),
            }
            entry = {**body, "entry_sha256": canonical_digest(body)}
            entry = json.loads(canonical_bytes(entry))
            stream.seek(0, os.SEEK_END)
            stream.write(canonical_bytes(entry) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            return entry

    def read(self) -> tuple[Mapping[str, object], ...]:
        self.snapshot.verify()
        if not self.path.exists():
            return ()
        with self.path.open("rb") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            return tuple(self._decode(stream.read()))

    def _decode(self, data: bytes) -> list[dict[str, object]]:
        if data and not data.endswith(b"\n"):
            raise ValueError("unterminated Phase 2 learning journal")
        entries: list[dict[str, object]] = []
        previous = GENESIS_HASH
        for sequence, line in enumerate(data.splitlines(), 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid Phase 2 learning journal JSON") from exc
            required = {
                "schema", "sequence", "snapshot_id", "destination_cell_id",
                "previous_sha256", "learning", "entry_sha256",
            }
            digest = entry.pop("entry_sha256", None)
            valid = (
                set(entry) == required - {"entry_sha256"}
                and entry["schema"] == _LEARNING_SCHEMA
                and entry["sequence"] == sequence
                and entry["snapshot_id"] == self.snapshot.snapshot_id
                and entry["destination_cell_id"]
                == self.snapshot.manifest["destination_cell_id"]
                and entry["previous_sha256"] == previous
                and digest == canonical_digest(entry)
            )
            try:
                learning = entry["learning"]
                Phase2Learning(
                    learning["kind"], learning["statement"],
                    learning["project_id"], learning["candidate_sha256"],
                    tuple(HostFact.from_mapping(fact) for fact in learning["host_facts"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid Phase 2 learning payload") from exc
            if not valid:
                raise ValueError("Phase 2 learning journal failed validation")
            self._validate_host_facts(Phase2Learning(
                learning["kind"], learning["statement"],
                learning["project_id"], learning["candidate_sha256"],
                tuple(HostFact.from_mapping(fact) for fact in learning["host_facts"]),
            ))
            entry["entry_sha256"] = digest
            previous = digest
            entries.append(entry)
        return entries

    def _validate_host_facts(self, learning: Phase2Learning) -> None:
        for fact in learning.host_facts:
            resolved = self._evidence_resolver.resolve(fact.evidence_sha256)
            if not isinstance(resolved, AuthoritativeEvidence):
                raise ValueError("host fact must resolve to authoritative Phase 2 evidence")
            checks = (
                ("kind", resolved.kind, fact.category),
                ("digest", resolved.evidence_sha256, fact.evidence_sha256),
                ("project", resolved.project_id, learning.project_id),
                ("candidate", resolved.candidate_sha256, learning.candidate_sha256),
                ("success", resolved.success, fact.success),
            )
            for label, actual, expected in checks:
                if actual != expected:
                    raise ValueError(
                        f"host fact {label} does not match authoritative Phase 2 evidence"
                    )

    def context(self) -> dict[str, object]:
        entries = self.read()
        phase1 = self.snapshot.memory_entries()
        return {
            "schema": "a3-phase2-memory-context-v1",
            "target": "Ascend910B4",
            "language": "ascend-c",
            "destination_cell_id": self.snapshot.manifest["destination_cell_id"],
            "phase1": {
                "snapshot_id": self.snapshot.snapshot_id,
                "entries": len(phase1),
                "projects": [entry["memory"] for entry in phase1],
            },
            "phase2": {
                "entries": len(entries),
                "learning": [entry["learning"] for entry in entries],
            },
        }
