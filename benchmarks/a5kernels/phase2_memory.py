"""A5-native immutable Phase 1 handoff and Phase 2 treatment identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping

from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_composition import (
    TERMINAL_SCHEMA,
    Phase1CellComposition,
)
from benchmarks.a5kernels.phase1_experiments import (
    A5Language,
    A5ModelIdentity,
    A5Phase1Cell,
    A5Target,
    KnowledgeMode,
    ProfilingGuidance,
    ProgrammingGuideIdentity,
    build_cells,
)
from benchmarks.a5kernels.phase1_memory import Phase1LearningJournal
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a5kernels.phase2_attention import SUITE_SHA256


CELL_SCHEMA = "a5-catlass-phase2-cell-v1"
CELL_SCHEMA_V2 = "a5-catlass-phase2-cell-v2"
SNAPSHOT_SCHEMA = "a5-catlass-phase2-phase1-snapshot-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class A5Phase2Cell:
    """One Phase 2 destination bound to an exact runnable Phase 1 treatment."""

    cell_id: str
    source_cell_id: str
    model: A5ModelIdentity
    knowledge: KnowledgeMode
    profiling: ProfilingGuidance
    guide: ProgrammingGuideIdentity
    qualification_sha256: str
    suite_sha256: str = SUITE_SHA256
    target: A5Target = A5Target.A5
    language: A5Language = A5Language.CATLASS_DSL
    schema: str = CELL_SCHEMA
    source_guide: ProgrammingGuideIdentity | None = None

    def __post_init__(self) -> None:
        if (
            self.target is not A5Target.A5
            or self.language is not A5Language.CATLASS_DSL
            or not isinstance(self.model, A5ModelIdentity)
            or not isinstance(self.knowledge, KnowledgeMode)
            or not isinstance(self.profiling, ProfilingGuidance)
            or not isinstance(self.guide, ProgrammingGuideIdentity)
        ):
            raise ValueError("Phase 2 cell must use registered A5 Catlass identities")
        if self.schema == CELL_SCHEMA:
            if self.source_guide is not None:
                raise ValueError("legacy Phase 2 cells cannot declare a source guide")
        elif self.schema == CELL_SCHEMA_V2:
            if (
                not isinstance(self.source_guide, ProgrammingGuideIdentity)
                or self.source_guide == self.guide
            ):
                raise ValueError(
                    "dual-guide Phase 2 cells require distinct source and destination guides"
                )
        else:
            raise ValueError("Phase 2 cell schema does not match its guide identity")
        if self.profiling is ProfilingGuidance.NEW:
            raise ValueError("pending profiler treatments cannot enter Phase 2")
        _require_digest(self.qualification_sha256, "qualification_sha256")
        if self.suite_sha256 != SUITE_SHA256:
            raise ValueError("Phase 2 cell changed the host-owned attention suite")
        source = _source_by_id(self.phase1_guide).get(self.source_cell_id)
        if source is None or (
            source.model,
            source.knowledge,
            source.profiling,
        ) != (self.model, self.knowledge, self.profiling):
            raise ValueError("Phase 2 cell does not match its runnable Phase 1 source")
        if self.cell_id != _cell_id(self.identity_values()):
            raise ValueError("Phase 2 cell ID does not match its treatment identity")

    @property
    def catlass_revision(self) -> str:
        return self.guide.catlass_revision

    @property
    def phase1_guide(self) -> ProgrammingGuideIdentity:
        """Guide identity that produced the admitted Phase 1 memory."""

        return self.source_guide or self.guide

    def identity_values(self) -> dict[str, object]:
        identity = {
            "schema": self.schema,
            "target": self.target.value,
            "language": self.language.value,
            "source_cell_id": self.source_cell_id,
            "model": self.model.as_dict(),
            "knowledge": self.knowledge.value,
            "profiling": self.profiling.value,
            "suite_sha256": self.suite_sha256,
            "guide": self.guide.as_dict(),
            "qualification_sha256": self.qualification_sha256,
            "catlass_revision": self.catlass_revision,
        }
        if self.source_guide is not None:
            identity["source_guide"] = self.source_guide.as_dict()
        return identity

    def as_dict(self) -> dict[str, object]:
        return {"cell_id": self.cell_id, **self.identity_values()}


def _cell_id(identity: Mapping[str, object]) -> str:
    return "a5-phase2-cell-" + canonical_digest(identity)[:16]


def _source_by_id(guide: ProgrammingGuideIdentity) -> dict[str, A5Phase1Cell]:
    return {cell.cell_id: cell for cell in build_cells(guide) if cell.runnable}


def phase2_cell_pairs(
    guide: ProgrammingGuideIdentity,
    qualification_sha256: str,
    *,
    destination_guide: ProgrammingGuideIdentity | None = None,
) -> tuple[tuple[A5Phase1Cell, A5Phase2Cell], ...]:
    """Return canonical Phase 1→Phase 2 pairs for source and current guides.

    Omitting ``destination_guide`` retains the legacy v1 identity byte for byte.
    """

    if not isinstance(guide, ProgrammingGuideIdentity):
        raise TypeError("guide must be a ProgrammingGuideIdentity")
    if destination_guide is None:
        destination_guide = guide
    if not isinstance(destination_guide, ProgrammingGuideIdentity):
        raise TypeError("destination_guide must be a ProgrammingGuideIdentity")
    qualification_sha256 = _require_digest(
        qualification_sha256, "qualification_sha256"
    )
    schema = CELL_SCHEMA if destination_guide == guide else CELL_SCHEMA_V2
    source_guide = None if schema == CELL_SCHEMA else guide
    pairs = []
    for source in build_cells(guide):
        if not source.runnable:
            continue
        identity = {
            "schema": schema,
            "target": source.target.value,
            "language": source.language.value,
            "source_cell_id": source.cell_id,
            "model": source.model.as_dict(),
            "knowledge": source.knowledge.value,
            "profiling": source.profiling.value,
            "suite_sha256": SUITE_SHA256,
            "guide": destination_guide.as_dict(),
            "qualification_sha256": qualification_sha256,
            "catlass_revision": destination_guide.catlass_revision,
        }
        if source_guide is not None:
            identity["source_guide"] = source_guide.as_dict()
        destination = A5Phase2Cell(
            _cell_id(identity),
            source.cell_id,
            source.model,
            source.knowledge,
            source.profiling,
            destination_guide,
            qualification_sha256,
            schema=schema,
            source_guide=source_guide,
        )
        pairs.append((source, destination))
    if len(pairs) != 8 or len({item.cell_id for _source, item in pairs}) != 8:
        raise RuntimeError("A5 Phase 2 requires exactly eight unique treatment cells")
    return tuple(pairs)


def phase2_cells(
    guide: ProgrammingGuideIdentity,
    qualification_sha256: str,
    *,
    destination_guide: ProgrammingGuideIdentity | None = None,
) -> tuple[A5Phase2Cell, ...]:
    return tuple(
        destination
        for _source, destination in phase2_cell_pairs(
            guide,
            qualification_sha256,
            destination_guide=destination_guide,
        )
    )


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
class A5Phase1Snapshot:
    """A verified byte-for-byte Phase 1 memory handoff for one A5 cell."""

    root: Path
    destination: A5Phase2Cell
    manifest: Mapping[str, object]

    @property
    def memory_path(self) -> Path:
        return self.root / "phase1-memory.jsonl"

    @property
    def terminal_path(self) -> Path:
        return self.root / "phase1-terminal.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "snapshot.json"

    @property
    def snapshot_id(self) -> str:
        return str(self.manifest["snapshot_id"])

    def verify(self) -> None:
        value = dict(self.manifest)
        required = {
            "schema",
            "destination_identity",
            "phase1_memory_sha256",
            "phase1_terminal_sha256",
            "phase1_memory_entries",
            "phase1_memory_head_sha256",
            "phase1_evidence_head_sha256",
            "snapshot_id",
        }
        snapshot_id = value.pop("snapshot_id", None)
        if (
            set(self.manifest) != required
            or self.manifest.get("schema") != SNAPSHOT_SCHEMA
            or snapshot_id != canonical_digest(value)
        ):
            raise ValueError("invalid A5 Phase 1 snapshot manifest")
        if self.manifest["destination_identity"] != self.destination.as_dict():
            raise ValueError("snapshot destination identity does not match")
        try:
            memory_bytes = self.memory_path.read_bytes()
            terminal_bytes = self.terminal_path.read_bytes()
        except OSError as exc:
            raise ValueError("immutable A5 Phase 1 snapshot is incomplete") from exc
        if (
            _sha256(memory_bytes) != self.manifest["phase1_memory_sha256"]
            or _sha256(terminal_bytes) != self.manifest["phase1_terminal_sha256"]
        ):
            raise ValueError("immutable A5 Phase 1 snapshot bytes changed")
        try:
            terminal = json.loads(terminal_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError("immutable A5 Phase 1 terminal is invalid") from exc
        terminal_fields = {
            "schema", "status", "cell_id", "model", "guide_sha256",
            "qualification_sha256", "completed_projects", "memory_head_sha256",
            "evidence_head_sha256", "record_sha256",
        }
        record_sha256 = terminal.get("record_sha256")
        terminal_body = {
            key: item for key, item in terminal.items() if key != "record_sha256"
        }
        if (
            set(terminal) != terminal_fields
            or record_sha256 != canonical_digest(terminal_body)
            or terminal.get("schema") != TERMINAL_SCHEMA
            or terminal.get("status") != "passed"
            or terminal.get("cell_id") != self.destination.source_cell_id
            or terminal.get("model") != self.destination.model.as_dict()
            or terminal.get("guide_sha256")
            != self.destination.phase1_guide.guide_sha256
            or terminal.get("qualification_sha256")
            != self.destination.qualification_sha256
            or terminal.get("completed_projects") != len(DEFAULT_PROPOSALS)
        ):
            raise ValueError("immutable A5 Phase 1 terminal identity is invalid")
        if not memory_bytes.endswith(b"\n"):
            raise ValueError("immutable Phase 1 memory journal is unterminated")
        journal = Phase1LearningJournal(self.memory_path, DEFAULT_PROPOSALS)
        entries = journal.read()
        state = journal.resume_state()
        if (
            len(entries) != len(DEFAULT_PROPOSALS)
            or state.next_ordinal is not None
            or self.manifest["phase1_memory_entries"] != len(entries)
            or self.manifest["phase1_memory_head_sha256"] != state.head_sha256
            or terminal.get("memory_head_sha256") != state.head_sha256
            or terminal.get("evidence_head_sha256")
            != self.manifest["phase1_evidence_head_sha256"]
        ):
            raise ValueError("immutable Phase 1 memory journal is not complete")

    @classmethod
    def open(
        cls, root: str | Path, destination: A5Phase2Cell
    ) -> "A5Phase1Snapshot":
        if not isinstance(destination, A5Phase2Cell):
            raise TypeError("destination must be an A5Phase2Cell")
        root = Path(root).resolve()
        try:
            manifest = json.loads((root / "snapshot.json").read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid A5 Phase 1 snapshot manifest") from exc
        snapshot = cls(root, destination, manifest)
        snapshot.verify()
        return snapshot

    def memory_entries(self) -> tuple[Mapping[str, object], ...]:
        self.verify()
        return tuple(json.loads(line) for line in self.memory_path.read_bytes().splitlines())


def admit_phase1_snapshot(
    composition: Phase1CellComposition,
    destination: A5Phase2Cell,
    phase2_root: str | Path,
) -> A5Phase1Snapshot:
    """Verify and byte-copy one completed A5 Phase 1 cell into Phase 2."""

    if not isinstance(composition, Phase1CellComposition):
        raise TypeError("composition must be a Phase1CellComposition")
    if not isinstance(destination, A5Phase2Cell):
        raise TypeError("destination must be an A5Phase2Cell")
    expected = phase2_cells(
        composition.guide,
        composition.qualification.report_sha256,
        destination_guide=destination.guide,
    )
    if destination not in expected:
        raise ValueError("Phase 2 destination identity does not match this composition")
    sources = {
        source.cell_id: source
        for source, _destination in phase2_cell_pairs(
            composition.guide,
            composition.qualification.report_sha256,
            destination_guide=destination.guide,
        )
    }
    source = sources[destination.source_cell_id]
    if tuple(composition.proposals) != tuple(DEFAULT_PROPOSALS):
        raise ValueError("Phase 1 source must contain the complete eight-project plan")

    terminal = composition.load_terminal(source)
    if terminal is None:
        raise ValueError("Phase 1 terminal is not complete")
    source_root = composition.root / "cells" / source.cell_id
    memory_path = source_root / "memory.jsonl"
    terminal_path = source_root / "terminal.json"
    memory_bytes = memory_path.read_bytes()
    terminal_bytes = terminal_path.read_bytes()
    if json.loads(terminal_bytes) != terminal:
        raise ValueError("Phase 1 terminal changed during snapshot admission")
    journal = Phase1LearningJournal(memory_path, DEFAULT_PROPOSALS)
    entries = journal.read()
    state = journal.resume_state()
    if len(entries) != 8 or state.next_ordinal is not None:
        raise ValueError("Phase 1 source does not have the complete eight-entry memory chain")

    body = {
        "schema": SNAPSHOT_SCHEMA,
        "destination_identity": destination.as_dict(),
        "phase1_memory_sha256": _sha256(memory_bytes),
        "phase1_terminal_sha256": _sha256(terminal_bytes),
        "phase1_memory_entries": len(entries),
        "phase1_memory_head_sha256": state.head_sha256,
        "phase1_evidence_head_sha256": terminal["evidence_head_sha256"],
    }
    manifest = {**body, "snapshot_id": canonical_digest(body)}
    root = (
        Path(phase2_root).resolve()
        / "cells"
        / destination.cell_id
        / "snapshot"
    )
    if root.exists() and any(root.iterdir()):
        retained = A5Phase1Snapshot.open(root, destination)
        if dict(retained.manifest) != manifest:
            raise ValueError("immutable snapshot manifest conflict")
        return retained
    _publish_exact(root / "phase1-memory.jsonl", memory_bytes)
    _publish_exact(root / "phase1-terminal.json", terminal_bytes)
    _publish_exact(root / "snapshot.json", canonical_bytes(manifest) + b"\n")
    return A5Phase1Snapshot.open(root, destination)


__all__ = [
    "A5Phase1Snapshot",
    "A5Phase2Cell",
    "CELL_SCHEMA",
    "CELL_SCHEMA_V2",
    "SNAPSHOT_SCHEMA",
    "admit_phase1_snapshot",
    "phase2_cell_pairs",
    "phase2_cells",
]
