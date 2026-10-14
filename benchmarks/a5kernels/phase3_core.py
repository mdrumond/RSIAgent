"""Authenticated frozen-memory handoff from A5 Phase 2 into Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_experiments import A5BackendModel
from benchmarks.a5kernels.phase2_live import Phase2Dependencies, Phase2LiveRunner
from benchmarks.a5kernels.phase2_memory import A5Phase1Snapshot, A5Phase2Cell


FROZEN_SCHEMA = "a5-catlass-phase3-frozen-memory-v1"
MANIFEST_SCHEMA = "a5-catlass-phase3-frozen-manifest-v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _publish_exact(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != value:
            raise ValueError(
                f"immutable Phase 3 artifact conflicts: {path.name}"
            ) from None
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _deep_freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _deep_freeze(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}")
    return value


@dataclass(frozen=True)
class Phase3FrozenMemory:
    """Immutable, Actor-visible memory copied from one passed Phase 2 cell."""

    root: Path
    cell: A5Phase2Cell
    manifest: Mapping[str, object]

    @property
    def memory_path(self) -> Path:
        return self.root / "actor-memory.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "snapshot.json"

    @property
    def snapshot_id(self) -> str:
        return str(self.manifest["snapshot_id"])

    def verify(self) -> None:
        body = dict(self.manifest)
        snapshot_id = body.pop("snapshot_id", None)
        required = {
            "schema", "source_cell", "phase2_status", "phase1_entries",
            "phase2_entries", "actor_memory_sha256", "source_snapshot_id",
            "source_protocol_sha256", "source_terminal_sha256",
            "source_memory_sha256",
        }
        if (
            set(body) != required
            or body.get("schema") != MANIFEST_SCHEMA
            or body.get("source_cell") != self.cell.as_dict()
            or body.get("phase2_status") != "passed"
            or snapshot_id != canonical_digest(body)
        ):
            raise ValueError("invalid Phase 3 frozen-memory manifest")
        try:
            payload = self.memory_path.read_bytes()
        except OSError as exc:
            raise ValueError("Phase 3 frozen memory is incomplete") from exc
        if _sha256(payload) != body["actor_memory_sha256"]:
            raise ValueError("Phase 3 frozen memory bytes changed")
        value = _read_object(self.memory_path, "Phase 3 Actor memory")
        if (
            set(value) != {"schema", "phase1", "phase2"}
            or value.get("schema") != FROZEN_SCHEMA
            or not isinstance(value.get("phase1"), list)
            or not isinstance(value.get("phase2"), list)
            or len(value["phase1"]) != body["phase1_entries"]
            or len(value["phase2"]) != body["phase2_entries"]
        ):
            raise ValueError("invalid Phase 3 Actor memory")

    def actor_memory(self) -> Mapping[str, object]:
        self.verify()
        return _deep_freeze(_read_object(self.memory_path, "Phase 3 Actor memory"))

    @classmethod
    def open(cls, root: str | Path, cell: A5Phase2Cell) -> "Phase3FrozenMemory":
        root = Path(root).resolve()
        manifest = _deep_freeze(
            _read_object(root / "snapshot.json", "Phase 3 manifest")
        )
        value = cls(root, cell, manifest)
        value.verify()
        return value


def admit_phase2_lineage(
    phase2_root: str | Path,
    cell: A5Phase2Cell,
    phase3_root: str | Path,
) -> Phase3FrozenMemory:
    """Authenticate a passed OpenAI Phase 2 lineage and freeze public memory."""

    if not isinstance(cell, A5Phase2Cell):
        raise TypeError("cell must be an A5Phase2Cell")
    if cell.model.backend_model is not A5BackendModel.GPT_5_6_SOL:
        raise ValueError("Phase 3 benchmark admission requires an OpenAI Phase 2 cell")
    phase2_root = Path(phase2_root).resolve()
    cell_root = phase2_root / "cells" / cell.cell_id
    snapshot = A5Phase1Snapshot.open(cell_root / "snapshot", cell)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Phase 3 admission must not execute Phase 2")

    runner = Phase2LiveRunner(
        phase2_root=phase2_root,
        snapshot=snapshot,
        dependencies=Phase2Dependencies(cell, forbidden, forbidden),
    )
    try:
        terminal = runner.load_terminal()
    except ValueError as exc:
        raise ValueError("invalid authenticated Phase 2 lineage") from exc
    if terminal is None or terminal.get("status") != "passed":
        raise ValueError("Phase 3 requires an authenticated passed Phase 2 terminal")

    protocol_bytes = runner.protocol_path.read_bytes()
    memory_bytes = runner.memory_path.read_bytes()
    terminal_bytes = runner.terminal_path.read_bytes()
    phase2_entries = [json.loads(line) for line in memory_bytes.splitlines()]
    visible_fields = (
        "attempt", "candidate_sha256", "evidence_sha256", "passed",
        "failed_cases", "entry_sha256",
    )
    actor_phase2 = [
        {key: entry[key] for key in visible_fields} for entry in phase2_entries
    ]
    actor_memory = {
        "schema": FROZEN_SCHEMA,
        "phase1": list(snapshot.memory_entries()),
        "phase2": actor_phase2,
    }
    actor_bytes = canonical_bytes(actor_memory) + b"\n"
    body = {
        "schema": MANIFEST_SCHEMA,
        "source_cell": cell.as_dict(),
        "phase2_status": "passed",
        "phase1_entries": len(actor_memory["phase1"]),
        "phase2_entries": len(actor_phase2),
        "actor_memory_sha256": _sha256(actor_bytes),
        "source_snapshot_id": snapshot.snapshot_id,
        "source_protocol_sha256": _sha256(protocol_bytes),
        "source_terminal_sha256": _sha256(terminal_bytes),
        "source_memory_sha256": _sha256(memory_bytes),
    }
    manifest = {**body, "snapshot_id": canonical_digest(body)}
    root = Path(phase3_root).resolve() / "lineages" / cell.cell_id
    _publish_exact(root / "actor-memory.json", actor_bytes)
    _publish_exact(root / "snapshot.json", canonical_bytes(manifest) + b"\n")
    return Phase3FrozenMemory.open(root, cell)


__all__ = ["Phase3FrozenMemory", "admit_phase2_lineage"]
