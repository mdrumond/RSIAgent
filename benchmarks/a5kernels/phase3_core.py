"""Authenticated frozen-memory handoff from A5 Phase 2 into Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_experiments import A5BackendModel
from benchmarks.a5kernels.phase2_live import Phase2Dependencies, Phase2LiveRunner
from benchmarks.a5kernels.phase2_memory import (
    A5Phase1Snapshot,
    A5Phase2Cell,
    QUALIFICATION_SCHEMA,
)


FROZEN_SCHEMA = "a5-catlass-phase3-frozen-memory-v1"
MANIFEST_SCHEMA = "a5-catlass-phase3-frozen-manifest-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


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


def _qualification_provenance(
    path: Path, cell: A5Phase2Cell,
) -> tuple[str, bytes, tuple[tuple[str, str], ...]]:
    """Validate the retained transport record without exposing its payload."""

    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Phase 2 transport qualification") from exc
    if not isinstance(value, dict):
        raise ValueError("invalid Phase 2 transport qualification")
    body = dict(value)
    record_sha256 = body.pop("record_sha256", None)
    required = {
        "schema", "cell_id", "catlass_source", "catlass_revision",
        "runtime_catlass_revision", "runtime_provenance", "physical_device",
        "case", "case_sha256", "candidate_sha256", "input_sha256", "status",
        "exit_code", "session_handle", "output", "output_sha256", "metadata",
    }
    legacy = required - {"runtime_catlass_revision"}
    try:
        provenance = tuple(tuple(item) for item in body["runtime_provenance"])
        provenance_map = dict(provenance)
        calculated_record_sha256 = canonical_digest(body)
        output_sha256 = canonical_digest(body["output"])
        case_sha256 = canonical_digest(body["case"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid Phase 2 transport qualification") from exc
    runtime_revision = body.get(
        "runtime_catlass_revision", body.get("catlass_revision")
    )
    metadata = body.get("metadata")
    digests = (
        record_sha256, body.get("case_sha256"), body.get("candidate_sha256"),
        body.get("input_sha256"), body.get("output_sha256"),
    )
    if (
        set(body) not in (required, legacy)
        or body.get("schema") != QUALIFICATION_SCHEMA
        or record_sha256 != calculated_record_sha256
        or body.get("cell_id") != cell.cell_id
        or body.get("catlass_revision") != cell.catlass_revision
        or not isinstance(runtime_revision, str)
        or _REVISION.fullmatch(runtime_revision) is None
        or provenance_map.get("catlass_revision") != runtime_revision
        or provenance_map.get("catlass_source") != body.get("catlass_source")
        or provenance_map.get("execution_profile") != "bz-a5"
        or len(provenance_map) != len(provenance)
        or any(
            len(item) != 2 or not all(isinstance(part, str) and part for part in item)
            for item in provenance
        )
        or not isinstance(body.get("catlass_source"), str)
        or not body["catlass_source"].startswith("/")
        or isinstance(body.get("physical_device"), bool)
        or not isinstance(body.get("physical_device"), int)
        or body["physical_device"] < 0
        or body.get("status") != "transport-qualified"
        or body.get("exit_code") != 0
        or not isinstance(body.get("session_handle"), str)
        or not body["session_handle"].startswith("bz-a5:")
        or not isinstance(body.get("case"), dict)
        or body.get("case_sha256") != case_sha256
        or not isinstance(body.get("output"), list)
        or body.get("output_sha256") != output_sha256
        or any(not isinstance(item, str) or _SHA256.fullmatch(item) is None
               for item in digests)
        or not isinstance(metadata, list)
        or any(
            not isinstance(item, list) or len(item) != 2
            or not all(isinstance(part, str) for part in item)
            for item in metadata
        )
    ):
        raise ValueError("invalid Phase 2 transport qualification")
    assert isinstance(record_sha256, str)
    return record_sha256, payload, provenance


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

    def _verified_actor_memory(self) -> dict[str, object]:
        body = dict(self.manifest)
        snapshot_id = body.pop("snapshot_id", None)
        required = {
            "schema", "source_cell", "phase2_status", "phase1_entries",
            "phase2_entries", "actor_memory_sha256", "source_snapshot_id",
            "source_protocol_sha256", "source_terminal_sha256",
            "source_memory_sha256", "transport_qualification_sha256",
            "qualification_artifact_sha256", "runtime_catlass_revision",
            "runtime_provenance",
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
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid Phase 3 Actor memory") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "phase1", "phase2"}
            or value.get("schema") != FROZEN_SCHEMA
            or not isinstance(value.get("phase1"), list)
            or not isinstance(value.get("phase2"), list)
            or len(value["phase1"]) != body["phase1_entries"]
            or len(value["phase2"]) != body["phase2_entries"]
        ):
            raise ValueError("invalid Phase 3 Actor memory")
        return value

    def verify(self) -> None:
        self._verified_actor_memory()

    def actor_memory(self) -> Mapping[str, object]:
        return _deep_freeze(self._verified_actor_memory())

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
    source_paths = (
        runner.protocol_path, runner.memory_path, runner.terminal_path,
    )
    try:
        captured = tuple(path.read_bytes() for path in source_paths)
        terminal = runner.load_terminal()
        rechecked = tuple(path.read_bytes() for path in source_paths)
    except (OSError, ValueError) as exc:
        raise ValueError("invalid authenticated Phase 2 lineage") from exc
    if captured != rechecked:
        raise ValueError("Phase 2 source changed during authentication")
    if terminal is None or terminal.get("status") != "passed":
        raise ValueError("Phase 3 requires an authenticated passed Phase 2 terminal")

    qualification_sha256, qualification_bytes, provenance = (
        _qualification_provenance(
            cell_root / "qualification.json", cell
        )
    )

    protocol_bytes, memory_bytes, terminal_bytes = captured
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
        "transport_qualification_sha256": qualification_sha256,
        "qualification_artifact_sha256": _sha256(qualification_bytes),
        "runtime_catlass_revision": dict(provenance)["catlass_revision"],
        "runtime_provenance": provenance,
    }
    manifest = {**body, "snapshot_id": canonical_digest(body)}
    root = Path(phase3_root).resolve() / "lineages" / cell.cell_id
    _publish_exact(root / "actor-memory.json", actor_bytes)
    _publish_exact(root / "snapshot.json", canonical_bytes(manifest) + b"\n")
    return Phase3FrozenMemory.open(root, cell)


__all__ = ["Phase3FrozenMemory", "admit_phase2_lineage"]
