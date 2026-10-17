"""A5-only, resumable Phase 2 protocol over host-owned attention evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping

from benchmarks.a5kernels.evidence import GENESIS_HASH, canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase2_attention import (
    CANDIDATE_ABI_SHA256,
    HOST_TARGET_GRADING_SHA256,
    SUITE_SHA256,
    AttentionTargetEvidence,
    _truncate_utf8,
)
from benchmarks.a5kernels.phase2_memory import A5Phase1Snapshot, A5Phase2Cell


PROTOCOL_SCHEMA = "a5-catlass-phase2-live-protocol-v3"
ATTEMPT_SCHEMA = "a5-catlass-phase2-attempt-v1"
MEMORY_SCHEMA = "a5-catlass-phase2-memory-v1"
TERMINAL_SCHEMA = "a5-catlass-phase2-terminal-v1"
MAX_ATTEMPTS = 3
_MAX_FAILURE_DETAIL = 160
_MAX_HOST_DIAGNOSTIC = 512
MAX_REPAIR_SOURCE_BYTES = 256 * 1024
REPAIR_INSTRUCTION = (
    "Diagnose the supplied host evidence and repair the preceding candidate. "
    "Preserve working structure and make focused changes instead of replacing "
    "the implementation blindly. Submit the complete revised source; only the "
    "host verifier can establish correctness."
)


@dataclass(frozen=True)
class Phase2AttemptRequest:
    """Treatment-bound context given to a candidate-producing agent."""

    cell: A5Phase2Cell
    protocol_sha256: str
    attempt: int
    request_id: str
    attempt_id: str
    phase1_memory: tuple[Mapping[str, object], ...]
    phase2_memory: tuple[Mapping[str, object], ...]
    repair_context: Mapping[str, object] | None = None


SourceProvider = Callable[[Phase2AttemptRequest], str]
TargetVerifier = Callable[[str, str, str, Path], AttentionTargetEvidence]
TraceSink = Callable[[Phase2AttemptRequest, str, AttentionTargetEvidence], None]


@dataclass(frozen=True)
class Phase2Dependencies:
    """Injected candidate and checked host-verifier seams for one A5 cell."""

    cell: A5Phase2Cell
    source: SourceProvider
    verify: TargetVerifier
    trace: TraceSink | None = None
    cell_type: type = A5Phase2Cell

    def __post_init__(self) -> None:
        if not isinstance(self.cell, self.cell_type):
            raise TypeError("Phase 2 dependencies require an A5Phase2Cell")
        if not callable(self.source) or not callable(self.verify):
            raise TypeError("Phase 2 source and verifier must be callable")
        if self.trace is not None and not callable(self.trace):
            raise TypeError("Phase 2 trace sink must be callable")


def _publish_exact(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != value:
            raise ValueError(f"immutable Phase 2 artifact conflicts: {path.name}") from None
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}")
    return value


class Phase2LiveRunner:
    """Run one isolated A5 treatment until host PASS or bounded exhaustion."""

    snapshot_type = A5Phase1Snapshot
    evidence_type = AttentionTargetEvidence
    execution_profile = "bz-a5"
    @property
    def candidate_abi_sha256(self):
        return CANDIDATE_ABI_SHA256
    protocol_schema = PROTOCOL_SCHEMA
    attempt_schema = ATTEMPT_SCHEMA
    memory_schema = MEMORY_SCHEMA
    terminal_schema = TERMINAL_SCHEMA
    source_filename = "candidate.py"

    def __init__(
        self, *, phase2_root: str | Path, snapshot: A5Phase1Snapshot,
        dependencies: Phase2Dependencies,
    ) -> None:
        if not isinstance(snapshot, self.snapshot_type):
            raise TypeError("snapshot must be an A5Phase1Snapshot")
        if not isinstance(dependencies, Phase2Dependencies):
            raise TypeError("dependencies must be Phase2Dependencies")
        if dependencies.cell != snapshot.destination:
            raise ValueError("snapshot and Phase 2 cell identity do not match")
        self.phase2_root = Path(phase2_root).resolve()
        self.snapshot = snapshot
        self.dependencies = dependencies
        self.root = self.phase2_root / "cells" / dependencies.cell.cell_id
        if snapshot.root.resolve() != self.root / "snapshot":
            raise ValueError("snapshot path does not match the Phase 2 cell root")

    @property
    def protocol_path(self) -> Path:
        return self.root / "protocol.json"

    @property
    def memory_path(self) -> Path:
        return self.root / "memory.jsonl"

    @property
    def terminal_path(self) -> Path:
        return self.root / "terminal.json"

    def run(self) -> Mapping[str, object]:
        # A snapshot is not a one-time admission token.  Every entry and resume
        # rechecks its immutable bytes before trusting any retained attempt.
        self.snapshot.verify()
        protocol = self._ensure_protocol()
        memories = self._read_memory(protocol)
        self._verify_committed_attempts(protocol, memories)
        if self.terminal_path.exists():
            return self._read_terminal(protocol, memories)

        for ordinal in range(len(memories) + 1, MAX_ATTEMPTS + 1):
            self.snapshot.verify()
            evidence = self._attempt(protocol, ordinal, memories)
            memories = self._commit_memory(protocol, ordinal, evidence, memories)
            if evidence.passed:
                return self._publish_terminal(protocol, memories, "passed")
        return self._publish_terminal(protocol, memories, "failed")

    def load_terminal(self) -> Mapping[str, object] | None:
        """Authenticate a retained terminal without creating campaign state."""

        self.snapshot.verify()
        if not self.terminal_path.exists():
            return None
        protocol = _read_object(self.protocol_path, "Phase 2 protocol")
        digest = protocol.pop("protocol_sha256", None)
        if protocol != self._protocol_body() or digest != canonical_digest(protocol):
            raise ValueError("Phase 2 protocol identity changed")
        protocol["protocol_sha256"] = digest
        memories = self._read_memory(protocol)
        self._verify_committed_attempts(protocol, memories)
        return self._read_terminal(protocol, memories)

    def _protocol_body(self) -> dict[str, object]:
        identity = {
            "target_suite_sha256": SUITE_SHA256,
            "candidate_abi_sha256": self.candidate_abi_sha256,
            "host_target_grading_sha256": HOST_TARGET_GRADING_SHA256,
            "source_snapshot_id": self.snapshot.snapshot_id,
            "cell": self.dependencies.cell.as_dict(),
        }
        return {
            "schema": self.protocol_schema,
            "identity": identity,
            "execution_profile": self.execution_profile,
            "maximum_semantic_attempts": MAX_ATTEMPTS,
            "semantic_retry": {
                "preceding_attempt_only": True,
                "candidate_source_max_bytes": MAX_REPAIR_SOURCE_BYTES,
                "instruction_sha256": hashlib.sha256(
                    REPAIR_INSTRUCTION.encode("utf-8")
                ).hexdigest(),
            },
        }

    def _ensure_protocol(self) -> dict[str, object]:
        body = self._protocol_body()
        value = {**body, "protocol_sha256": canonical_digest(body)}
        _publish_exact(self.protocol_path, canonical_bytes(value) + b"\n")
        retained = _read_object(self.protocol_path, "Phase 2 protocol")
        digest = retained.pop("protocol_sha256", None)
        if retained != body or digest != canonical_digest(retained):
            raise ValueError("Phase 2 protocol identity changed")
        return {**retained, "protocol_sha256": digest}

    def _attempt_identity(
        self, protocol: Mapping[str, object], ordinal: int,
    ) -> dict[str, object]:
        prefix = f"{self.snapshot.snapshot_id[:16]}-a{ordinal:02d}"
        body = {
            "schema": self.attempt_schema,
            "protocol_sha256": protocol["protocol_sha256"],
            "cell_id": self.dependencies.cell.cell_id,
            "snapshot_id": self.snapshot.snapshot_id,
            "suite_sha256": SUITE_SHA256,
            "candidate_abi_sha256": self.candidate_abi_sha256,
            "host_target_grading_sha256": HOST_TARGET_GRADING_SHA256,
            "attempt": ordinal,
            "request_id": prefix,
            "attempt_id": f"target-{prefix}",
        }
        return {**body, "attempt_sha256": canonical_digest(body)}

    def _attempt(
        self, protocol: Mapping[str, object], ordinal: int,
        memories: tuple[Mapping[str, object], ...],
    ) -> AttentionTargetEvidence:
        identity = self._attempt_identity(protocol, ordinal)
        attempt_root = self.root / "attempts" / f"attempt-{ordinal:02d}"
        identity_path = attempt_root / "attempt.json"
        _publish_exact(identity_path, canonical_bytes(identity) + b"\n")

        source_path = attempt_root / self.source_filename
        request = Phase2AttemptRequest(
            self.dependencies.cell,
            str(protocol["protocol_sha256"]),
            ordinal,
            str(identity["request_id"]),
            str(identity["attempt_id"]),
            self.snapshot.memory_entries(),
            tuple(self._memory_context(item) for item in memories),
            self._repair_context(memories),
        )
        if source_path.exists():
            source = source_path.read_text(encoding="utf-8")
        else:
            source = self.dependencies.source(request)
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Phase 2 candidate provider must return source text")
            _publish_exact(source_path, source.encode("utf-8"))

        evidence_path = attempt_root / "evidence.json"
        if evidence_path.exists():
            evidence = self.evidence_type.read(evidence_path)
        else:
            # Infrastructure errors intentionally propagate before evidence or
            # memory is committed.  Resume therefore retries this same attempt.
            evidence = self.dependencies.verify(
                source, request.request_id, request.attempt_id, attempt_root / "host",
            )
            if not isinstance(evidence, self.evidence_type):
                raise TypeError("Phase 2 verifier must return AttentionTargetEvidence")
            self._validate_evidence(evidence, source, request)
            _publish_exact(evidence_path, canonical_bytes(asdict(evidence)) + b"\n")
            evidence = self.evidence_type.read(evidence_path)
        self._validate_evidence(evidence, source, request)
        if self.dependencies.trace is not None:
            self.dependencies.trace(request, source, evidence)
        return evidence

    def _validate_evidence(
        self, evidence: AttentionTargetEvidence, source: str,
        request: Phase2AttemptRequest,
    ) -> None:
        if (
            evidence.request_id != request.request_id
            or evidence.attempt_id != request.attempt_id
            or evidence.execution_profile != self.execution_profile
            or evidence.suite_sha256 != SUITE_SHA256
            or evidence.candidate_sha256
            != hashlib.sha256(source.encode("utf-8")).hexdigest()
        ):
            raise ValueError("host attention evidence changed attempt identity")

    def _read_memory(
        self, protocol: Mapping[str, object],
    ) -> tuple[Mapping[str, object], ...]:
        if not self.memory_path.exists():
            return ()
        entries: list[Mapping[str, object]] = []
        previous = GENESIS_HASH
        for ordinal, line in enumerate(self.memory_path.read_bytes().splitlines(), 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid Phase 2 memory journal") from exc
            digest = entry.pop("entry_sha256", None)
            if (
                entry.get("schema") != self.memory_schema
                or entry.get("protocol_sha256") != protocol["protocol_sha256"]
                or entry.get("attempt") != ordinal
                or entry.get("previous_sha256") != previous
                or digest != canonical_digest(entry)
            ):
                raise ValueError("invalid Phase 2 memory chain")
            entry["entry_sha256"] = digest
            entries.append(entry)
            previous = str(digest)
        return tuple(entries)

    def _verify_committed_attempts(
        self, protocol: Mapping[str, object],
        memories: tuple[Mapping[str, object], ...],
    ) -> None:
        for ordinal, memory in enumerate(memories, 1):
            identity = self._attempt_identity(protocol, ordinal)
            root = self.root / "attempts" / f"attempt-{ordinal:02d}"
            if _read_object(root / "attempt.json", "Phase 2 attempt") != identity:
                raise ValueError("committed Phase 2 attempt identity changed")
            try:
                source = (root / self.source_filename).read_text(encoding="utf-8")
                evidence = self.evidence_type.read(root / "evidence.json")
            except OSError as exc:
                raise ValueError("committed Phase 2 attempt is incomplete") from exc
            request = Phase2AttemptRequest(
                self.dependencies.cell, str(protocol["protocol_sha256"]), ordinal,
                str(identity["request_id"]), str(identity["attempt_id"]), (), (),
            )
            self._validate_evidence(evidence, source, request)
            if (
                memory["attempt_identity_sha256"] != identity["attempt_sha256"]
                or memory["evidence_sha256"] != evidence.evidence_sha256
                or memory["candidate_sha256"] != evidence.candidate_sha256
                or memory["passed"] is not evidence.passed
            ):
                raise ValueError("committed Phase 2 memory changed host evidence")

    def _commit_memory(
        self, protocol: Mapping[str, object], ordinal: int,
        evidence: AttentionTargetEvidence,
        memories: tuple[Mapping[str, object], ...],
    ) -> tuple[Mapping[str, object], ...]:
        if len(memories) >= ordinal:
            retained = memories[ordinal - 1]
            if retained["evidence_sha256"] != evidence.evidence_sha256:
                raise ValueError("committed Phase 2 attempt changed evidence")
            return memories
        if len(memories) != ordinal - 1:
            raise ValueError("Phase 2 memory attempts are not contiguous")
        failures = [self._failure_context(item) for item in evidence.cases
                    if not item.passed]
        body = {
            "schema": self.memory_schema,
            "protocol_sha256": protocol["protocol_sha256"],
            "attempt": ordinal,
            "attempt_identity_sha256": self._attempt_identity(
                protocol, ordinal
            )["attempt_sha256"],
            "candidate_sha256": evidence.candidate_sha256,
            "evidence_sha256": evidence.evidence_sha256,
            "passed": evidence.passed,
            "failed_cases": failures,
            "previous_sha256": (
                memories[-1]["entry_sha256"] if memories else GENESIS_HASH
            ),
        }
        entry = {**body, "entry_sha256": canonical_digest(body)}
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_path.open("ab") as stream:
            stream.write(canonical_bytes(entry) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return self._read_memory(protocol)

    @staticmethod
    def _failure_context(item) -> Mapping[str, object]:
        failure = {
            "case_id": item.case_id,
            "failure": str(item.failure)[:_MAX_FAILURE_DETAIL],
            "exit_code": item.exit_code,
            "max_abs_error": item.max_abs_error,
            "max_rel_error": item.max_rel_error,
            "mismatch_index": item.mismatch_index,
            "mismatch_expected": item.mismatch_expected,
            "mismatch_actual": item.mismatch_actual,
        }
        diagnostic = getattr(item, "diagnostic", None)
        if diagnostic is not None:
            failure["diagnostic"] = _truncate_utf8(
                str(diagnostic), _MAX_HOST_DIAGNOSTIC
            )
        return failure

    @staticmethod
    def _memory_context(entry: Mapping[str, object]) -> Mapping[str, object]:
        return {
            key: entry[key]
            for key in (
                "attempt", "candidate_sha256", "evidence_sha256", "passed",
                "failed_cases", "entry_sha256",
            )
        }

    def _repair_context(
        self, memories: tuple[Mapping[str, object], ...],
    ) -> Mapping[str, object] | None:
        """Rebuild the preceding candidate/evidence pair for a focused retry."""

        if not memories:
            return None
        memory = memories[-1]
        ordinal = int(memory["attempt"])
        source = (
            self.root / "attempts" / f"attempt-{ordinal:02d}" / self.source_filename
        ).read_text(encoding="utf-8")
        source_bytes = source.encode("utf-8")
        candidate_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if candidate_sha256 != memory["candidate_sha256"]:
            raise ValueError("preceding Phase 2 candidate changed host evidence")
        bounded_source = _truncate_utf8(source, MAX_REPAIR_SOURCE_BYTES)
        return {
            "instruction": REPAIR_INSTRUCTION,
            "prior_attempt": ordinal,
            "candidate_sha256": candidate_sha256,
            "evidence_sha256": memory["evidence_sha256"],
            "memory_entry_sha256": memory["entry_sha256"],
            "host_evidence": self._memory_context(memory),
            "candidate_source": bounded_source,
            "candidate_source_bytes": len(source_bytes),
            "candidate_source_truncated": bounded_source != source,
            "candidate_context_sha256": hashlib.sha256(
                bounded_source.encode("utf-8")
            ).hexdigest(),
        }

    def _publish_terminal(
        self, protocol: Mapping[str, object],
        memories: tuple[Mapping[str, object], ...], status: str,
    ) -> Mapping[str, object]:
        last = memories[-1]
        if status == "passed" and last["passed"] is not True:
            raise ValueError("only host PASS evidence can admit Phase 2")
        if status == "failed" and (
            len(memories) != MAX_ATTEMPTS or last["passed"] is not False
        ):
            raise ValueError("failed terminal requires exhausted host evidence")
        body = {
            "schema": self.terminal_schema,
            "status": status,
            "cell_id": self.dependencies.cell.cell_id,
            "protocol_sha256": protocol["protocol_sha256"],
            "completed_attempts": len(memories),
            "memory_head_sha256": last["entry_sha256"],
            "evidence_head_sha256": last["evidence_sha256"],
        }
        terminal = {**body, "record_sha256": canonical_digest(body)}
        _publish_exact(self.terminal_path, canonical_bytes(terminal) + b"\n")
        return self._read_terminal(protocol, memories)

    def _read_terminal(
        self, protocol: Mapping[str, object],
        memories: tuple[Mapping[str, object], ...],
    ) -> Mapping[str, object]:
        value = _read_object(self.terminal_path, "Phase 2 terminal")
        digest = value.pop("record_sha256", None)
        status = value.get("status")
        valid_completion = (
            status == "passed" and bool(memories) and memories[-1]["passed"] is True
        ) or (
            status == "failed" and len(memories) == MAX_ATTEMPTS
            and memories[-1]["passed"] is False
        )
        if (
            value.get("schema") != self.terminal_schema
            or value.get("cell_id") != self.dependencies.cell.cell_id
            or value.get("protocol_sha256") != protocol["protocol_sha256"]
            or value.get("completed_attempts") != len(memories)
            or not memories
            or value.get("memory_head_sha256") != memories[-1]["entry_sha256"]
            or value.get("evidence_head_sha256") != memories[-1]["evidence_sha256"]
            or digest != canonical_digest(value)
            or not valid_completion
        ):
            raise ValueError("invalid Phase 2 terminal")
        return {**value, "record_sha256": digest}


__all__ = [
    "MAX_ATTEMPTS", "MAX_REPAIR_SOURCE_BYTES", "REPAIR_INSTRUCTION",
    "Phase2AttemptRequest", "Phase2Dependencies", "Phase2LiveRunner",
]
