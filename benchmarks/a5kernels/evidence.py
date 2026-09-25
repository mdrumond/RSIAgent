"""Host-owned, tamper-evident records for A5 kernel runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


GENESIS_HASH = "0" * 64


class EvidenceKind(str, Enum):
    REQUEST = "request"
    ACTION = "action"
    ARTIFACT = "artifact"
    RESULT = "result"


def canonical_bytes(value: Any) -> bytes:
    """Serialize JSON data identically or reject ambiguous/non-portable data."""

    normalized = _normalize(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _normalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical evidence cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical evidence mappings require string keys")
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(f"unsupported canonical evidence type: {type(value).__name__}")


@dataclass(frozen=True)
class EvidenceEntry:
    sequence: int
    kind: str
    payload: Mapping[str, Any]
    previous_sha256: str
    entry_sha256: str

    @classmethod
    def create(
        cls,
        sequence: int,
        kind: EvidenceKind | str,
        payload: Mapping[str, Any],
        previous_sha256: str,
    ) -> "EvidenceEntry":
        body = {
            "sequence": sequence,
            "kind": EvidenceKind(kind).value,
            "payload": payload,
            "previous_sha256": previous_sha256,
        }
        return cls(**body, entry_sha256=canonical_digest(body))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "payload": self.payload,
            "previous_sha256": self.previous_sha256,
            "entry_sha256": self.entry_sha256,
        }


class EvidenceLedger:
    """Append-only JSONL ledger whose complete existing chain is verified."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries = tuple(self._read())
        self.verify(self._entries)

    @property
    def entries(self) -> tuple[EvidenceEntry, ...]:
        return self._entries

    @property
    def head_sha256(self) -> str:
        return self._entries[-1].entry_sha256 if self._entries else GENESIS_HASH

    def append(
        self, kind: EvidenceKind | str, payload: Mapping[str, Any]
    ) -> EvidenceEntry:
        normalized = _normalize(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            current = tuple(self._decode(stream.read()))
            self.verify(current)
            previous = current[-1].entry_sha256 if current else GENESIS_HASH
            entry = EvidenceEntry.create(len(current), kind, normalized, previous)
            stream.seek(0, os.SEEK_END)
            stream.write(canonical_bytes(entry.as_dict()) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._entries = current + (entry,)
        return entry

    def _read(self) -> Iterable[EvidenceEntry]:
        if not self.path.exists():
            return ()
        return self._decode(self.path.read_bytes())

    @staticmethod
    def _decode(data: bytes) -> Iterable[EvidenceEntry]:
        entries = []
        for line_number, line in enumerate(data.splitlines(), 1):
            try:
                value = json.loads(line)
                entries.append(EvidenceEntry(**value))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid evidence at line {line_number}") from exc
        return entries

    @staticmethod
    def verify(entries: Iterable[EvidenceEntry]) -> None:
        previous = GENESIS_HASH
        for expected_sequence, entry in enumerate(entries):
            if entry.sequence != expected_sequence or entry.previous_sha256 != previous:
                raise ValueError(f"broken evidence chain at sequence {expected_sequence}")
            expected = EvidenceEntry.create(
                entry.sequence, entry.kind, entry.payload, entry.previous_sha256
            )
            if entry.entry_sha256 != expected.entry_sha256:
                raise ValueError(f"invalid evidence digest at sequence {expected_sequence}")
            previous = entry.entry_sha256
