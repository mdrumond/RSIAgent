"""Prospective, host-verified trace retention and Fireworks SFT export."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence


SOURCE_SCHEMA = "a5-sft-sources-v1"
TRACE_SCHEMA = "a5-sft-trace-v1"
DATASET_SCHEMA = "a5-fireworks-sft-dataset-v1"
REQUIRED_SOURCE_NAMES = {
    "ops-transformer", "ops-nn", "ops-cv", "recsdk", "pypto", "pto-isa",
    "catlass", "cannbench",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
_SECRET_KEY = re.compile(r"(?i)(api[_-]?key|authorization|credential|password|secret|token)")
_SECRET_VALUE = re.compile(
    r"(?i)(?:(?:api[_-]?key|password|secret|token)\s*[=:]\s*)\S+|"
    r"Bearer\s+[A-Za-z0-9._~+/-]+|AKIA[0-9A-Z]{16}"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _redact(value: object, *, key: str = "") -> object:
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    if isinstance(value, Mapping):
        return {str(k): _redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


@dataclass(frozen=True)
class Source:
    name: str
    repository: str
    revision: str
    license: str
    include: tuple[str, ...]

    def validate(self) -> None:
        if self.name not in REQUIRED_SOURCE_NAMES:
            raise ValueError(f"unknown source name: {self.name}")
        if not self.repository.startswith("https://") or self.repository.endswith("/"):
            raise ValueError(f"invalid repository for {self.name}")
        if not _REVISION.fullmatch(self.revision):
            raise ValueError(f"invalid pinned revision for {self.name}")
        if not self.license.strip() or not self.include:
            raise ValueError(f"license and include globs are required for {self.name}")
        for pattern in self.include:
            pure = PurePosixPath(pattern)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe include glob for {self.name}")


@dataclass(frozen=True)
class SourceManifest:
    sources: tuple[Source, ...]
    sha256: str

    @classmethod
    def load(cls, path: Path) -> "SourceManifest":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) != {"schema", "sources"} or raw["schema"] != SOURCE_SCHEMA:
            raise ValueError("invalid source manifest schema")
        sources = tuple(
            Source(
                name=item["name"], repository=item["repository"],
                revision=item["revision"], license=item["license"],
                include=tuple(item["include"]),
            )
            for item in raw["sources"]
        )
        for source in sources:
            source.validate()
        names = [source.name for source in sources]
        if names != sorted(set(names)) or set(names) != REQUIRED_SOURCE_NAMES:
            raise ValueError("manifest must contain all required sources exactly once")
        return cls(sources, hashlib.sha256(_canonical(raw)).hexdigest())


class TraceRecorder:
    """Append a complete redacted event stream and seal it with a hash chain."""

    def __init__(
        self, path: Path, *, trace_id: str, phase: str, round_id: str,
        operator_family: str, model: str, source_manifest_sha256: str,
        tools: Sequence[Mapping[str, object]] = (),
    ) -> None:
        for label, value in (("trace_id", trace_id), ("round_id", round_id),
                             ("operator_family", operator_family)):
            if not _SAFE_ID.fullmatch(value):
                raise ValueError(f"invalid {label}")
        if phase not in {"phase1", "phase2"}:
            raise ValueError("phase must be phase1 or phase2")
        if not _SHA256.fullmatch(source_manifest_sha256):
            raise ValueError("invalid source manifest digest")
        self.path = Path(path)
        self._rows: list[dict[str, object]] = []
        self._sealed = False
        self._append("header", {
            "trace_id": trace_id, "phase": phase, "round_id": round_id,
            "operator_family": operator_family, "model": model,
            "source_manifest_sha256": source_manifest_sha256,
            "tools": list(tools),
        })

    def _append(self, kind: str, payload: object) -> None:
        if self._sealed:
            raise ValueError("trace is already sealed")
        body = {
            "schema": TRACE_SCHEMA, "sequence": len(self._rows), "kind": kind,
            "payload": _redact(payload),
            "previous_sha256": self._rows[-1]["record_sha256"] if self._rows else None,
        }
        self._rows.append({**body, "record_sha256": _digest(body)})

    def message(
        self, role: str, content: str, *, weight: int | None = None,
        reasoning_content: str | None = None,
        tool_calls: Sequence[Mapping[str, object]] | None = None,
        tool_call_id: str | None = None, name: str | None = None,
    ) -> None:
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            raise ValueError("invalid conversation message")
        payload: dict[str, object] = {"role": role, "content": content}
        if weight is not None:
            if role != "assistant" or weight not in {0, 1}:
                raise ValueError("weight is only valid for assistant messages")
            payload["weight"] = weight
        if reasoning_content is not None:
            payload["reasoning_content"] = reasoning_content
        if tool_calls is not None:
            if role != "assistant" or not tool_calls:
                raise ValueError("tool_calls require an assistant message")
            payload["tool_calls"] = [dict(call) for call in tool_calls]
        if tool_call_id is not None or name is not None:
            if role != "tool" or not tool_call_id or not name:
                raise ValueError("tool_call_id and name require a tool result message")
            payload.update({"tool_call_id": tool_call_id, "name": name})
        self._append("message", payload)

    def tool_event(self, name: str, payload: Mapping[str, object]) -> None:
        if not _SAFE_ID.fullmatch(name):
            raise ValueError("invalid tool event name")
        self._append("tool", {"name": name, **dict(payload)})

    def candidate(self, source: str, *, candidate_id: str) -> None:
        if not _SAFE_ID.fullmatch(candidate_id) or not source.strip():
            raise ValueError("candidate requires an id and source")
        self._append("candidate", {
            "candidate_id": candidate_id, "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        })

    def validation(
        self, *, candidate_id: str, validator: str, status: str,
        evidence_sha256: str, details: Mapping[str, object] | None = None,
    ) -> None:
        if status not in {"passed", "failed", "infrastructure-failure"}:
            raise ValueError("invalid validation status")
        if not _SHA256.fullmatch(evidence_sha256):
            raise ValueError("invalid evidence digest")
        self._append("validation", {
            "candidate_id": candidate_id, "validator": validator, "status": status,
            "evidence_sha256": evidence_sha256, "details": dict(details or {}),
        })

    def finish(self) -> Path:
        if not any(row["kind"] == "validation" for row in self._rows):
            raise ValueError("cannot seal a trace without validation")
        self._append("terminal", {
            "event_count": len(self._rows),
            "events_sha256": _digest([row["record_sha256"] for row in self._rows]),
        })
        self._sealed = True
        _publish(self.path, b"".join(_canonical(row) + b"\n" for row in self._rows))
        return self.path


def _read_trace(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < 3:
        raise ValueError(f"incomplete trace: {path}")
    previous = None
    for sequence, row in enumerate(rows):
        digest = row.pop("record_sha256", None)
        if (row.get("schema") != TRACE_SCHEMA or row.get("sequence") != sequence
                or row.get("previous_sha256") != previous or digest != _digest(row)):
            raise ValueError(f"trace integrity failure: {path}")
        row["record_sha256"] = digest
        previous = digest
    if rows[-1]["kind"] != "terminal" or rows[-1]["payload"]["event_count"] != len(rows) - 1:
        raise ValueError(f"unsealed trace: {path}")
    expected_events = _digest([row["record_sha256"] for row in rows[:-1]])
    if rows[-1]["payload"]["events_sha256"] != expected_events:
        raise ValueError(f"terminal trace digest failure: {path}")
    return rows


def _normalized_candidate(source: str) -> str:
    no_comments = re.sub(r"#.*", "", source)
    return " ".join(no_comments.lower().split())


def _is_near_duplicate(value: str, retained: Sequence[str]) -> bool:
    """Collapse extremely close edits while retaining real kernel variants."""
    return any(SequenceMatcher(None, value, item).ratio() >= 0.985 for item in retained)


def _example(rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]] | None:
    header = rows[0]["payload"]
    validations = [row["payload"] for row in rows if row["kind"] == "validation"]
    candidates = [row["payload"] for row in rows if row["kind"] == "candidate"]
    if not validations or not candidates or validations[-1]["status"] != "passed":
        return None
    final = candidates[-1]
    if validations[-1]["candidate_id"] != final["candidate_id"]:
        return None
    messages = [dict(row["payload"]) for row in rows if row["kind"] == "message"]
    assistants = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if not assistants:
        return None
    for index in assistants:
        messages[index]["weight"] = 0
    messages[assistants[-1]]["content"] = final["source"]
    messages[assistants[-1]]["weight"] = 1
    example: dict[str, object] = {"messages": messages}
    if header.get("tools"):
        example["tools"] = header["tools"]
    metadata = {
        "trace_id": header["trace_id"], "round_id": header["round_id"],
        "operator_family": header["operator_family"],
        "candidate_sha256": final["source_sha256"],
        "repair": (
            any(item["status"] == "failed" for item in validations[:-1])
            or any(
                row["kind"] == "tool" and row["payload"].get("status") == "failed"
                for row in rows
            )
        ),
        "terminal_evidence_sha256": validations[-1]["evidence_sha256"],
        "normalized": _normalized_candidate(final["source"]),
    }
    return example, metadata


def build_dataset(
    traces: Path, output: Path, *, validation_fraction: float = 0.1,
) -> dict[str, object]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    eligible: list[tuple[dict[str, object], dict[str, object]]] = []
    rejected = 0
    duplicates = 0
    seen: set[str] = set()
    for path in sorted(Path(traces).glob("*.jsonl")):
        candidate = _example(_read_trace(path))
        if candidate is None:
            rejected += 1
            continue
        normalized = candidate[1]["normalized"]
        if normalized in seen or _is_near_duplicate(normalized, tuple(seen)):
            duplicates += 1
            continue
        seen.add(normalized)
        eligible.append(candidate)
    groups: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {}
    for item in eligible:
        groups.setdefault(str(item[1]["operator_family"]), []).append(item)
    train = []
    validation = []
    for family in sorted(groups):
        group = sorted(groups[family], key=lambda item: _digest(item[1]["trace_id"]))
        validation_count = max(1, round(len(group) * validation_fraction)) if len(group) > 1 else 0
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])
    train.sort(key=lambda item: str(item[1]["trace_id"]))
    validation.sort(key=lambda item: str(item[1]["trace_id"]))
    output = Path(output)
    train_bytes = b"".join(_canonical(item[0]) + b"\n" for item in train)
    validation_bytes = b"".join(_canonical(item[0]) + b"\n" for item in validation)
    _publish(output / "train.jsonl", train_bytes)
    _publish(output / "validation.jsonl", validation_bytes)
    counts = lambda items: {
        family: sum(item[1]["operator_family"] == family for item in items)
        for family in sorted(groups)
        if any(item[1]["operator_family"] == family for item in items)
    }
    body = {
        "schema": DATASET_SCHEMA, "eligible": len(eligible), "rejected": rejected,
        "duplicates": duplicates, "validation_fraction": validation_fraction,
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_bytes).hexdigest(),
        "split_counts_by_operator": {"train": counts(train), "validation": counts(validation)},
        "examples": [
            {k: v for k, v in item[1].items() if k != "normalized"}
            for item in train + validation
        ],
    }
    manifest = {**body, "dataset_sha256": _digest(body)}
    _publish(output / "manifest.json", _canonical(manifest) + b"\n")
    return manifest


def saturation_decision(
    novel_examples: Sequence[int], novel_repair_patterns: Sequence[int],
) -> dict[str, object]:
    if len(novel_examples) != len(novel_repair_patterns):
        raise ValueError("example and repair novelty rounds must align")
    if any(type(value) is not int or value < 0 for value in (*novel_examples, *novel_repair_patterns)):
        raise ValueError("novelty counts must be nonnegative integers")
    if len(novel_examples) < 3:
        return {"stop": False, "reason": "minimum-rounds"}
    if len(novel_examples) >= 5:
        return {"stop": True, "reason": "maximum-rounds"}
    def below(values: Sequence[int]) -> bool:
        prior, current = values[-2:]
        return current == 0 if prior == 0 else current / prior < 0.1

    if below(novel_examples) and below(novel_repair_patterns):
        return {"stop": True, "reason": "novelty-below-10-percent"}
    return {"stop": False, "reason": "continue"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-sources")
    validate.add_argument("manifest", type=Path)
    build = commands.add_parser("build")
    build.add_argument("--traces", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-sources":
        print(SourceManifest.load(args.manifest).sha256)
    else:
        print(json.dumps(build_dataset(args.traces, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
