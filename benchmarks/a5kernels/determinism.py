"""Exactly-three-replay comparison for host-captured A5 evidence.

Three matching observations are evidence of repeatability for those attempts;
they never prove that an implementation is deterministic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from benchmarks.a5kernels.evidence import (
    EvidenceEntry,
    EvidenceKind,
    EvidenceLedger,
    canonical_bytes,
    canonical_digest,
)
from benchmarks.a5kernels.protocol import RunRequest


AUDITED_FIELDS = ("actions", "source_artifacts", "evidence", "output", "metrics")
_AUDIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}")


@dataclass(frozen=True)
class ReplaySnapshot:
    request_id: str
    attempt_id: str
    actions: tuple[Mapping[str, Any], ...]
    source_artifacts: tuple[Mapping[str, Any], ...]
    evidence: tuple[str, ...]
    output: Mapping[str, Any]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class FieldComparison:
    matches: bool
    unique_value_count: int
    divergent_pairs: int
    value_sha256_by_attempt: Mapping[str, str]


@dataclass(frozen=True)
class AuditReport:
    request_id: str
    run_count: int
    compared_attempt_ids: tuple[str, ...]
    fields_with_observed_divergence: tuple[str, ...]
    comparisons: Mapping[str, FieldComparison]
    interpretation: str = (
        "Three matching replays are consistent with repeatability for these "
        "attempts; they do not prove determinism."
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


ReplayCapture = Callable[[RunRequest, str], ReplaySnapshot]


@dataclass(frozen=True)
class MetricsPacket:
    """Supplemental host metrics bound to one verified ledger head."""

    request_id: str
    attempt_id: str
    ledger_head_sha256: str
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not all(isinstance(item, str) and item for item in (self.request_id, self.attempt_id)):
            raise ValueError("metrics packet identity must contain non-empty strings")
        if not re.fullmatch(r"[0-9a-f]{64}", self.ledger_head_sha256):
            raise ValueError("metrics packet ledger head must be a sha256 digest")
        if not isinstance(self.metrics, Mapping):
            raise ValueError("metrics packet metrics must be an object")
        reserved = {"max_abs_error", "max_abs_error_status", "exit_code"}
        collisions = reserved.intersection(self.metrics)
        if collisions:
            raise ValueError(
                "supplemental metrics use reserved ledger keys: "
                + ", ".join(sorted(collisions))
            )
        canonical_bytes(self.metrics)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MetricsPacket":
        required = {"request_id", "attempt_id", "ledger_head_sha256", "metrics"}
        if set(value) != required or not isinstance(value.get("metrics"), Mapping):
            raise ValueError("metrics packet has an invalid schema")
        return cls(
            request_id=value["request_id"],
            attempt_id=value["attempt_id"],
            ledger_head_sha256=value["ledger_head_sha256"],
            metrics=value["metrics"],
        )


def run_three_replays(
    request: RunRequest, capture: ReplayCapture, *, audit_id: str
) -> AuditReport:
    """Capture exactly three unique host-owned attempts, then compare them."""

    if not _AUDIT_ID.fullmatch(audit_id):
        raise ValueError("audit_id must be a safe 1-40 character identifier")
    snapshots = []
    for replay in range(1, 4):
        attempt_id = f"{audit_id}-replay-{replay}"
        snapshot = capture(request, attempt_id)
        if snapshot.request_id != request.request_id:
            raise ValueError("capture returned a different request_id")
        if snapshot.attempt_id != attempt_id:
            raise ValueError("capture returned a different attempt_id")
        snapshots.append(snapshot)
    return audit_replays(snapshots)


def snapshot_from_ledger(
    entries: Sequence[EvidenceEntry], *, metrics: MetricsPacket | None = None
) -> ReplaySnapshot:
    """Create one comparison unit from a verified single-attempt ledger."""

    EvidenceLedger.verify(entries)
    by_kind = {
        kind.value: [entry for entry in entries if entry.kind == kind.value]
        for kind in EvidenceKind
    }
    requests = by_kind[EvidenceKind.REQUEST.value]
    actions = by_kind[EvidenceKind.ACTION.value]
    artifacts = by_kind[EvidenceKind.ARTIFACT.value]
    results = by_kind[EvidenceKind.RESULT.value]
    if len(requests) != 1 or len(results) != 1:
        raise ValueError("a replay requires exactly one request and one result entry")
    if len(actions) != 1:
        raise ValueError("a replay requires exactly one action entry")
    if not artifacts:
        raise ValueError("a replay requires at least one artifact entry")
    request = requests[0].payload
    result = results[0].payload
    request_id = str(request.get("request_id", ""))
    attempt_id = str(request.get("attempt_id", ""))
    if not request_id or not attempt_id:
        raise ValueError("request evidence is missing replay identity")
    if any(
        entry.payload.get("request_id") != request_id
        or entry.payload.get("attempt_id") != attempt_id
        for entry in entries
    ):
        raise ValueError("ledger entries must belong to one request attempt")

    ledger_head = entries[-1].entry_sha256 if entries else ""
    if metrics is not None and (
        metrics.request_id != request_id
        or metrics.attempt_id != attempt_id
        or metrics.ledger_head_sha256 != ledger_head
    ):
        raise ValueError("metrics packet does not match the replay ledger identity")
    observed_metrics = {
        key: result.get(key)
        for key in ("max_abs_error", "max_abs_error_status", "exit_code")
        if key in result
    }
    snapshot = ReplaySnapshot(
        request_id=request_id,
        attempt_id=attempt_id,
        actions=tuple(
            {
                key: value
                for key, value in entry.payload.items()
                if key not in ("request_id", "attempt_id")
            }
            for entry in by_kind[EvidenceKind.ACTION.value]
        ),
        source_artifacts=tuple(
            {
                "sequence": entry.sequence,
                "source_sha256": entry.payload.get("source_sha256", {}),
                "artifact_sha256": entry.payload.get("artifact_sha256", {}),
            }
            for entry in by_kind[EvidenceKind.ARTIFACT.value]
        ),
        evidence=tuple(
            canonical_digest(
                {
                    "kind": entry.kind,
                    "payload": {
                        key: value
                        for key, value in entry.payload.items()
                        if key
                        not in {
                            "request_id",
                            "attempt_id",
                            "attestation_sha256",
                            "evidence_sha256",
                            "session_handle",
                        }
                    },
                }
            )
            for entry in entries
        ),
        output={
            key: result.get(key)
            for key in ("output_sha256", "passed")
            if key in result
        },
        metrics={
            "verified": observed_metrics,
            "supplemental": dict(metrics.metrics) if metrics else {},
        },
    )
    canonical_bytes(asdict(snapshot))
    return snapshot


def audit_replays(snapshots: Sequence[ReplaySnapshot]) -> AuditReport:
    """Compare exactly three attempts without converting equality into proof."""

    if len(snapshots) != 3:
        raise ValueError("a determinism audit requires exactly three replays")
    request_ids = {snapshot.request_id for snapshot in snapshots}
    if len(request_ids) != 1:
        raise ValueError("replay attempts must share one request_id")
    attempt_ids = tuple(snapshot.attempt_id for snapshot in snapshots)
    if len(set(attempt_ids)) != 3:
        raise ValueError("the three attempt_id values must be unique")

    comparisons: dict[str, FieldComparison] = {}
    for field in AUDITED_FIELDS:
        digests = tuple(
            canonical_digest(getattr(snapshot, field)) for snapshot in snapshots
        )
        divergent_pairs = sum(
            left != right
            for index, left in enumerate(digests)
            for right in digests[index + 1 :]
        )
        comparisons[field] = FieldComparison(
            matches=divergent_pairs == 0,
            unique_value_count=len(set(digests)),
            divergent_pairs=divergent_pairs,
            value_sha256_by_attempt=dict(zip(attempt_ids, digests)),
        )
    return AuditReport(
        request_id=next(iter(request_ids)),
        run_count=3,
        compared_attempt_ids=attempt_ids,
        fields_with_observed_divergence=tuple(
            field for field in AUDITED_FIELDS if not comparisons[field].matches
        ),
        comparisons=comparisons,
    )


def _load_metrics(path: str) -> MetricsPacket:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: metrics must be a JSON object")
    return MetricsPacket.from_mapping(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledgers", nargs=3, help="exactly three evidence JSONL files")
    parser.add_argument("--metrics", nargs=3, metavar=("ONE", "TWO", "THREE"))
    parser.add_argument("--output", help="write JSON report here instead of stdout")
    args = parser.parse_args(argv)
    metrics = [_load_metrics(path) for path in args.metrics] if args.metrics else [None] * 3
    snapshots = [
        snapshot_from_ledger(EvidenceLedger(path).entries, metrics=measurement)
        for path, measurement in zip(args.ledgers, metrics)
    ]
    rendered = json.dumps(
        audit_replays(snapshots).as_dict(), sort_keys=True, separators=(",", ":")
    ) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
