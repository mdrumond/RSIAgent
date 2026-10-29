"""Deterministic, secret-safe archives of historical A5 campaign evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Mapping

from .evidence import canonical_bytes, canonical_digest


_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SAFE_TREATMENT = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_SECRET_NAMES = re.compile(r"(^|/)(\.env(?:\..*)?|[^/]*\.(?:pem|key))$", re.I)
_SECRET_CONTENT = (
    re.compile(rb"(?:OPENAI|DEEPSEEK|ANTHROPIC)_API_KEY\s*[:=]\s*\S+", re.I),
    re.compile(rb"(?:password|access_token|secret_key)\s*[\"']?\s*[:=]\s*[\"']?\S+", re.I),
    re.compile(rb"(?:Bearer\s+|gh[pousr]_)[A-Za-z0-9_-]{16,}", re.I),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_SENSITIVE_WORDS = ("secret", "token", "password", "api_key", "credential")


class ArchiveError(RuntimeError):
    """The requested campaign cannot be archived without losing trust."""


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    phase: str
    root: Path
    model: str
    treatments: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MissingArtifact:
    scope: str
    artifact: str
    reason: str


@dataclass(frozen=True)
class ArchiveSpec:
    campaigns: tuple[Campaign, ...]
    known_missing: tuple[MissingArtifact, ...]


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _metadata_text(value: object, field: str) -> str:
    text = _text(value, field)
    encoded = text.encode("utf-8")
    if any(pattern.search(encoded) for pattern in _SECRET_CONTENT):
        raise ValueError(f"{field} contains secret-like metadata")
    return text


def load_archive_spec(value: Mapping[str, object]) -> ArchiveSpec:
    """Validate an untrusted JSON-like archive request."""

    if not isinstance(value, Mapping) or set(value) != {"campaigns", "known_missing"}:
        raise ValueError("archive spec must contain campaigns and known_missing")
    raw_campaigns = value["campaigns"]
    raw_missing = value["known_missing"]
    if not isinstance(raw_campaigns, list) or not raw_campaigns:
        raise ValueError("campaigns must be a non-empty array")
    if not isinstance(raw_missing, list):
        raise ValueError("known_missing must be an array")
    campaigns = []
    for item in raw_campaigns:
        required = {"id", "phase", "root", "model", "treatments"}
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError("each campaign must use the documented fields")
        campaign_id = _metadata_text(item["id"], "campaign id")
        phase = _text(item["phase"], "phase")
        model = _metadata_text(item["model"], "model")
        if not _ID.fullmatch(campaign_id) or phase not in {"phase1"}:
            raise ValueError("campaign id or phase is invalid")
        treatments = item["treatments"]
        if not isinstance(treatments, Mapping):
            raise ValueError("campaign treatments must be an object")
        normalized = []
        for key, treatment_value in treatments.items():
            if (
                not isinstance(key, str)
                or not _SAFE_TREATMENT.fullmatch(key)
                or any(word in key.lower() for word in _SENSITIVE_WORDS)
            ):
                raise ValueError("treatment name is invalid or sensitive")
            normalized.append((
                key, _metadata_text(treatment_value, "treatment value")
            ))
        campaigns.append(Campaign(
            campaign_id, phase, Path(_text(item["root"], "campaign root")).resolve(),
            model, tuple(sorted(normalized)),
        ))
    ids = [item.campaign_id for item in campaigns]
    if len(ids) != len(set(ids)):
        raise ValueError("campaign ids must be unique")
    missing = []
    for item in raw_missing:
        if not isinstance(item, Mapping) or set(item) != {"scope", "artifact", "reason"}:
            raise ValueError("known-missing declarations use scope, artifact, and reason")
        declaration = MissingArtifact(*(
            _metadata_text(item[field], f"known-missing {field}")
            for field in ("scope", "artifact", "reason")
        ))
        if declaration.scope != "all" and declaration.scope not in ids:
            raise ValueError("known-missing scope must name a campaign or all")
        missing.append(declaration)
    return ArchiveSpec(
        tuple(sorted(campaigns, key=lambda item: item.campaign_id)),
        tuple(sorted(missing, key=lambda item: (item.scope, item.artifact, item.reason))),
    )


def _contains_secret(relative: str, data: bytes) -> bool:
    if _SECRET_NAMES.search(relative):
        return True
    if b"\0" in data[:8192]:
        return False
    return any(pattern.search(data) for pattern in _SECRET_CONTENT)


def _read_campaign(campaign: Campaign) -> list[tuple[str, bytes, os.stat_result]]:
    if not campaign.root.is_dir():
        raise ArchiveError(f"campaign {campaign.campaign_id} is not a directory")
    result = []
    for path in sorted(campaign.root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(campaign.root).as_posix()
        if path.is_symlink():
            raise ArchiveError(f"campaign {campaign.campaign_id} contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArchiveError(f"campaign {campaign.campaign_id} contains a special file")
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity or len(data) != before.st_size:
            raise ArchiveError(f"campaign {campaign.campaign_id} changed during inventory")
        if _contains_secret(relative, data):
            raise ArchiveError(
                f"campaign {campaign.campaign_id} contains secret-like material"
            )
        result.append((relative, data, before))
    return result


def _json_values(path: str, data: bytes) -> list[object]:
    if path.endswith(".jsonl"):
        values = []
        for line in data.splitlines():
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        return values
    try:
        if path.endswith(".json"):
            return [json.loads(data)]
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    return []


def _interpretation_count(value: object) -> int:
    if isinstance(value, Mapping):
        count = 0
        for key, item in value.items():
            if key == "agent_interpretations" and isinstance(item, list):
                count += len(item)
            count += _interpretation_count(item)
        return count
    if isinstance(value, list):
        return sum(_interpretation_count(item) for item in value)
    return 0


_CLAIM_FIELDS = {"claim", "conclusion", "interpretation", "lesson", "statement", "summary"}
_CLAIM_WORDS = {
    "correctness": re.compile(r"\b(?:correct|match|pass(?:ed|es)?|verif(?:y|ied))\b", re.I),
    "performance": re.compile(
        r"\b(?:bandwidth|faster|latency|performance|speedup|throughput)\b", re.I
    ),
}

_ARTIFACT_LAYER_BY_ROLE = {
    "actor-instruction": "prompts",
    "actor-system-prompt": "prompts",
    "action-payload": "actions",
    "host-observation": "host_observations",
    "candidate-source": "source",
    "final-source": "source",
    "candidate-diff": "source",
    "final-profile": "profile",
    "performance-study": "profile",
    "agent-distillation": "memory",
    "knowledge-query": "knowledge_queries",
}

_ARTIFACT_LAYER_BY_FIELD = {
    "instruction_artifact_sha256": "prompts",
    "payload_artifact_sha256": "actions",
}


def _walk(value: object, pointer: str = ""):
    if isinstance(value, Mapping):
        yield pointer or "/", value
        for key, item in value.items():
            yield from _walk(item, f"{pointer}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{pointer}/{index}")


def _content_review(
    campaign: Campaign,
    files: list[tuple[str, bytes, os.stat_result]],
    inventory: list[dict[str, object]],
    known_missing: tuple[MissingArtifact, ...],
) -> dict[str, object]:
    names = (
        "prompts", "actions", "host_observations", "source", "profile",
        "memory", "knowledge_queries",
    )
    paths = {name: set() for name in names}
    records = {name: 0 for name in names}
    definitions = {item["sha256"] for item in inventory}
    references: list[str] = []
    referenced = {name: 0 for name in _CLAIM_WORDS}
    unsupported = {name: 0 for name in _CLAIM_WORDS}
    findings = []
    for relative, data, _stat in files:
        lower = relative.lower()
        base = PurePosixPath(lower).name
        if base.endswith((".py", ".cpp", ".cc", ".c", ".h", ".hpp")):
            paths["source"].add(relative)
        if "profile" in lower or "timing" in base or "msprof" in lower:
            paths["profile"].add(relative)
        if "memory" in base:
            paths["memory"].add(relative)
        for record_index, value in enumerate(_json_values(relative, data)):
            matched = set()
            for pointer, node in _walk(value):
                role = node.get("role")
                role_digest = node.get("sha256")
                if (
                    isinstance(role, str)
                    and isinstance(role_digest, str)
                    and (layer := _ARTIFACT_LAYER_BY_ROLE.get(role)) is not None
                ):
                    references.append(role_digest)
                    records[layer] += 1
                    matched.add(layer)
                for key, item in node.items():
                    if key in {"entry_sha256", "record_sha256"} and isinstance(item, str):
                        definitions.add(item)
                    if (
                        isinstance(item, str)
                        and key != "sha256"
                        and (key.endswith("evidence_sha256")
                             or key.endswith("artifact_sha256"))
                    ):
                        references.append(item)
                        if (layer := _ARTIFACT_LAYER_BY_FIELD.get(key)) is not None:
                            records[layer] += 1
                            matched.add(layer)
                    if key in {"prompt", "system_prompt", "instructions"}:
                        records["prompts"] += 1
                        matched.add("prompts")
                    if key == "action" and isinstance(item, str):
                        records["actions"] += 1
                        matched.add("actions")
                    if key == "actions" and isinstance(item, list):
                        records["actions"] += len(item)
                        matched.add("actions")
                    if key == "host_facts" and isinstance(item, list):
                        records["host_observations"] += len(item)
                        matched.add("host_observations")
                    if key == "kind" and item == "knowledge_query":
                        records["knowledge_queries"] += 1
                        matched.add("knowledge_queries")
                    if key not in _CLAIM_FIELDS or not isinstance(item, str):
                        continue
                    evidence_count = sum(
                        isinstance(v, str) for k, v in node.items()
                        if k.endswith("evidence_sha256")
                    )
                    for category, pattern in _CLAIM_WORDS.items():
                        if not pattern.search(item):
                            continue
                        target = referenced if evidence_count else unsupported
                        target[category] += 1
                        if not evidence_count:
                            findings.append({
                                "category": category,
                                "json_pointer": pointer + "/" + key,
                                "json_record": record_index,
                                "path": relative,
                                "statement_sha256": hashlib.sha256(item.encode()).hexdigest(),
                            })
            for name in matched:
                paths[name].add(relative)
    layers = {
        name: {
            "example_paths": sorted(paths[name])[:3],
            "file_count": len(paths[name]),
            "record_count": records[name],
        }
        for name in names
    }
    scoped_missing = [
        item.__dict__ for item in known_missing
        if item.scope in {"all", campaign.campaign_id}
    ]
    return {
        "campaign_id": campaign.campaign_id,
        "claims": {
            "with_evidence_reference": referenced,
            "unsupported": unsupported,
            "unsupported_findings": sorted(
                findings, key=lambda item: (item["path"], item["json_record"], item["json_pointer"])
            ),
        },
        "evidence_linkage": {
            "archive_resolved_count": sum(item in definitions for item in references),
            "reference_count": len(references),
            "unresolved_count": sum(item not in definitions for item in references),
        },
        "information_loss": {
            "absent_layers": [name for name in names if not paths[name]],
            "declared": scoped_missing,
        },
        "retained_layers": layers,
    }


def archive_baselines(spec: ArchiveSpec, destination: Path) -> dict[str, object]:
    """Create raw content objects plus a separate model-neutral derived audit."""

    destination = destination.resolve()
    if destination.exists():
        raise ArchiveError("archive destination already exists")
    for campaign in spec.campaigns:
        try:
            destination.relative_to(campaign.root)
        except ValueError:
            pass
        else:
            raise ArchiveError("archive destination cannot be inside a campaign")

    # Complete all trust and secret checks before creating the destination.
    captured = [(campaign, _read_campaign(campaign)) for campaign in spec.campaigns]
    inventory = []
    audits = []
    objects: dict[str, bytes] = {}
    for campaign, files in captured:
        campaign_inventory = []
        signals = {
            "candidate_sources": 0, "evidence_records": 0, "interpretations": 0,
            "json_records": 0, "memory_records": 0, "profile_artifacts": 0,
        }
        total_bytes = 0
        for relative, data, _stat in files:
            digest = hashlib.sha256(data).hexdigest()
            objects.setdefault(digest, data)
            inventory_item = {
                "campaign_id": campaign.campaign_id,
                "path": PurePosixPath(relative).as_posix(),
                "sha256": digest,
                "size": len(data),
            }
            inventory.append(inventory_item)
            campaign_inventory.append(inventory_item)
            total_bytes += len(data)
            values = _json_values(relative, data)
            signals["json_records"] += len(values)
            base = PurePosixPath(relative).name.lower()
            if "memory" in base and relative.endswith(".jsonl"):
                signals["memory_records"] += len(values)
            if "evidence" in base and relative.endswith((".json", ".jsonl")):
                signals["evidence_records"] += len(values)
            signals["interpretations"] += sum(_interpretation_count(v) for v in values)
            signals["candidate_sources"] += base in {"candidate.py", "kernel.py"}
            signals["profile_artifacts"] += "profile" in base
        audits.append({
            "campaign_id": campaign.campaign_id,
            "phase": campaign.phase,
            "model": campaign.model,
            "treatments": dict(campaign.treatments),
            "file_count": len(files),
            "byte_count": total_bytes,
            "inventory_sha256": canonical_digest(campaign_inventory),
            "signals": signals,
        })

    missing = [item.__dict__ for item in spec.known_missing]
    manifest = {
        "schema": "a5-baseline-content-archive-v1",
        "inventory": inventory,
        "known_missing": missing,
    }
    audit = {
        "schema": "a5-baseline-neutral-audit-v1",
        "campaigns": audits,
        "known_missing": missing,
    }
    reviews = []
    for campaign, files in captured:
        campaign_inventory = [
            item for item in inventory if item["campaign_id"] == campaign.campaign_id
        ]
        reviews.append(_content_review(campaign, files, campaign_inventory, spec.known_missing))
    review = {
        "schema": "a5-baseline-content-review-v1",
        "method": {
            "claims": "keyword-bearing claim fields; unsupported means no adjacent evidence_sha256",
            "evidence_linkage": "references resolve only to hashes retained in this archive",
            "information_loss": "declared losses plus structurally absent retained layers",
        },
        "campaigns": reviews,
    }
    for digest, data in sorted(objects.items()):
        target = destination / "raw/objects/sha256" / digest[:2] / digest[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for relative, value in (
        ("raw/manifest.json", manifest),
        ("derived/audit.json", audit),
        ("derived/review.json", review),
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_bytes(value) + b"\n")
    return manifest
