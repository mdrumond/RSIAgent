"""Read-only campaign evidence inventory and portable memory/evidence archives.

This module preserves retained records; it never manufactures training traces.
Remote report descriptors are consumed by a separate transport worker.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def report_digest(root):
    """Digest a raw msprof directory using the native profiler's convention."""
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError("empty raw report")
    for path in files:
        if path.is_symlink():
            raise ValueError("report contains symlink")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


_EXCLUDED = {".git", "__pycache__", "reference_repos", "weights", "private",
             "profiles", "private-config", ".ssh", "credentials.json",
             "credentials", "secrets.json"}
_ROOT_FILES = {"state.json", "pending.json", "campaign-identity.json",
               "campaign-run.json",
               "project-evidence.jsonl", "initial-memory.bin", "memory.jsonl",
               "terminal.json", "protocol.json", "review.json", "snapshot.json",
               "phase1-memory.jsonl", "phase1-terminal.json"}
_TREES = {"waves", "native-evidence", "measurements", "cells", "attempts",
          "provider", "provider-evidence", "source-evidence", "artifacts", "traces",
          "sft-traces", "snapshot"}


def _retained(root):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _EXCLUDED or part.startswith(".env") for part in relative.parts):
            continue
        if relative.parts[0] not in _TREES and relative.as_posix() not in _ROOT_FILES:
            continue
        if path.is_symlink():
            raise ValueError(f"campaign contains symlink: {relative}")
        if path.is_file():
            yield relative.as_posix(), path


def inventory_campaign(root, *, artifact_store=None):
    """Inventory every retained attempt, CAS object and measurement without writes.

    Paths in ``files`` and ``records`` are campaign-relative. Remote report
    locations are provenance descriptors, never local archive destinations.
    ``complete`` means a committed boundary with all referenced local/remote
    evidence available; it does not assert that a campaign reached its budget.
    """
    root = Path(root).resolve(strict=True)
    store = Path(artifact_store).resolve() if artifact_store is not None else None
    files, records, reports, issues = [], [], [], []
    documents = {}
    for relative, path in _retained(root):
        data = path.read_bytes()
        files.append({"path": relative, "sha256": _sha(data), "size": len(data)})
        if path.suffix not in {".json", ".jsonl"}:
            continue
        try:
            rows = ([json.loads(line) for line in data.splitlines() if line.strip()]
                    if path.suffix == ".jsonl" else [json.loads(data)])
        except (ValueError, UnicodeDecodeError):
            issues.append({"kind": "invalid-record", "path": relative})
            continue
        documents[relative] = rows
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            origin = relative + (f":{index + 1}" if path.suffix == ".jsonl" else "")
            records.append({"origin": origin, "schema": row.get("schema"),
                            "kind": row.get("kind", row.get("stage")),
                            "classification": row.get("classification"),
                            "project_id": row.get("project_id"),
                            "attempt_id": row.get("attempt_id")})
            if (row.get("kind") == "artifact" and isinstance(row.get("payload"), dict)
                    and "sha256" in row["payload"]):
                digest = row["payload"].get("sha256", "")
                objects = path.parent / "artifacts" / "objects"
                obj = objects / digest[:2] / digest[2:]
                if not obj.is_file() or _sha(obj.read_bytes()) != digest:
                    issues.append({"kind": "missing-or-corrupt-object", "origin": origin,
                                   "sha256": digest})
            evidence = row.get("evidence", {})
            if not isinstance(evidence, dict) or not evidence.get("remote_report"):
                continue
            result = row.get("result", {})
            identity = result.get("identity", {})
            descriptor = {"target": evidence.get("profile") or identity.get("execution_profile"),
                          "remote_path": evidence["remote_report"],
                          "expected_sha256": result.get("summary", {}).get("report_sha256")}
            descriptor["id"] = _sha(_json(descriptor))
            descriptor.update(record_origin=origin, kind="directory",
                              digest_algorithm="relative-path-sha256-v1")
            local = store / descriptor["id"] if store else None
            descriptor["status"] = "missing"
            if local and local.is_dir():
                try:
                    actual = report_digest(local)
                    descriptor["status"] = ("available" if actual == descriptor["expected_sha256"]
                                            else "digest-mismatch")
                except ValueError:
                    descriptor["status"] = "digest-mismatch"
            if not descriptor["target"] or not descriptor["expected_sha256"]:
                descriptor["status"] = "incomplete-descriptor"
            if descriptor["status"] != "available":
                issues.append({"kind": "raw-report-" + descriptor["status"],
                               "origin": origin, "id": descriptor["id"]})
            reports.append(descriptor)

    state = documents.get("state.json", [{}])[0]
    identity = documents.get("campaign-identity.json", [{}])[0]
    if not state and not any(p.endswith("protocol.json") for p in documents):
        issues.append({"kind": "missing-campaign-state"})
    pending = [p for p in documents if p.endswith("pending.json")]
    # Phase 1 publications are the memory commit boundary, not mere outcomes.
    for wave in state.get("completed_waves", []):
        if f"waves/{wave}/published.json" not in documents:
            issues.append({"kind": "missing-wave-publication", "wave": wave})
    for relative in documents:
        if relative.startswith("waves/") and relative.endswith("/plan.json"):
            if relative.split("/")[1] not in state.get("completed_waves", []):
                pending.append(relative.rsplit("/", 1)[0])
    for relative in documents:
        if relative.endswith("attempt.json"):
            parent = relative.rsplit("/", 1)[0]
            cell = parent.split("/attempts/", 1)[0] if "/attempts/" in parent else ""
            memories = documents.get((cell + "/" if cell else "") + "memory.jsonl", [])
            attempt = documents[relative][0]
            evidence = documents.get(parent + "/evidence.json", [{}])[0]
            committed = any(m.get("attempt_identity_sha256") == attempt.get("attempt_sha256")
                            and m.get("evidence_sha256") == evidence.get("evidence_sha256")
                            and m.get("attempt_identity_sha256") for m in memories)
            if not evidence or not committed:
                pending.append(parent)
            source = next((root / parent / name for name in ("candidate.py", "candidate.cpp")
                           if (root / parent / name).is_file()), None)
            expected = evidence.get("candidate_sha256")
            if expected and (source is None or _sha(source.read_bytes()) != expected):
                issues.append({"kind": "missing-or-corrupt-source", "origin": relative})
        if relative.endswith("snapshot.json"):
            snapshot = documents[relative][0]
            for name, expected in snapshot.get("files", {}).items():
                path = root / Path(relative).parent / name
                if not path.is_relative_to(root) or not path.is_file() or _sha(path.read_bytes()) != expected:
                    issues.append({"kind": "missing-or-corrupt-snapshot-file", "path": name})
    if pending:
        issues.append({"kind": "pending-boundary", "paths": sorted(pending)})
    return {"schema": "ascend-training-annex-v1", "identity": identity,
            "boundary": {"state_sha256": next((f["sha256"] for f in files
                         if f["path"] == "state.json"), None),
                         "completed_waves": state.get("completed_waves", []),
                         "pending": sorted(pending)},
            "files": files, "records": records, "remote_reports": reports,
            "issues": issues, "complete": not issues,
            "inventory_sha256": _sha(_json(files))}


def package_campaign(root, destination, *, artifact_store=None, allow_pending=False):
    """Create a new tar.gz containing memory and evidence at one frozen boundary.

    Default rejects incomplete evidence. ``allow_pending`` explicitly creates a
    checkpoint listing every gap. Concurrent changes always abort publication.
    """
    root, destination = Path(root).resolve(strict=True), Path(destination).resolve()
    if destination.is_relative_to(root):
        raise ValueError("archive destination must be outside campaign")
    manifest = inventory_campaign(root, artifact_store=artifact_store)
    if not manifest["complete"] and not allow_pending:
        raise ValueError("campaign is incomplete; inspect inventory issues or allow_pending")
    manifest["package_kind"] = "complete" if manifest["complete"] else "pending-checkpoint"
    # Write exclusively, deleting only our own partial output on failure.
    output = destination.open("xb")
    try:
        with output, tarfile.open(fileobj=output, mode="w:gz") as archive:
            def add(name, data):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(data))

            for entry in manifest["files"]:
                data = (root / entry["path"]).read_bytes()
                if _sha(data) != entry["sha256"]:
                    raise ValueError("campaign changed during packaging")
                add("campaign/" + entry["path"], data)
            raw_files = []
            for report in manifest["remote_reports"]:
                if report["status"] != "available":
                    continue
                local = Path(artifact_store) / report["id"]
                captured_digest = hashlib.sha256()
                for path in sorted(p for p in local.rglob("*") if p.is_file()):
                    data = path.read_bytes()
                    captured_digest.update(path.relative_to(local).as_posix().encode() + b"\0")
                    captured_digest.update(hashlib.sha256(data).digest())
                    name = "reports/" + report["id"] + "/" + path.relative_to(local).as_posix()
                    raw_files.append({"path": name, "sha256": _sha(data), "size": len(data)})
                    add(name, data)
                if captured_digest.hexdigest() != report["expected_sha256"]:
                    raise ValueError("raw report changed during packaging")
            current = inventory_campaign(root, artifact_store=artifact_store)
            if current != {k: v for k, v in manifest.items() if k != "package_kind"}:
                raise ValueError("campaign changed during packaging")
            manifest["raw_files"] = raw_files
            add("manifest.json", _json(manifest) + b"\n")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return manifest
