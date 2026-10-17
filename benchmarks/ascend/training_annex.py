"""Read-only campaign evidence inventory and portable memory/evidence archives.

This module preserves retained records; it never manufactures training traces.
Remote report descriptors are consumed by a separate transport worker.
"""
from __future__ import annotations

import hashlib
import base64
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
    paths = sorted(root.rglob("*"))
    if root.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("report contains symlink")
    files = [p for p in paths if p.is_file()]
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


def _captures(value):
    if isinstance(value, dict):
        if isinstance(value.get("capture"), dict):
            yield value["capture"]
        for child in value.values():
            yield from _captures(child)
    elif isinstance(value, list):
        for child in value:
            yield from _captures(child)


def _report(target, remote_path, expected, origin, store):
    descriptor = {"target": target, "remote_path": remote_path, "expected_sha256": expected}
    descriptor["id"] = _sha(_json(descriptor))
    descriptor.update(record_origin=origin, kind="directory",
                      digest_algorithm="relative-path-sha256-v1",
                      digest_origin="capture-time" if expected else "unavailable",
                      status="missing" if expected else "digest-unavailable")
    local = store / descriptor["id"] if store else None
    if local and local.is_dir():
        try:
            actual = report_digest(local)
            receipt_path = store / (descriptor["id"] + ".receipt.json")
            if expected is None and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                if (receipt.get("descriptor_id") == descriptor["id"]
                        and receipt.get("digest_origin") == "collection-time"
                        and receipt.get("source_sha256") == actual
                        and receipt.get("downloaded_sha256") == actual):
                    descriptor.update(resolved_sha256=actual, digest_origin="collection-time")
                    expected = actual
            if expected:
                descriptor["status"] = "available" if actual == expected else "digest-mismatch"
        except (ValueError, OSError):
            descriptor["status"] = "digest-mismatch"
    if not target:
        descriptor["status"] = "incomplete-descriptor"
    return descriptor


def _verify_wave(root, *, closed_failure=False):
    from benchmarks.a5kernels.phase1_curriculum import (
        WaveSpec, ProjectOutcome, consolidate_wave,
    )
    read = lambda path: json.loads(path.read_bytes())
    wave = WaveSpec.from_mapping(read(root / "plan.json")["wave"])
    publication = read(root / ("closed-failure.json" if closed_failure else "published.json"))
    if closed_failure and len(wave.projects) != 1:
        raise ValueError("closed failure requires a single project")
    outcomes = []
    for assignment in wave.assignments():
        branch = root / "branches" / assignment.project_id
        outcome = read(branch / "outcome.json")
        if (outcome["wave_fingerprint"] != wave.fingerprint
                or outcome["project_id"] != assignment.project_id
                or outcome["authored_ordinal"] != assignment.authored_ordinal
                or outcome["pre_wave_memory_sha256"] != wave.pre_wave_memory_sha256
                or outcome["verified"] is not (not closed_failure)
                or outcome["verified"] != (outcome["classification"] == "host-verified")):
            raise ValueError("branch differs from committed wave")
        for name, field in (("memory-delta.bin", "memory_delta_sha256"),
                            ("execution-evidence.bin", "evidence_sha256")):
            if _sha((branch / name).read_bytes()) != outcome[field]:
                raise ValueError("committed branch digest mismatch")
        outcomes.append(ProjectOutcome(assignment.project_id, outcome["memory_delta_sha256"],
                                       wave.fingerprint, assignment.authored_ordinal, outcome["verified"]))
    if closed_failure:
        if (publication["wave_fingerprint"] != wave.fingerprint
                or publication["outcome_sha256"] != _sha((branch / "outcome.json").read_bytes())):
            raise ValueError("closed failure marker differs from retained outcome")
        return
    consolidation = consolidate_wave(wave, tuple(outcomes))
    from dataclasses import asdict
    consolidated = read(root / "consolidated-memory.json")
    expected_memory = {
        "schema": "a5-phase1-consolidated-memory-v1",
        "pre_wave_memory_sha256": wave.pre_wave_memory_sha256,
        "deltas": [{"project_id": commit.project_id,
                    "memory_delta_hex": (root / "branches" / commit.project_id
                                         / "memory-delta.bin").read_bytes().hex()}
                   for commit in consolidation.commits],
    }
    if (publication["wave_id"] != wave.wave_id
            or publication["wave_fingerprint"] != wave.fingerprint
            or publication["pre_wave_memory_sha256"] != wave.pre_wave_memory_sha256
            or publication["commits"] != [asdict(c) for c in consolidation.commits]
            or consolidated != expected_memory
            or publication["consolidated_memory_sha256"] != _sha(
                (root / "consolidated-memory.json").read_bytes())):
        raise ValueError("publication differs from retained committed evidence")


def _verify_memory_attempt(root, relative, memory):
    parent = Path(relative).parent / "attempts" / f"attempt-{memory['attempt']:02d}"
    attempt = json.loads((root / parent / "attempt.json").read_bytes())
    evidence = json.loads((root / parent / "evidence.json").read_bytes())
    source = next(root / parent / name for name in ("candidate.py", "candidate.cpp")
                  if (root / parent / name).is_file())
    for record, field in ((attempt, "attempt_sha256"), (evidence, "evidence_sha256")):
        if record[field] != _sha(_json({k: v for k, v in record.items() if k != field})):
            raise ValueError("committed attempt record digest mismatch")
    if (memory["attempt_identity_sha256"] != attempt["attempt_sha256"]
            or memory["evidence_sha256"] != evidence["evidence_sha256"]
            or memory["candidate_sha256"] != evidence["candidate_sha256"]
            or memory["candidate_sha256"] != _sha(source.read_bytes())
            or memory["passed"] is not evidence["passed"]):
        raise ValueError("committed memory differs from retained attempt")


def _project_memory(branch):
    """Decode adaptive deltas without treating legacy opaque deltas as JSON."""
    from benchmarks.a5kernels.phase1_adaptive import DELTA_SCHEMA
    from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory
    try:
        delta = json.loads((branch / "memory-delta.bin").read_bytes())
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(delta, dict) or delta.get("schema") != DELTA_SCHEMA:
        return None
    memory = Phase1ProjectMemory.from_mapping(json.loads(
        base64.b64decode(delta["memory_base64"], validate=True)))
    return delta, memory


def _verify_adaptive_state(root, state):
    """Bind the published state to ordered deltas and every pre-wave snapshot."""
    from benchmarks.a5kernels.phase1_adaptive import MEMORY_SCHEMA
    from benchmarks.a5kernels.phase1_curriculum import WaveSpec
    seed = root / "initial-memory.bin"
    initial = seed.read_bytes() if seed.is_file() else None
    if initial is not None and state["initial_memory_sha256"] != _sha(initial):
        raise ValueError("initial seed differs from adaptive state")
    memories, outcomes = [], []
    for wave_id in state["completed_waves"]:
        wave_root = root / "waves" / wave_id
        wave = WaveSpec.from_mapping(json.loads((wave_root / "plan.json").read_bytes())["wave"])
        if initial is not None:
            snapshot = {"schema": MEMORY_SCHEMA,
                        "initial_memory_base64": base64.b64encode(initial).decode("ascii"),
                        "project_memory_base64": memories}
            if wave.pre_wave_memory_sha256 != _sha(_json(snapshot)):
                raise ValueError("published pre-wave memory chain differs")
        for project in wave.projects:
            delta, memory = _project_memory(wave_root / "branches" / project.spec.project_id)
            if (delta["project_id"] != project.spec.project_id
                    or memory.project_spec != project.spec
                    or _json(memory.as_dict()) != base64.b64decode(delta["memory_base64"], validate=True)):
                raise ValueError("published memory differs from project")
            memories.append(delta["memory_base64"])
            outcomes.append({"project_id": project.spec.project_id,
                             "spec": project.spec.as_dict(), "verified": True,
                             "summary": delta["summary"]})
    if memories != state["memory_bytes_base64"] or outcomes != state["prior_outcomes"]:
        raise ValueError("adaptive state differs from ordered published deltas")


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
    identity_path = root / "campaign-identity.json"
    campaign_identity = json.loads(identity_path.read_bytes()) if identity_path.is_file() else {}
    fallback_target = (campaign_identity.get("execution_profile")
                       or campaign_identity.get("cell", {}).get("execution_profile"))
    for relative, path in _retained(root):
        data = path.read_bytes()
        files.append({"path": relative, "sha256": _sha(data), "size": len(data)})
        is_object = "objects" in path.parts
        if path.suffix not in {".json", ".jsonl"} and not is_object:
            continue
        try:
            rows = ([json.loads(line) for line in data.splitlines() if line.strip()]
                    if path.suffix == ".jsonl" else [json.loads(data)])
        except (ValueError, UnicodeDecodeError):
            if not is_object:
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
            found = []
            if isinstance(evidence, dict) and evidence.get("remote_report"):
                result = row.get("result", {})
                target = evidence.get("profile") or result.get("identity", {}).get("execution_profile")
                found.append(_report(target, evidence["remote_report"],
                                     result.get("report_sha256") or result.get("summary", {}).get("report_sha256"),
                                     origin, store))
            for capture in _captures(row):
                archive = capture.get("evidence", {})
                if archive.get("remote_tree"):
                    target = (capture.get("execution_profile") or archive.get("execution_profile")
                              or row.get("execution_profile") or fallback_target)
                    found.append(_report(target, archive["remote_tree"], None, origin, store))
            for descriptor in found:
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
        else:
            try:
                _verify_wave(root / "waves" / wave)
            except (OSError, ValueError, KeyError, TypeError):
                issues.append({"kind": "invalid-wave-publication", "wave": wave})
    from benchmarks.a5kernels.phase1_adaptive import STATE_SCHEMA
    if state.get("schema") == STATE_SCHEMA:
        # Phase 2 uses its separately authenticated reviewed snapshot as seed;
        # this file is the Phase 1 launcher's portable seed convention.
        if state.get("phase") != "adaptive-phase2" and not (root / "initial-memory.bin").is_file():
            issues.append({"kind": "missing-initial-memory"})
        try:
            _verify_adaptive_state(root, state)
        except (OSError, ValueError, KeyError, TypeError):
            issues.append({"kind": "invalid-adaptive-memory-chain"})
    for relative in documents:
        if relative.startswith("waves/") and relative.endswith("/outcome.json"):
            branch = (root / relative).parent
            try:
                decoded = _project_memory(branch)
                if decoded is not None:
                    _, memory = decoded
                    digests = {memory.source_revision, *memory.admitted_artifact_sha256,
                               *(fact.evidence_sha256 for fact in memory.host_facts)}
                    objects = branch / (memory.artifact_store or "artifacts") / "objects"
                    for digest in sorted(digests):
                        obj = objects / digest[:2] / digest[2:]
                        if not obj.is_file() or _sha(obj.read_bytes()) != digest:
                            issues.append({"kind": "missing-or-corrupt-object",
                                           "origin": relative, "sha256": digest})
            except (OSError, ValueError, KeyError, TypeError):
                issues.append({"kind": "invalid-project-memory", "origin": relative})
        if relative.startswith("waves/") and relative.endswith("/plan.json"):
            if relative.split("/")[1] not in state.get("completed_waves", []):
                wave_root = relative.rsplit("/", 1)[0]
                if wave_root + "/closed-failure.json" in documents:
                    try:
                        _verify_wave(root / wave_root, closed_failure=True)
                    except (OSError, ValueError, KeyError, TypeError):
                        issues.append({"kind": "invalid-closed-failure", "path": wave_root})
                else:
                    pending.append(wave_root)
    for relative in documents:
        if relative == "memory.jsonl" or relative.endswith("/memory.jsonl"):
            for memory in documents[relative]:
                if "attempt_identity_sha256" in memory:
                    try:
                        _verify_memory_attempt(root, relative, memory)
                    except (OSError, ValueError, KeyError, TypeError, StopIteration):
                        issues.append({"kind": "missing-or-corrupt-committed-attempt",
                                       "origin": relative, "attempt": memory.get("attempt")})
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
            expected_files = dict(snapshot.get("files", {}))
            for field, name in (("phase1_memory_sha256", "phase1-memory.jsonl"),
                                ("phase1_terminal_sha256", "phase1-terminal.json")):
                if field in snapshot:
                    expected_files[name] = snapshot[field]
            for name, expected in expected_files.items():
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
            packaged_reports = set()
            for report in manifest["remote_reports"]:
                if report["status"] != "available" or report["id"] in packaged_reports:
                    continue
                packaged_reports.add(report["id"])
                local = Path(artifact_store) / report["id"]
                captured_digest = hashlib.sha256()
                for path in sorted(p for p in local.rglob("*") if p.is_file()):
                    data = path.read_bytes()
                    captured_digest.update(path.relative_to(local).as_posix().encode() + b"\0")
                    captured_digest.update(hashlib.sha256(data).digest())
                    name = "reports/" + report["id"] + "/" + path.relative_to(local).as_posix()
                    raw_files.append({"path": name, "sha256": _sha(data), "size": len(data)})
                    add(name, data)
                if captured_digest.hexdigest() != (report["expected_sha256"] or report.get("resolved_sha256")):
                    raise ValueError("raw report changed during packaging")
                if report["digest_origin"] == "collection-time":
                    receipt = Path(artifact_store) / (report["id"] + ".receipt.json")
                    data = receipt.read_bytes()
                    name = "reports/" + receipt.name
                    raw_files.append({"path": name, "sha256": _sha(data), "size": len(data)})
                    add(name, data)
            current = inventory_campaign(root, artifact_store=artifact_store)
            if current != {k: v for k, v in manifest.items() if k != "package_kind"}:
                raise ValueError("campaign changed during packaging")
            manifest["raw_files"] = raw_files
            add("manifest.json", _json(manifest) + b"\n")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return manifest
