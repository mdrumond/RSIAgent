"""Bounded, non-authoritative views over immutable agent evidence artifacts."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from collections.abc import Mapping

from benchmarks.a5kernels.evidence import canonical_bytes

PAGE_BYTES = 8192


class MemoryArtifactReader:
    """Read only memory-reachable historical objects; never admit host facts."""

    def __init__(self, request, current):
        from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
        from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory
        self.current, self.sources = current, {}
        try:
            envelope = json.loads(request.pre_wave_memory)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(envelope, dict) or "project_memory_base64" not in envelope:
            return
        root = Path(request.memory_root) if getattr(request, "memory_root", None) else Path(request.branch_root).parents[3]
        entries = [(base64.b64decode(row, validate=True), None)
                   for row in envelope["project_memory_base64"]]
        try:
            seed = json.loads(base64.b64decode(envelope["initial_memory_base64"], validate=True))
        except (ValueError, UnicodeDecodeError):
            seed = None
        if isinstance(seed, dict) and seed.get("schema") == "ascend-phase2-curriculum-seed-v1":
            from .phase2_curriculum import reviewed_seed
            authenticated, snapshot_root = reviewed_seed(root, seed["snapshot_id"])
            if json.loads(authenticated) != seed:
                raise ValueError("reviewed seed differs from current memory")
            entries += [(canonical_bytes(row), snapshot_root) for row in seed["phase1_memory"]]
        for raw, snapshot_root in entries:
            memory = Phase1ProjectMemory.from_mapping(json.loads(raw))
            if snapshot_root is not None:
                stores = [ContentAddressedArtifacts(snapshot_root / "artifacts")]
            else:
                stores = []
                if memory.project_spec is None:
                    continue
                project_id = memory.project_spec.project_id
                for delta_path in (root / "waves").glob(f"*/branches/{project_id}/memory-delta.bin"):
                    delta = json.loads(delta_path.read_bytes())
                    if base64.b64decode(delta["memory_base64"], validate=True) == raw:
                        stores.append(ContentAddressedArtifacts(
                            delta_path.parent / (memory.artifact_store or "artifacts")))
            digests = {memory.source_revision, *memory.admitted_artifact_sha256,
                       *(fact.evidence_sha256 for fact in memory.host_facts),
                       *(digest for lesson in memory.lessons for digest in lesson.evidence_artifact_sha256)}
            for digest in digests:
                self.sources.setdefault(digest, []).extend(stores)

    def read_bytes(self, digest):
        try:
            return self.current.read_bytes(digest)
        except FileNotFoundError:
            for store in self.sources.get(digest, ()):
                try:
                    return store.read_bytes(digest)
                except FileNotFoundError:
                    continue
            raise

    def put_json(self, value):
        return self.current.put_json(value)


def _preview(value, *, depth=0):
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if len(encoded) <= 2048:
            return value
        return {"prefix": encoded[:1536].decode("utf-8", errors="replace"),
                "suffix": encoded[-512:].decode("utf-8", errors="replace"),
                "bytes": len(encoded), "truncated": True}
    if isinstance(value, Mapping):
        if depth > 8:
            return {"omitted": True, "keys": list(value)[:16]}
        return {key: _preview(item, depth=depth + 1)
                for key, item in list(value.items())[:48]}
    if isinstance(value, (list, tuple)):
        if len(value) <= 16:
            return [_preview(item, depth=depth + 1) for item in value]
        result = {"count": len(value), "sample": [
            _preview(item, depth=depth + 1) for item in value[:4]], "truncated": True}
        numbers = [item for item in value if type(item) in (int, float)
                   and math.isfinite(item)]
        if len(numbers) == len(value):
            result.update(min=min(numbers), max=max(numbers))
        return result
    return value


def compact_observation(payload, artifacts):
    """Retain exact host bytes before presenting a bounded, hash-linked view.

    This view never replaces the host result used by verification or publication.
    """
    digest = artifacts.put_json(payload)
    if len(canonical_bytes(payload)) <= 12 * 1024:
        return dict(payload)
    preview = _preview(payload)
    artifact = preview.get("artifact")
    if isinstance(artifact, dict) and isinstance(artifact.get("payload"), dict):
        if artifact["payload"].get("compile_diagnostic") == preview.get("diagnostic"):
            artifact["payload"].pop("compile_diagnostic", None)
    # Rare wide reports are still accessible by hash, not recursively duplicated.
    if len(canonical_bytes(preview)) > 12 * 1024:
        preview = {key: _preview(payload[key]) for key in (
            "event", "source_sha256", "classification", "detail", "passed", "error",
            "diagnostic", "numerical_summary",
        ) if key in payload}
        for key in ("artifact", "artifacts"):
            if isinstance(payload.get(key), Mapping):
                value = payload[key]
                preview[key] = ({"sha256": value["sha256"]} if "sha256" in value else
                                {name: {"sha256": row["sha256"]} for name, row in value.items()
                                 if isinstance(row, Mapping) and "sha256" in row})
    if len(canonical_bytes(preview)) > 11 * 1024:
        preview = {key: _preview(payload[key]) for key in (
            "event", "source_sha256", "classification", "passed", "error", "diagnostic",
        ) if key in payload}
    preview["full_observation_sha256"] = digest
    preview["presentation_only"] = True
    return preview


def mismatch_summary(expected, observed):
    """Numerical repair clues only; authoritative tolerance remains the host's."""
    if observed is None:
        return None
    samples, maximum, unequal = [], 0.0, 0
    for index, (want, got) in enumerate(zip(expected, observed)):
        if want != got:
            unequal += 1
            error = abs(want - got) if type(got) in (int, float) else None
            if error is not None and math.isfinite(error):
                maximum = max(maximum, error)
            if len(samples) < 8:
                samples.append({"index": index, "expected": want, "observed": got,
                                "absolute_error": error if error is None or math.isfinite(error) else None})
    unequal += abs(len(expected) - len(observed))
    tail_start = min(len(expected), len(observed))
    tail_end = min(max(len(expected), len(observed)), tail_start + 8 - len(samples))
    for index in range(tail_start, tail_end):
        missing_side = "expected" if index >= len(expected) else "observed"
        samples.append({"index": index,
                        "expected": expected[index] if index < len(expected) else None,
                        "observed": observed[index] if index < len(observed) else None,
                        "missing_side": missing_side, "absolute_error": None})
    return {"expected_count": len(expected), "observed_count": len(observed),
            "unequal_count": unequal, "max_finite_absolute_error": maximum,
            "samples": samples, "comparison": "exact differences, not host tolerance"}


def memory_presentation(raw, artifacts):
    """Index every original memory; show readable bounded lessons, not base64."""
    digest = artifacts.put_bytes(raw)
    result = {"sha256": digest, "bytes": len(raw), "projects": [],
              "note": "Presentation only. Full immutable memories available via read-artifact."}
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        result["preview"] = raw[:2048].decode("utf-8", errors="replace")
        return result
    if isinstance(value, dict) and "usefulness_lesson" in value:
        result["usefulness_lesson"] = value["usefulness_lesson"]
    if isinstance(value, dict) and isinstance(value.get("memory_entries"), list):
        result["memory_entries"] = [dict(sha256=artifacts.put_bytes(canonical_bytes(entry)),
            lesson=_preview(entry)) for entry in value["memory_entries"]]
    if not isinstance(value, dict) or "project_memory_base64" not in value:
        result["preview"] = _preview(value)
        if len(canonical_bytes(result)) > 8192:
            result.pop("preview")
        return result
    if "phase2_feedback" in value:
        result["phase2_feedback"] = _preview(value["phase2_feedback"])
    initial = base64.b64decode(value["initial_memory_base64"], validate=True)
    result["initial_memory_sha256"] = artifacts.put_bytes(initial)
    try:
        seed = json.loads(initial)
    except (ValueError, UnicodeDecodeError):
        seed = None
    seed_entries = seed.get("phase1_memory", []) if isinstance(seed, dict) else []
    if not seed_entries:
        initial_preview = _preview(seed) if seed is not None else initial[:2048].decode("utf-8", errors="replace")
        if len(canonical_bytes(initial_preview)) <= 2048:
            result["initial_memory_preview"] = initial_preview
    entries = [(canonical_bytes(entry), "reviewed-phase1") for entry in seed_entries]
    entries.extend((base64.b64decode(encoded, validate=True), "progressive")
                   for encoded in value["project_memory_base64"])
    remaining = 8192
    optimizer_remaining = 8192
    for content, origin in entries:
        project = json.loads(content)
        row = {"sha256": artifacts.put_bytes(content),
               "origin": origin,
               "project_id": project.get("project_spec", {}).get("project_id", project.get("task_id"))}
        if project.get("task_id"):
            summary = {"host_fact": str(project.get("statement", ""))[:500],
                       "interpretation": str(project.get("interpretation", ""))[:800]}
            size = len(canonical_bytes(summary))
            if size <= optimizer_remaining:
                row.update(summary, origin="optimizer-phase2")
                optimizer_remaining -= size
        lessons = _preview(project.get("agent_lessons", []))
        size = len(canonical_bytes(lessons))
        if size <= remaining:
            row["lessons"] = lessons
            remaining -= size
        result["projects"].append(row)
    return result


def read_artifact(payload, artifacts):
    """Read up to 8 KiB from a hash-verified artifact, using byte offsets."""
    content = artifacts.read_bytes(payload["sha256"])
    offset, limit = payload.get("offset", 0), payload.get("limit", PAGE_BYTES)
    page = content[offset:offset + limit]
    # Base64 pages are exact even when a UTF-8 character crosses a byte boundary.
    return {"event": "artifact-page", "sha256": payload["sha256"],
            "offset": offset, "size_bytes": len(content),
            "next_offset": offset + len(page) if offset + len(page) < len(content) else None,
            "text": page.decode("utf-8", errors="replace"),
            "base64": base64.b64encode(page).decode("ascii")}


def artifact_page_presentation(payload, artifacts):
    """Readable Actor view; byte-exact pages remain available in the CAS.

    Valid UTF-8 pages end on a codepoint boundary. Binary content and offsets
    inside a codepoint return metadata, never an opaque encoded blob.
    """
    raw = read_artifact(payload, artifacts)
    content = artifacts.read_bytes(payload["sha256"])
    offset, limit = raw["offset"], payload.get("limit", PAGE_BYTES)
    view = {key: value for key, value in raw.items() if key not in {"text", "base64"}}
    try:
        content.decode("utf-8")
        content[:offset].decode("utf-8")
    except UnicodeDecodeError:
        view.update(encoding="binary", text_available=False,
                    note="No UTF-8 preview at this byte offset; exact bytes remain in the retained artifact.")
    else:
        page = content[offset:offset + limit]
        # Only a trailing partial codepoint can remain after the checks above.
        text = page.decode("utf-8", errors="ignore")
        consumed = len(text.encode("utf-8"))
        if page and not consumed:
            raise ValueError("page limit cannot fit the next UTF-8 codepoint; use limit >= 4")
        view.update(encoding="utf-8", text_available=True, text=text,
                    next_offset=offset + consumed if offset + consumed < len(content) else None)
        raw = read_artifact({**payload, "limit": consumed or limit}, artifacts)
    view["page_bytes"] = len(base64.b64decode(raw["base64"]))
    view["full_observation_sha256"] = artifacts.put_json(raw)
    return view
