"""Reviewed adaptive memory handoff, without manufacturing a legacy journal."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_adaptive import MEMORY_SCHEMA, STATE_SCHEMA
from benchmarks.a5kernels.phase1_curriculum import WaveSpec
from benchmarks.a5kernels.phase1_experiments import build_cells
from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory
from benchmarks.a5kernels.phase1_wave_runtime import LiveWaveCoordinator
from benchmarks.a5kernels.phase2_memory import (
    ADAPTIVE_CELL_SCHEMA, A5Phase1Snapshot, A5Phase2Cell, _cell_id, _publish_exact,
)


SCHEMA = "a5-phase2-adaptive-snapshot-v1"
REVIEW_SCHEMA = "a5-phase1-usefulness-review-v1"


def _read(path):
    return json.loads(Path(path).read_bytes())


def _memory(state, initial):
    if (state.get("schema") != STATE_SCHEMA
            or state.get("initial_memory_sha256") != hashlib.sha256(initial).hexdigest()
            or not state.get("terminal")
            or state["terminal"].get("status") not in {"SATURATED", "PROJECT_LIMIT_REACHED"}):
        raise ValueError("adaptive Phase 1 must be complete with matching initial memory")
    memories = tuple(base64.b64decode(value, validate=True)
                     for value in state["memory_bytes_base64"])
    outcomes = state["prior_outcomes"]
    if not memories or len(memories) != len(outcomes):
        raise ValueError("adaptive Phase 1 requires published projects")
    for payload, outcome in zip(memories, outcomes, strict=True):
        memory = Phase1ProjectMemory.from_mapping(json.loads(payload))
        if (canonical_bytes(memory.as_dict()) != payload
                or memory.project_spec is None
                or outcome.get("verified") is not True
                or outcome.get("spec") != memory.project_spec.as_dict()
                or outcome.get("project_id") != memory.project_spec.project_id):
            raise ValueError("adaptive memory differs from verified published projects")
    snapshot = canonical_bytes({
        "schema": MEMORY_SCHEMA,
        "initial_memory_base64": base64.b64encode(initial).decode("ascii"),
        "project_memory_base64": state["memory_bytes_base64"],
    })
    return memories, hashlib.sha256(snapshot).hexdigest()


def adaptive_destination(source, review):
    identity = {
        "schema": ADAPTIVE_CELL_SCHEMA, "target": source.target.value,
        "language": source.language.value, "source_cell_id": source.cell_id,
        "model": source.model.as_dict(), "knowledge": source.knowledge.value,
        "profiling": source.profiling.value, "guide": source.guide.as_dict(),
        "catlass_revision": source.guide.catlass_revision,
        "qualification_sha256": canonical_digest(review),
    }
    from benchmarks.a5kernels.phase2_attention import SUITE_SHA256
    identity["suite_sha256"] = SUITE_SHA256
    if source.profiling_guide is not None:
        identity["profiling_guide"] = source.profiling_guide.as_dict()
    return A5Phase2Cell(
        _cell_id(identity), source.cell_id, source.model, source.knowledge,
        source.profiling, source.guide, canonical_digest(review),
        schema=ADAPTIVE_CELL_SCHEMA, profiling_guide=source.profiling_guide,
    )


def _published_evidence(root, state, initial):
    """Reuse the coordinator's read-only publication checks before handoff."""
    files, published = {}, []
    for wave_id in state["completed_waves"]:
        wave_root = root / "waves" / wave_id
        wave = WaveSpec.from_mapping(_read(wave_root / "plan.json")["wave"])
        pre_memory = canonical_bytes({
            "schema": MEMORY_SCHEMA,
            "initial_memory_base64": base64.b64encode(initial).decode("ascii"),
            "project_memory_base64": published,
        })
        reader = LiveWaveCoordinator(wave_root, wave, pre_memory, lambda _: None, 1)
        reader.read_published_memory()  # authenticates outcomes and both evidence digests
        for name in ("plan.json", "published.json", "consolidated-memory.json"):
            files[f"waves/{wave_id}/{name}"] = (wave_root / name).read_bytes()
        for project in wave.projects:
            branch = wave_root / "branches" / project.spec.project_id
            delta = _read(branch / "memory-delta.bin")
            memory = Phase1ProjectMemory.from_mapping(json.loads(base64.b64decode(delta["memory_base64"])))
            evidence = _read(branch / "execution-evidence.bin")
            if (delta.get("project_id") != project.spec.project_id
                    or memory.project_spec != project.spec
                    or evidence.get("classification") != "host-verified"
                    or evidence.get("source_sha256") != memory.source_revision
                    or evidence.get("artifacts", {}).get("classification") not in {
                        fact.evidence_sha256 for fact in memory.host_facts
                        if fact.category == "host-verification"}):
                raise ValueError("published adaptive host evidence differs from memory")
            published.append(delta["memory_base64"])
            for name in ("outcome.json", "memory-delta.bin", "execution-evidence.bin"):
                files[f"waves/{wave_id}/branches/{project.spec.project_id}/{name}"] = (branch / name).read_bytes()
    if published != state["memory_bytes_base64"]:
        raise ValueError("adaptive state differs from published branch memory")
    return files


class AdaptivePhase1Snapshot(A5Phase1Snapshot):
    """Snapshot understood by the existing bounded Phase 2 runner."""

    def verify(self):
        manifest = dict(self.manifest)
        digest = manifest.pop("snapshot_id", None)
        if (manifest.get("schema") != SCHEMA or digest != canonical_digest(manifest)
                or manifest["destination_identity"] != self.destination.as_dict()):
            raise ValueError("invalid adaptive snapshot identity")
        for relative, expected in manifest["files"].items():
            if hashlib.sha256((self.root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("adaptive snapshot bytes changed")
        state = _read(self.root / "state.json")
        memories, memory_sha = _memory(state, (self.root / "initial-memory.bin").read_bytes())
        _published_evidence(self.root, state, (self.root / "initial-memory.bin").read_bytes())
        review = _read(self.root / "review.json")
        _review(review, memory_sha, self.destination.source_cell_id)
        if canonical_digest(review) != self.destination.qualification_sha256:
            raise ValueError("adaptive review identity changed")
        identity = _read(self.root / "campaign-identity.json")
        sources = build_cells(self.destination.guide,
                              profiling_guide=self.destination.profiling_guide)
        source = next(item for item in sources
                      if item.cell_id == self.destination.source_cell_id)
        if identity.get("cell") != source.as_dict():
            raise ValueError("adaptive snapshot treatment changed")
        if self.memory_path.read_bytes() != b"".join(row + b"\n" for row in memories):
            raise ValueError("adaptive snapshot memory changed")

    def memory_entries(self):
        self.verify()
        return tuple(json.loads(row) for row in self.memory_path.read_bytes().splitlines())


def _review(review, memory_sha, cell_id):
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("memory_sha256") != memory_sha
            or review.get("cell_id") != cell_id
            or any(review.get(key) is not True for key in (
                "accepted", "useful", "diverse", "profiler_guidance_observed"))
            or not isinstance(review.get("rationale"), str)
            or not review["rationale"].strip()):
        raise ValueError("adaptive memory needs an accepted usefulness/diversity review")


def admit_adaptive_snapshot(source_root, initial_memory, review_path, source, phase2_root):
    """Copy exact published memory and all explicitly cited project evidence."""
    source_root = Path(source_root)
    if (source_root / "pending.json").exists():
        raise ValueError("adaptive Phase 1 has a pending wave")
    state_bytes = (source_root / "state.json").read_bytes()
    state = json.loads(state_bytes)
    memories, memory_sha = _memory(state, initial_memory)
    review_bytes = Path(review_path).read_bytes()
    review = json.loads(review_bytes)
    _review(review, memory_sha, source.cell_id)
    destination = adaptive_destination(source, review)
    files = {
        "state.json": state_bytes,
        "initial-memory.bin": initial_memory,
        "review.json": review_bytes,
        "campaign-identity.json": (source_root / "campaign-identity.json").read_bytes(),
        "phase1-memory.jsonl": b"".join(row + b"\n" for row in memories),
    }
    files.update(_published_evidence(source_root, state, initial_memory))
    for payload in memories:
        memory = Phase1ProjectMemory.from_mapping(json.loads(payload))
        project_id = memory.project_spec.project_id
        branches = [source_root / "waves" / wave / "branches" / project_id
                    for wave in state["completed_waves"]]
        branches = [branch for branch in branches if branch.is_dir()]
        if len(branches) != 1:
            raise ValueError("published adaptive project evidence is unavailable")
        artifacts = ContentAddressedArtifacts(branches[0] / "artifacts")
        digests = set(memory.admitted_artifact_sha256)
        digests.add(memory.source_revision)
        digests.update(fact.evidence_sha256 for fact in memory.host_facts)
        for digest in digests:
            files[f"artifacts/objects/{digest[:2]}/{digest[2:]}"] = artifacts.read_bytes(digest)
    body = {
        "schema": SCHEMA, "destination_identity": destination.as_dict(),
        "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
    }
    manifest = {**body, "snapshot_id": canonical_digest(body)}
    root = Path(phase2_root).resolve() / "cells" / destination.cell_id / "snapshot"
    for relative, payload in files.items():
        _publish_exact(root / relative, payload)
    _publish_exact(root / "snapshot.json", canonical_bytes(manifest) + b"\n")
    snapshot = AdaptivePhase1Snapshot(root, destination, manifest)
    snapshot.verify()
    return snapshot
