"""Reviewed Catlass Phase 1 memory, independent of any learning stage."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_experiments import build_cells
from .reviewed_memory import _read, _memory, _published_evidence, _review, collect_adaptive_snapshot, _publish_exact
SCHEMA = "catlass-reviewed-memory-snapshot-v1"
REVIEW_SCHEMA = "a5-phase1-usefulness-review-v1"

@dataclass(frozen=True)
class CatlassReviewedIdentity:
    source: object
    qualification_sha256: str

    @property
    def source_cell_id(self): return self.source.cell_id
    @property
    def cell_id(self): return "reviewed-memory-" + canonical_digest(self.as_dict())[:16]
    @property
    def guide(self): return self.source.guide
    @property
    def profiling_guide(self): return self.source.profiling_guide
    def as_dict(self):
        return dict(schema="catlass-reviewed-memory-identity-v1", target="a5", language="catlass-dsl",
                    source_cell_id=self.source.cell_id, source_identity=self.source.as_dict(),
                    qualification_sha256=self.qualification_sha256)

def adaptive_destination(source, review):
    return CatlassReviewedIdentity(source, canonical_digest(review))

@dataclass(frozen=True)
class ReviewedSnapshot:
    root: Path
    destination: CatlassReviewedIdentity
    manifest: dict
    @property
    def memory_path(self): return self.root / "phase1-memory.jsonl"
    @property
    def snapshot_id(self): return self.manifest["snapshot_id"]

class AdaptivePhase1Snapshot(ReviewedSnapshot):
    """Authenticated copy of reviewed Catlass learning."""

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
        _review(review, memory_sha, self.destination.source_cell_id, schema=REVIEW_SCHEMA)
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

def admit_adaptive_snapshot(source_root, initial_memory, review_path, source, snapshot_root):
    """Copy exact published memory and all explicitly cited project evidence."""
    files, review, memory_sha = collect_adaptive_snapshot(source_root, initial_memory, review_path)
    _review(review, memory_sha, source.cell_id, schema=REVIEW_SCHEMA)
    destination = adaptive_destination(source, review)
    body = {
        "schema": SCHEMA, "destination_identity": destination.as_dict(),
        "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
    }
    manifest = {**body, "snapshot_id": canonical_digest(body)}
    root = Path(snapshot_root).resolve() / "cells" / destination.cell_id / "snapshot"
    for relative, payload in files.items():
        _publish_exact(root / relative, payload)
    _publish_exact(root / "snapshot.json", canonical_bytes(manifest) + b"\n")
    snapshot = AdaptivePhase1Snapshot(root, destination, manifest)
    snapshot.verify()
    return snapshot
