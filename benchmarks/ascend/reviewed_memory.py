"""Authenticated reviewed Phase 1 memory snapshots for current workflows."""
from dataclasses import dataclass
import base64
import hashlib
import json
import os
from pathlib import Path

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_adaptive import MEMORY_SCHEMA, STATE_SCHEMA
from benchmarks.a5kernels.phase1_curriculum import WaveSpec
from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory
from benchmarks.a5kernels.phase1_wave_runtime import LiveWaveCoordinator

REVIEW_SCHEMA = "ascend-phase1-usefulness-review-v1"
CELL_SCHEMA = "ascend-reviewed-memory-identity-v1"
SNAPSHOT_SCHEMA = "ascend-reviewed-memory-snapshot-v1"

def _read(path):
    return json.loads(Path(path).read_bytes())

def _memory(state, initial):
    terminal = state.get("terminal")
    target = state.get("project_target")
    target_complete = (terminal is None and type(target) is int and target > 0
                       and len(state.get("prior_outcomes", [])) >= target)
    if (state.get("schema") != STATE_SCHEMA
            or state.get("initial_memory_sha256") != hashlib.sha256(initial).hexdigest()
            or not (target_complete or terminal and terminal.get("status") in {
                "SATURATED", "PROJECT_LIMIT_REACHED", "TARGET_REACHED"})):
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
        reader = LiveWaveCoordinator(
            wave_root, wave, pre_memory, lambda _: None, 1,
            allow_semantic_failures=True,
            allow_partial=_read(wave_root / "published.json").get("completion") in {"partial", "empty"},
        )
        reader.read_published_memory()  # authenticates outcomes and both evidence digests
        for name in ("plan.json", "published.json", "consolidated-memory.json"):
            files[f"waves/{wave_id}/{name}"] = (wave_root / name).read_bytes()
        for project in wave.projects:
            branch = wave_root / "branches" / project.spec.project_id
            if not (branch / "outcome.json").exists():
                # The publication reader authenticates the skipped interruption record.
                files[f"waves/{wave_id}/branches/{project.spec.project_id}/interruption.json"] = (
                    branch / "interruption.json").read_bytes()
                continue
            for name in ("outcome.json", "memory-delta.bin", "execution-evidence.bin"):
                files[f"waves/{wave_id}/branches/{project.spec.project_id}/{name}"] = (branch / name).read_bytes()
            if not _read(branch / "outcome.json")["verified"]:
                continue
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
    if published != state["memory_bytes_base64"]:
        raise ValueError("adaptive state differs from published branch memory")
    return files

def _review(review, memory_sha, cell_id, *, schema=REVIEW_SCHEMA):
    if (review.get("schema") != schema
            or review.get("memory_sha256") != memory_sha
            or review.get("cell_id") != cell_id
            or any(review.get(key) is not True for key in (
                "accepted", "useful", "diverse", "profiler_guidance_observed"))
            or not isinstance(review.get("rationale"), str)
            or not review["rationale"].strip()):
        raise ValueError("adaptive memory needs an accepted usefulness/diversity review")

def collect_adaptive_snapshot(source_root, initial_memory, review_path):
    """Read/authenticate exact published memory and referenced evidence for any target."""
    source_root = Path(source_root)
    if (source_root / "pending.json").exists():
        raise ValueError("adaptive Phase 1 has a pending wave")
    state_bytes = (source_root / "state.json").read_bytes()
    state = json.loads(state_bytes)
    memories, memory_sha = _memory(state, initial_memory)
    review_bytes = Path(review_path).read_bytes()
    review = json.loads(review_bytes)
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
    return files, review, memory_sha

def _publish_exact(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != value:
            raise ValueError(f"immutable snapshot conflict: {path.name}") from None
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())

@dataclass(frozen=True)
class ReviewedMemoryIdentity:
    source_identity: dict
    qualification_sha256: str

    def __post_init__(self):
        _native_identity(self.source_identity)

    @property
    def source_cell_id(self):
        return source_cell_id(self.source_identity)

    @property
    def cell_id(self):
        return "reviewed-memory-" + canonical_digest(self.as_dict())[:16]

    @property
    def guide(self):
        return NativeGuide(self.source_identity["guide_sha256"], self.source_identity["cpl_skills_revision"])

    def as_dict(self):
        return dict(schema=CELL_SCHEMA, target="a3", language="ascend-c",
                    source_identity=self.source_identity, qualification_sha256=self.qualification_sha256)

def source_cell_id(identity):
    return "a3-phase1-cell-" + canonical_digest(identity)[:16]

def _native_identity(identity):
    if (identity.get("schema") not in {"ascend-native-adaptive-identity-v1", "ascend-native-reviewed-phase1-identity-v1"}
            or identity.get("language") != "ascend-c"
            or identity.get("execution_profile") not in {"bz-a3-1", "bz-a3-2", "local"}
            or not identity.get("target") or not identity.get("arch")):
        raise ValueError("handoff requires an explicitly native A3 source identity")

@dataclass(frozen=True)
class NativeGuide:
    guide_sha256: str
    cpl_skills_revision: str

    def as_dict(self):
        return vars(self)

@dataclass(frozen=True)
class NativePhase1Snapshot:
    root: Path
    destination: ReviewedMemoryIdentity
    manifest: dict

    @property
    def memory_path(self):
        return self.root / "phase1-memory.jsonl"

    @property
    def snapshot_id(self):
        return self.manifest["snapshot_id"]

    def verify(self):
        manifest = dict(self.manifest)
        digest = manifest.pop("snapshot_id", None)
        if (manifest.get("schema") != SNAPSHOT_SCHEMA or digest != canonical_digest(manifest)
                or manifest["destination_identity"] != self.destination.as_dict()):
            raise ValueError("native snapshot identity changed")
        for relative, expected in manifest["files"].items():
            if hashlib.sha256((self.root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("native snapshot bytes changed")
        initial = (self.root / "initial-memory.bin").read_bytes()
        state = json.loads((self.root / "state.json").read_bytes())
        memories, memory_sha = _memory(state, initial)
        _published_evidence(self.root, state, initial)
        review = json.loads((self.root / "review.json").read_bytes())
        _review(review, memory_sha, self.destination.source_cell_id, schema=REVIEW_SCHEMA)
        if canonical_digest(review) != self.destination.qualification_sha256:
            raise ValueError("native review identity changed")
        identity = json.loads((self.root / "campaign-identity.json").read_bytes())
        if identity != self.destination.source_identity:
            raise ValueError("native source treatment changed")
        if self.memory_path.read_bytes() != b"".join(row + b"\n" for row in memories):
            raise ValueError("native snapshot memory changed")

    def memory_entries(self):
        self.verify()
        return tuple(json.loads(row) for row in self.memory_path.read_bytes().splitlines())

def admit_native_snapshot(source_root, initial_memory, review_path, snapshot_root):
    identity = json.loads((Path(source_root) / "campaign-identity.json").read_bytes())
    _native_identity(identity)
    files, review, memory_sha = collect_adaptive_snapshot(source_root, initial_memory, review_path)
    _review(review, memory_sha, source_cell_id(identity), schema=REVIEW_SCHEMA)
    cell = ReviewedMemoryIdentity(identity, canonical_digest(review))
    body = dict(schema=SNAPSHOT_SCHEMA, destination_identity=cell.as_dict(),
                files={name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()})
    manifest = {**body, "snapshot_id": canonical_digest(body)}
    root = Path(snapshot_root).resolve() / "cells" / cell.cell_id / "snapshot"
    for relative, payload in files.items():
        _publish_exact(root / relative, payload)
    _publish_exact(root / "snapshot.json", canonical_bytes(manifest) + b"\n")
    snapshot = NativePhase1Snapshot(root, cell, manifest)
    snapshot.verify()
    return snapshot

def reviewed_seed(root, snapshot_id):
    """Recover the seed from its pinned, authenticated portable snapshot."""
    from benchmarks.a5kernels.evidence import canonical_digest
    matches = []
    for manifest_path in Path(root).glob("cells/*/snapshot/snapshot.json"):
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("snapshot_id") == snapshot_id:
            matches.append((manifest_path.parent, manifest))
    if len(matches) != 1:
        raise ValueError("reviewed snapshot is missing or ambiguous")
    directory, manifest = matches[0]
    if manifest.get("schema") not in {
            "catlass-reviewed-memory-snapshot-v1", SNAPSHOT_SCHEMA}:
        raise ValueError("reviewed snapshot schema changed")
    if canonical_digest({k: v for k, v in manifest.items() if k != "snapshot_id"}) != snapshot_id:
        raise ValueError("reviewed snapshot identity changed")
    for relative, digest in manifest["files"].items():
        if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("reviewed snapshot bytes changed")
    state = json.loads((directory / "state.json").read_bytes())
    initial = (directory / "initial-memory.bin").read_bytes()
    memories, memory_sha = _memory(state, initial)
    _published_evidence(directory, state, initial)
    destination = manifest["destination_identity"]
    native = destination.get("language") == "ascend-c"
    source_id = ("a3-phase1-cell-" + canonical_digest(destination["source_identity"])[:16]
                 if native else destination["source_cell_id"])
    review = json.loads((directory / "review.json").read_bytes())
    _review(review, memory_sha, source_id, schema=(
        "ascend-phase1-usefulness-review-v1" if native else "a5-phase1-usefulness-review-v1"))
    if canonical_digest(review) != destination["qualification_sha256"]:
        raise ValueError("reviewed snapshot review changed")
    entries = [json.loads(row) for row in memories]
    if (directory / "phase1-memory.jsonl").read_bytes() != b"".join(row + b"\n" for row in memories):
        raise ValueError("reviewed snapshot memory changed")
    return canonical_bytes({"schema": "ascend-reviewed-memory-seed-v1",
                            "snapshot_id": snapshot_id, "phase1_memory": entries}), directory


def reviewed_phase1_coordinator(root, objective, reviewed_snapshot, curriculum, execute_project,
            *, project_target=None, trace_hook=None, max_seconds=7200):
    from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
    reviewed_snapshot.verify()
    entries = list(reviewed_snapshot.memory_entries())
    if not entries:
        raise ValueError("reviewed seed requires published project memory")
    initial = canonical_bytes(dict(schema="ascend-reviewed-memory-seed-v1",
        snapshot_id=reviewed_snapshot.snapshot_id, phase1_memory=entries))
    coordinator = AdaptivePhase1Coordinator(root, objective, initial, curriculum,
        execute_project, max_parallel=1, project_target=project_target, max_seconds=max_seconds,
        allow_partial=True)
    seed_path = coordinator.root / "initial-memory.bin"
    state_path = coordinator.root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_bytes())
        if (state.get("initial_memory_sha256") != hashlib.sha256(initial).hexdigest()
                or not seed_path.is_file() or seed_path.read_bytes() != initial):
            raise ValueError("retained reviewed seed differs from admitted memory")
    else:
        from optimizer.jobs import atomic
        coordinator.root.mkdir(parents=True, exist_ok=True)
        atomic(seed_path, initial)
    coordinator.snapshot = reviewed_snapshot
    coordinator.trace_hook = trace_hook
    return coordinator
