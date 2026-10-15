"""Reviewed native handoff bindings over the shared resumable attention loop."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from benchmarks.a3kernels import phase2_attention as native
from benchmarks.a5kernels.evidence import canonical_bytes, canonical_digest
from benchmarks.a5kernels.phase1_experiments import (
    A5BackendModel, A5ModelIdentity, KnowledgeMode, ProfilingGuidance,
)
from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider
from benchmarks.a5kernels.phase2_adaptive_memory import (
    _memory, _published_evidence, _review, collect_adaptive_snapshot,
)
from benchmarks.a5kernels.phase2_live import Phase2Dependencies, Phase2LiveRunner
from benchmarks.a5kernels.phase2_memory import _publish_exact
from benchmarks.a5kernels.phase2_source_provider import Phase2SourceProvider, _target_contract


REVIEW_SCHEMA = "ascend-phase1-usefulness-review-v1"
CELL_SCHEMA = "a3-native-phase2-adaptive-cell-v1"
SNAPSHOT_SCHEMA = "a3-native-phase2-adaptive-snapshot-v1"


def source_cell_id(identity):
    return "a3-phase1-cell-" + canonical_digest(identity)[:16]


def _native_identity(identity):
    if (identity.get("schema") != "ascend-native-adaptive-identity-v1"
            or identity.get("language") != "ascend-c"
            or identity.get("execution_profile") not in {"bz-a3-1", "bz-a3-2"}
            or not identity.get("target") or not identity.get("arch")):
        raise ValueError("handoff requires an explicitly native A3 source identity")


@dataclass(frozen=True)
class NativeGuide:
    guide_sha256: str
    cpl_skills_revision: str

    def as_dict(self):
        return vars(self)


@dataclass(frozen=True)
class NativePhase2Cell:
    source_identity: dict
    qualification_sha256: str
    suite_sha256: str = native.SUITE_SHA256
    schema: str = CELL_SCHEMA

    def __post_init__(self):
        _native_identity(self.source_identity)
        if self.suite_sha256 != native.SUITE_SHA256 or self.schema != CELL_SCHEMA:
            raise ValueError("native Phase2 suite identity changed")

    @property
    def cell_id(self):
        return "a3-phase2-cell-" + canonical_digest(self.as_dict())[:16]

    @property
    def source_cell_id(self):
        return source_cell_id(self.source_identity)

    @property
    def guide(self):
        return NativeGuide(self.source_identity["guide_sha256"], self.source_identity["cpl_skills_revision"])

    @property
    def model(self):
        model = dict(self.source_identity["model"])
        model["backend_model"] = A5BackendModel(model["backend_model"])
        return A5ModelIdentity(**model)

    @property
    def knowledge(self):
        return KnowledgeMode.WITH_KDB if self.source_identity.get("kdb_collection") else KnowledgeMode.WITHOUT_KDB

    @property
    def profiling(self):
        return ProfilingGuidance(self.source_identity.get("profiling_mode", (
            "new" if self.source_identity.get("profiling_guidance") else "without-profiling-guidance")))

    def as_dict(self):
        return dict(schema=self.schema, target="a3", language="ascend-c", source_identity=self.source_identity,
                    qualification_sha256=self.qualification_sha256, suite_sha256=self.suite_sha256)


@dataclass(frozen=True)
class NativePhase1Snapshot:
    root: Path
    destination: NativePhase2Cell
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


def admit_native_snapshot(source_root, initial_memory, review_path, phase2_root):
    identity = json.loads((Path(source_root) / "campaign-identity.json").read_bytes())
    _native_identity(identity)
    files, review, memory_sha = collect_adaptive_snapshot(source_root, initial_memory, review_path)
    _review(review, memory_sha, source_cell_id(identity), schema=REVIEW_SCHEMA)
    cell = NativePhase2Cell(identity, canonical_digest(review))
    body = dict(schema=SNAPSHOT_SCHEMA, destination_identity=cell.as_dict(),
                files={name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()})
    manifest = {**body, "snapshot_id": canonical_digest(body)}
    root = Path(phase2_root).resolve() / "cells" / cell.cell_id / "snapshot"
    for relative, payload in files.items():
        _publish_exact(root / relative, payload)
    _publish_exact(root / "snapshot.json", canonical_bytes(manifest) + b"\n")
    snapshot = NativePhase1Snapshot(root, cell, manifest)
    snapshot.verify()
    return snapshot


class NativeSourceProvider(Phase2SourceProvider):
    adaptive_schema = CELL_SCHEMA
    provider_schema = "a3-native-phase2-source-provider-v1"
    source_filename = "candidate.cpp"
    task_prompt = "Produce Ascend C source for the host-owned attention suite."
    system_prompt = Phase2SourceProvider.system_prompt.replace("Catlass DSL", "Ascend C").replace("complete Python source", "complete Ascend C device source")

    def _validate_profiling(self, request):
        expected = request.cell.source_identity.get("profiling_guidance")
        if self.profiling_guidance != expected:
            raise ValueError("native profiling guidance changed")
        if request.cell.profiling is ProfilingGuidance.WITHOUT_GUIDANCE:
            if expected is not None:
                raise ValueError("profiling-off treatment cannot receive guidance")
        elif not isinstance(expected, str) or not expected.strip():
            raise ValueError("profiling treatment requires exact native guidance")

    @staticmethod
    def target_contract():
        contract = _target_contract()
        contract.update(target_id=native.TARGET_ID, candidate_abi=dict(native.CANDIDATE_ABI),
                        candidate_abi_sha256=native.CANDIDATE_ABI_SHA256)
        contract["candidate_module"] = {
            "source": "Ascend C device source including kernel_operator.h; no host code or Torch registration",
            "only_export": dict(native.CANDIDATE_ABI)["kernel_export"],
            "host_bound_constants": dict(native.CANDIDATE_ABI)["host_bound_constants"],
            "dtype": "DTYPE is half or bfloat16_t matching the case",
            "launch": "one block writes complete contiguous output",
        }
        contract["semantics"]["evaluation"] = (
            "candidate Ascend C device source is compiled and launched by the host; "
            "unsupported constructs surface as host compiler/runtime evidence")
        return contract


class NativePhase2Runner(Phase2LiveRunner):
    snapshot_type = NativePhase1Snapshot
    evidence_type = native.AttentionTargetEvidence
    candidate_abi_sha256 = native.CANDIDATE_ABI_SHA256
    protocol_schema = "a3-native-phase2-live-protocol-v1"
    attempt_schema = "a3-native-phase2-attempt-v1"
    memory_schema = "a3-native-phase2-memory-v1"
    terminal_schema = "a3-native-phase2-terminal-v1"
    source_filename = "candidate.cpp"

    @property
    def execution_profile(self):
        return self.dependencies.cell.source_identity["execution_profile"]


def build_native_phase2(snapshot, guide_bytes, runtime, *, environment,
                        provider_transport=None, knowledge_factory=None):
    snapshot.verify()
    cell = snapshot.destination
    if hashlib.sha256(guide_bytes).hexdigest() != cell.guide.guide_sha256:
        raise ValueError("native programming guide changed")
    if cell.knowledge is KnowledgeMode.WITH_KDB and knowledge_factory is None:
        raise ValueError("native KDB treatment requires a gated Knowledge Agent")
    provider = DirectCompletionProvider(environment, **(
        {"transport": provider_transport} if provider_transport is not None else {}))
    source = NativeSourceProvider(provider, evidence_root=snapshot.root.parent / "provider",
                                  guide_text=guide_bytes.decode(), knowledge_factory=knowledge_factory,
                                  profiling_guidance=cell.source_identity.get("profiling_guidance"))

    def verify(source, request_id, attempt_id, workdir):
        identity = cell.source_identity
        backend = native.NativeAttentionBackend(runtime, request_id=request_id, attempt_id=attempt_id,
                    execution_profile=identity["execution_profile"], target=identity["target"], arch=identity["arch"])
        return native.AttentionTargetVerifier(backend).verify(source, request_id=request_id,
                    attempt_id=attempt_id, execution_profile=identity["execution_profile"], workdir=workdir)

    return NativePhase2Runner(phase2_root=snapshot.root.parents[2], snapshot=snapshot,
        dependencies=Phase2Dependencies(cell, source, verify, cell_type=NativePhase2Cell))
