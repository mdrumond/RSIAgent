from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_adaptive import DELTA_SCHEMA, MEMORY_SCHEMA, STATE_SCHEMA
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec, WaveProject, WaveSpec
from benchmarks.a5kernels.phase1_experiments import build_cells
from benchmarks.a5kernels.phase1_memory import HostFact, Phase1ProjectMemory
from benchmarks.a5kernels.phase1_provider import DirectCompletion
from benchmarks.a5kernels.phase1_project import EvidenceClassification, ExecutionEvidence
from benchmarks.a5kernels.phase1_wave_runtime import LiveWaveCoordinator, WaveProjectResult
from benchmarks.a5kernels.phase2_adaptive_memory import (
    REVIEW_SCHEMA, admit_adaptive_snapshot, adaptive_destination,
)
from benchmarks.a5kernels.phase2_live import DEFAULT_MAX_SECONDS, Phase2AttemptRequest
from benchmarks.a5kernels.phase2_production import Phase2ProductionConfig, build_phase2_production
from tests.test_a5_phase1_adaptive import _project
from tests.test_a5_phase2_source_provider import GUIDE, GUIDE_ID, ENVIRONMENT
from tests.test_a5_phase2_production import FakeExecutor, CopyVBackend


def _source(tmp_path, *, new=False, historical_citation=False):
    profiling_guide = None
    if new:
        from benchmarks.a5kernels.profiling_guidance import ProfilingGuidanceIdentity
        profiling_guide = ProfilingGuidanceIdentity(
            "a5-new-profiling-guidance-v1", hashlib.sha256(b"New profiler instructions").hexdigest(), "1" * 40,
        )
    source = next(cell for cell in build_cells(GUIDE_ID, profiling_guide=profiling_guide)
                  if cell.knowledge.value == "without-kdb"
                  and cell.profiling.value == ("new" if new else "without-profiling-guidance"))
    root = tmp_path / "adaptive"
    artifacts = ContentAddressedArtifacts(root / "waves/wave-01/branches/copy/artifacts")
    evidence = artifacts.put_json({"host": "verified", "runtime": "9.3-weekly"})
    source_sha = artifacts.put_bytes(b"actual candidate source")
    spec = ProjectSpec.from_mapping(_project("copy", "gm-ub-movement"))
    memory = Phase1ProjectMemory(
        spec, source_sha, ("run",), (HostFact("host-verification", "passed", evidence),),
        admitted_artifact_sha256=(evidence,), artifact_store="artifacts",
    ).as_dict()
    if historical_citation:
        memory["kdb_citations"] = [dict(collection="docs", path="guide.py", start_line=1,
                                       end_line=2, chunk_hash="c" * 64)]
        if historical_citation == "empty-provenance":
            memory["kdb_citations"][0].update(repository=None, revision=None,
                                            source_sha256=None, reference_exposed=False)
        elif historical_citation == "populated-provenance":
            memory["kdb_citations"][0].update(repository="https://example.invalid/repo",
                revision="d" * 40, source_sha256="e" * 64, reference_exposed=True)
    encoded = base64.b64encode(canonical_bytes(memory)).decode()
    pre_memory = canonical_bytes({
        "schema": MEMORY_SCHEMA, "initial_memory_base64": "", "project_memory_base64": [],
    })
    wave = WaveSpec("wave-01", hashlib.sha256(pre_memory).hexdigest(), (), (WaveProject(spec),))
    LiveWaveCoordinator(root / "waves/wave-01", wave, pre_memory, lambda _: WaveProjectResult(
        canonical_bytes({"schema": DELTA_SCHEMA, "project_id": "copy", "summary": "host verified",
                         "memory_base64": encoded}),
        ExecutionEvidence(EvidenceClassification.HOST_VERIFIED, "test fixture"),
        canonical_bytes({"classification": "host-verified", "source_sha256": source_sha,
                         "artifacts": {"classification": evidence}}),
    ), 1).run()
    state = {
        "schema": STATE_SCHEMA, "initial_memory_sha256": hashlib.sha256(b"").hexdigest(),
        "terminal": {"status": "SATURATED", "reason": "done"},
        "memory_bytes_base64": [encoded], "completed_waves": ["wave-01"],
        "prior_outcomes": [{"project_id": "copy", "spec": spec.as_dict(), "verified": True}],
    }
    (root / "state.json").write_bytes(canonical_bytes(state) + b"\n")
    (root / "campaign-identity.json").write_bytes(canonical_bytes({"cell": source.as_dict()}))
    memory_sha = hashlib.sha256(canonical_bytes({
        "schema": MEMORY_SCHEMA, "initial_memory_base64": "", "project_memory_base64": [encoded],
    })).hexdigest()
    review = {"schema": REVIEW_SCHEMA, "cell_id": source.cell_id, "memory_sha256": memory_sha,
              "accepted": True, "useful": True, "diverse": True,
              "profiler_guidance_observed": True, "rationale": "Human-reviewed fixture, not NPU evidence."}
    review_path = tmp_path / "review.json"
    review_path.write_bytes(canonical_bytes(review))
    return root, source, review_path, memory


def test_snapshot_copies_exact_memory_and_evidence_and_survives_source_removal(tmp_path):
    root, source, review, memory = _source(tmp_path)
    before = (root / "state.json").read_bytes()
    snapshot = admit_adaptive_snapshot(root, b"", review, source, tmp_path / "phase2")
    assert snapshot.memory_entries() == (memory,)
    assert (snapshot.root / "state.json").read_bytes() == before
    assert (root / "state.json").read_bytes() == before
    retained = ContentAddressedArtifacts(snapshot.root / "artifacts")
    assert retained.read_bytes(memory["source_revision"]) == b"actual candidate source"
    assert retained.read_json(memory["host_facts"][0]["evidence_sha256"])["host"] == "verified"
    (root / "state.json").unlink()
    snapshot.verify()
    snapshot.memory_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        snapshot.verify()


@pytest.mark.parametrize("representation", [True, "empty-provenance", "populated-provenance"])
def test_historical_nonempty_citation_resumes_and_hands_off_exact_bytes(tmp_path, representation):
    from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
    root, source, review, memory = _source(tmp_path, historical_citation=representation)
    coordinator = object.__new__(AdaptivePhase1Coordinator)
    coordinator.root = root
    delta = coordinator._read_delta("wave-01", "copy")
    original = base64.b64decode(delta["memory_base64"])
    assert json.loads(original)["kdb_citations"] == memory["kdb_citations"]
    snapshot = admit_adaptive_snapshot(root, b"", review, source, tmp_path / "phase2")
    assert snapshot.memory_entries() == (memory,)
    assert snapshot.memory_path.read_bytes() == original + b"\n"


@pytest.mark.parametrize("failure", ["running", "pending", "rejected", "wrong-memory", "missing-evidence"])
def test_admission_rejects_incomplete_or_unreviewed_memory(tmp_path, failure):
    root, source, review, memory = _source(tmp_path)
    if failure == "running":
        state = json.loads((root / "state.json").read_bytes())
        state["terminal"] = None
        (root / "state.json").write_bytes(canonical_bytes(state))
    elif failure == "pending":
        (root / "pending.json").write_text("{}")
    elif failure in {"rejected", "wrong-memory"}:
        value = json.loads(review.read_bytes())
        value["accepted" if failure == "rejected" else "memory_sha256"] = False
        review.write_bytes(canonical_bytes(value))
    else:
        digest = memory["host_facts"][0]["evidence_sha256"]
        next((root / "waves").glob(f"**/objects/{digest[:2]}/{digest[2:]}" )).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        admit_adaptive_snapshot(root, b"", review, source, tmp_path / "phase2")


@pytest.mark.parametrize("status", ["PROJECT_LIMIT_REACHED", "TARGET_REACHED"])
def test_completed_terminal_is_admitted(tmp_path, status):
    root, source, review, _ = _source(tmp_path)
    state = json.loads((root / "state.json").read_bytes())
    state["terminal"]["status"] = status
    (root / "state.json").write_bytes(canonical_bytes(state))
    admit_adaptive_snapshot(root, b"", review, source, tmp_path / "phase2").verify()


@pytest.mark.parametrize("name", ["outcome.json", "memory-delta.bin", "execution-evidence.bin"])
@pytest.mark.parametrize("mutation", ["missing", "stale"])
def test_admission_requires_actual_published_branch_evidence(tmp_path, name, mutation):
    root, source, review, _ = _source(tmp_path)
    path = root / "waves/wave-01/branches/copy" / name
    if mutation == "missing":
        path.unlink()
    else:
        path.write_bytes(b'{}')
    with pytest.raises(ValueError):
        admit_adaptive_snapshot(root, b"", review, source, tmp_path / "phase2")


def test_new_guidance_reaches_phase2_and_current_qualification_is_required(tmp_path):
    root, source, review, memory = _source(tmp_path, new=True)
    destination = adaptive_destination(source, json.loads(review.read_bytes()))
    guide_path = tmp_path / "guide.md"
    guide_path.write_text(GUIDE)
    initial = tmp_path / "initial.bin"
    initial.write_bytes(b"")
    phase2 = tmp_path / "phase2"
    config = Phase2ProductionConfig(
        root, None, guide_path, phase2, phase2 / "cells" / destination.cell_id / "snapshot",
        destination.cell_id, tmp_path, "/remote/catlass", 0,
        adaptive_review=review, initial_memory=initial,
        profiling_guidance_text="New profiler instructions",
    )
    executor = FakeExecutor(source.guide.catlass_revision)
    with pytest.raises(ValueError, match="9.3"):
        build_phase2_production(config, environment=ENVIRONMENT, transport=lambda **_: None,
                                executor=executor, backend=CopyVBackend())
    executor.runtime_provenance = tuple((key, "9.3.0-weekly" if key == "cann_version" else value)
                                        for key, value in executor.runtime_provenance)
    calls = []
    def transport(**kwargs):
        calls.append(kwargs)
        return DirectCompletion('{"candidate_source":{"source":"candidate"}}', 1)
    snapshot, dependencies = build_phase2_production(
        config, environment=ENVIRONMENT, transport=transport,
        executor=executor, backend=CopyVBackend(),
    )
    request = Phase2AttemptRequest(destination, "3" * 64, 1, "req-1", "attempt-1", (memory,), ())
    assert dependencies.source(request) == "candidate"
    assert "New profiler instructions" in json.dumps(calls)
    assert snapshot.memory_entries() == (memory,)
    assert DEFAULT_MAX_SECONDS == 7200
    with pytest.raises(ValueError):
        build_phase2_production(replace(config, profiling_guidance_text="wrong"),
                                environment=ENVIRONMENT, transport=transport,
                                executor=executor, backend=CopyVBackend())
