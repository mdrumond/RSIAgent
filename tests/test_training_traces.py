import hashlib
import json

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.sft_dataset import SourceManifest, TraceRecorder
from benchmarks.ascend.training_traces import export_annex


def _campaign(tmp_path, *, evidence=True):
    root = tmp_path / "campaign"
    branch = root / "waves" / "wave-1" / "project-1"
    store = ContentAddressedArtifacts(branch / "artifacts")
    source = "token = 0\nreturn token\n"
    source_sha = store.put_bytes(source.encode())
    digest = store.put_json({"source_sha256": source_sha, "verified": True})
    if not evidence:
        (store.objects / digest[:2] / digest[2:]).unlink()
    root.joinpath("state.json").write_text('{"completed_waves":[]}')
    trace = TraceRecorder(branch / "sft-traces" / "one.jsonl", trace_id="one",
                          phase="phase1", round_id="round-1", operator_family="relu",
                          model="fixture", source_manifest_sha256="a" * 64)
    trace.message("user", "implement relu")
    trace.message("assistant", "earlier attempt")
    trace.candidate(source, candidate_id="candidate")
    trace.validation(candidate_id="candidate", validator="fixture", status="passed",
                     evidence_sha256=digest, candidate_sha256=source_sha)
    trace.supervised_source(source)
    trace.finish()
    return root


def test_offline_export_binds_exact_source_evidence_and_is_deterministic(tmp_path):
    root = _campaign(tmp_path)
    first = export_annex(root, tmp_path / "first")
    second = export_annex(root, tmp_path / "second")
    assert first == second
    assert first["dataset"]["eligible"] == 1
    row = json.loads((tmp_path / "first" / "train.jsonl").read_text())
    assert row["messages"][-1]["content"] == "token = 0\nreturn token\n"
    assert row["messages"][-1]["weight"] == 1
    assert row["messages"][1]["weight"] == 0
    with pytest.raises(ValueError, match="outside"):
        export_annex(root, root / "dataset")


def test_missing_exact_evidence_and_historical_summary_are_rejected(tmp_path):
    root = _campaign(tmp_path, evidence=False)
    branch = root / "waves" / "wave-1" / "historical"
    branch.mkdir()
    (branch / "evidence.jsonl").write_text('{"summary":"passed"}\n')
    manifest = export_annex(root, tmp_path / "output")
    assert manifest["dataset"]["eligible"] == 0
    assert len(manifest["rejected"]) == 2
    assert "evidence absent" in manifest["rejected"][0]["reason"]


def test_observed_inventory_does_not_require_unavailable_repositories(tmp_path):
    path = tmp_path / "sources.json"
    source = {"name": "actual-kdb", "repository": "https://example.invalid/kdb",
              "revision": hashlib.sha1(b"revision").hexdigest(),
              "license": "Apache-2.0", "include": ["**/*.cpp"]}
    path.write_text(json.dumps({"schema": "ascend-sft-sources-v2", "sources": [source]}))
    assert SourceManifest.load(path).sources[0].name == "actual-kdb"
