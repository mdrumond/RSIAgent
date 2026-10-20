from types import SimpleNamespace

import pytest

from benchmarks.ascend import campaign_cli
from optimizer import memory, worker
from optimizer.jobs import canonical, sha
from test_ascend_phase2_optimizer_tasks import seed


@pytest.mark.parametrize("collection,manifest", [
    ("kernels", "a" * 64),
    (["kernels", "examples"], {"kernels": "a" * 64, "examples": "b" * 64}),
])
def test_optimizer_composition_binds_reviewed_kdb_manifest(tmp_path, monkeypatch, collection, manifest):
    guide = tmp_path / "guide.md"
    guide.write_text("Ascend C")
    config = dict(model_profile="openai/gpt-5.6-sol", profile="bz-a3-1", target="bz-a3-1", runtime="py311-torch", physical_device=10,
                  arch="dav-2201", cpl_remote="/configured/transport", validation_wrapper="/configured/adapter",
                  remote_workspace="/configured/workspace", guide=str(guide),
                  knowledge=dict(collection=collection, manifest=manifest))
    path = tmp_path / "worker.json"
    job = dict(root=str(tmp_path / "campaign"), objective="new", model="openai/gpt-5.6-sol",
               target_attempt_budget=3, practice_attempt_budget=2)
    native = SimpleNamespace(execution_profile="bz-a3-1", arch="dav-2201",
                             guide_sha256=sha(guide.read_bytes()), collection=collection)
    composed = SimpleNamespace(curriculum=SimpleNamespace(_complete=lambda _: ""),
        execute_project=SimpleNamespace(executor=SimpleNamespace(runtime_identity={"device": "10"})))
    snapshot = SimpleNamespace(destination=SimpleNamespace(source_identity={"kdb_manifest": manifest}))
    queries = []
    monkeypatch.setattr(memory, "freeze_reviewed_phase1", lambda _: seed())
    monkeypatch.setattr(worker, "knowledge_query", lambda *args: queries.append(args))

    def compose():
        path.write_bytes(canonical(config))
        job["optimizer_worker_config_sha256"] = sha(path.read_bytes())
        return campaign_cli._optimizer_controller(job, {"optimizer_worker_config": str(path)},
                                                   snapshot, composed, native)

    assert compose().knowledge is None
    assert len(queries) == 1
    config["knowledge"]["manifest"] = "c" * 64 if isinstance(manifest, str) else {**manifest, "kernels": "c" * 64}
    with pytest.raises(ValueError, match="KDB manifest differs from reviewed Phase 1"):
        compose()
    del config["knowledge"]["manifest"]
    with pytest.raises(ValueError, match="KDB manifest differs from reviewed Phase 1"):
        compose()
    config["knowledge"]["manifest"] = manifest
    snapshot.destination.source_identity.clear()
    with pytest.raises(ValueError, match="KDB manifest differs from reviewed Phase 1"):
        compose()
    assert len(queries) == 1  # Reject drift before opening or querying another database.
