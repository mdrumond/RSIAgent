"""Shared remote settings select the same machine inputs as local settings."""
from argparse import Namespace
import importlib
import json
from pathlib import Path

import pytest

from benchmarks.ascend.worker_bindings import apply_a5_worker_arguments


@pytest.mark.parametrize("entrypoint", ["run_a3_phase1"])
def test_fixed_a3_remote_worker_supplies_transport_fields(tmp_path, entrypoint):
    from run_a3_phase1 import _worker_settings
    worker = dict(execution="remote", profile="bz-a3-2", cpl_remote="/shared/cpl-remote",
                  validation_wrapper="/checked/catlass-validation.sh", remote_workspace="/remote/jobs", physical_device=7)
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(worker))
    args = importlib.import_module(entrypoint)._parser().parse_args([
        "preflight", "--state-root", str(tmp_path / "state"), "--worker-config", str(path)])
    _worker_settings(args)
    assert args.profile == "bz-a3-2"
    assert args.cpl_remote == worker["cpl_remote"]
    assert args.validation_wrapper == Path(worker["validation_wrapper"])
    if hasattr(args, "remote_workspace"):
        assert args.remote_workspace == worker["remote_workspace"]
        assert args.physical_device == 7


@pytest.mark.parametrize("source,revision", [("catlass_source", "catlass_revision"),
    ("catlass_src", "catlass_revision"), ("catlass_source", "runtime_catlass_revision")])
def test_a5_remote_worker_overrides_machine_fields_and_preserves_omissions(source, revision):
    args = Namespace(tla_root=Path("/old"), device=1, profiling_skill_root=Path("/skills"))
    setattr(args, source, "/old/catlass")
    setattr(args, revision, "a" * 40)
    apply_a5_worker_arguments(args, dict(execution="remote", profile="bz-a5",
        catlass_source="/new/catlass", catlass_revision="b" * 40, physical_device=3,
        validation_wrapper="/checked/execution-profiles/catlass-validation.sh"), source=source, revision=revision)
    assert getattr(args, source) == "/new/catlass"
    assert getattr(args, revision) == "b" * 40
    assert args.device == 3
    assert args.tla_root == Path("/checked")
    assert args.profiling_skill_root == Path("/skills")


def test_a5_remote_worker_rejects_foreign_profile():
    with pytest.raises(ValueError, match="bz-a5"):
        apply_a5_worker_arguments(Namespace(), dict(execution="remote", profile="bz-a3-1"))


@pytest.mark.parametrize("module,entry", [("run_a5_phase1", "_paths"),
    ("run_a5kernels", "_production_paths"),
    ("benchmarks.a5kernels.phase1_adaptive_cli", "main")])
def test_each_a5_launcher_applies_shared_remote_worker(tmp_path, monkeypatch, module, entry):
    import benchmarks.ascend.worker_bindings as bindings
    worker = tmp_path / "worker.json"
    worker.write_text(json.dumps(dict(execution="remote", profile="bz-a5", physical_device=3,
                                    catlass_source="/worker/catlass", catlass_revision="b" * 40)))
    calls = []
    def capture(args, settings, **options):
        calls.append((settings, options))
        raise ValueError("captured worker binding")
    monkeypatch.setattr(bindings, "apply_a5_worker_arguments", capture)
    function = getattr(importlib.import_module(module), entry)
    with pytest.raises(ValueError, match="captured worker binding"):
        if entry == "main":
            memory = tmp_path / "memory"
            memory.write_text("")
            function(["run", "--root", str(tmp_path / "run"), "--objective", "copy",
                      "--initial-memory", str(memory), "--worker-config", str(worker)])
        elif entry == "_paths":
            function(Namespace(worker_config=worker), "b" * 40)
        else:
            function(Namespace(worker_config=worker))
    assert calls[0][0]["physical_device"] == 3
    assert calls[0][0]["execution"] == "remote"
