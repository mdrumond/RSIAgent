import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[2] / "ascend_tools" / "actor_setup.py"
spec = importlib.util.spec_from_file_location("actor_setup", MODULE)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def config(root, attempt="a", runtime="runtime"):
    return {"binding": {"target": "target", "runtime": runtime, "product": "a3"},
            "cache_root": str(root / "cache"), "scratch": str(root / attempt)}


def test_compatible_actors_reuse_cache_with_isolated_persistent_scratch(tmp_path):
    first = setup.prepare(config(tmp_path), {})
    Path(first["TRITON_CACHE_DIR"], "compiled").write_text("retained")
    Path(first["RSI_SCRATCH_DIR"], "probe").write_text("a")
    second = setup.prepare(config(tmp_path, "b"), {})
    assert first["TRITON_CACHE_DIR"] == second["TRITON_CACHE_DIR"]
    assert Path(second["TRITON_CACHE_DIR"], "compiled").read_text() == "retained"
    assert not Path(second["RSI_SCRATCH_DIR"], "probe").exists()
    resumed = setup.prepare(config(tmp_path), {})
    assert Path(resumed["RSI_SCRATCH_DIR"], "probe").read_text() == "a"
    assert json.loads(Path(second["RSI_SETUP_REPORT"]).read_text())["environment"]["RSI_SCRATCH_DIR"] == second["RSI_SCRATCH_DIR"]


def test_runtime_and_compiler_changes_separate_cache(tmp_path):
    first = setup.prepare(config(tmp_path), {})
    other = setup.prepare(config(tmp_path, runtime="other"), {})
    assert first["TRITON_CACHE_DIR"] != other["TRITON_CACHE_DIR"]
    with patch.object(setup, "runtime_identity", return_value={"compiler": "new-build"}):
        changed = setup.prepare(config(tmp_path), {})
    assert changed["TRITON_CACHE_DIR"] != first["TRITON_CACHE_DIR"]


def test_explicit_cache_overrides_and_initialized_catlass_are_preserved(tmp_path):
    env = {"TRITON_CACHE_DIR": str(tmp_path / "cold"),
           "CATLASS_DSL_CACHE_DIR": str(tmp_path / "initialized-catlass")}
    result = setup.prepare(config(tmp_path), env)
    assert result["TRITON_CACHE_DIR"] == env["TRITON_CACHE_DIR"]
    assert result["CATLASS_DSL_CACHE_DIR"] == env["CATLASS_DSL_CACHE_DIR"]
    assert not Path(result["CATLASS_DSL_CACHE_DIR"]).exists()


def test_bootstrap_passes_command_output_status_and_environment(tmp_path):
    command = 'printf "%s" "$RSI_SCRATCH_DIR"; echo retained > "$RSI_SCRATCH_DIR/probe"; exit 7'
    result = subprocess.run([sys.executable, str(MODULE), json.dumps(config(tmp_path)), command],
                            capture_output=True, text=True)
    assert result.returncode == 7
    assert result.stdout == str(tmp_path / "a")
    assert (tmp_path / "a" / "probe").read_text() == "retained\n"


def test_existing_report_refreshes_after_runtime_change(tmp_path):
    first = setup.prepare(config(tmp_path), {})
    with patch.object(setup, "runtime_identity", return_value={"compiler": "replacement"}):
        second = setup.prepare(config(tmp_path), {})
    assert first["RSI_SETUP_REPORT"] == second["RSI_SETUP_REPORT"]
    report = json.loads(Path(second["RSI_SETUP_REPORT"]).read_text())
    assert report["identity"] == {"compiler": "replacement"}
    assert report["environment"]["TRITON_CACHE_DIR"] == second["TRITON_CACHE_DIR"]
