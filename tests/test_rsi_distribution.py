"""Exercise installed runtime dependencies outside the source checkout."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def test_wheel_runs_legacy_helpers_outside_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    built = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--wheel-dir", str(tmp_path / "wheels"), str(root)],
        capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    installed = tmp_path / "installed"
    with zipfile.ZipFile(next((tmp_path / "wheels").glob("rsi-*.whl"))) as wheel:
        wheel.extractall(installed)
    env = {**os.environ, "PYTHONPATH": str(installed)}
    imported = subprocess.run(
        [sys.executable, "-c", "import tools.exam_fence; import explore.practice_loop"],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert imported.returncode == 0, imported.stderr
    launched = subprocess.run(
        [sys.executable, str(installed / "run_ascend_campaign.py"), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert launched.returncode == 0, launched.stderr
