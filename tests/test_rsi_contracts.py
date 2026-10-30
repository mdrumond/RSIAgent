"""Behavioral contracts for portable RSI specs and the foundation CLI."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from rsi.contracts import BudgetLimits, CycleSpec, ProjectSpec, Stage, StepEvidence, Mode


def project():
    return ProjectSpec(project_id="sum", goal="Accurately sum floats", command=("python", "verify.py"))


def test_roundtrip_freezes_nested_inputs():
    cycle = CycleSpec(cycle_id="cpu", mode=Mode.PHASE1, projects=(project(),),
                      baseline_kdb="kdb-sha", baseline_skills="skills-sha")
    payload = cycle.to_dict()
    restored = CycleSpec.from_dict(json.loads(json.dumps(payload)))
    assert restored == cycle
    payload["projects"][0]["command"][0] = "changed"
    assert restored.projects[0].command[0] == "python"
    with pytest.raises(FrozenInstanceError):
        restored.projects[0].goal = "changed"


def test_caps_are_independently_disableable_and_time_domains_separate():
    default = BudgetLimits()
    assert (default.design_iterations, default.implementation_iterations,
            default.execution_tries, default.actor_seconds, default.wall_seconds) == (3, 5, 20, 1800, None)
    assert replace(default, actor_seconds=None, wall_seconds=30).actor_seconds is None
    disabled = BudgetLimits(None, None, None, None, None)
    assert BudgetLimits.from_dict(disabled.to_dict()) == disabled


@pytest.mark.parametrize("values", [
    {"design_iterations": 0}, {"execution_tries": True}, {"actor_seconds": -1},
    {"wall_seconds": float("nan")}, {"actor_seconds": float("inf")},
    {"implementation_iterations": 1.5},
])
def test_invalid_limits_rejected(values):
    with pytest.raises(ValueError):
        BudgetLimits(**values)


def test_spec_rejects_unknown_fields_and_wrong_types():
    payload = project().to_dict()
    for changes in ({"goal": ""}, {"command": "python verify.py"}, {"typo": 2}):
        with pytest.raises(ValueError):
            ProjectSpec.from_dict({**payload, **changes})
    with pytest.raises(ValueError):
        CycleSpec("x", "unknown", (project(),), "kdb", "skills")
    with pytest.raises(ValueError):
        CycleSpec("x", Mode.PHASE1, (project(), project()), "kdb", "skills")


def test_evidence_preserves_empty_retrieval_and_artifact_references():
    evidence = StepEvidence(Stage.DESIGN, "step-1", artifact_refs=("sha256:report",),
                            retrieval_refs=("query:empty",), retrospective_ref="sha256:retro")
    assert StepEvidence.from_dict(evidence.to_dict()) == evidence
    assert evidence.retrieval_refs == ("query:empty",)


def test_cli_validates_fixture_and_reports_invalid_input(tmp_path):
    fixture = Path(__file__).resolve().parents[1] / "examples/cpu/cycle.json"
    result = subprocess.run([sys.executable, "-m", "rsi", "validate-spec", str(fixture)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert CycleSpec.from_dict(json.loads(result.stdout)).projects
    bad = tmp_path / "bad.json"
    bad.write_text('{"cycle_id": "missing fields"}')
    result = subprocess.run([sys.executable, "-m", "rsi", "validate-spec", str(bad)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert not result.stdout


def test_cpu_fixture_runs_without_device_support():
    root = Path(__file__).resolve().parents[1] / "examples/cpu"
    result = subprocess.run([sys.executable, "verify.py"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_direct_constructor_detaches_mutable_command():
    command = ["python", "verify.py"]
    spec = ProjectSpec("p", "check", command)
    command[0] = "changed"
    assert spec.command == ("python", "verify.py")


def test_project_cli_roundtrip(tmp_path):
    path = tmp_path / "project.json"
    path.write_text(json.dumps(project().to_dict()))
    result = subprocess.run([sys.executable, "-m", "rsi", "validate-spec", "--kind", "project", str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert ProjectSpec.from_dict(json.loads(result.stdout)) == project()


def test_numeric_fixture_rejects_naive_replacement(tmp_path):
    root = Path(__file__).resolve().parents[1] / "examples/cpu"
    (tmp_path / "verify.py").write_bytes((root / "verify.py").read_bytes())
    (tmp_path / "solution.py").write_text("def stable_sum(values):\n    total = 0.0\n    for value in values:\n        total += value\n    return total\n")
    result = subprocess.run([sys.executable, "verify.py"], cwd=tmp_path, capture_output=True)
    assert result.returncode != 0
