"""Repeated collection selectors must reach actual production configuration."""

import pytest

import run_a5_phase1
import run_a5_phase2
import run_a5kernels


@pytest.mark.parametrize("names", [("legacy",), ("ops-nn", "triton")])
@pytest.mark.parametrize("entrypoint", ["phase1", "phase2", "pilot"])
def test_collection_cli_preserves_every_selected_collection(names, entrypoint, monkeypatch):
    flags = [flag for name in names for flag in ("--collection", name)]
    if entrypoint == "phase1":
        args = run_a5_phase1._parser().parse_args([
            "run", "--gate", "gate.json", "--guide", "guide.md", "--root", "results",
            "--tla-root", "/workspace", "--profiling-skill-root", "/profiling",
            "--catlass-source", "/catlass", "--bge-cache", "cache", "--kdb", "db",
            "--device", "0", *flags,
        ])
        config = run_a5_phase1._paths(args, "a" * 40)
    elif entrypoint == "phase2":
        args = run_a5_phase2._parser().parse_args([
            "run", "--root", "results", "--snapshot", "snapshot.json", "--cell-id", "cell",
            "--phase1-root", "phase1", "--guide", "guide.md", "--tla-root", "/workspace",
            "--catlass-source", "/catlass", "--device", "0",
            "--adaptive-review", "review.json", "--initial-memory", "memory.md", *flags,
        ])
        captured = []
        monkeypatch.setattr(run_a5_phase2, "build_phase2_production",
                            lambda value: captured.append(value))
        run_a5_phase2._production_factory(args)
        config = captured[0]
    else:
        args = run_a5kernels._parser().parse_args([
            "preflight", "--tla-root", "/workspace", "--profiling-skill-root", "/profiling",
            "--catlass-src", "/catlass", "--catlass-revision", "a" * 40,
            "--bge-cache", "cache", "--kdb", "db", "--results-root", "results",
            "--device", "0", *flags,
        ])
        config = run_a5kernels._production_paths(args)
    assert config.collection == (names[0] if len(names) == 1 else names)


def test_phase2_kdb_off_cli_keeps_collection_none(monkeypatch):
    args = run_a5_phase2._parser().parse_args([
        "run", "--root", "results", "--snapshot", "snapshot.json", "--cell-id", "cell",
        "--phase1-root", "phase1", "--guide", "guide.md", "--tla-root", "/workspace",
        "--catlass-source", "/catlass", "--device", "0",
        "--adaptive-review", "review.json", "--initial-memory", "memory.md",
    ])
    captured = []
    monkeypatch.setattr(run_a5_phase2, "build_phase2_production",
                        lambda value: captured.append(value))
    run_a5_phase2._production_factory(args)
    assert captured[0].collection is None
    assert captured[0].kdb is None
