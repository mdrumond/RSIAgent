"""Repeated collection selectors must reach actual production configuration."""

import pytest

import run_a5_phase1
import run_a5kernels


@pytest.mark.parametrize("names", [("legacy",), ("ops-nn", "triton")])
@pytest.mark.parametrize("entrypoint", ["phase1", "pilot"])
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
    else:
        args = run_a5kernels._parser().parse_args([
            "preflight", "--tla-root", "/workspace", "--profiling-skill-root", "/profiling",
            "--catlass-src", "/catlass", "--catlass-revision", "a" * 40,
            "--bge-cache", "cache", "--kdb", "db", "--results-root", "results",
            "--device", "0", *flags,
        ])
        config = run_a5kernels._production_paths(args)
    assert config.collection == (names[0] if len(names) == 1 else names)
