import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.a3_experiments import (
    BackendModel, KnowledgeMode, ProfilingGuidance,
)
from benchmarks.a3kernels.candidate import profile_driver_asset
from benchmarks.a3kernels.phase1_evidence import canonical_digest
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.phase1_wave import (
    Phase1Config, Phase1Wave, foundation_cells, full_dry_run,
)


def config(tmp_path):
    wrapper = tmp_path / "catlass-validation.sh"
    wrapper.write_text("#!/bin/sh\n")
    wrapper.chmod(0o755)
    embedding = tmp_path / "embedding"; embedding.mkdir()
    corpus = tmp_path / "corpus"; corpus.mkdir()
    database = tmp_path / "kdb.sqlite"; database.write_bytes(b"db")
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    return Phase1Config(
        tmp_path / "state", wrapper, embedding, corpus, database, manifest
    )


def terminal_outcome(digest="a" * 64):
    return {
        "status": "passed", "terminal_reason": "completed",
        "completed_projects": 8, "failed_project_id": None,
        "evidence_sha256": digest,
    }


def trial_protocol_sha256(openai_turns=24):
    return canonical_digest({
        "schema": "a3-trial-protocol-v3",
        "verification_feedback_schema": "a3-verification-mismatch-v1",
        "candidate_validation_exception": "rewrite-required",
        "recovery_starter_ambiguous_outcome": "infrastructure-unverified",
        "attempt_completion": {
            "schema": "a3-attempt-completion-v1",
            "source_frozen": True,
            "allowed_operations": ["compile", "run", "profile", "submit"],
            "call_limits": {
                ProfilingGuidance.WITHOUT_GUIDANCE.value: 3,
                ProfilingGuidance.WITH_GUIDANCE.value: 4,
            },
            "token_ceiling": "absolute",
            "rewrite_required": "terminate",
            "rewrite_terminal_status": "attempt-rewrite-required",
        },
        "model_budgets": {
            BackendModel.GPT_5_6_SOL.value: {
                "max_turns": openai_turns, "max_tokens": 32768,
            },
            BackendModel.DEEPSEEK_FLASH.value: {
                "max_turns": 24, "max_tokens": 65536,
            },
        },
    })


def test_all_eight_project_dry_run_has_no_side_effects(tmp_path):
    cfg = config(tmp_path)
    before = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")}
    plan = full_dry_run()
    after = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")}
    assert len(plan["projects"]) == 8 and plan["mode"] == "dry-run"
    assert before == after and not cfg.state_root.exists()


def test_foundation_selection_is_exact_two_by_two_by_two():
    cells = foundation_cells()
    assert len(cells) == len({cell.cell_id for cell in cells}) == 8
    assert len({cell.backend_model for cell in cells}) == 2
    assert len({cell.knowledge for cell in cells}) == 2
    assert len({cell.profiling for cell in cells}) == 2
    assert {cell.programming_level.value for cell in cells} == {"foundation"}


def test_preflight_checks_credentials_paths_and_fixed_adapter(tmp_path):
    cfg = config(tmp_path)
    report = cfg.preflight({"OPENAI_API_KEY": "gpt", "DEEPSEEK_API_KEY": "ds"})
    assert report["ready"] is True and all(report["checks"].values())
    assert report["profile_driver_sha256"] == profile_driver_asset().sha256
    assert report["trial_protocol_sha256"] == trial_protocol_sha256()
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        cfg.preflight({"OPENAI_API_KEY": "gpt"})


def test_preflight_uses_registered_profile_credential_names(monkeypatch, tmp_path):
    import benchmarks.a3kernels.phase1_wave as phase1_wave

    credential_names = {
        BackendModel.GPT_5_6_SOL: "OPENAI_API_KEY",
        BackendModel.DEEPSEEK_FLASH: "DEEPSEEK_API_KEY",
    }
    monkeypatch.setattr(
        phase1_wave,
        "load_a3_model_profile",
        lambda model: SimpleNamespace(credential_env=credential_names[model]),
    )

    cfg = config(tmp_path)
    report = cfg.preflight(
        {
            "OPENAI_API_KEY": "direct-openai-secret",
            "DEEPSEEK_API_KEY": "deepseek-secret",
        }
    )

    assert report["checks"]["OPENAI_API_KEY"] is True
    assert "OPENROUTER_API_KEY" not in report["checks"]
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        cfg.preflight(
            {
                "OPENROUTER_API_KEY": "stale-openrouter-secret",
                "DEEPSEEK_API_KEY": "deepseek-secret",
            }
        )


@pytest.mark.parametrize(
    ("model", "credential"),
    [
        (BackendModel.GPT_5_6_SOL, "OPENAI_API_KEY"),
        (BackendModel.DEEPSEEK_FLASH, "DEEPSEEK_API_KEY"),
    ],
)
def test_preflight_requires_only_selected_cell_credentials(tmp_path, model, credential):
    cell = next(
        cell for cell in foundation_cells()
        if cell.backend_model is model
    )

    report = config(tmp_path).preflight({credential: "secret"}, cells=(cell,))

    assert report["checks"][credential] is True
    other = {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"} - {credential}
    assert other.isdisjoint(report["checks"])


def test_preflight_requires_kdb_artifacts_only_for_selected_kdb_treatment(tmp_path):
    cfg = config(tmp_path)
    cfg.embedding_cache.rmdir()
    cfg.corpus_artifacts.rmdir()
    cfg.knowledge_database.unlink()
    cfg.knowledge_manifest.unlink()
    without_kdb = next(
        cell for cell in foundation_cells()
        if cell.knowledge is KnowledgeMode.WITHOUT_KDB
    )
    with_kdb = next(
        cell for cell in foundation_cells()
        if cell.backend_model is without_kdb.backend_model
        and cell.knowledge is KnowledgeMode.WITH_KDB
    )
    credential = (
        "OPENAI_API_KEY"
        if without_kdb.backend_model is BackendModel.GPT_5_6_SOL
        else "DEEPSEEK_API_KEY"
    )

    report = cfg.preflight({credential: "secret"}, cells=(without_kdb,))

    assert {
        "embedding_cache", "corpus_artifacts", "knowledge_database",
        "knowledge_manifest",
    }.isdisjoint(report["checks"])
    with pytest.raises(ValueError, match="embedding_cache"):
        cfg.preflight({credential: "secret"}, cells=(with_kdb,))


def test_fake_eight_cell_wave_is_isolated_terminal_and_reportable(tmp_path):
    cfg = config(tmp_path)
    seen = []
    def execute(cell, paths):
        seen.append((cell.cell_id, paths))
        assert paths.workspace.parent == cfg.state_root / "cells" / cell.cell_id
        assert paths.profile_driver.read_text() == profile_driver_asset().content
        return terminal_outcome(cell.cell_id.removeprefix("a3-cell-") * 4)
    wave = Phase1Wave(cfg, execute)
    records = wave.run()
    assert len(records) == len(seen) == 8
    assert {record["schema"] for record in records} == {
        "a3-phase1-cell-terminal-v4"
    }
    assert all(record["completed_projects"] == 8 for record in records)
    assert {record["status"] for record in records} == {"passed"}
    assert {record["profile_driver_sha256"] for record in records} == {
        profile_driver_asset().sha256
    }
    assert len({record["plan_fingerprint"] for record in records}) == 1
    assert len({record["host_fixture_sha256"] for record in records}) == 1
    report = wave.report()
    assert report["counts"] == {"passed": 8}
    assert len({str(paths.memory) for _, paths in seen}) == 8


def test_interruption_resume_does_not_duplicate_terminal_cells(tmp_path):
    cfg = config(tmp_path)
    calls = []
    def interrupted(cell, paths):
        calls.append(cell.cell_id)
        if len(calls) == 4: raise KeyboardInterrupt()
        return terminal_outcome()
    wave = Phase1Wave(cfg, interrupted)
    with pytest.raises(KeyboardInterrupt): wave.run()
    assert len(wave.report()["records"]) == 3
    resumed_calls = []
    records = Phase1Wave(
        cfg, lambda cell, paths: (
            resumed_calls.append(cell.cell_id) or terminal_outcome("b" * 64)
        ),
    ).resume()
    assert len(records) == 8 and len(resumed_calls) == 5
    assert not set(calls[:3]) & set(resumed_calls)


def test_resume_rejects_terminal_from_other_execution_profile(tmp_path):
    cfg = config(tmp_path)
    execute = lambda *_: terminal_outcome()
    Phase1Wave(cfg, execute, execution_profile="bz-a3-1").run()
    terminal = Phase1Wave(cfg, execute).paths(foundation_cells()[0]).terminal
    assert json.loads(terminal.read_text())["execution_profile"] == "bz-a3-1"

    with pytest.raises(ValueError, match="foreign|conflicting"):
        Phase1Wave(cfg, execute, execution_profile="bz-a3-2").resume()


def test_foreign_or_conflicting_terminal_record_is_rejected(tmp_path):
    cfg = config(tmp_path)
    wave = Phase1Wave(cfg, lambda *_: terminal_outcome())
    first = foundation_cells()[0]
    paths = wave.paths(first); paths.root.mkdir(parents=True)
    paths.terminal.write_text(json.dumps({"cell_id": first.cell_id, "target": "a5", "language": "catlass-dsl", "status": "passed", "evidence_sha256": "a" * 64}))
    with pytest.raises(ValueError, match="foreign"):
        wave.resume()


def test_full_terminal_rejects_contradictory_project_progress(tmp_path):
    cfg = config(tmp_path)
    wave = Phase1Wave(cfg, lambda *_: terminal_outcome())
    wave.run()
    terminal = wave.paths(foundation_cells()[0]).terminal
    record = json.loads(terminal.read_text())
    record["completed_projects"] = 7
    terminal.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="foreign|conflicting"):
        wave.report()


def test_full_terminal_rejects_failed_project_out_of_sequence(tmp_path):
    cfg = config(tmp_path)
    failure = {
        "status": "failed", "terminal_reason": "turn-budget-exhausted",
        "completed_projects": 2,
        "failed_project_id": DEFAULT_PROPOSALS[2].project_id,
        "evidence_sha256": "a" * 64,
    }
    wave = Phase1Wave(cfg, lambda *_: failure)
    wave.run()
    terminal = wave.paths(foundation_cells()[0]).terminal
    record = json.loads(terminal.read_text())
    record["failed_project_id"] = DEFAULT_PROPOSALS[3].project_id
    terminal.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="foreign|conflicting"):
        wave.resume()


@pytest.mark.parametrize(
    "field", ["plan_fingerprint", "profile_driver_sha256", "host_fixture_sha256"]
)
def test_resume_rejects_terminal_from_stale_research_identity(tmp_path, field):
    cfg = config(tmp_path)
    wave = Phase1Wave(
        cfg, lambda *_: terminal_outcome()
    )
    wave.run()
    terminal = wave.paths(foundation_cells()[0]).terminal
    record = json.loads(terminal.read_text())
    record[field] = canonical_digest({"stale": field})
    terminal.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="foreign|conflicting"):
        wave.resume()


def test_full_resume_rejects_retained_twelve_turn_trial_protocol(tmp_path):
    cfg = config(tmp_path)
    wave = Phase1Wave(cfg, lambda *_: terminal_outcome())
    wave.run()
    terminal = wave.paths(foundation_cells()[0]).terminal
    record = json.loads(terminal.read_text())
    record["trial_protocol_sha256"] = trial_protocol_sha256(12)
    terminal.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="foreign|conflicting"):
        wave.resume()


def test_cli_dry_run_and_report(capsys, tmp_path):
    import run_a3_phase1
    assert run_a3_phase1.main(["dry-run"]) == 0
    assert len(json.loads(capsys.readouterr().out)["projects"]) == 8
    cfg = config(tmp_path)
    assert run_a3_phase1.main(["report", "--state-root", str(cfg.state_root)]) == 0
    assert json.loads(capsys.readouterr().out)["counts"] == {}
