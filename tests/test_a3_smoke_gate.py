import json
from types import SimpleNamespace

import pytest

from benchmarks.a3_experiments import (
    BackendModel, KnowledgeMode, ProfilingGuidance,
)
from benchmarks.a3kernels.candidate import CANDIDATE_SOURCE_CONTRACT
from benchmarks.a3kernels.phase1_evidence import canonical_digest
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.phase1_wave import (
    LENGTH_KNEE_16_SMOKE_PROPOSALS,
    SMOKE_PROPOSALS,
    TRIAL_RELIABILITY_SMOKE_PROPOSALS,
    Phase1Config,
    SmokeWave,
    foundation_cells,
    select_foundation_cells,
    smoke_dry_run,
)
from benchmarks.a3kernels.knowledge import CollectionManifest


def config(tmp_path):
    wrapper = tmp_path / "catlass-validation.sh"
    wrapper.write_text("#!/bin/sh\n")
    wrapper.chmod(0o755)
    embedding = tmp_path / "embedding"; embedding.mkdir()
    corpus = tmp_path / "corpus"; corpus.mkdir()
    database = tmp_path / "knowledge.sqlite3"; database.write_bytes(b"db")
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    return Phase1Config(
        tmp_path / "state", wrapper, embedding, corpus, database, manifest
    )


def outcome(*_args):
    return {
        "status": "passed", "terminal_reason": "completed",
        "completed_projects": 1, "failed_project_id": None,
        "evidence_sha256": "a" * 64,
        "infrastructure_retries_used": 0,
        "infrastructure_retry_evidence_sha256": canonical_digest([]),
    }


def knowledge_identity(seed="a"):
    return {
        "embedding_snapshot_sha256": seed * 64,
        "knowledge_database_sha256": "b" * 64,
        "knowledge_manifest_sha256": "c" * 64,
        "knowledge_collection_sha256": "d" * 64,
        "knowledge_probe_sha256": "e" * 64,
    }


def trial_protocol_sha256(openai_turns=24, *, mismatch_feedback=True):
    value = {
        "schema": "a3-trial-protocol-v5",
        "candidate_source_contract": CANDIDATE_SOURCE_CONTRACT.as_dict(),
        "infrastructure_retry": {
            "max_retries": 3, "backoff_seconds": 120,
            "retryable_status": "infrastructure-unverified",
            "journal_schema": "a3-infrastructure-retry-v1",
            "attempt_outcome_required_before_decision": True,
            "ambiguous_candidate_outcome": "infrastructure-unverified",
            "prepare_runtime_outcome": "infrastructure-unverified",
            "interrupted_started_outcome": "terminal-infrastructure-unverified",
            "cell_execution": "exclusive-nonblocking",
            "memory_commit_reconciliation": "authenticated-passed-outcome",
        },
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
    }
    if mismatch_feedback:
        value["verification_feedback_schema"] = "a3-verification-mismatch-v1"
    return canonical_digest(value)


def test_smoke_plan_is_one_registered_baseline_and_not_phase1_plan():
    plan = smoke_dry_run()
    assert plan["schema"] == "a3-ascendc-foundation-smoke-plan-v1"
    assert plan["mode"] == "smoke"
    assert len(plan["projects"]) == len(SMOKE_PROPOSALS) == 1
    assert plan["projects"][0]["family"] == "vector-add-baseline"


def test_length_knee_16_smoke_plan_is_the_registered_project():
    plan = smoke_dry_run(LENGTH_KNEE_16_SMOKE_PROPOSALS)
    assert LENGTH_KNEE_16_SMOKE_PROPOSALS == (DEFAULT_PROPOSALS[4],)
    assert len(plan["projects"]) == 1
    assert plan["projects"][0]["family"] == "length-knee"
    assert plan["projects"][0]["parameters"] == {"length": 16}


def test_smoke_paths_and_terminals_are_isolated_from_full_phase1(tmp_path):
    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITHOUT_KDB)
    wave = SmokeWave(
        cfg, outcome, cells=(cell,), execution_profile="bz-a3-1"
    )
    record = wave.run()[0]
    assert wave.paths(cell).root.parent == cfg.state_root / "smoke-cells"
    assert record["schema"] == "a3-phase1-smoke-cell-terminal-v3"
    assert record["terminal_reason"] == "completed"
    assert record["completed_projects"] == 1
    assert record["failed_project_id"] is None
    assert record["knowledge_identity"] is None
    assert record["plan_fingerprint"] == canonical_digest(smoke_dry_run())
    assert record["proposal_set_sha256"] == canonical_digest(
        [proposal.as_dict() for proposal in SMOKE_PROPOSALS]
    )
    assert not (cfg.state_root / "cells" / cell.cell_id).exists()


def test_runtime_recovery_preflight_and_terminal_share_proposal_identity(tmp_path):
    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITHOUT_KDB)
    preflight = cfg.smoke_preflight(
        {"OPENAI_API_KEY": "secret"}, cells=(cell,),
        proposals=TRIAL_RELIABILITY_SMOKE_PROPOSALS,
    )
    wave = SmokeWave(
        cfg, outcome, cells=(cell,), execution_profile="bz-a3-1",
        proposals=TRIAL_RELIABILITY_SMOKE_PROPOSALS,
    )
    record = wave.run()[0]

    assert preflight["smoke_plan_fingerprint"] == record["plan_fingerprint"]
    assert preflight["trial_protocol_sha256"] == record["trial_protocol_sha256"]
    assert record["trial_protocol_sha256"] == trial_protocol_sha256()
    assert record["plan_fingerprint"] == canonical_digest(
        smoke_dry_run(TRIAL_RELIABILITY_SMOKE_PROPOSALS)
    )
    assert wave.report()["records"] == [record]


def test_smoke_persists_and_reloads_infrastructure_unverified_terminal(tmp_path):
    cfg = config(tmp_path)
    cell = next(
        item for item in foundation_cells()
        if item.knowledge is KnowledgeMode.WITHOUT_KDB
    )
    proposal = TRIAL_RELIABILITY_SMOKE_PROPOSALS[0]
    failure = {
        "status": "failed",
        "terminal_reason": "infrastructure-unverified",
        "completed_projects": 0,
        "failed_project_id": proposal.project_id,
        "evidence_sha256": "f" * 64,
        "infrastructure_retries_used": 3,
        "infrastructure_retry_evidence_sha256": "e" * 64,
    }
    wave = SmokeWave(
        cfg, lambda *_: failure, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    )

    record = wave.run()[0]
    reloaded = SmokeWave(
        cfg, lambda *_: pytest.fail("persisted smoke failure was replayed"),
        cells=(cell,), execution_profile="bz-a3-1", proposals=(proposal,),
    )

    assert reloaded.resume() == (record,)
    assert reloaded.report() == {
        "schema": "a3-phase1-foundation-smoke-report-v2",
        "counts": {"failed": 1},
        "records": [record],
    }


def test_smoke_terminal_rejects_contradictory_project_progress(tmp_path):
    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITHOUT_KDB)
    wave = SmokeWave(cfg, outcome, cells=(cell,), execution_profile="bz-a3-1")
    wave.run()
    terminal = wave.paths(cell).terminal
    record = json.loads(terminal.read_text())
    record["completed_projects"] = 0
    terminal.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="foreign|conflicting|corrupt"):
        wave.report()


def test_no_kdb_smoke_never_authenticates_or_opens_artifacts(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    for path in (
        cfg.embedding_cache, cfg.corpus_artifacts, cfg.knowledge_database,
        cfg.knowledge_manifest,
    ):
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITHOUT_KDB)
    monkeypatch.setattr(
        "benchmarks.a3kernels.phase1_wave.authenticate_smoke_knowledge",
        lambda *_: pytest.fail("no-KDB selection touched knowledge artifacts"),
    )
    credential = (
        "OPENAI_API_KEY" if "openai" in cell.backend_model.value
        else "DEEPSEEK_API_KEY"
    )
    report = cfg.smoke_preflight({credential: "secret"}, cells=(cell,))
    assert report["knowledge_identity"] is None
    SmokeWave(cfg, outcome, cells=(cell,), execution_profile="bz-a3-1").run()


def test_kdb_authentication_happens_before_executor(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITH_KDB)
    events = []
    monkeypatch.setattr(
        "benchmarks.a3kernels.phase1_wave.authenticate_smoke_knowledge",
        lambda *_: events.append("authenticated-query") or knowledge_identity(),
    )
    wave = SmokeWave(
        cfg,
        lambda *_: events.append("provider-executor") or outcome(),
        cells=(cell,), execution_profile="bz-a3-1",
    )
    wave.run()
    assert events == ["authenticated-query", "provider-executor"]


def test_injected_smoke_knowledge_identity_must_have_exact_digests(tmp_path):
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITH_KDB)
    with pytest.raises(ValueError, match="knowledge identity"):
        SmokeWave(
            config(tmp_path), outcome, cells=(cell,), execution_profile="bz-a3-1",
            knowledge_identity={"knowledge_database_sha256": "not-a-digest"},
        )


def test_smoke_rejects_non_bz_compatibility_profile(tmp_path):
    with pytest.raises(ValueError, match="BZ-A3"):
        SmokeWave(config(tmp_path), outcome, execution_profile="gz-a3")


def test_smoke_kdb_authentication_opens_read_only_and_runs_fixed_query(
    tmp_path, monkeypatch,
):
    import benchmarks.a3kernels.phase1_wave as wave_module

    cfg = config(tmp_path)
    manifest = CollectionManifest(
        "a3", "a3", "model", "revision", 3, "schema", 80, 20, (), "f" * 64
    )
    events = []

    class Database:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            events.append("closed")
        def manifest(self, collection):
            events.append(("manifest", collection))
            return manifest
        def query(self, collection, query, *, limit):
            events.append(("query", collection, query, limit))
            return [SimpleNamespace(chunk_id="1" * 64, content_sha256="2" * 64)]

    monkeypatch.setattr(
        wave_module, "authenticated_snapshot_identity", lambda _path: "a" * 64
    )
    monkeypatch.setattr(
        wave_module.CollectionManifest, "from_json", lambda _value: manifest
    )
    monkeypatch.setattr(wave_module, "PinnedBGEEmbeddings", lambda **_kw: object())
    monkeypatch.setattr(
        wave_module, "validate_manifest_contract",
        lambda _spec, selected, _embeddings: events.append(
            ("contract", selected.collection)
        ),
    )
    monkeypatch.setattr(
        wave_module.KnowledgeDB, "open_read_only",
        lambda path, embeddings: events.append(("open", path, embeddings)) or Database(),
    )

    identity = wave_module.authenticate_smoke_knowledge(cfg)

    assert identity["embedding_snapshot_sha256"] == "a" * 64
    assert events[0] == ("contract", "a3")
    assert events[1][0] == "open"
    assert events[2] == ("manifest", "a3")
    assert events[3] == (
        "query", "a3", "A3 Ascend C vector addition tensor movement", 1,
    )
    assert events[4] == "closed"


def test_smoke_kdb_probe_fails_closed_when_query_has_no_result(tmp_path, monkeypatch):
    import benchmarks.a3kernels.phase1_wave as wave_module

    cfg = config(tmp_path)
    manifest = SimpleNamespace(collection="a3", fingerprint="f" * 64)

    class Database:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def manifest(self, _collection): return manifest
        def query(self, *_args, **_kwargs): return []

    monkeypatch.setattr(
        wave_module, "authenticated_snapshot_identity", lambda _path: "a" * 64
    )
    monkeypatch.setattr(
        wave_module.CollectionManifest, "from_json", lambda _value: manifest
    )
    monkeypatch.setattr(wave_module, "PinnedBGEEmbeddings", lambda **_kw: object())
    monkeypatch.setattr(
        wave_module, "validate_manifest_contract", lambda *_args: None
    )
    monkeypatch.setattr(
        wave_module.KnowledgeDB, "open_read_only", lambda *_args: Database()
    )
    with pytest.raises(ValueError, match="no authenticated result"):
        wave_module.authenticate_smoke_knowledge(cfg)


def test_smoke_kdb_fails_before_open_when_manifest_is_not_pinned(
    tmp_path, monkeypatch,
):
    import benchmarks.a3kernels.phase1_wave as wave_module

    cfg = config(tmp_path)
    manifest = SimpleNamespace(collection="other", fingerprint="f" * 64)
    monkeypatch.setattr(
        wave_module, "authenticated_snapshot_identity", lambda _path: "a" * 64
    )
    monkeypatch.setattr(
        wave_module.CollectionManifest, "from_json", lambda _value: manifest
    )
    monkeypatch.setattr(wave_module, "PinnedBGEEmbeddings", lambda **_kw: object())
    monkeypatch.setattr(
        wave_module, "validate_manifest_contract",
        lambda *_args: (_ for _ in ()).throw(ValueError("not pinned")),
    )
    monkeypatch.setattr(
        wave_module.KnowledgeDB, "open_read_only",
        lambda *_args: pytest.fail("untrusted database was opened"),
    )
    with pytest.raises(ValueError, match="not pinned"):
        wave_module.authenticate_smoke_knowledge(cfg)


@pytest.mark.parametrize(
    "field",
    [
        "execution_profile", "model_profile_sha256", "proposal_set_sha256",
        "profile_driver_sha256", "host_fixture_sha256", "knowledge_identity",
    ],
)
def test_smoke_resume_rejects_changed_bound_identity(tmp_path, monkeypatch, field):
    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITH_KDB)
    identity = knowledge_identity()
    monkeypatch.setattr(
        "benchmarks.a3kernels.phase1_wave.authenticate_smoke_knowledge",
        lambda *_: identity,
    )
    wave = SmokeWave(
        cfg, outcome, cells=(cell,), execution_profile="bz-a3-1"
    )
    wave.run()
    terminal = wave.paths(cell).terminal
    record = json.loads(terminal.read_text())
    record[field] = (
        knowledge_identity("f") if field == "knowledge_identity" else "f" * 64
    )
    terminal.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="foreign|conflicting"):
        wave.resume()


def test_smoke_resume_rejects_retained_twelve_turn_trial_protocol(tmp_path):
    cfg = config(tmp_path)
    cell = next(
        c for c in foundation_cells()
        if c.knowledge is KnowledgeMode.WITHOUT_KDB
    )
    wave = SmokeWave(
        cfg, outcome, cells=(cell,), execution_profile="bz-a3-1"
    )
    wave.run()
    terminal = wave.paths(cell).terminal
    record = json.loads(terminal.read_text())
    record["trial_protocol_sha256"] = trial_protocol_sha256(12)
    terminal.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="foreign|conflicting"):
        wave.resume()


def test_smoke_resume_rejects_protocol_without_mismatch_feedback(tmp_path):
    cfg = config(tmp_path)
    cell = next(
        c for c in foundation_cells()
        if c.knowledge is KnowledgeMode.WITHOUT_KDB
    )
    wave = SmokeWave(cfg, outcome, cells=(cell,), execution_profile="bz-a3-1")
    wave.run()
    terminal = wave.paths(cell).terminal
    record = json.loads(terminal.read_text())
    record["trial_protocol_sha256"] = trial_protocol_sha256(
        mismatch_feedback=False
    )
    terminal.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="foreign|conflicting"):
        wave.resume()


def test_interrupted_smoke_state_rejects_changed_identity_before_executor(
    tmp_path, monkeypatch,
):
    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITH_KDB)
    first = knowledge_identity()
    monkeypatch.setattr(
        "benchmarks.a3kernels.phase1_wave.authenticate_smoke_knowledge",
        lambda *_: first,
    )
    calls = []
    wave = SmokeWave(
        cfg,
        lambda *_: calls.append("first") or (_ for _ in ()).throw(KeyboardInterrupt()),
        cells=(cell,), execution_profile="bz-a3-1",
    )
    with pytest.raises(KeyboardInterrupt):
        wave.run()
    assert calls == ["first"]
    assert wave.paths(cell).terminal.exists() is False
    assert (wave.paths(cell).root / "identity.json").is_file()

    with pytest.raises(ValueError, match="conflicting"):
        SmokeWave(
            cfg, lambda *_: calls.append("second") or outcome(), cells=(cell,),
            execution_profile="bz-a3-1", knowledge_identity=knowledge_identity("f"),
        ).resume()
    assert calls == ["first"]


def test_smoke_report_rejects_non_bz_terminal_profile(tmp_path):
    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITHOUT_KDB)
    wave = SmokeWave(cfg, outcome, cells=(cell,), execution_profile="bz-a3-1")
    wave.run()
    terminal = wave.paths(cell).terminal
    record = json.loads(terminal.read_text())
    record["execution_profile"] = "gz-a3"
    terminal.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="foreign|conflicting"):
        wave.report()


def test_smoke_shards_are_deterministic_disjoint_and_use_unique_paths(tmp_path):
    cfg = config(tmp_path)
    first = select_foundation_cells(shard_count=2, shard_index=0)
    second = select_foundation_cells(shard_count=2, shard_index=1)
    assert not set(first) & set(second)
    assert set(first) | set(second) == set(foundation_cells())
    first_paths = {
        SmokeWave(cfg, outcome, cells=(cell,), execution_profile="bz-a3-1").paths(cell).root
        for cell in first
    }
    second_paths = {
        SmokeWave(cfg, outcome, cells=(cell,), execution_profile="bz-a3-2").paths(cell).root
        for cell in second
    }
    assert first_paths.isdisjoint(second_paths)


@pytest.mark.parametrize(
    ("smoke_project", "proposals"),
    [
        ("baseline", SMOKE_PROPOSALS),
        ("runtime-recovery", TRIAL_RELIABILITY_SMOKE_PROPOSALS),
        ("length-knee-16", LENGTH_KNEE_16_SMOKE_PROPOSALS),
    ],
)
def test_cli_smoke_selects_one_project_composition_and_smoke_terminal(
    tmp_path, monkeypatch, capsys, smoke_project, proposals,
):
    import run_a3_phase1

    cfg = config(tmp_path)
    cell = next(c for c in foundation_cells() if c.knowledge is KnowledgeMode.WITHOUT_KDB)
    selected = {}
    original_preflight = Phase1Config.smoke_preflight

    def capture_preflight(self, environ, **kwargs):
        selected["preflight_proposals"] = tuple(kwargs["proposals"])
        return original_preflight(self, environ, **kwargs)

    class Composition:
        def __init__(self, _cfg, _deps, *, proposals, lineage_prefix):
            selected["proposals"] = tuple(proposals)
            selected["lineage_prefix"] = lineage_prefix
        def execute(self, *_args):
            return outcome()

    monkeypatch.setattr(run_a3_phase1, "LiveComposition", Composition)
    monkeypatch.setattr(Phase1Config, "smoke_preflight", capture_preflight)
    monkeypatch.setattr(
        run_a3_phase1, "bz_live_dependencies", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        run_a3_phase1, "_bz_preflight", lambda *_args: {"state": "completed"}
    )
    monkeypatch.setattr(
        "benchmarks.a3kernels.phase1_wave.authenticate_smoke_knowledge",
        lambda *_args: knowledge_identity(),
    )
    credential = (
        "OPENAI_API_KEY" if "openai" in cell.backend_model.value
        else "DEEPSEEK_API_KEY"
    )
    monkeypatch.setenv(credential, "secret")
    argv = [
        "smoke", "--state-root", str(cfg.state_root),
        "--validation-wrapper", str(cfg.validation_wrapper),
        "--profile", "bz-a3-1", "--cpl-remote", "/checked/cpl-remote",
        "--remote-workspace", "/remote/rsi", "--physical-device", "0",
        "--cell-id", cell.cell_id,
        "--smoke-project", smoke_project,
    ]
    assert run_a3_phase1.main(argv) == 0
    record = json.loads(capsys.readouterr().out)[0]
    assert record["schema"] == "a3-phase1-smoke-cell-terminal-v3"
    assert selected == {
        "proposals": proposals,
        "preflight_proposals": proposals,
        "lineage_prefix": "smoke",
    }


def test_cli_smoke_preflight_selects_runtime_recovery_identity(
    tmp_path, monkeypatch, capsys,
):
    import run_a3_phase1

    cfg = config(tmp_path)
    monkeypatch.setattr(
        run_a3_phase1, "_bz_preflight", lambda *_args: {"state": "completed"}
    )
    monkeypatch.setattr(
        "benchmarks.a3kernels.phase1_wave.authenticate_smoke_knowledge",
        lambda *_args: knowledge_identity(),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    argv = [
        "smoke-preflight", "--state-root", str(cfg.state_root),
        "--validation-wrapper", str(cfg.validation_wrapper),
        "--embedding-cache", str(cfg.embedding_cache),
        "--corpus-artifacts", str(cfg.corpus_artifacts),
        "--knowledge-database", str(cfg.knowledge_database),
        "--knowledge-manifest", str(cfg.knowledge_manifest),
        "--profile", "bz-a3-1", "--cpl-remote", "/checked/cpl-remote",
        "--smoke-project", "runtime-recovery",
    ]

    assert run_a3_phase1.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["smoke_plan_fingerprint"] == canonical_digest(
        smoke_dry_run(TRIAL_RELIABILITY_SMOKE_PROPOSALS)
    )


def test_cli_smoke_preflight_selects_length_knee_16_identity(
    tmp_path, monkeypatch, capsys,
):
    import run_a3_phase1

    cfg = config(tmp_path)
    monkeypatch.setattr(
        run_a3_phase1, "_bz_preflight", lambda *_args: {"state": "completed"}
    )
    monkeypatch.setattr(
        "benchmarks.a3kernels.phase1_wave.authenticate_smoke_knowledge",
        lambda *_args: knowledge_identity(),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    argv = [
        "smoke-preflight", "--state-root", str(cfg.state_root),
        "--validation-wrapper", str(cfg.validation_wrapper),
        "--embedding-cache", str(cfg.embedding_cache),
        "--corpus-artifacts", str(cfg.corpus_artifacts),
        "--knowledge-database", str(cfg.knowledge_database),
        "--knowledge-manifest", str(cfg.knowledge_manifest),
        "--profile", "bz-a3-1", "--cpl-remote", "/checked/cpl-remote",
        "--smoke-project", "length-knee-16",
    ]

    assert run_a3_phase1.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["smoke_plan_fingerprint"] == canonical_digest(
        smoke_dry_run(LENGTH_KNEE_16_SMOKE_PROPOSALS)
    )


def test_cli_smoke_project_choices_reject_unregistered_alias(tmp_path):
    import run_a3_phase1

    with pytest.raises(SystemExit):
        run_a3_phase1._parser().parse_args([
            "smoke-report", "--state-root", str(tmp_path),
            "--smoke-project", "length-knee-400",
        ])
