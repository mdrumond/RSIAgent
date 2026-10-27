from __future__ import annotations

import json
from dataclasses import replace

import pytest

import run_a5_phase2
import benchmarks.a5kernels.phase2_production as production_module
from benchmarks.a5kernels.bz import RuntimeUnavailableError
from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.phase1_composition import (
    A5_PROFILING_GUIDANCE,
    CellDisposition,
    Phase1CellComposition,
)
from benchmarks.a5kernels.phase1_experiments import KnowledgeMode, ProfilingGuidance
from benchmarks.a5kernels.phase1_provider import DirectCompletion
from benchmarks.a5kernels.phase2_attention import (
    ATTENTION_CASES,
    AttentionInfrastructureError,
    AttentionReceipt,
    attention_reference,
)
from benchmarks.a5kernels.phase2_attention_bz import ATTENTION_CANDIDATE_FIXTURE
from benchmarks.a5kernels.phase2_live import Phase2LiveRunner
from benchmarks.a5kernels.phase2_production import (
    Phase2ProductionConfig,
    build_phase2_production,
)
from benchmarks.a5kernels.phase2_memory import phase2_cell_pairs
from benchmarks.a5kernels.sft_dataset import build_dataset
from tests.test_a5_phase1_composition import (
    FakeEmbeddings,
    _dependencies as phase1_dependencies,
    _guide_and_report,
    _knowledge_database,
)
from tests.test_a5_phase2_cli import _dependencies
from tests.test_a5_phase2_memory import _completed_source, _snapshot


ENVIRONMENT = {
    "OPENAI_API_KEY": "openai-secret",
    "DEEPSEEK_API_KEY": "deepseek-secret",
}


class FakeExecutor:
    def __init__(self, revision, source="/remote/catlass"):
        self.probes = []
        self.runtime_provenance = (
            ("catlass_revision", revision),
            ("catlass_source", source),
            ("execution_profile", "bz-a5"),
            ("manifest_sha256", "a" * 64),
            ("cann_version", "9.1"),
        )

    def probe_device(self, device):
        self.probes.append(device)


class CopyVBackend:
    def __init__(self, handle_prefix="bz-a5"):
        self.handle_prefix = handle_prefix
        self.calls = []
        self.compile_calls = []

    def execute(self, source, execution, workdir):
        self.calls.append((source, execution, workdir))
        return AttentionReceipt(
            0, execution.v, session_handle=f"{self.handle_prefix}:qualification",
            metadata=(("result_exit_code", "0"),),
        )

    def compile(self, source, execution, workdir):
        self.compile_calls.append((source, execution, workdir))
        return AttentionReceipt(
            0, (), stdout="compiled", session_handle="bz-a5:compile",
        )


class QueryEmbeddings(FakeEmbeddings):
    def embed_query(self, texts):
        return self.embed(texts)


def _config(tmp_path, **changes):
    composition, _source, destination = _completed_source(tmp_path)
    root = tmp_path / "phase2-production"
    values = {
        "phase1_root": composition.root,
        "gate": composition.qualification,
        "guide_path": tmp_path / "agent-guide.md",
        "phase2_root": root,
        "snapshot_path": root / "cells" / destination.cell_id / "snapshot",
        "cell_id": destination.cell_id,
        "tla_root": tmp_path / "tla",
        "catlass_source": "/remote/catlass",
        "device": 3,
    }
    values.update(changes)
    return composition, destination, Phase2ProductionConfig(**values)


def _treatment_config(tmp_path, knowledge, profiling, **changes):
    guide_path, guide, report = _guide_and_report(tmp_path)
    selected = next(
        cell for cell in dict(phase2_cell_pairs(guide, report.report_sha256))
        if cell.knowledge is knowledge and cell.profiling is profiling
    )

    def disposition(cell):
        return (
            CellDisposition.runnable()
            if cell == selected else CellDisposition.pending("test isolation")
        )

    composition = Phase1CellComposition(
        tmp_path / "phase1", guide_path, guide, report,
        phase1_dependencies(tmp_path / "dependencies", []),
        cell_disposition=disposition, sleeper=lambda _seconds: None,
    )
    composition.run()
    destination = dict(phase2_cell_pairs(guide, report.report_sha256))[selected]
    root = tmp_path / "phase2-production"
    values = {
        "phase1_root": composition.root,
        "gate": report,
        "guide_path": guide_path,
        "phase2_root": root,
        "snapshot_path": root / "cells" / destination.cell_id / "snapshot",
        "cell_id": destination.cell_id,
        "tla_root": tmp_path / "tla",
        "catlass_source": "/remote/catlass",
        "device": 5,
    }
    values.update(changes)
    return destination, Phase2ProductionConfig(**values)


def _transport(**_kwargs):
    return DirectCompletion('{"candidate_source":{"source":"source"}}', 1)


class RepairBackend(CopyVBackend):
    def execute(self, source, execution, workdir):
        self.calls.append((source, execution, workdir))
        if "qualification" in str(workdir):
            output = execution.v
        else:
            output = attention_reference(execution)
            if "attempt-01" in str(workdir):
                output = (output[0] + 1.0, *output[1:])
        return AttentionReceipt(0, output, session_handle="bz-a5:repair-test")


def test_production_trace_and_dataset_classify_successful_repair(tmp_path):
    _composition, destination, config = _config(tmp_path)
    backend = RepairBackend()
    snapshot, dependencies = build_phase2_production(
        config, environment=ENVIRONMENT, transport=_transport,
        executor=FakeExecutor(destination.catlass_revision), backend=backend,
    )
    terminal = Phase2LiveRunner(
        phase2_root=config.phase2_root,
        snapshot=snapshot,
        dependencies=dependencies,
    ).run()
    assert terminal["status"] == "passed"
    assert terminal["completed_attempts"] == 2

    traces = config.phase2_root / "sft-traces"
    passed_trace = next(
        path for path in traces.glob("*.jsonl")
        if any(
            row["kind"] == "validation" and row["payload"]["status"] == "passed"
            for row in map(json.loads, path.read_text().splitlines())
        )
    )
    rows = [json.loads(line) for line in passed_trace.read_text().splitlines()]
    failed = [row for row in rows if row["kind"] == "tool" and row["payload"]["status"] == "failed"]
    assert failed and failed[0]["payload"]["failed_cases"]

    manifest = build_dataset(traces, tmp_path / "dataset")
    assert manifest["eligible"] == 1
    example = manifest["examples"][0]
    assert example["repair"] is True
    assert len(example["repair_pattern_sha256"]) == 64
    assert manifest["rounds"][0]["novel_repair_patterns"] == 1


def test_without_kdb_qualifies_copy_v_once_and_composes_treatments(tmp_path):
    composition, destination, config = _config(tmp_path)
    assert destination.knowledge is KnowledgeMode.WITHOUT_KDB
    executor = FakeExecutor(destination.catlass_revision)
    backend = CopyVBackend()

    snapshot, dependencies = build_phase2_production(
        config, environment=ENVIRONMENT, transport=_transport,
        executor=executor, backend=backend,
    )

    assert snapshot.root == config.snapshot_path.resolve()
    assert dependencies.cell == destination
    assert executor.probes == [3]
    assert len(backend.calls) == 1
    qualification = json.loads(
        (config.phase2_root / "cells" / destination.cell_id / "qualification.json").read_text()
    )
    assert qualification["status"] == "transport-qualified"
    assert qualification["physical_device"] == 3
    assert qualification["catlass_revision"] == composition.guide.catlass_revision
    assert qualification["runtime_catlass_revision"] == destination.catlass_revision
    assert qualification["session_handle"] == "bz-a5:qualification"
    assert "correct" not in qualification and "attention_passed" not in qualification
    assert dependencies.source.knowledge_factory is None
    assert dependencies.source.profiling_guidance is None

    retained, _ = build_phase2_production(
        config, environment=ENVIRONMENT, transport=_transport,
        executor=executor, backend=backend,
    )
    assert retained == snapshot
    assert len(backend.calls) == 1


def test_opt_in_compile_probe_uses_host_compile_only_and_rejects_bad_abi(tmp_path):
    _composition, destination, config = _config(tmp_path, compile_probes=2)
    backend = CopyVBackend()
    dependencies = build_phase2_production(
        config, environment=ENVIRONMENT, transport=_transport,
        executor=FakeExecutor(destination.catlass_revision), backend=backend,
    )[1]
    probe = dependencies.source.compile_probe

    invalid = probe("not Python", tmp_path / "invalid")
    compiled = probe(ATTENTION_CANDIDATE_FIXTURE, tmp_path / "compiled")

    assert invalid["status"] == "invalid-source"
    assert compiled == {
        "status": "compiled", "case_id": ATTENTION_CASES[0].case_id,
        "exit_code": 0, "diagnostic": "compiled",
        "session_handle": "bz-a5:compile",
    }
    assert len(backend.compile_calls) == 1


def test_historical_snapshot_can_use_explicit_current_runtime(tmp_path):
    composition, destination, config = _config(
        tmp_path, runtime_catlass_revision="b" * 40,
    )
    executor = FakeExecutor("b" * 40)

    snapshot, dependencies = build_phase2_production(
        config, environment=ENVIRONMENT, transport=_transport,
        executor=executor, backend=CopyVBackend(),
    )

    assert snapshot.destination.catlass_revision == composition.guide.catlass_revision
    assert dependencies.cell == destination
    qualification = json.loads(
        (config.phase2_root / "cells" / destination.cell_id / "qualification.json").read_text()
    )
    assert qualification["catlass_revision"] == destination.catlass_revision
    assert qualification["runtime_catlass_revision"] == "b" * 40
    assert dict(qualification["runtime_provenance"])["catlass_revision"] == "b" * 40


def test_same_runtime_accepts_legacy_v1_qualification_without_runtime_field(tmp_path):
    _composition, destination, config = _config(tmp_path)
    executor = FakeExecutor(destination.catlass_revision)
    backend = CopyVBackend()
    build_phase2_production(
        config, environment=ENVIRONMENT, transport=_transport,
        executor=executor, backend=backend,
    )
    path = config.phase2_root / "cells" / destination.cell_id / "qualification.json"
    legacy = json.loads(path.read_text())
    legacy.pop("runtime_catlass_revision")
    legacy.pop("record_sha256")
    legacy["record_sha256"] = canonical_digest(legacy)
    path.write_text(json.dumps(legacy))

    build_phase2_production(
        config, environment=ENVIRONMENT, transport=_transport,
        executor=executor, backend=backend,
    )

    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "field,replacement",
    [("manifest_sha256", "b" * 64), ("cann_version", "9.1-patch")],
)
def test_retained_qualification_rejects_changed_live_runtime_provenance(
    tmp_path, field, replacement,
):
    _composition, destination, config = _config(tmp_path)
    original = FakeExecutor(destination.catlass_revision)
    backend = CopyVBackend()
    build_phase2_production(
        config, environment=ENVIRONMENT, transport=_transport,
        executor=original, backend=backend,
    )
    qualification = (
        config.phase2_root / "cells" / destination.cell_id / "qualification.json"
    )
    retained = qualification.read_bytes()
    changed = FakeExecutor(destination.catlass_revision)
    changed.runtime_provenance = tuple(
        (key, replacement if key == field else value)
        for key, value in changed.runtime_provenance
    )

    with pytest.raises(ValueError, match="runtime provenance changed"):
        build_phase2_production(
            config, environment=ENVIRONMENT, transport=_transport,
            executor=changed, backend=backend,
        )

    assert qualification.read_bytes() == retained
    assert changed.probes == []
    assert len(backend.calls) == 1


@pytest.mark.parametrize("selector", ["catlass", ["catlass"]])
def test_with_kdb_preflights_read_only_access_and_supplies_exact_guidance(tmp_path, selector):
    database = _knowledge_database(tmp_path / "knowledge")
    cache = tmp_path / "bge-cache"
    cache.mkdir()
    destination, config = _treatment_config(
        tmp_path,
        KnowledgeMode.WITH_KDB,
        ProfilingGuidance.WITH_GUIDANCE,
        bge_cache=cache,
        kdb=database,
        collection=selector,
    )
    dependencies = build_phase2_production(
        config,
        environment=ENVIRONMENT,
        transport=_transport,
        embeddings=QueryEmbeddings(),
        executor=FakeExecutor(destination.catlass_revision),
        backend=CopyVBackend(),
    )[1]

    assert dependencies.source.profiling_guidance == A5_PROFILING_GUIDANCE
    assert dependencies.source.knowledge_factory is not None
    agent = dependencies.source.knowledge_factory(tmp_path / "journal.jsonl", destination)
    try:
        assert agent.enabled is True
        assert agent.collection == "catlass"
    finally:
        agent.close()


def test_qualification_rejects_semantic_copy_failure_and_tampering(tmp_path):
    _composition, destination, config = _config(tmp_path)
    executor = FakeExecutor(destination.catlass_revision)

    class WrongBackend(CopyVBackend):
        def execute(self, source, execution, workdir):
            receipt = super().execute(source, execution, workdir)
            return AttentionReceipt(0, receipt.output[:-1], session_handle=receipt.session_handle)

    with pytest.raises(ValueError, match="qualification"):
        build_phase2_production(
            config, environment=ENVIRONMENT, transport=_transport,
            executor=executor, backend=WrongBackend(),
        )
    qualification = config.phase2_root / "cells" / destination.cell_id / "qualification.json"
    assert not qualification.exists()

    backend = CopyVBackend()
    build_phase2_production(
        config, environment=ENVIRONMENT, transport=_transport,
        executor=executor, backend=backend,
    )
    value = json.loads(qualification.read_text())
    value["output"][0] += 1
    qualification.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="qualification"):
        build_phase2_production(
            config, environment=ENVIRONMENT, transport=_transport,
            executor=executor, backend=backend,
        )
    assert len(backend.calls) == 1


def test_qualification_infrastructure_uncertainty_publishes_no_record(tmp_path):
    _composition, destination, config = _config(tmp_path)

    class UnavailableBackend:
        def execute(self, *_args):
            raise AttentionInfrastructureError("indeterminate retained operation")

    with pytest.raises(AttentionInfrastructureError, match="indeterminate"):
        build_phase2_production(
            config, environment=ENVIRONMENT, transport=_transport,
            executor=FakeExecutor(destination.catlass_revision),
            backend=UnavailableBackend(),
        )
    assert not (
        config.phase2_root / "cells" / destination.cell_id / "qualification.json"
    ).exists()


def test_identity_mismatch_fails_before_checked_runtime(tmp_path):
    _composition, destination, config = _config(
        tmp_path, snapshot_path=tmp_path / "wrong-snapshot",
    )
    executor = FakeExecutor(destination.catlass_revision)
    with pytest.raises(ValueError, match="snapshot path"):
        build_phase2_production(
            config, environment=ENVIRONMENT, transport=_transport,
            executor=executor, backend=CopyVBackend(),
        )
    assert executor.probes == []


@pytest.mark.parametrize(
    "environment,error",
    [
        ({"OPENAI_API_KEY": "present"}, "DEEPSEEK_API_KEY"),
        ({"DEEPSEEK_API_KEY": "present"}, "OPENAI_API_KEY"),
    ],
)
def test_missing_local_credentials_fail_before_bz(
    tmp_path, environment, error,
):
    _composition, destination, config = _config(tmp_path)
    executor = FakeExecutor(destination.catlass_revision)
    backend = CopyVBackend()

    with pytest.raises(ValueError, match=error):
        build_phase2_production(
            config, environment=environment, transport=_transport,
            executor=executor, backend=backend,
        )
    assert executor.probes == []
    assert backend.calls == []


def test_missing_openai_package_fails_before_bz(tmp_path, monkeypatch):
    _composition, destination, config = _config(tmp_path)
    executor = FakeExecutor(destination.catlass_revision)
    backend = CopyVBackend()
    monkeypatch.setattr(production_module.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="openai package"):
        build_phase2_production(
            config, environment=ENVIRONMENT,
            executor=executor, backend=backend,
        )
    assert executor.probes == []
    assert backend.calls == []


def test_missing_with_kdb_inputs_fail_before_bz(tmp_path):
    destination, config = _treatment_config(
        tmp_path, KnowledgeMode.WITH_KDB, ProfilingGuidance.WITHOUT_GUIDANCE,
    )
    executor = FakeExecutor(destination.catlass_revision)
    backend = CopyVBackend()

    with pytest.raises(ValueError, match="requires cache"):
        build_phase2_production(
            config, environment=ENVIRONMENT, transport=_transport,
            executor=executor, backend=backend,
        )
    assert executor.probes == []
    assert backend.calls == []


@pytest.mark.parametrize('failure', [RuntimeUnavailableError('offline'), ValueError('identity')])
def test_retired_cli_does_not_dispatch_or_retry_production(tmp_path, failure):
    calls, sleeps = [], []
    def factory(args):
        calls.append(args)
        raise failure
    with pytest.raises(SystemExit) as error:
        run_a5_phase2.main(['run', '--root', str(tmp_path / 'phase2'),
            '--snapshot', 'absent', '--cell-id', 'old-cell'],
            runtime_factory=factory, sleeper=sleeps.append)
    assert error.value.code == 2
    assert calls == [] and sleeps == []


pytestmark = pytest.mark.usefixtures("scripted_pi")


def _qualification_for_mode(tmp_path, mode):
    worker = (dict(execution="local", workspace=str(tmp_path / "workspace"),
                   physical_device=3, arch="test-arch") if mode == "local" else None)
    _, cell, config = _config(tmp_path, worker_config=worker)
    executor = FakeExecutor(cell.catlass_revision)
    executor.runtime_provenance = tuple(
        (key, "local" if mode == "local" else value)
        if key == "execution_profile" else (key, value)
        for key, value in executor.runtime_provenance
    )
    return config, executor


@pytest.mark.parametrize("mode,prefix,accepted", [
    ("local", "local", True), ("remote", "bz-a5", True),
    ("local", "bz-a5", False), ("remote", "local", False),
])
def test_qualification_handle_matches_execution_mode(tmp_path, mode, prefix, accepted):
    config, executor = _qualification_for_mode(tmp_path, mode)
    backend = CopyVBackend(prefix)
    options = dict(environment=ENVIRONMENT, transport=_transport,
                   executor=executor, backend=backend)
    path = config.phase2_root / "cells" / config.cell_id / "qualification.json"
    if not accepted:
        with pytest.raises(ValueError, match="invalid Phase 2 transport qualification"):
            build_phase2_production(config, **options)
        assert not path.exists()
        return

    snapshot, dependencies = build_phase2_production(config, **options)
    assert dependencies.execution_profile == ("local" if mode == "local" else None)
    assert json.loads(path.read_text())["session_handle"] == f"{prefix}:qualification"
    resumed, _ = build_phase2_production(config, **options)
    assert resumed == snapshot
    assert len(backend.calls) == 1


@pytest.mark.parametrize("mode,prefix,wrong_prefix", [
    ("local", "local", "bz-a5"), ("remote", "bz-a5", "local"),
])
def test_retained_qualification_rejects_other_mode_handle(tmp_path, mode, prefix, wrong_prefix):
    config, executor = _qualification_for_mode(tmp_path, mode)
    backend = CopyVBackend(prefix)
    options = dict(environment=ENVIRONMENT, transport=_transport,
                   executor=executor, backend=backend)
    build_phase2_production(config, **options)
    path = config.phase2_root / "cells" / config.cell_id / "qualification.json"
    value = json.loads(path.read_text())
    value["session_handle"] = f"{wrong_prefix}:qualification"
    value.pop("record_sha256")
    value["record_sha256"] = canonical_digest(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="invalid Phase 2 transport qualification"):
        build_phase2_production(config, **options)
    assert len(backend.calls) == 1


def test_local_qualification_rejects_changed_execution_identity(tmp_path):
    config, executor = _qualification_for_mode(tmp_path, "local")
    backend = CopyVBackend("local")
    options = dict(environment=ENVIRONMENT, transport=_transport,
                   executor=executor, backend=backend)
    build_phase2_production(config, **options)
    changed = replace(config, worker_config={**config.worker_config,
                      "workspace": str(tmp_path / "different-workspace")})
    with pytest.raises(ValueError, match="invalid Phase 2 transport qualification"):
        build_phase2_production(changed, **options)
    assert len(backend.calls) == 1


def test_local_qualification_still_checks_numerical_output(tmp_path):
    config, executor = _qualification_for_mode(tmp_path, "local")
    backend = CopyVBackend("local")
    options = dict(environment=ENVIRONMENT, transport=_transport,
                   executor=executor, backend=backend)
    build_phase2_production(config, **options)
    path = config.phase2_root / "cells" / config.cell_id / "qualification.json"
    value = json.loads(path.read_text())
    value["output"][0] += 1
    value["output_sha256"] = canonical_digest(value["output"])
    value.pop("record_sha256")
    value["record_sha256"] = canonical_digest(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="transport fixture did not reproduce input V"):
        build_phase2_production(config, **options)
    assert len(backend.calls) == 1
