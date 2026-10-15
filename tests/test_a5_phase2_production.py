from __future__ import annotations

import json

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
    AttentionInfrastructureError,
    AttentionReceipt,
    attention_reference,
)
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
    def __init__(self):
        self.calls = []

    def execute(self, source, execution, workdir):
        self.calls.append((source, execution, workdir))
        return AttentionReceipt(
            0, execution.v, session_handle="bz-a5:qualification",
            metadata=(("result_exit_code", "0"),),
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


@pytest.mark.parametrize(
    "failure",
    [RuntimeUnavailableError("offline"), AttentionInfrastructureError("lost")],
)
def test_cli_retries_infrastructure_exactly_three_times(tmp_path, failure, capsys):
    _composition, _source, destination, snapshot = _snapshot(tmp_path)
    dependencies = _dependencies(snapshot)
    calls = []
    sleeps = []

    def factory(_args):
        calls.append(1)
        if len(calls) <= 3:
            raise type(failure)(str(failure))
        return snapshot, dependencies

    assert run_a5_phase2.main(
        ["run", "--root", str(tmp_path / "phase2"),
         "--snapshot", str(snapshot.root), "--cell-id", destination.cell_id],
        runtime_factory=factory, sleeper=sleeps.append,
    ) == 0
    assert len(calls) == 4
    assert sleeps == [120, 120, 120]
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_cli_does_not_retry_semantic_configuration_failure(tmp_path):
    calls = []

    def factory(_args):
        calls.append(1)
        raise ValueError("identity mismatch")

    with pytest.raises(ValueError, match="identity"):
        run_a5_phase2.main(
            ["run", "--root", str(tmp_path / "phase2"),
             "--snapshot", str(tmp_path / "snapshot"), "--cell-id", "cell"],
            runtime_factory=factory, sleeper=pytest.fail,
        )
    assert calls == [1]


def test_cli_stops_after_three_infrastructure_retries(tmp_path):
    calls = []
    sleeps = []

    def factory(_args):
        calls.append(1)
        raise RuntimeUnavailableError("still offline")

    with pytest.raises(RuntimeUnavailableError, match="offline"):
        run_a5_phase2.main(
            ["run", "--root", str(tmp_path / "phase2"),
             "--snapshot", str(tmp_path / "snapshot"), "--cell-id", "cell"],
            runtime_factory=factory, sleeper=sleeps.append,
        )
    assert len(calls) == 4
    assert sleeps == [120, 120, 120]
