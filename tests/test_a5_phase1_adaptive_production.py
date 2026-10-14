from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.bz import CommandResult
from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActionContext
from benchmarks.a5kernels.phase1_adaptive_production import (
    A5_PROFILING_GUIDANCE,
    AdaptiveProductionPaths,
    build_adaptive_phase1,
)
import benchmarks.a5kernels.phase1_adaptive_production as production
from benchmarks.a5kernels.phase1_adaptive_cli import main as cli_main
from benchmarks.a5kernels.phase1_curriculum import Capability
from benchmarks.a5kernels.phase1_experiments import (
    KnowledgeMode,
    ProfilingGuidance,
    ProgrammingGuideIdentity,
    build_cells,
)
from benchmarks.a5kernels.phase1_project_bz import CatlassProjectBackend
from benchmarks.a5kernels.phase1_provider import DirectActorDriver, DirectCompletion


GUIDE_BYTES = b"Pinned Catlass DSL programming guide.\n"


class FakeCommandExecutor:
    runtime_provenance = (("catlass_revision", "a" * 40),)

    def run(self, _invocation):
        return CommandResult(1, "")

    def inspect(self, _argv):
        return CommandResult(1, "")

    def observe(self, _handle):
        return CommandResult(1, "")


class FakeEmbeddings:
    model = "fake"
    revision = "fake-v1"

    def embed(self, texts):
        return [[1.0, float("movement" in text)] for text in texts]


def _guide():
    return ProgrammingGuideIdentity(
        "catlass-dsl-programming-guide-v1", hashlib.sha256(GUIDE_BYTES).hexdigest(),
        "c" * 40,
        "9a6ac627b5f4078060287844189730cf0d184800",
    )


def _cell(knowledge=KnowledgeMode.WITHOUT_KDB,
          profiling=ProfilingGuidance.WITHOUT_GUIDANCE):
    return next(
        cell for cell in build_cells(_guide())
        if cell.knowledge is knowledge and cell.profiling is profiling
    )


def _paths(tmp_path, **changes):
    values = {
        "root": tmp_path / "campaign",
        "tla_root": tmp_path,
        "catlass_source": "/opt/catlass",
        "catlass_revision": "9a6ac627b5f4078060287844189730cf0d184800",
        "device": 2,
    }
    values.update(changes)
    return AdaptiveProductionPaths(**values)


def _provider_transport(**_kwargs):
    raise AssertionError("construction must not call the provider")


def _build(tmp_path, cell, **kwargs):
    return build_adaptive_phase1(
        cell=cell,
        paths=_paths(tmp_path, **kwargs.pop("path_changes", {})),
        objective="Learn one generic elementwise Catlass capability.",
        initial_memory=kwargs.pop("initial_memory", b""),
        guide_bytes=kwargs.pop("guide_bytes", GUIDE_BYTES),
        environment={"OPENAI_API_KEY": "x", "DEEPSEEK_API_KEY": "y"},
        provider_transport=_provider_transport,
        command_executor=kwargs.pop("command_executor", FakeCommandExecutor()),
        **kwargs,
    )


def test_factory_binds_registered_direct_actor_checked_backend_and_coordinator(tmp_path):
    campaign = _build(tmp_path, _cell())

    assert isinstance(campaign.actor, DirectActorDriver)
    assert campaign.profile.identity == campaign.cell.model
    assert isinstance(campaign.backend, CatlassProjectBackend)
    assert isinstance(campaign.coordinator, AdaptivePhase1Coordinator)
    assert campaign.coordinator.max_parallel == 1
    assert campaign.backend._device == 2
    assert campaign.coordinator.execute_project is not campaign.project_actor
    assert campaign.coordinator.curriculum.unavailable_capabilities == (
        Capability.TIMING_SWEEP, Capability.PROFILING,
    )


def test_selected_model_uses_its_registered_direct_provider_route(tmp_path):
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return DirectCompletion("DECISION: STALLED\n{\"reason\":\"observed failure\"}", 3)

    campaign = build_adaptive_phase1(
        cell=_cell(), paths=_paths(tmp_path), objective="Learn one project.",
        initial_memory=b"", guide_bytes=GUIDE_BYTES, environment={
            "OPENAI_API_KEY": "openai-secret", "DEEPSEEK_API_KEY": "deepseek-secret",
        }, provider_transport=transport, command_executor=FakeCommandExecutor(),
    )

    campaign.actor.provider.complete(campaign.profile, ({"role": "user", "content": "x"},))

    assert calls[0]["base_url"] == "https://api.openai.com/v1"
    assert calls[0]["request"]["model"] == "gpt-5.6-sol"
    assert calls[0]["api_key"] == "openai-secret"


def _checked_wrappers(tmp_path):
    paths = _paths(tmp_path)
    for wrapper in (
        paths.validation_wrapper, paths.upload_wrapper, paths.session_wrapper,
    ):
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
        wrapper.chmod(0o755)
    return paths


def test_default_backend_proves_provenance_then_device_before_provider(tmp_path, monkeypatch):
    events = []
    paths = _checked_wrappers(tmp_path)

    class Checked(FakeCommandExecutor):
        def __init__(self, **_kwargs):
            events.append("construct-command")

        @property
        def runtime_provenance(self):
            events.append("provenance")
            return (
                ("catlass_revision", paths.catlass_revision),
                ("catlass_source", paths.catlass_source),
                ("execution_profile", "bz-a5"),
            )

        def probe_device(self, device):
            events.append(("device", device))

    real_provider = production.DirectCompletionProvider

    class Provider(real_provider):
        def __init__(self, *args, **kwargs):
            events.append("provider")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(production, "CatlassValidationExecutor", Checked)
    monkeypatch.setattr(production, "DirectCompletionProvider", Provider)

    build_adaptive_phase1(
        cell=_cell(), paths=paths, objective="Learn.", initial_memory=b"",
        guide_bytes=GUIDE_BYTES,
        environment={"OPENAI_API_KEY": "x", "DEEPSEEK_API_KEY": "y"},
        provider_transport=_provider_transport,
    )

    assert events == [
        "construct-command", "provenance", ("device", 2), "provider",
    ]


def test_default_backend_rejects_mismatched_provenance_before_provider(tmp_path, monkeypatch):
    paths = _checked_wrappers(tmp_path)

    class Mismatched(FakeCommandExecutor):
        def __init__(self, **_kwargs):
            pass

        runtime_provenance = (
            ("catlass_revision", "f" * 40),
            ("catlass_source", "/wrong/source"),
            ("execution_profile", "bz-a5"),
        )

        def probe_device(self, _device):
            raise AssertionError("mismatched provenance must fail before device probe")

    monkeypatch.setattr(production, "CatlassValidationExecutor", Mismatched)
    monkeypatch.setattr(
        production, "DirectCompletionProvider",
        lambda *_args, **_kwargs: pytest.fail("provider constructed before preflight"),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        build_adaptive_phase1(
            cell=_cell(), paths=paths, objective="Learn.", initial_memory=b"",
            guide_bytes=GUIDE_BYTES,
            environment={"OPENAI_API_KEY": "x", "DEEPSEEK_API_KEY": "y"},
        )


def test_injected_command_executor_is_not_preflighted(tmp_path):
    class Injected(FakeCommandExecutor):
        @property
        def runtime_provenance(self):
            raise AssertionError("injected seam must remain lazy")

        def probe_device(self, _device):
            raise AssertionError("injected seam must not be probed")

    _build(tmp_path, _cell(), command_executor=Injected())


def test_actor_context_binds_guide_and_guidance_without_wrapping_memory(tmp_path):
    user_memory = b'{"prior":"host-owned"}'
    control = _build(tmp_path / "control", _cell(), initial_memory=user_memory)
    treatment = _build(
        tmp_path / "treatment",
        _cell(profiling=ProfilingGuidance.WITH_GUIDANCE),
        initial_memory=user_memory,
    )
    control_context = json.loads(control.project_actor.immutable_context)
    treatment_context = json.loads(treatment.project_actor.immutable_context)

    assert control_context["programming_guide"] == treatment_context["programming_guide"]
    assert control_context["programming_guide"]["text"].encode() == GUIDE_BYTES
    assert control_context["programming_guide"]["sha256"] == _guide().guide_sha256
    assert control.coordinator.initial_memory == user_memory
    assert treatment.coordinator.initial_memory == user_memory
    assert user_memory not in control.project_actor.immutable_context
    assert control.project_actor.immutable_context.decode().count(
        GUIDE_BYTES.decode().rstrip()
    ) == 1
    assert control_context["profiling_guidance"] is None
    assert treatment_context["profiling_guidance"] == A5_PROFILING_GUIDANCE


def test_guide_identity_mismatch_fails_before_composition(tmp_path):
    with pytest.raises(ValueError, match="guide bytes"):
        _build(tmp_path, _cell(), guide_bytes=b"different guide")


def test_runtime_revision_must_match_admitted_guide_before_external_work(tmp_path):
    with pytest.raises(ValueError, match="runtime Catlass revision"):
        _build(
            tmp_path,
            _cell(),
            path_changes={"catlass_revision": "a" * 40},
        )


def test_campaign_identity_is_persisted_and_unchanged_reconstruction_works(tmp_path):
    first = _build(tmp_path, _cell())
    identity_path = first.coordinator.root / "campaign-identity.json"
    identity = json.loads(identity_path.read_text())

    second = _build(tmp_path, _cell())

    assert identity == {
        "schema": "a5-phase1-production-identity-v1",
        "cell": first.cell.as_dict(),
    }
    assert second.cell == first.cell
    assert identity_path.stat().st_mode & 0o777 == 0o600


def test_changed_cell_and_legacy_state_fail_before_external_probes(tmp_path, monkeypatch):
    _build(tmp_path, _cell())
    monkeypatch.setattr(
        production, "_checked_command",
        lambda _paths: pytest.fail("remote probe ran before identity rejection"),
    )
    with pytest.raises(ValueError, match="different treatment cell"):
        _build(
            tmp_path,
            _cell(profiling=ProfilingGuidance.WITH_GUIDANCE),
            command_executor=None,
        )

    legacy = tmp_path / "legacy"
    (legacy / "campaign").mkdir(parents=True)
    (legacy / "campaign" / "state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy adaptive state"):
        _build(legacy, _cell())


def test_control_treatments_are_explicitly_disabled(tmp_path):
    campaign = _build(tmp_path, _cell())
    context = object()

    knowledge = campaign.query_knowledge({"query": "movement"}, context)
    profile = campaign.request_profile({}, context)

    assert knowledge.observation["available"] is False
    assert knowledge.observation["reason"] == "disabled-by-treatment"
    assert profile.observation["available"] is False
    assert profile.observation["reason"] == "disabled-by-treatment"


def test_guidance_treatment_returns_guidance_without_claiming_measurement(tmp_path):
    campaign = _build(
        tmp_path, _cell(profiling=ProfilingGuidance.WITH_GUIDANCE)
    )

    result = campaign.request_profile({}, object())

    assert result.observation == {
        "event": "profiling-guidance",
        "available": True,
        "guidance": A5_PROFILING_GUIDANCE,
        "measurement_available": False,
    }
    assert result.host_facts == ()
    assert result.admitted_artifacts == {}


def test_kdb_treatment_uses_read_only_agent_and_retains_exact_citation(tmp_path):
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    source = tmp_path / "sources"
    source.mkdir()
    guide = source / "guide.md"
    guide.write_text("movement uses a host verified copy\n", encoding="utf-8")
    kdb = tmp_path / "knowledge.sqlite"
    with KnowledgeDB(kdb, FakeEmbeddings()) as database:
        database.index(source, [guide], collection="catlass", language="catlass-dsl")
    campaign = _build(
        tmp_path,
        _cell(knowledge=KnowledgeMode.WITH_KDB),
        embeddings=FakeEmbeddings(),
        path_changes={"kdb": kdb, "collection": "catlass"},
    )
    artifact_root = tmp_path / "action-artifacts"
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    context = AdaptiveActionContext(
        request=SimpleNamespace(branch_root=tmp_path / "branch"), prepared=None,
        source=None, latest_execution=None,
        artifacts=ContentAddressedArtifacts(artifact_root),
    )

    result = campaign.query_knowledge(
        {"query": "movement", "limit": 1}, context
    )

    assert result.observation["event"] == "knowledge-query"
    assert result.observation["available"] is True
    assert len(result.citations) == 1
    retained = result.observation["results"][0]
    assert retained["citation"]["path"].endswith("guide.md")
    digest = retained["artifact_sha256"]
    assert context.artifacts.read_bytes(digest).decode() == retained["text"]
    assert result.admitted_artifacts == {digest: "knowledge"}


def test_kdb_manifest_must_match_supplied_embedding_backend(tmp_path):
    from benchmarks.a5kernels.knowledge import KnowledgeDB

    source = tmp_path / "sources"
    source.mkdir()
    guide = source / "guide.md"
    guide.write_text("movement guidance\n", encoding="utf-8")
    kdb = tmp_path / "knowledge.sqlite"
    with KnowledgeDB(kdb, FakeEmbeddings()) as database:
        database.index(source, [guide], collection="catlass", language="catlass-dsl")

    class WrongEmbeddings(FakeEmbeddings):
        revision = "wrong-revision"

    with pytest.raises(ValueError, match="manifest"):
        _build(
            tmp_path,
            _cell(knowledge=KnowledgeMode.WITH_KDB),
            embeddings=WrongEmbeddings(),
            path_changes={"kdb": kdb, "collection": "catlass"},
        )


@pytest.mark.parametrize(
    "spec",
    [
        SimpleNamespace(timing_lengths=(64, 128), profile_metric=None),
        SimpleNamespace(timing_lengths=(), profile_metric="timeline"),
    ],
)
def test_unsupported_measurement_projects_stop_before_actor(tmp_path, spec):
    campaign = _build(tmp_path, _cell())

    with pytest.raises(ValueError, match="does not support"):
        campaign.coordinator.execute_project(SimpleNamespace(spec=spec))


@pytest.mark.parametrize(
    ("cell", "path_changes", "message"),
    [
        (_cell(profiling=ProfilingGuidance.NEW), {}, "pending"),
        (_cell(knowledge=KnowledgeMode.WITH_KDB), {}, "KDB-on"),
        (_cell(), {"kdb": "unexpected.sqlite", "collection": "catlass"}, "KDB-off"),
    ],
)
def test_configuration_fails_closed(tmp_path, cell, path_changes, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        _build(tmp_path, cell, path_changes=path_changes, embeddings=FakeEmbeddings())


def test_resume_and_report_delegate_to_same_retained_coordinator(tmp_path, monkeypatch):
    campaign = _build(tmp_path, _cell())
    expected = {"status": "RUNNING", "memory_sha256": canonical_digest([])}
    calls = []
    monkeypatch.setattr(campaign.coordinator, "resume", lambda: calls.append("resume") or expected)
    monkeypatch.setattr(campaign.coordinator, "report", lambda: calls.append("report") or expected)

    assert campaign.resume() == expected
    assert campaign.report() == expected
    assert calls == ["resume", "report"]


@pytest.mark.parametrize("action", ["run", "resume"])
def test_cli_wires_each_action_to_the_production_factory(
    tmp_path, capsys, action,
):
    guide_path = tmp_path / "guide.md"
    guide_path.write_bytes(GUIDE_BYTES)
    memory_path = tmp_path / "memory.json"
    memory_path.write_bytes(b'{"prior":1}')
    captured = {}

    class Campaign:
        def __getattr__(self, name):
            return lambda: {"action": name, "status": "ok"}

    def builder(**kwargs):
        captured.update(kwargs)
        return Campaign()

    cli_main([
        action,
        "--root", str(tmp_path / "results"),
        "--objective", "Learn exactly one generic project.",
        "--initial-memory", str(memory_path),
        "--guide", str(guide_path),
        "--cpl-skills-revision", "c" * 40,
        "--tla-root", str(tmp_path),
        "--catlass-source", "/opt/catlass",
        "--catlass-revision", "9a6ac627b5f4078060287844189730cf0d184800",
        "--model", "openai/gpt-5.6-sol",
        "--knowledge", "without-kdb",
        "--profiling", "with-profiling-guidance",
        "--device", "3",
        "--max-parallel", "2",
        "--env-file", str(tmp_path / ".env"),
    ], builder=builder, environment_loader=lambda **_kwargs: {"loaded": "env"})

    assert json.loads(capsys.readouterr().out) == {"action": action, "status": "ok"}
    assert captured["guide_bytes"] == GUIDE_BYTES
    assert captured["initial_memory"] == b'{"prior":1}'
    assert captured["environment"] == {"loaded": "env"}
    assert captured["cell"].profiling is ProfilingGuidance.WITH_GUIDANCE
    assert captured["paths"].device == 3
    assert captured["max_parallel"] == 2


def test_cli_report_reads_retained_state_without_live_dependencies(tmp_path, capsys):
    memory_path = tmp_path / "memory.json"
    memory_path.write_bytes(b'{"prior":1}')
    campaign = _build(tmp_path, _cell(), initial_memory=memory_path.read_bytes())
    expected = campaign.coordinator.report()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline report constructed a live dependency")

    cli_main([
        "report",
        "--root", str(campaign.coordinator.root),
        "--objective", "Learn one generic elementwise Catlass capability.",
        "--initial-memory", str(memory_path),
    ], builder=forbidden, environment_loader=forbidden, embedding_factory=forbidden)

    assert json.loads(capsys.readouterr().out) == expected
