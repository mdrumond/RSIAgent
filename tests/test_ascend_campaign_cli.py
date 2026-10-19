import json
import io
import threading
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.ascend import campaign_cli as cli


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def test_harness_revision_uses_checkout_or_real_export_substitution(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / ".gitattributes").write_text(".git_archival.txt export-subst\n")
    (checkout / ".git_archival.txt").write_text("node: $Format:%H$\n")
    subprocess.run(["git", "add", ".gitattributes", ".git_archival.txt"], cwd=checkout, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "Source export fixture"], cwd=checkout, check=True)
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    assert cli._harness_revision(checkout) == expected
    archive = subprocess.check_output(["git", "archive", "HEAD"], cwd=checkout)
    exported = tmp_path / "exported"
    exported.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        (exported / ".git_archival.txt").write_bytes(tree.extractfile(".git_archival.txt").read())
    assert not (exported / ".git").exists()
    assert cli._harness_revision(exported) == expected


@pytest.mark.parametrize("metadata", [None, "node: $Format:%H$\n", "node: unknown\n"])
def test_unknown_harness_revision_is_not_fabricated(tmp_path, metadata):
    if metadata is not None:
        (tmp_path / ".git_archival.txt").write_text(metadata)
    with pytest.raises(ValueError, match="Git checkout or git archive"):
        cli._harness_revision(tmp_path)


def start(tmp_path, *, phase=1, extra=(), screen=lambda *a: "session", seed=b"seed"):
    manifest = write(tmp_path / "sources.json", {
        "schema": "ascend-sft-sources-v2", "sources": [{"name": "catlass",
        "repository": "https://example.invalid/catlass", "revision": "a" * 40,
        "license": "Apache-2.0", "include": ["**/*.py"]}]})
    config = write(tmp_path / "settings.json", {"model": "openai/gpt-5.6-sol",
        "phase1_arguments": list(extra), "source_manifest": str(manifest)})
    memory = tmp_path / "seed"
    memory.write_bytes(seed)
    argv = ["start", "--control", str(tmp_path / "control"), "--root", str(tmp_path / "campaign"),
            "--target", "a3", "--phase", str(phase), "--objective", "Learn kernels",
            "--config", str(config), "--initial-memory", str(memory), "--project-target", "2"]
    if phase == 2:
        argv += ["--phase1-root", str(tmp_path / "old"), "--review", str(tmp_path / "review.json")]
    cli.main(argv, screen=screen)
    return cli._read(tmp_path / "control" / "job.json")


def test_start_retains_public_provenance_not_private_settings_and_separate_workers(tmp_path):
    calls = []
    job = start(tmp_path, screen=lambda job, role: calls.append(role) or role)
    assert calls == ["collector", "campaign"]
    root = Path(job["root"])
    assert (root / "initial-memory.bin").read_bytes() == b"seed"
    public = cli._read(root / "campaign-run.json")
    assert "config" not in public and "control" not in public
    assert public["phase"] == 1
    assert len(public["harness_revision"]) == 40
    assert (root / "source-evidence" / "manifest.json").is_file()
    with pytest.raises(SystemExit):
        start(tmp_path)


@pytest.mark.parametrize("legacy_seed", [False, True])
def test_phase2_cli_start_complete_and_strict_package_uses_reviewed_seed(tmp_path, legacy_seed):
    from dataclasses import replace
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
    from benchmarks.a5kernels.phase1_memory import HostFact
    from benchmarks.ascend.phase2_curriculum import AdaptivePhase2Coordinator
    from tests.test_a3_reviewed_phase2 import native_source
    from tests.test_a5_phase1_adaptive import _project, _result, _wave
    source, review, _ = native_source(tmp_path)
    source.rename(tmp_path / "old")
    job = start(tmp_path, phase=2, seed=b"")
    root = Path(job["root"])
    snapshot = cli._snapshot(job, {})
    # Older launchers retained the Phase 1 input, not the reviewed Phase 2 seed.
    if legacy_seed:
        (root / "initial-memory.bin").write_bytes(b"old unrelated launch input")
    original = (root / "initial-memory.bin").read_bytes()
    counter = 0
    def select(messages):
        nonlocal counter
        counter += 1
        spec = _project(f"phase2-{counter}", "gm-ub-movement")
        spec["learning_focus"] = ["pipe-mte2" if counter == 1 else "pipe-mte3"]
        return _wave(messages[-1]["content"], [spec])
    def execute(request):
        objects = ContentAddressedArtifacts(request.branch_root / "artifacts")
        source = objects.put_bytes(b"source")
        host = objects.put_bytes(b"host verification")
        result = _result(request.spec, True, "verified")
        return replace(result, memory=replace(result.memory, source_revision=source,
            host_facts=(HostFact("host-verification", "host passed", host),),
            lessons=(), no_new_knowledge=True, admitted_artifact_sha256=(host,), artifact_store="artifacts"))
    coordinator = AdaptivePhase2Coordinator(root, job["objective"], snapshot,
        CurriculumAgent(select), execute, project_target=2)
    assert coordinator.run()["successful_projects"] == 2
    output = tmp_path / "phase2.tar.gz"
    cli.main(["package", "--control", job["control"], "--output", str(output)])
    assert output.is_file()
    assert (root / "initial-memory.bin").read_bytes() == original
    with tarfile.open(output) as archive:
        assert archive.extractfile("campaign/initial-memory.bin").read() == original


def test_optional_collector_start_failure_does_not_prevent_compute(tmp_path):
    calls = []
    def screen(job, role):
        calls.append(role)
        if role == "collector":
            raise OSError("downloader unavailable")
        return role
    start(tmp_path, screen=screen)
    assert calls == ["collector", "campaign"]


@pytest.mark.parametrize("equals_form", [False, True])
def test_collector_uses_same_native_transport_executable_as_compute(tmp_path, equals_form):
    native = write(tmp_path / "native-config.json", {"cpl_remote": "/configured/transport/cpl-remote"})
    arguments = [f"--config={native}"] if equals_form else ["--config", str(native)]
    job = start(tmp_path, extra=["--model", "openai/gpt-5.6-sol", *arguments])
    captured = []
    class Downloader:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)
        def collect_once(self):
            return {"counts": {}}
    cli.collector_worker(job, once=True, factory=Downloader)
    assert captured == [{"cpl_remote": "/configured/transport/cpl-remote"}]


def test_resume_does_not_extend_and_continue_is_explicit(tmp_path):
    job = start(tmp_path)
    control = Path(job["control"])
    cli.main(["resume", "--control", str(control)], screen=lambda *a: "session")
    assert cli._read(control / "job.json")["project_target"] == 2
    cli.main(["continue", "--control", str(control), "--project-target", "4", "--reopen-terminal"],
             screen=lambda *a: "session")
    job = cli._read(control / "job.json")
    calls = []
    coordinator = SimpleNamespace(continue_campaign=lambda n, **kw: calls.append((n, kw)),
                                  resume=lambda: {"successful_projects": 4})
    cli.campaign_worker(job, builder=lambda job: coordinator)
    assert calls == [(4, {"reopen_terminal": True})]
    assert cli._read(control / "job.json")["project_target"] == 4
    assert cli._read(Path(job["root"]) / "campaign-run.json")["project_target"] == 4


def test_approval_lifecycle_selects_subset_and_runs_once(tmp_path):
    job = start(tmp_path)
    job["approval_mode"] = True
    cli._write(Path(job["control"]) / "job.json", job)
    proposals = [{"base_wave_index": 1, "base_memory_sha256": "a" * 64,
                  "wave": {"projects": [
                      {"spec": {"project_id": "one"}},
                      {"spec": {"project_id": "two"}},
                  ]}, "session": {"messages": [], "turns": 0}}]
    calls = []

    class Coordinator:
        def propose_wave(self):
            return proposals[0]
        def approve_wave(self, proposal, selected):
            calls.append(("approve", selected))
            return {**proposal, "wave": {"projects": [proposal["wave"]["projects"][1]]}}
        def run_approved_wave(self):
            calls.append(("run",))
            return {"successful_projects": 1}

    builder = lambda _job: Coordinator()
    sampled = cli.sample_curriculum(job, builder=builder)
    assert sampled["status"] == "proposed"
    assert [row["project_id"] for row in sampled["projects"]] == ["one", "two"]
    approved = cli.approve_curriculum(job, ["two"], sampled["proposal_id"], builder=builder)
    assert approved["status"] == "approved"
    result = cli.run_approved_curriculum(job, sampled["proposal_id"], builder=builder)
    assert result["result"] == {"successful_projects": 1}
    assert calls == [("approve", ("two",)), ("run",)]
    # Retrying the same request returns its retained result without re-execution.
    assert cli.run_approved_curriculum(job, sampled["proposal_id"], builder=builder) == result
    assert calls == [("approve", ("two",)), ("run",)]


def test_approval_lifecycle_rejects_stale_or_wrong_proposal(tmp_path):
    job = start(tmp_path)
    job["approval_mode"] = True
    cli._write(Path(job["control"]) / "job.json", job)

    class Coordinator:
        def propose_wave(self):
            return {"base_wave_index": 1, "base_memory_sha256": "a" * 64,
                    "wave": {"projects": [{"spec": {"project_id": "one"}}]},
                    "session": {"messages": [], "turns": 0}}
        def discard_proposal(self, proposal):
            raise ValueError("proposal no longer matches published memory")

    sampled = cli.sample_curriculum(job, builder=lambda _job: Coordinator())
    with pytest.raises(ValueError, match="proposal ID"):
        cli.reject_curriculum(job, "0" * 64, builder=lambda _job: Coordinator())
    with pytest.raises(ValueError, match="published memory"):
        cli.reject_curriculum(job, sampled["proposal_id"], builder=lambda _job: Coordinator())


def test_phase2_approval_requires_exactly_one_project(tmp_path):
    job = start(tmp_path, phase=2)
    job["approval_mode"] = True
    cli._write(Path(job["control"]) / "job.json", job)
    proposal = {"base_wave_index": 1, "base_memory_sha256": "a" * 64,
                "wave": {"projects": [{"spec": {"project_id": "one"}},
                                        {"spec": {"project_id": "two"}}]},
                "session": {"messages": [], "turns": 0}}

    class Coordinator:
        def propose_wave(self):
            return proposal
        def approve_wave(self, proposal, selected):
            assert len(selected) == 1
            return proposal

    sampled = cli.sample_curriculum(job, builder=lambda _job: Coordinator())
    with pytest.raises(ValueError, match="exactly one"):
        cli.approve_curriculum(job, ["one", "two"], sampled["proposal_id"],
                               builder=lambda _job: Coordinator())


def test_run_approved_recovers_after_wave_published_before_control_update(tmp_path):
    job = start(tmp_path)
    job["approval_mode"] = True
    control = Path(job["control"])
    cli._write(control / "job.json", job)
    proposal = {"base_wave_index": 3, "base_memory_sha256": "a" * 64,
                "wave": {"projects": [{"spec": {"project_id": "one"}}]}}
    proposal_id = cli._proposal_id(proposal)
    cli._write(control / "curriculum-approval.json", {
        "proposal_id": proposal_id, "status": "running", "proposal": proposal,
    })
    coordinator = SimpleNamespace(report=lambda: {"wave_index": 4, "successful_projects": 1})
    recovered = cli.run_approved_curriculum(
        job, proposal_id, builder=lambda _job: coordinator)
    assert recovered["status"] == "complete" and recovered["recovered"] is True


def test_running_published_proposal_cannot_be_overwritten_before_reconciliation(tmp_path):
    job = start(tmp_path)
    job["approval_mode"] = True
    control = Path(job["control"])
    cli._write(control / "job.json", job)
    proposal = {"base_wave_index": 3, "base_memory_sha256": "a" * 64,
                "wave": {"projects": [{"spec": {"project_id": "one"}}]}}
    proposal_id = cli._proposal_id(proposal)
    cli._write(control / "curriculum-approval.json", {
        "proposal_id": proposal_id, "status": "running", "proposal": proposal,
    })
    coordinator = SimpleNamespace(report=lambda: {"wave_index": 4, "status": "RUNNING"})
    with pytest.raises(ValueError, match="resolve the retained"):
        cli.sample_curriculum(job, builder=lambda _job: pytest.fail("must not sample"))
    recovered = cli.run_approved_curriculum(
        job, proposal_id, builder=lambda _job: coordinator)
    assert recovered["status"] == "complete" and recovered["recovered"] is True


def test_terminal_proposal_requires_empty_approval_and_commits_without_wave(tmp_path):
    from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
    from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
    from tests.test_a5_phase1_adaptive import SEED, _project, _result, _wave

    job = start(tmp_path)
    job["approval_mode"] = True
    cli._write(Path(job["control"]) / "job.json", job)
    selections, executions = [], []

    def select(messages):
        selections.append(1)
        if len(selections) == 1:
            return _wave(messages[-1]["content"], [
                _project("observed", "gm-ub-movement")])
        return 'DECISION: STALLED\n{"reason":"no useful next experiment"}'

    coordinator = AdaptivePhase1Coordinator(
        Path(job["root"]), "Learn kernels", SEED, CurriculumAgent(select),
        lambda request: executions.append(request) or _result(
            request.spec, True, "verified"), max_parallel=1, project_target=2)
    first = coordinator.propose_wave()
    coordinator.approve_wave(first, ("observed",))
    coordinator.run_approved_proposal(first)
    builder = lambda _job: coordinator
    sampled = cli.sample_curriculum(job, builder=builder)
    assert sampled["projects"] == [] and sampled["proposal"]["terminal"]["status"] == "STALLED"
    with pytest.raises(ValueError, match="zero project IDs"):
        cli.approve_curriculum(job, ["unexpected"], sampled["proposal_id"], builder=builder)
    approved = cli.approve_curriculum(job, [], sampled["proposal_id"], builder=builder)
    assert approved["status"] == "approved"
    completed = cli.run_approved_curriculum(job, sampled["proposal_id"], builder=builder)
    assert completed["result"]["status"] == "STALLED"
    assert len(executions) == 1
    assert len(list((Path(job["root"]) / "waves").iterdir())) == 1
    assert coordinator.resume()["status"] == "STALLED"
    assert len(selections) == 2


def test_approval_mutations_are_serialized_for_non_cli_callers(tmp_path):
    job = start(tmp_path)
    job["approval_mode"] = True
    cli._write(Path(job["control"]) / "job.json", job)
    entered, release = threading.Event(), threading.Event()

    class Coordinator:
        def propose_wave(self):
            entered.set()
            assert release.wait(2)
            return {"base_wave_index": 1, "base_memory_sha256": "a" * 64,
                    "wave": {"projects": [{"spec": {"project_id": "one"}}]}}

    thread = threading.Thread(target=lambda: cli.sample_curriculum(
        job, builder=lambda _job: Coordinator()))
    thread.start()
    assert entered.wait(2)
    with pytest.raises(RuntimeError, match="campaign control is busy"):
        cli.sample_curriculum(job, builder=lambda _job: Coordinator())
    release.set()
    thread.join(2)
    assert not thread.is_alive()

    class ReentrantCoordinator:
        def propose_wave(self):
            return cli.sample_curriculum(job, builder=lambda _job: self)

    (Path(job["control"]) / "curriculum-approval.json").unlink()
    with pytest.raises(RuntimeError, match="campaign control is busy"):
        cli.sample_curriculum(job, builder=lambda _job: ReentrantCoordinator())


def test_interrupted_approval_can_be_rejected_then_resampled(tmp_path, monkeypatch):
    from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
    from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
    from tests.test_a5_phase1_adaptive import SEED, _project, _result, _wave

    job = start(tmp_path)
    job["approval_mode"] = True
    cli._write(Path(job["control"]) / "job.json", job)
    selections = iter(["one", "two"])
    coordinator = AdaptivePhase1Coordinator(
        Path(job["root"]), "Learn kernels", SEED,
        CurriculumAgent(lambda messages: _wave(messages[-1]["content"], [
            _project(next(selections), "gm-ub-movement")
        ])), lambda request: _result(request.spec, True, "verified"),
        max_parallel=1, project_target=2)
    builder = lambda _job: coordinator
    sampled = cli.sample_curriculum(job, builder=builder)
    original_write = cli._write

    def interrupt(path, value):
        if Path(path) == Path(job["control"]) / "curriculum-approval.json" \
                and value.get("status") == "approved":
            raise OSError("control write interrupted")
        original_write(path, value)

    monkeypatch.setattr(cli, "_write", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        cli.approve_curriculum(job, ["one"], sampled["proposal_id"], builder=builder)
    assert (Path(job["root"]) / "pending.json").is_file()
    monkeypatch.setattr(cli, "_write", original_write)
    rejected = cli.reject_curriculum(job, sampled["proposal_id"], builder=builder)
    assert rejected["status"] == "rejected"
    assert not (Path(job["root"]) / "pending.json").exists()
    replacement = cli.sample_curriculum(job, builder=builder)
    assert replacement["projects"][0]["project_id"] == "two"


def test_approval_worker_waits_and_collector_packages_only_after_final_run(tmp_path, monkeypatch):
    job = start(tmp_path)
    job["approval_mode"] = True
    control = Path(job["control"])
    cli._write(control / "job.json", job)
    coordinator = SimpleNamespace(report=lambda: {"status": "RUNNING"})
    cli.campaign_worker(job, builder=lambda _job: coordinator)
    assert cli._read(control / "worker.json")["status"] == "waiting-approval"

    class Downloader:
        def __init__(self, *args, **kwargs):
            pass
        def collect_once(self):
            return {"counts": {}}

    packaged = []
    monkeypatch.setattr(cli, "package_campaign", lambda *args, **kwargs: packaged.append(args))
    cli.collector_worker(job, once=True, factory=Downloader)
    assert packaged == []
    proposal = {"base_wave_index": 1, "base_memory_sha256": "a" * 64,
                "wave": {"projects": [{"spec": {"project_id": "one"}}]}}
    proposal_id = cli._proposal_id(proposal)
    cli._write(control / "curriculum-approval.json", {
        "proposal_id": proposal_id, "status": "approved", "proposal": proposal,
    })
    runner = SimpleNamespace(run_approved_wave=lambda: {
        "status": "TARGET_REACHED", "wave_index": 2})
    cli.run_approved_curriculum(job, proposal_id, builder=lambda _job: runner)
    assert cli._read(control / "worker.json")["status"] == "complete"
    monkeypatch.setattr(cli, "inventory_campaign", lambda *args, **kwargs: {
        "complete": True, "inventory_sha256": "a" * 64, "remote_reports": []})
    cli.collector_worker(job, once=True, factory=Downloader)
    assert len(packaged) == 1


def test_worker_failure_and_unavailable_downloader_are_independent(tmp_path):
    job = start(tmp_path)
    def fail(job):
        raise OSError("infra interrupted")
    with pytest.raises(OSError):
        cli.campaign_worker(job, builder=fail)
    class Downloader:
        def __init__(self, *a):
            pass
        def collect_once(self):
            raise OSError("horrible connection")
    cli.collector_worker(job, once=True, factory=Downloader)
    control = Path(job["control"])
    assert cli._read(control / "worker.json")["status"] == "failed"
    assert cli._read(control / "package-status.json")["status"] == "pending"
    assert (Path(job["root"]) / "initial-memory.bin").read_bytes() == b"seed"


def test_compute_completion_queues_package_without_waiting_for_downloader(tmp_path):
    job = start(tmp_path)
    write(Path(job["root"]) / "state.json", {"completed_waves": []})
    cli.campaign_worker(job, builder=lambda j: SimpleNamespace(resume=lambda: {"done": True}))
    class Downloader:
        def __init__(self, *a):
            pass
        def collect_once(self):
            return {"counts": {"complete": 0}}
    cli.collector_worker(job, once=True, factory=Downloader)
    control = Path(job["control"])
    packaged = cli._read(control / "package-status.json")
    assert packaged["status"] == "complete"
    assert Path(packaged["archive"]).is_file()


def test_screen_uses_distinct_durable_local_commands_and_reuses_existing(tmp_path):
    job = start(tmp_path)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="No Sockets found.\n")
    campaign = cli.start_screen(job, "campaign", runner=run)
    collector = cli.start_screen(job, "collector", runner=run)
    assert campaign != collector
    assert commands[1][:2] == ["screen", "-dmS"]
    assert "_campaign" in commands[1] and "_collector" in commands[3]
    commands.clear()
    cli.start_screen(job, "campaign", runner=lambda command, **kw:
                     commands.append(command) or SimpleNamespace(stdout=f"123.{campaign}\t(Detached)\n"))
    assert commands == [["screen", "-ls"]]


@pytest.mark.skipif(shutil.which("screen") is None, reason="GNU Screen unavailable")
def test_real_screen_start_returns_while_worker_is_active(tmp_path):
    job = start(tmp_path)
    started = []
    def runner(command, **kwargs):
        if command[1] != "-ls":
            started.append(command[2])
            command = command[:6] + ["sleep", "30"]
        return subprocess.run(command, timeout=2, **kwargs)
    try:
        name = cli.start_screen(job, "collector", runner=runner)
        sessions = subprocess.run(["screen", "-ls"], capture_output=True, text=True, timeout=2)
        assert name in sessions.stdout
    finally:
        for name in started:
            subprocess.run(["screen", "-S", name, "-X", "quit"], capture_output=True, timeout=2)


@pytest.mark.parametrize("stale_current_archive", [False, True])
def test_live_collector_reloads_continuation_target_and_preserves_old_archive(tmp_path, stale_current_archive):
    job = start(tmp_path)
    control, root = Path(job["control"]), Path(job["root"])
    old_archive = control / "campaign-2.tar.gz"
    old_archive.write_bytes(b"previous completed campaign")
    write(root / "state.json", {"completed_waves": [], "project_target": 4})
    write(control / "worker.json", {"status": "complete"})
    current = {**job, "project_target": 4}
    write(control / "job.json", current)
    public = cli._read(root / "campaign-run.json")
    write(root / "campaign-run.json", {**public, "project_target": 4})
    stale = control / "campaign-4.tar.gz"
    if stale_current_archive:
        stale.write_bytes(b"incomplete old archive")
    class Downloader:
        def __init__(self, *args):
            pass
        def collect_once(self):
            return {"counts": {}}
    cli.collector_worker(job, once=True, factory=Downloader)
    packaged = cli._read(control / "package-status.json")
    assert packaged["status"] == "complete"
    assert Path(packaged["archive"]).name.startswith("campaign-4")
    assert old_archive.read_bytes() == b"previous completed campaign"
    if stale_current_archive:
        assert Path(packaged["archive"]) != stale
        assert stale.read_bytes() == b"incomplete old archive"


@pytest.mark.parametrize("target", ["a3", "a5"])
@pytest.mark.parametrize("phase", [1, 2])
def test_shared_builder_dispatch_and_phase2_snapshot(tmp_path, monkeypatch, target, phase):
    from benchmarks.ascend import phase1_cli, production
    from benchmarks.a5kernels import phase1_adaptive_production as a5production
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    job = start(tmp_path, phase=phase)
    job["target"] = target
    model = MODEL_IDENTITIES[0]
    captured = []
    source = dict(target="native", arch="native-arch", execution_profile="bz-a3-1",
        guide_sha256="a" * 64, cpl_skills_revision="b" * 40, environment_identity={},
        model=model.as_dict(), profiling_mode="without-profiling-guidance")
    native = SimpleNamespace(**{k: v for k, v in source.items() if k != "model"}, collection=None)
    snapshot = SimpleNamespace(destination=SimpleNamespace(source_identity=source))
    monkeypatch.setattr(cli, "_snapshot", lambda *a: snapshot)
    def old_cli(argv, *, native_builder, a5_main):
        assert argv[:2] == ["--target", target]
        native_builder(**({"config": native, "profile": SimpleNamespace(identity=model)}
                           if target == "a3" else {"cell": SimpleNamespace(model=model)}))
    monkeypatch.setattr(phase1_cli, "main", old_cli)
    coordinator = SimpleNamespace()
    def build(**options):
        captured.append(options)
        return coordinator if target == "a3" else SimpleNamespace(coordinator=coordinator)
    monkeypatch.setattr(production, "build_native_adaptive_phase1", build)
    monkeypatch.setattr(a5production, "build_adaptive_phase1", build)
    assert cli.build_controller(job) is coordinator
    assert callable(captured[0]["trace_recorder_factory"])
    assert ("reviewed_snapshot" in captured[0]) is (phase == 2)


def test_edited_configuration_refused_and_status_separates_verified_from_published(tmp_path):
    job = start(tmp_path)
    Path(job["config"]).write_text("{}")
    with pytest.raises(ValueError, match="settings changed"):
        cli.build_controller(job)
    root = Path(job["root"])
    write(root / "state.json", {"prior_outcomes": [{"project_id": "published"}], "completed_waves": []})
    for name in ("one", "two"):
        branch = root / "waves" / "wave" / "branches" / name
        branch.mkdir(parents=True)
        write(branch / "outcome.json", {"verified": True})
    result = cli.status(job)
    assert result["verified"] == 2 and result["published"] == 1
    assert result["published_per_hour"] > 0


def test_invalid_phase2_or_identity_override_fails_before_start(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["start", "--control", str(tmp_path / "control"), "--root", str(tmp_path / "campaign"),
                  "--target", "a3", "--phase", "2", "--objective", "Learn", "--config", "absent",
                  "--initial-memory", "absent", "--project-target", "1"])
    with pytest.raises(SystemExit):
        start(tmp_path, extra=["--project-target=100"])


def test_real_native_composition_runs_actor_host_verification_trace_and_package(tmp_path, monkeypatch):
    from benchmarks.ascend import phase1_cli
    from benchmarks.a3kernels.candidate import CandidateCompilation
    from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt
    from benchmarks.a5kernels.phase1_provider import DirectCompletion
    from benchmarks.ascend.training_traces import export_annex
    from pathlib import PurePosixPath
    guide = tmp_path / "guide.md"
    guide.write_text("Native programming guide")
    config = write(tmp_path / "native.json", dict(guide=str(guide), target="native-a3", arch="native-arch",
        execution_profile="bz-a3-1", cpl_skills_revision="a" * 40, cpl_remote="/tools/cpl-remote",
        validation_wrapper="/checked/validation", remote_workspace="/owned", physical_device=0,
        environment_identity=dict(cann_version="actual-cann", profiler_version="actual-msprof")))
    job = start(tmp_path, extra=["--config", str(config), "--model", "openai/gpt-5.6-sol",
        "--knowledge", "without-kdb", "--profiling", "without-profiling-guidance"])
    job["project_target"] = 1
    class Transport:
        def __init__(self, **kwargs):
            pass
        def compile(self, plan, directory):
            return CandidateCompilation(plan, "b" * 64, "compiled", "", "a" * 64)
        def execute_receipt(self, compiled):
            return ExecutionReceipt(0, compiled.plan.inputs[0].values, "synchronized")
        def remote_candidate_directory(self, plan):
            return PurePosixPath("/owned") / plan.execution_id
    source = 'extern "C" __global__ __aicore__ void project_kernel(GM_ADDR gm_input0, GM_ADDR gm_output) {}'
    actions = iter([json.dumps({"write": {"slot": "kernel", "content": source}}),
                    '{"compile":{}}', '{"run":{}}', '{"submit":{}}'])
    def provider(**kwargs):
        messages = kwargs["request"]["messages"]
        if "Curriculum Agent" in messages[0]["content"]:
            context = json.loads(messages[-1]["content"].rsplit("\n", 1)[-1])
            spec = dict(project_id="copy-study", capabilities=["gm-ub-movement"], shape=[4], dtype="float32",
                block_count=1, padding_multiple=1, failure_mode="none", timing_lengths=[], profile_metric=None,
                hypothesis="Study native copy", graph=dict(inputs=[dict(name="x", shape=[4])],
                nodes=[dict(name="out", op="copy", operands=["x"], axis=None)], output="out"))
            text = "DECISION: WAVE\n" + json.dumps(dict(wave_id=context["next_wave_id"],
                pre_wave_memory_sha256=context["pre_wave_memory_sha256"], pre_wave_coverage=context["pre_wave_coverage"],
                projects=[dict(spec=spec, requires=[])]))
        elif "Distill only" in messages[0]["content"] or "Distill only" in messages[-1]["content"]:
            text = '{"distillation":{"no_new_knowledge":true,"lessons":[]}}'
        else:
            text = next(actions)
        return DirectCompletion(text, 1)
    original = phase1_cli.main
    monkeypatch.setattr(phase1_cli, "main", lambda argv, **kwargs: original(argv, **kwargs,
        transport_factory=Transport, provider_transport=provider,
        environment_loader=lambda **kw: dict(OPENAI_API_KEY="fixture", DEEPSEEK_API_KEY="fixture")))
    result = cli.campaign_worker(job)
    assert result["successful_projects"] == 1
    assert len(list(Path(job["root"]).rglob("sft-traces/*.jsonl"))) == 1
    exported = export_annex(job["root"], tmp_path / "dataset")
    assert exported["dataset"]["eligible"] == 1
    archive = tmp_path / "complete.tar.gz"
    manifest = cli.package_campaign(job["root"], archive)
    assert manifest["complete"] and archive.exists()


def test_phase2_closed_semantic_failure_and_success_are_strictly_packaged(tmp_path):
    from dataclasses import replace
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.ascend.phase2_curriculum import AdaptivePhase2Coordinator
    from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
    from test_ascend_phase2_curriculum import project
    from tests.test_a3_reviewed_phase2 import native_source
    from benchmarks.ascend.phase2 import admit_native_snapshot
    from test_a5_phase1_adaptive import _result, _wave
    calls, executions = [], []
    def select(messages):
        calls.append(messages)
        return _wave(messages[-1]["content"], [project(len(calls))])
    def execute(request):
        executions.append(request)
        result = _result(request.spec, len(executions) > 1, "host outcome")
        objects = ContentAddressedArtifacts(request.branch_root / "artifacts")
        source = objects.put_bytes(b"retained candidate source")
        evidence = objects.put_bytes(result.evidence_payload)
        memory = replace(result.memory, source_revision=source,
            host_facts=tuple(replace(fact, evidence_sha256=evidence) for fact in result.memory.host_facts),
            admitted_artifact_sha256=(evidence,), lessons=(), no_new_knowledge=True)
        return replace(result, memory=memory)
    root = tmp_path / "phase2"
    source, review, _ = native_source(tmp_path)
    reviewed = admit_native_snapshot(source, b"", review, root)
    coordinator = AdaptivePhase2Coordinator(root, "Learn from failures and success", reviewed,
        CurriculumAgent(select), execute, project_target=1)
    result = coordinator.run()
    assert result["successful_projects"] == 1 and result["failed_projects"] == 1
    manifest = cli.package_campaign(root, tmp_path / "phase2.tar.gz")
    assert manifest["complete"]
    assert any(row["path"].endswith("closed-failure.json") for row in manifest["files"])
    assert sum(row["path"].startswith("waves/") and row["path"].endswith("outcome.json")
               for row in manifest["files"]) == 2
