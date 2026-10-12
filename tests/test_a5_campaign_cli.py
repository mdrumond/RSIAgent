import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

import run_a5_phase1
import run_a5_phase2
import run_a5_campaign
from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.phase1_live import InfrastructureFailure
from benchmarks.a5kernels.priority_controller import Job, JobKind, Observation
from benchmarks.a5kernels.screen_workers import ScreenWorkerPool, run_request
from benchmarks.a5kernels.phase2_live import Phase2LiveRunner


def test_phase2_qualify_command_stops_before_model_or_live_runner(tmp_path, capsys):
    from tests.test_a5_phase2_cli import _dependencies
    from tests.test_a5_phase2_memory import _snapshot

    _composition, _source, destination, snapshot = _snapshot(tmp_path)
    dependencies = _dependencies(snapshot)
    dependencies = type(dependencies)(
        dependencies.cell,
        lambda *_args: pytest.fail("qualification called the model"),
        dependencies.verify,
    )
    qualification = snapshot.root.parent / "qualification.json"

    def factory(_args):
        qualification.parent.mkdir(parents=True, exist_ok=True)
        qualification.write_text('{"status":"transport-qualified"}\n')
        return snapshot, dependencies

    assert run_a5_phase2.main([
        "qualify", "--root", str(tmp_path / "phase2"),
        "--snapshot", str(snapshot.root), "--cell-id", destination.cell_id,
    ], runtime_factory=factory, sleeper=pytest.fail) == 0
    assert json.loads(capsys.readouterr().out) == {
        "cell_id": destination.cell_id, "status": "transport-qualified"
    }
    assert not (qualification.parent / "terminal.json").exists()


def test_screen_worker_consumes_exact_argv_and_retains_result(tmp_path):
    calls = []

    def screen(argv, **kwargs):
        calls.append((argv, kwargs))
        return type("Completed", (), {"returncode": 0, "stdout": ""})()

    root = tmp_path / "controller"
    worker = root / "workers" / "phase1-run" / "p1-a"
    pool = ScreenWorkerPool(root, screen_runner=screen,
                            script=Path("/repo/run_a5_campaign.py"))
    job = Job(JobKind.PHASE1_RUN, "p1-a")
    assignment = pool.start(job, 3, worker, ("python", "run.py"))
    request = worker / "attempt-0001" / "request.json"
    assert calls[0][0][:3] == ("screen", "-dmS", assignment.worker_handle)
    assert calls[0][0][-3:] == ("worker", "--request", str(request))
    value = json.loads(request.read_text())
    assert value["command"] == ["python", "run.py"]
    assert value["physical_device"] == 3

    def execute(argv, **kwargs):
        assert argv == ["python", "run.py"]
        assert kwargs == {"check": False}
        return type("Completed", (), {"returncode": 19})()

    run_request(request, process_runner=execute)
    assert pool.observe(assignment) == Observation.completed(19)


def test_screen_worker_start_does_not_wait_for_detached_output_holder(tmp_path):
    """A detached Screen descendant must not keep start's capture pipes open."""

    def screen(_argv, **kwargs):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess, sys; "
                    "subprocess.Popen([sys.executable, '-c', "
                    "'import time; time.sleep(1)'], "
                    "stdout=sys.stdout, stderr=sys.stderr)"
                ),
            ],
            timeout=0.25,
            **kwargs,
        )

    root = tmp_path / "controller"
    worker = root / "workers" / "phase1-run" / "p1-detach"
    pool = ScreenWorkerPool(
        root, screen_runner=screen, script=Path("/repo/run_a5_campaign.py")
    )

    started = time.monotonic()
    pool.start(Job(JobKind.PHASE1_RUN, "p1-detach"), 2, worker, ("run",))

    assert time.monotonic() - started < 0.25


def test_screen_worker_start_reports_screen_diagnostics(tmp_path):
    def screen(_argv, **kwargs):
        kwargs["stdout"].write("Cannot make directory /run/screen\n")
        return subprocess.CompletedProcess(_argv, 1)

    root = tmp_path / "controller"
    worker = root / "workers" / "phase1-run" / "p1-failed"
    pool = ScreenWorkerPool(
        root, screen_runner=screen, script=Path("/repo/run_a5_campaign.py")
    )

    with pytest.raises(RuntimeError, match="Cannot make directory /run/screen"):
        pool.start(Job(JobKind.PHASE1_RUN, "p1-failed"), 2, worker, ("run",))


def test_screen_worker_real_screen_returns_while_worker_is_active(
    tmp_path, monkeypatch,
):
    if shutil.which("screen") is None:
        pytest.skip("GNU Screen is unavailable")
    screen_dir = tmp_path / "screen"
    screen_dir.mkdir(mode=0o700)
    screen_dir.chmod(0o700)
    monkeypatch.setenv("SCREENDIR", str(screen_dir))
    root = tmp_path / "controller"
    worker = root / "workers" / "phase1-run" / "p1-real-screen"
    pool = ScreenWorkerPool(root, script=Path(run_a5_campaign.__file__))
    assignment = None

    try:
        started = time.monotonic()
        assignment = pool.start(
            Job(JobKind.PHASE1_RUN, "p1-real-screen"),
            2,
            worker,
            (sys.executable, "-c", "import time; time.sleep(2)"),
        )
        assert time.monotonic() - started < 0.5
        assert pool.observe(assignment) == Observation.active()
    finally:
        if assignment is not None:
            subprocess.run(
                ("screen", "-S", assignment.worker_handle, "-X", "quit"),
                text=True,
                capture_output=True,
                check=False,
                timeout=3,
            )


def test_screen_worker_observes_retained_screen_without_relaunch(tmp_path):
    outputs = []

    def screen(argv, **kwargs):
        if argv[:2] == ("screen", "-dmS"):
            return type("Completed", (), {"returncode": 0, "stdout": ""})()
        outputs.append(argv)
        return type("Completed", (), {
            "returncode": 0, "stdout": f"1234.{assignment.worker_handle}\n"
        })()

    root = tmp_path / "controller"
    worker = root / "workers" / "phase2-run" / "p2-a"
    pool = ScreenWorkerPool(root, screen_runner=screen,
                            script=Path("/repo/run_a5_campaign.py"))
    assignment = pool.start(
        Job(JobKind.PHASE2_RUN, "p2-a"), 4, worker, ("run",),
    )

    assert pool.observe(assignment) == Observation.active()
    assert outputs == [("screen", "-ls", assignment.worker_handle)]


def test_screen_worker_recovers_unrecorded_active_envelope(tmp_path):
    def screen(argv, **_kwargs):
        if argv[:2] == ("screen", "-dmS"):
            return type("Completed", (), {"returncode": 0, "stdout": ""})()
        return type("Completed", (), {
            "returncode": 0, "stdout": f"1234.{assignment.worker_handle}\n"
        })()

    root = tmp_path / "controller"
    worker = root / "workers" / "phase1-run" / "p1-a"
    pool = ScreenWorkerPool(root, screen_runner=screen,
                            script=Path("/repo/run_a5_campaign.py"))
    assignment = pool.start(
        Job(JobKind.PHASE1_RUN, "p1-a"), 2, worker, ("run",),
    )

    assert pool.recover() == (assignment,)


def test_until_complete_reconciles_again_after_bounded_poll(monkeypatch, capsys):
    reports = iter([
        type("Report", (), {"status": "incomplete", "as_dict": lambda self: {"status": self.status}})(),
        type("Report", (), {"status": "complete", "as_dict": lambda self: {"status": self.status}})(),
    ])
    controller = type("Controller", (), {"reconcile": lambda self: next(reports)})()
    monkeypatch.setattr(run_a5_campaign, "_controller", lambda _args: controller)
    sleeps = []
    required = [
        "--controller-root", "/c", "--phase1-root", "/p1", "--phase2-root", "/p2",
        "--gate", "/gate", "--guide", "/guide", "--tla-root", "/tla",
        "--profiling-skill-root", "/skill", "--bge-cache", "/bge", "--kdb", "/kdb",
        "--catlass-source", "/remote/catlass", "--collection", "catlass",
    ]

    assert run_a5_campaign.main(
        ["reconcile", *required, "--until-complete", "--poll-seconds", "2"],
        sleeper=sleeps.append,
    ) == 0
    assert sleeps == [2.0]
    assert [json.loads(line)["status"] for line in capsys.readouterr().out.splitlines()] == [
        "incomplete", "complete",
    ]


def test_until_complete_stops_on_blocked_and_retry_is_explicit(monkeypatch, capsys):
    calls = []
    report = type("Report", (), {
        "status": "blocked", "as_dict": lambda self: {"status": self.status},
    })()

    class Controller:
        def reconcile(self, **kwargs):
            calls.append(kwargs)
            return report

    monkeypatch.setattr(run_a5_campaign, "_controller", lambda _args: Controller())
    required = [
        "--controller-root", "/c", "--phase1-root", "/p1", "--phase2-root", "/p2",
        "--gate", "/gate", "--guide", "/guide", "--tla-root", "/tla",
        "--profiling-skill-root", "/skill", "--bge-cache", "/bge", "--kdb", "/kdb",
        "--catlass-source", "/remote/catlass", "--collection", "catlass",
    ]

    assert run_a5_campaign.main([
        "reconcile", *required, "--until-complete",
        "--retry-job", "phase1-run:p1-a",
    ], sleeper=pytest.fail) == 2
    assert calls == [{"retry_job_ids": ("phase1-run:p1-a",)}]
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_authenticated_qualification_returns_bound_physical_device(tmp_path):
    destination = type("Destination", (), {
        "cell_id": "p2-a", "catlass_revision": "a" * 40,
    })()
    args = type("Args", (), {
        "phase2_root": tmp_path, "catlass_source": "/remote/catlass",
    })()
    path = tmp_path / "cells" / "p2-a" / "qualification.json"
    path.parent.mkdir(parents=True)
    body = {
        "schema": "a5-catlass-phase2-transport-qualification-v1",
        "status": "transport-qualified", "cell_id": "p2-a",
        "catlass_revision": "a" * 40, "catlass_source": "/remote/catlass",
        "physical_device": 3,
    }
    path.write_text(json.dumps({**body, "record_sha256": canonical_digest(body)}))

    assert run_a5_campaign._qualification(args, destination) == 3


@pytest.mark.parametrize("module", [run_a5_phase1, run_a5_phase2])
def test_top_level_infrastructure_exhaustion_uses_exit_75(module, monkeypatch):
    monkeypatch.setattr(module, "main", lambda: (_ for _ in ()).throw(
        InfrastructureFailure("offline")
    ))
    assert module._entry() == 75


def test_top_level_semantic_error_is_not_reclassified(monkeypatch):
    monkeypatch.setattr(run_a5_phase2, "main", lambda: (_ for _ in ()).throw(
        ValueError("bad configuration")
    ))
    with pytest.raises(ValueError, match="configuration"):
        run_a5_phase2._entry()


def test_public_phase2_terminal_loader_authenticates_without_mutation(tmp_path):
    from tests.test_a5_phase2_cli import _dependencies
    from tests.test_a5_phase2_memory import _snapshot

    _composition, _source, _destination, snapshot = _snapshot(tmp_path)
    dependencies = _dependencies(snapshot)
    runner = Phase2LiveRunner(
        phase2_root=tmp_path / "phase2", snapshot=snapshot,
        dependencies=dependencies,
    )
    terminal = runner.run()
    before = {path: path.read_bytes() for path in runner.root.rglob("*") if path.is_file()}

    assert runner.load_terminal() == terminal
    assert {path: path.read_bytes() for path in runner.root.rglob("*") if path.is_file()} == before
