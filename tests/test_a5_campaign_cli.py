import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

import run_a5_phase1
import run_a5_phase2
import run_a5_campaign
from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.device_inventory import (
    DeviceInventoryError, DeviceInventoryUnavailable,
)
from benchmarks.a5kernels.phase1_live import InfrastructureFailure
from benchmarks.a5kernels.priority_controller import (
    Job, JobKind, Observation, RetainedAssignment,
)
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
            "returncode": 0,
            "stdout": (
                f"\t1234.{assignment.worker_handle}\t"
                "(10/12/26 12:00:00)\t(Detached)\n"
            ),
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


@pytest.mark.parametrize("listing", [
    "\t1234.{handle}\t(10/12/26 12:00:00)\t(Dead ???)\n",
    "\t1234.{handle}\t(Dead ???)\n",
    "\t1234.{handle}-suffix\t(10/12/26 12:00:00)\t(Detached)\n",
    "\t1234.{handle}-suffix\t(Detached)\n",
])
def test_screen_worker_rejects_non_live_or_inexact_sessions(tmp_path, listing):
    root = tmp_path / "controller"
    assignment = RetainedAssignment(
        Job(JobKind.PHASE1_RUN, "p1-stale"), 2, "rsi-a5-stale", 1,
    )

    def screen(_argv, **_kwargs):
        return subprocess.CompletedProcess(
            _argv, 0, stdout=listing.format(handle=assignment.worker_handle),
        )

    pool = ScreenWorkerPool(
        root, screen_runner=screen, script=Path("/repo/run_a5_campaign.py")
    )

    assert pool.observe(assignment) == Observation.incomplete()


@pytest.mark.parametrize(
    "state", ["Attached", "Detached", "Multi, attached", "Multi, detached"]
)
@pytest.mark.parametrize("metadata", ["(10/12/26 12:00:00)\t", ""])
def test_screen_worker_treats_exact_live_session_as_active(
    tmp_path, state, metadata
):
    handle = "rsi-a5-attached"

    def screen(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=f"\t1234.{handle}\t{metadata}({state})\n",
        )

    pool = ScreenWorkerPool(
        tmp_path, screen_runner=screen, script=Path("/repo/run_a5_campaign.py")
    )
    assignment = RetainedAssignment(
        Job(JobKind.PHASE1_RUN, "attached"), 1, handle, 1
    )

    assert pool.observe(assignment) == Observation.active()


def test_screen_worker_authenticated_result_precedes_dead_session(tmp_path):
    calls = []

    def screen(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, stdout="\t1234.rsi-a5-result\t(Dead ???)\n",
        )

    root = tmp_path / "controller"
    worker = root / "workers" / "phase1-run" / "p1-result"
    pool = ScreenWorkerPool(
        root, screen_runner=screen, script=Path("/repo/run_a5_campaign.py")
    )
    assignment = pool.start(
        Job(JobKind.PHASE1_RUN, "p1-result"), 2, worker, ("run",),
    )
    request = worker / "attempt-0001" / "request.json"
    run_request(
        request,
        process_runner=lambda _argv, **_kwargs: subprocess.CompletedProcess(
            _argv, 7,
        ),
    )

    assert pool.observe(assignment) == Observation.completed(7)
    assert len(calls) == 1


def test_screen_worker_real_dead_session_is_incomplete(tmp_path, monkeypatch):
    if shutil.which("screen") is None:
        pytest.skip("GNU Screen is unavailable")
    screen_dir = tmp_path / "screen"
    screen_dir.mkdir(mode=0o700)
    screen_dir.chmod(0o700)
    monkeypatch.setenv("SCREENDIR", str(screen_dir))
    handle = "rsi-a5-real-dead"
    assignment = RetainedAssignment(
        Job(JobKind.PHASE1_RUN, "p1-real-dead"), 2, handle, 1,
    )
    pool = ScreenWorkerPool(
        tmp_path / "controller", script=Path(run_a5_campaign.__file__)
    )

    subprocess.run(
        ("screen", "-dmS", handle, "sleep", "60"), check=True, timeout=3,
    )
    socket = next(screen_dir.glob(f"*.{handle}"))
    try:
        os.kill(int(socket.name.partition(".")[0]), signal.SIGKILL)
        for _ in range(30):
            listing = subprocess.run(
                ("screen", "-ls", handle), text=True, capture_output=True,
                check=False, timeout=3,
            )
            if "(Dead ???)" in listing.stdout:
                break
            time.sleep(0.01)
        assert "(Dead ???)" in listing.stdout
        assert pool.observe(assignment) == Observation.incomplete()
    finally:
        subprocess.run(
            ("screen", "-wipe"), text=True, capture_output=True,
            check=False, timeout=3,
        )


def test_screen_worker_recovers_unrecorded_active_envelope(tmp_path):
    def screen(argv, **_kwargs):
        if argv[:2] == ("screen", "-dmS"):
            return type("Completed", (), {"returncode": 0, "stdout": ""})()
        return type("Completed", (), {
            "returncode": 0,
            "stdout": (
                f"\t1234.{assignment.worker_handle}\t"
                "(10/12/26 12:00:00)\t(Detached)\n"
            ),
        })()

    root = tmp_path / "controller"
    worker = root / "workers" / "phase1-run" / "p1-a"
    pool = ScreenWorkerPool(root, screen_runner=screen,
                            script=Path("/repo/run_a5_campaign.py"))
    assignment = pool.start(
        Job(JobKind.PHASE1_RUN, "p1-a"), 2, worker, ("run",),
    )

    assert pool.recover() == (assignment,)


def test_screen_worker_does_not_recover_dead_envelope(tmp_path):
    def screen(argv, **_kwargs):
        if argv[:2] == ("screen", "-dmS"):
            return subprocess.CompletedProcess(argv, 0, stdout="")
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                f"\t1234.{assignment.worker_handle}\t"
                "(10/12/26 12:00:00)\t(Dead ???)\n"
            ),
        )

    root = tmp_path / "controller"
    worker = root / "workers" / "phase1-run" / "p1-dead"
    pool = ScreenWorkerPool(
        root, screen_runner=screen, script=Path("/repo/run_a5_campaign.py")
    )
    assignment = pool.start(
        Job(JobKind.PHASE1_RUN, "p1-dead"), 2, worker, ("run",),
    )

    assert pool.recover() == ()


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


def test_campaign_forwards_compile_probe_limit_only_to_phase1_workers():
    args = type("Args", (), {
        "gate": Path("/gate"), "guide": Path("/guide"),
        "phase1_root": Path("/p1"), "phase2_root": Path("/p2"),
        "tla_root": Path("/tla"), "profiling_skill_root": Path("/profile"),
        "catlass_source": "/remote/catlass", "bge_cache": Path("/bge"),
        "kdb": Path("/kdb"), "collection": "catlass", "env_file": None,
        "phase2_guide": None, "phase2_cpl_skills_revision": None,
        "compile_probes": 8,
    })()
    destination = type("Destination", (), {"cell_id": "p2-a"})()
    command = run_a5_campaign._command_factory(args, (destination,))

    phase1 = command(Job(JobKind.PHASE1_RUN, "p1-a"), 2)
    phase2 = command(Job(JobKind.PHASE2_QUALIFY, "p2-a"), 3)

    assert phase1[-2:] == ("--compile-probes", "8")
    assert "--compile-probes" not in phase2


def test_until_complete_retries_inventory_unavailability_without_scheduling(
    monkeypatch, capsys,
):
    complete = type("Report", (), {
        "status": "complete", "as_dict": lambda self: {"status": self.status},
    })()

    class Controller:
        calls = 0

        def reconcile(self):
            self.calls += 1
            if self.calls == 1:
                raise DeviceInventoryUnavailable("transport details are private")
            return complete

    controller = Controller()
    monkeypatch.setattr(run_a5_campaign, "_controller", lambda _args: controller)
    required = [
        "--controller-root", "/c", "--phase1-root", "/p1", "--phase2-root", "/p2",
        "--gate", "/gate", "--guide", "/guide", "--tla-root", "/tla",
        "--profiling-skill-root", "/skill", "--bge-cache", "/bge", "--kdb", "/kdb",
        "--catlass-source", "/remote/catlass", "--collection", "catlass",
    ]
    sleeps = []

    assert run_a5_campaign.main(
        ["reconcile", *required, "--until-complete", "--poll-seconds", "2"],
        sleeper=sleeps.append,
    ) == 0
    assert controller.calls == 2
    assert sleeps == [2.0]
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [
        {
            "reason": "device-inventory-unavailable",
            "retry_in_seconds": 2.0,
            "status": "retrying",
        },
        {"status": "complete"},
    ]


def test_one_shot_inventory_unavailability_fails_closed(monkeypatch):
    class Controller:
        def reconcile(self):
            raise DeviceInventoryUnavailable("offline")

    monkeypatch.setattr(run_a5_campaign, "_controller", lambda _args: Controller())
    required = [
        "--controller-root", "/c", "--phase1-root", "/p1", "--phase2-root", "/p2",
        "--gate", "/gate", "--guide", "/guide", "--tla-root", "/tla",
        "--profiling-skill-root", "/skill", "--bge-cache", "/bge", "--kdb", "/kdb",
        "--catlass-source", "/remote/catlass", "--collection", "catlass",
    ]

    with pytest.raises(DeviceInventoryUnavailable, match="offline"):
        run_a5_campaign.main(["reconcile", *required], sleeper=pytest.fail)


def test_until_complete_does_not_retry_semantic_inventory_error(monkeypatch):
    class Controller:
        def reconcile(self):
            raise DeviceInventoryError("malformed checked evidence")

    monkeypatch.setattr(run_a5_campaign, "_controller", lambda _args: Controller())
    required = [
        "--controller-root", "/c", "--phase1-root", "/p1", "--phase2-root", "/p2",
        "--gate", "/gate", "--guide", "/guide", "--tla-root", "/tla",
        "--profiling-skill-root", "/skill", "--bge-cache", "/bge", "--kdb", "/kdb",
        "--catlass-source", "/remote/catlass", "--collection", "catlass",
    ]

    with pytest.raises(DeviceInventoryError, match="malformed checked evidence"):
        run_a5_campaign.main(
            ["reconcile", *required, "--until-complete"], sleeper=pytest.fail,
        )


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
def test_top_level_infrastructure_exhaustion_uses_exit_75(
    module, monkeypatch, capsys,
):
    monkeypatch.setattr(module, "main", lambda: (_ for _ in ()).throw(
        InfrastructureFailure("offline\nwithout traceback")
    ))
    assert module._entry() == 75
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "InfrastructureFailure: offline without traceback\n"


@pytest.mark.parametrize("module", [run_a5_phase1, run_a5_phase2])
def test_top_level_infrastructure_diagnostic_is_bounded(module, monkeypatch, capsys):
    monkeypatch.setattr(module, "main", lambda: (_ for _ in ()).throw(
        InfrastructureFailure("x" * 1024)
    ))
    assert module._entry() == 75
    diagnostic = capsys.readouterr().err.rstrip("\n")
    assert diagnostic.startswith("InfrastructureFailure: ")
    assert diagnostic.endswith("...")
    assert len(diagnostic.encode("utf-8")) <= 512


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
