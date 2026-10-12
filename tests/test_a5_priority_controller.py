import json
from pathlib import Path

import pytest

from benchmarks.a5kernels.priority_controller import (
    CampaignController,
    ControllerBusy,
    Job,
    JobKind,
    Observation,
    RetainedAssignment,
)


class Inventory:
    def __init__(self, *devices):
        self.devices = devices

    def profiling_snapshot(self):
        return type("Snapshot", (), {
            "eligible_devices": tuple(
                type("Device", (), {"physical_device": item})()
                for item in self.devices
            )
        })()


class Campaign:
    def __init__(self, root, cells=("p1-b", "p1-a"), completed=()):
        self.root = Path(root)
        self.cells = tuple(type("Cell", (), {"cell_id": item})() for item in cells)
        self.completed = set(completed)
        self.qualification = type("Gate", (), {"report_sha256": "a" * 64})()
        self.guide = object()

    def load_terminal(self, cell):
        return {"status": "passed"} if cell.cell_id in self.completed else None


class Workers:
    def __init__(self, observations=None):
        self.observations = observations or {}
        self.started = []

    def recover(self):
        return ()

    def observe(self, assignment):
        return self.observations.get(assignment.job.job_id, Observation.active())

    def start(self, job, device, worker_root, command):
        self.started.append((job, device, worker_root, command))
        return RetainedAssignment(job, device, f"worker-{len(self.started)}", 100)


def controller(tmp_path, *, devices=(1, 2), completed=(), workers=None,
               phase1_progress=(), qualified=(), phase2_terminal=()):
    campaign = Campaign(tmp_path / "phase1", completed=completed)
    destinations = tuple(
        type("Destination", (), {"cell_id": f"p2-{name[-1]}",
                                  "source_cell_id": name})()
        for name in ("p1-b", "p1-a")
    )
    return CampaignController(
        root=tmp_path / "controller",
        phase1=campaign,
        phase2_root=tmp_path / "phase2",
        destinations=destinations,
        inventory=Inventory(*devices),
        workers=workers or Workers(),
        command=lambda job, device: ("worker", job.kind.value, job.cell_id,
                                     str(device)),
        phase1_progress=lambda cell_id: cell_id in phase1_progress,
        snapshot=lambda destination: tmp_path / "phase2" / destination.cell_id / "snapshot",
        qualified=lambda destination: destination.cell_id in qualified,
        phase2_terminal=lambda destination: (
            {"status": "passed"} if destination.cell_id in phase2_terminal else None
        ),
    )


def test_deterministic_phase1_queue_fills_idle_devices_without_leases(tmp_path):
    subject = controller(tmp_path)

    report = subject.reconcile()

    assert [(item.job.cell_id, item.physical_device) for item in report.active] == [
        ("p1-a", 1), ("p1-b", 2)
    ]
    assert [item.job.kind for item in report.active] == [JobKind.PHASE1_RUN] * 2
    assert not list((tmp_path / "controller").glob("*lease*"))
    assert report.status == "incomplete"


def test_observes_retained_work_before_filling_newly_idle_device(tmp_path):
    workers = Workers()
    first = controller(tmp_path, devices=(1,), workers=workers)
    first.reconcile()
    second = controller(tmp_path, devices=(1, 2), workers=workers)

    report = second.reconcile()

    assert [(item.job.cell_id, item.physical_device) for item in report.active] == [
        ("p1-a", 1), ("p1-b", 2)
    ]
    assert len(workers.started) == 2


def test_dead_phase1_worker_is_truthfully_incomplete_then_resumed(tmp_path):
    workers = Workers()
    subject = controller(tmp_path, devices=(1,), workers=workers)
    subject.reconcile()
    workers.observations["phase1-run:p1-a"] = Observation.incomplete(17)
    resumed = controller(
        tmp_path, devices=(1,), workers=workers, phase1_progress=("p1-a",)
    ).reconcile()

    assert resumed.incomplete[0]["exit_code"] == 17
    assert resumed.incomplete[0]["classification"] == "infrastructure-incomplete"
    assert resumed.active[0].job == Job(JobKind.PHASE1_RESUME, "p1-a")
    assert workers.started[-1][2].parts[-2:] == ("phase1-resume", "p1-a")
    assert workers.started[-1][3] == ("worker", "phase1-resume", "p1-a", "1")


def test_phase1_has_priority_then_qualification_then_phase2(tmp_path):
    workers = Workers()
    report = controller(
        tmp_path, devices=(1, 2), workers=workers,
        completed=("p1-a",), qualified=("p2-a",),
    ).reconcile()

    assert [(item.job.kind, item.job.cell_id) for item in report.active] == [
        (JobKind.PHASE1_RUN, "p1-b"),
        (JobKind.PHASE2_RUN, "p2-a"),
    ]

    workers = Workers()
    report = controller(
        tmp_path / "qualification", devices=(1, 2), workers=workers,
        completed=("p1-a", "p1-b"), qualified=("p2-a",),
    ).reconcile()
    assert [(item.job.kind, item.job.cell_id) for item in report.active] == [
        (JobKind.PHASE2_QUALIFY, "p2-b"),
        (JobKind.PHASE2_RUN, "p2-a"),
    ]


def test_matching_phase2_is_not_enqueued_before_successful_phase1(tmp_path):
    report = controller(tmp_path, devices=(1,), completed=("p1-a",)).reconcile()

    assert report.active[0].job == Job(JobKind.PHASE1_RUN, "p1-b")
    assert all(item.job.cell_id != "p2-b" for item in report.active)


def test_terminal_report_preserves_phase2_failed_semantic_status(tmp_path):
    subject = controller(
        tmp_path, completed=("p1-a", "p1-b"),
        qualified=("p2-a", "p2-b"), phase2_terminal=("p2-a", "p2-b"),
    )

    report = subject.reconcile()

    assert report.status == "complete"
    assert report.active == ()
    assert report.phase1_completed == ("p1-a", "p1-b")
    assert report.phase2_terminal == ("p2-a", "p2-b")


def test_malformed_state_fails_closed_and_second_controller_cannot_lock(tmp_path):
    subject = controller(tmp_path)
    subject.root.mkdir(parents=True)
    subject.state_path.write_text('{"schema":"wrong"}')
    with pytest.raises(ValueError, match="controller state"):
        subject.reconcile()

    subject.state_path.unlink()
    with subject.lock():
        with pytest.raises(ControllerBusy):
            subject.reconcile()


def test_state_is_stable_json_and_rejects_duplicate_device_assignments(tmp_path):
    subject = controller(tmp_path, devices=(1, 2))
    subject.reconcile()
    value = json.loads(subject.state_path.read_text())
    value["active"][1]["physical_device"] = value["active"][0]["physical_device"]
    subject.state_path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="duplicate active device"):
        subject.reconcile()


@pytest.mark.parametrize("mutation", ["device", "incomplete"])
def test_state_rejects_out_of_range_devices_and_malformed_incomplete(tmp_path, mutation):
    subject = controller(tmp_path, devices=(1, 2))
    subject.reconcile()
    value = json.loads(subject.state_path.read_text())
    if mutation == "device":
        value["active"][0]["physical_device"] = 8
    else:
        value["incomplete"] = [{"classification": "semantic-failure"}]
    subject.state_path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="controller state"):
        subject.reconcile()


def test_observation_invariants_reject_ambiguous_active_exit_code():
    with pytest.raises(ValueError, match="active"):
        Observation("active", 0)


def test_retained_unknown_job_fails_closed_before_observation(tmp_path):
    subject = controller(tmp_path)
    subject.root.mkdir(parents=True)
    subject.state_path.write_text(json.dumps({
        "schema": "a5-priority-controller-state-v1",
        "active": [RetainedAssignment(
            Job(JobKind.PHASE1_RUN, "unknown"), 1, "stale-screen", 1,
        ).as_dict()],
        "incomplete": [],
    }))

    with pytest.raises(ValueError, match="job universe"):
        subject.reconcile()


def test_real_composition_only_schedules_eight_runnable_sources(tmp_path):
    from benchmarks.a5kernels.phase1_composition import (
        LiveDependencies, Phase1CellComposition,
    )
    from benchmarks.a5kernels.phase2_memory import phase2_cells
    from tests.test_a5_phase1_composition import _guide_and_report

    guide_path, guide, gate = _guide_and_report(tmp_path)
    def unavailable(*_args, **_kwargs):
        return None
    phase1 = Phase1CellComposition(
        tmp_path / "phase1", guide_path, guide, gate,
        LiveDependencies(unavailable, unavailable),
    )
    destinations = phase2_cells(guide, gate.report_sha256)
    workers = Workers()
    subject = CampaignController(
        root=tmp_path / "controller", phase1=phase1,
        phase2_root=tmp_path / "phase2", destinations=destinations,
        inventory=Inventory(*range(1, 8)), workers=workers,
        command=lambda job, device: (job.job_id, str(device)),
        phase1_progress=lambda _cell_id: False,
        snapshot=lambda destination: tmp_path / destination.cell_id,
        qualified=lambda _destination: False,
        phase2_terminal=lambda _destination: None,
    )

    report = subject.reconcile()
    assert len(report.active) == 7
    first = report.active[0]
    workers.observations[first.job.job_id] = Observation.completed()
    phase1.load_terminal = lambda cell: (
        {"status": "passed"} if cell.cell_id == first.job.cell_id else None
    )
    report = subject.reconcile()

    assert len(workers.started) == 8
    assert {item[0].cell_id for item in workers.started} == {
        item.source_cell_id for item in destinations
    }
    assert all(item[0].kind is JobKind.PHASE1_RUN for item in workers.started)

