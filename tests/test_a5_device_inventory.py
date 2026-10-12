import json
import re
import subprocess

import pytest

from benchmarks.a5kernels.device_inventory import (
    BZDeviceInventory,
    DeviceInventoryError,
    DeviceInventoryUnavailable,
)


def candidate(device, *, hbm=0, utilization=0):
    return (
        f"BZ_A5_DEVICE_CANDIDATE id={device} state=eligible health=OK "
        f"faults=0 hbm_percent={hbm} utilization_percent={utilization} "
        "processes=0 process_memory_mb=0"
    )


def terminal(device):
    return "\n".join(
        (
            f"BZ_A5_DEVICE={device}",
            "BZ_A5_DEVICE_STATE=selected purpose=profiling "
            f"physical={device} application_device=physical-{device}",
        )
    )


def output(*lines, selected=2, complete=True):
    records = list(lines)
    if complete:
        present = {
            int(match.group(1))
            for line in records
            if (match := re.search(r"(?:^| )id=([0-9]+)(?: |$)", line))
        }
        records.extend(
            "BZ_A5_DEVICE_CANDIDATE "
            f"id={device} state=excluded reason=health-unresponsive"
            for device in range(1, 8)
            if device not in present
        )
    return "\n".join((*records, terminal(selected))) + "\n"


def inventory(stdout, *, returncode=0, stderr=""):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    result = BZDeviceInventory(
        selector_wrapper="execution-profiles/bz-a5/select-device.sh",
        process_runner=run,
    )
    return result, calls


def test_inventory_returns_every_idle_profiling_device_in_physical_order():
    probe, calls = inventory(output(
        "BZ_A5_DEVICE_CANDIDATE id=1 state=excluded reason=current-fault count=2",
        candidate(6, hbm=1, utilization=7),
        "BZ_A5_DEVICE_CANDIDATE id=3 state=excluded reason=active-processes "
        "hbm_percent=4 processes=1 process_memory_mb=739",
        candidate(2, hbm=4, utilization=0),
        "BZ_A5_DEVICE_CANDIDATE id=4 state=excluded reason=hbm-limit hbm_percent=51",
        "BZ_A5_DEVICE_CANDIDATE id=5 state=excluded reason=health-unresponsive",
        "BZ_A5_DEVICE_CANDIDATE id=7 state=excluded reason=health-Alarm",
        selected=6,
    ))

    snapshot = probe.profiling_snapshot()

    assert calls == [(final_args, {
        "text": True,
        "capture_output": True,
        "check": False,
    }) for final_args in [(
        "execution-profiles/bz-a5/select-device.sh", "--purpose", "profiling"
    )]]
    assert snapshot.selected_physical_device == 6
    assert [item.physical_device for item in snapshot.eligible_devices] == [2, 6]
    assert snapshot.eligible_devices[0].hbm_percent == 4
    assert snapshot.eligible_devices[1].utilization_percent == 7
    assert json.loads(snapshot.to_json()) == {
        "eligible_devices": [
            {
                "fault_count": 0,
                "hbm_percent": 4,
                "health": "OK",
                "physical_device": 2,
                "process_count": 0,
                "process_memory_mb": 0,
                "utilization_percent": 0,
            },
            {
                "fault_count": 0,
                "hbm_percent": 1,
                "health": "OK",
                "physical_device": 6,
                "process_count": 0,
                "process_memory_mb": 0,
                "utilization_percent": 7,
            },
        ],
        "purpose": "profiling",
        "selected_physical_device": 6,
    }


@pytest.mark.parametrize(
    "excluded",
    [
        "BZ_A5_DEVICE_CANDIDATE id=1 state=excluded reason=health-unresponsive",
        "BZ_A5_DEVICE_CANDIDATE id=1 state=excluded reason=fault-query-unresponsive",
        "BZ_A5_DEVICE_CANDIDATE id=1 state=excluded reason=current-fault count=1",
        "BZ_A5_DEVICE_CANDIDATE id=1 state=excluded reason=usage-query-unresponsive",
        "BZ_A5_DEVICE_CANDIDATE id=1 state=excluded reason=invalid-hbm-usage",
        "BZ_A5_DEVICE_CANDIDATE id=1 state=excluded reason=hbm-limit hbm_percent=50",
        "BZ_A5_DEVICE_CANDIDATE id=1 state=excluded reason=process-query-unresponsive",
        "BZ_A5_DEVICE_CANDIDATE id=1 state=excluded reason=active-processes "
        "hbm_percent=4 processes=2 process_memory_mb=1024",
    ],
)
def test_inventory_accepts_each_checked_selector_exclusion(excluded):
    probe, _ = inventory(output(excluded, candidate(2)))

    snapshot = probe.profiling_snapshot()

    assert [item.physical_device for item in snapshot.eligible_devices] == [2]


def test_inventory_reports_checked_selector_failure_without_parsing_partial_output():
    probe, _ = inventory(candidate(2), returncode=1, stderr="no isolated device")

    with pytest.raises(DeviceInventoryError, match="selector failed with exit 1"):
        probe.profiling_snapshot()


def test_inventory_classifies_checked_transport_exit_as_unavailable():
    probe, _ = inventory("", returncode=75)

    with pytest.raises(DeviceInventoryUnavailable, match="temporarily unavailable"):
        probe.profiling_snapshot()


def test_inventory_rejects_successful_but_truncated_candidate_coverage():
    probe, _ = inventory(output(candidate(2), selected=2, complete=False))

    with pytest.raises(DeviceInventoryError, match="complete devices 1 through 7"):
        probe.profiling_snapshot()


def test_inventory_rejects_unexpected_candidate_device():
    probe, _ = inventory(output(candidate(2), candidate(8), selected=2))

    with pytest.raises(DeviceInventoryError, match="candidate device"):
        probe.profiling_snapshot()


def test_inventory_normalizes_selector_launch_failure():
    def fail(_argv, **_kwargs):
        raise OSError("selector disappeared")

    probe = BZDeviceInventory(
        selector_wrapper="execution-profiles/bz-a5/select-device.sh",
        process_runner=fail,
    )

    with pytest.raises(DeviceInventoryUnavailable, match="could not run"):
        probe.profiling_snapshot()


@pytest.mark.parametrize(
    "stdout, message",
    [
        (output("BZ_A5_DEVICE_CANDIDATE id=oops state=eligible", candidate(2)), "malformed"),
        (output("BZ_A5_DEVICE_CANDIDATE", candidate(2)), "malformed candidate"),
        (output(candidate(2), candidate(2)), "duplicate candidate"),
        (output(candidate(0), selected=0), "automatic profiling inventory excludes device 0"),
        (output(candidate(2).replace("health=OK", "health=Alarm")), "eligible candidate"),
        (output(candidate(2).replace("faults=0", "faults=1")), "eligible candidate"),
        (output(candidate(2).replace("processes=0", "processes=1")), "eligible candidate"),
        (output(candidate(2).replace("hbm_percent=0", "hbm_percent=50")), "eligible candidate"),
        (output(candidate(2)).replace(terminal(2), ""), "terminal"),
        (output(candidate(2)) + terminal(2) + "\n", "terminal"),
        (output(candidate(2)).replace("purpose=profiling", "purpose=regular"), "profiling"),
        (output(candidate(2)) + "BZ_A5_DEVICE_STATE\n", "terminal"),
        (output(candidate(2)) + "BZ_A5_DEVICE_STATE=\n", "STATE"),
        (output(candidate(2)).replace("BZ_A5_DEVICE=2", "BZ_A5_DEVICE=3"), "inconsistent"),
        (output(candidate(2)).replace("physical=2", "physical=3"), "inconsistent"),
        (output(candidate(2)).replace("physical-2", "logical-0"), "inconsistent"),
        (output(candidate(2), selected=3), "not eligible"),
        (
            output("BZ_A5_DEVICE_CANDIDATE id=1 state=excluded reason=mystery", candidate(2)),
            "excluded candidate",
        ),
    ],
)
def test_inventory_fails_closed_on_ambiguous_or_malformed_output(stdout, message):
    probe, _ = inventory(stdout)

    with pytest.raises(DeviceInventoryError, match=message):
        probe.profiling_snapshot()


@pytest.mark.parametrize(
    "path",
    ["execution-profiles/bz-a5/run.sh", "select-device", ""],
)
def test_inventory_requires_the_checked_selector_interface(path):
    with pytest.raises(ValueError, match="select-device.sh"):
        BZDeviceInventory(selector_wrapper=path)
