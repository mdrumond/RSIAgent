"""Strict consumer for the checked BZ-A5 device selector.

The selector owns remote probing and selection policy.  This module turns its
line-oriented evidence into a stable Python/JSON inventory for controllers;
it never performs its own device probing or reservation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Callable


_CANDIDATE = "BZ_A5_DEVICE_CANDIDATE"
_DEVICE = "BZ_A5_DEVICE="
_STATE = "BZ_A5_DEVICE_STATE"
_PROFILING_UNAVAILABLE = (
    "BZ_A5_DEVICE_STATE=unavailable purpose=profiling requested=auto"
)
_UNRESPONSIVE_REASONS = frozenset(
    {
        "health-unresponsive",
        "fault-query-unresponsive",
        "usage-query-unresponsive",
        "process-query-unresponsive",
        "invalid-hbm-usage",
    }
)


class DeviceInventoryError(RuntimeError):
    """The checked selector could not prove an unambiguous inventory."""


class DeviceInventoryUnavailable(DeviceInventoryError):
    """The checked selector's infrastructure is temporarily unavailable."""


@dataclass(frozen=True, order=True)
class ProfilingDevice:
    """One healthy, fault-free, idle physical NPU."""

    physical_device: int
    health: str
    fault_count: int
    hbm_percent: int
    utilization_percent: int
    process_count: int
    process_memory_mb: int


@dataclass(frozen=True)
class ProfilingDeviceSnapshot:
    """A deterministic snapshot proven by one successful selector call."""

    purpose: str
    selected_physical_device: int
    eligible_devices: tuple[ProfilingDevice, ...]

    def to_json(self) -> str:
        """Return the stable machine-readable representation."""

        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class BZDeviceInventory:
    """Invoke and strictly validate the configured checked selector."""

    def __init__(
        self,
        *,
        selector_wrapper: str,
        process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if Path(selector_wrapper).name != "select-device.sh":
            raise ValueError("selector_wrapper must identify checked select-device.sh")
        self._selector = selector_wrapper
        self._run = process_runner

    def profiling_snapshot(self) -> ProfilingDeviceSnapshot:
        """Return every profiling-eligible device from one bounded probe pass."""

        try:
            completed = self._run(
                (self._selector, "--purpose", "profiling"),
                text=True,
                capture_output=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DeviceInventoryUnavailable(
                "checked BZ-A5 selector could not run"
            ) from exc
        if completed.returncode == 75:
            raise DeviceInventoryUnavailable(
                "checked BZ-A5 selector infrastructure is temporarily unavailable"
            )
        has_unavailable_record = _has_canonical_unavailable_record(
            completed.stdout, completed.stderr
        )
        has_complete_excluded_inventory = (
            completed.returncode == 1
            and isinstance(completed.stdout, str)
            and _is_complete_excluded_inventory(completed.stdout)
        )
        if completed.returncode == 1 and (
            has_unavailable_record or has_complete_excluded_inventory
        ):
            raise DeviceInventoryUnavailable(
                "checked BZ-A5 profiling inventory is temporarily unavailable"
            )
        if completed.returncode != 0:
            detail = (
                " without canonical unavailability evidence"
                if completed.returncode == 1 else ""
            )
            raise DeviceInventoryError(
                "checked BZ-A5 selector failed with exit "
                f"{completed.returncode}{detail}"
            )
        if not isinstance(completed.stdout, str):
            raise DeviceInventoryError("checked BZ-A5 selector returned invalid output")
        return _parse_profiling_snapshot(completed.stdout)


def _has_canonical_unavailable_record(stdout: object, stderr: object) -> bool:
    """Accept one unavailable terminal without contradictory selector evidence."""

    streams = tuple(stream for stream in (stdout, stderr) if isinstance(stream, str))
    if sum(
        line == _PROFILING_UNAVAILABLE
        for stream in streams
        for line in stream.splitlines()
    ) != 1:
        return False
    if not isinstance(stdout, str):
        return True
    selector_lines = tuple(
        line
        for line in stdout.splitlines()
        if line.startswith((_CANDIDATE, _DEVICE, _STATE))
        and line != _PROFILING_UNAVAILABLE
    )
    candidate_lines = tuple(
        line for line in selector_lines if line.startswith(_CANDIDATE)
    )
    if any(line.startswith((_DEVICE, _STATE)) for line in selector_lines):
        return False
    if not candidate_lines:
        return True
    return _is_complete_excluded_inventory("\n".join(candidate_lines))


def _is_complete_excluded_inventory(stdout: str) -> bool:
    """Prove that a failed profiling probe exhausted all automatic devices."""

    candidates: dict[int, ProfilingDevice | None] = {}
    has_terminal_record = False
    for line in stdout.splitlines():
        if line.startswith(_CANDIDATE):
            if not line.startswith(f"{_CANDIDATE} "):
                raise DeviceInventoryError("malformed candidate record")
            fields = _fields(line, _CANDIDATE)
            device = _candidate_device(fields)
            if device in candidates:
                raise DeviceInventoryError(f"duplicate candidate device {device}")
            candidates[device] = _eligible_device(fields, device)
        elif line.startswith(_DEVICE) or line.startswith(_STATE):
            has_terminal_record = True
    return (
        not has_terminal_record
        and set(candidates) == set(range(1, 8))
        and all(candidate is None for candidate in candidates.values())
    )


def _parse_profiling_snapshot(stdout: str) -> ProfilingDeviceSnapshot:
    candidates: dict[int, ProfilingDevice | None] = {}
    selected_records: list[int] = []
    terminal_records: list[dict[str, str]] = []

    for line in stdout.splitlines():
        if line.startswith(_CANDIDATE):
            if not line.startswith(f"{_CANDIDATE} "):
                raise DeviceInventoryError("malformed candidate record")
            fields = _fields(line, _CANDIDATE)
            device = _candidate_device(fields)
            if device in candidates:
                raise DeviceInventoryError(f"duplicate candidate device {device}")
            candidates[device] = _eligible_device(fields, device)
        elif line.startswith(_DEVICE):
            encoded = line.removeprefix(_DEVICE)
            selected_records.append(_canonical_int(encoded, "selected device"))
        elif line.startswith(_STATE):
            if not line.startswith(f"{_STATE}="):
                raise DeviceInventoryError("malformed profiling terminal record")
            terminal_records.append(_fields(line, _STATE))

    if len(selected_records) != 1 or len(terminal_records) != 1:
        raise DeviceInventoryError("selector output must contain one unique terminal record")
    if set(candidates) != set(range(1, 8)):
        raise DeviceInventoryError(
            "profiling inventory must contain complete devices 1 through 7"
        )
    selected = selected_records[0]
    terminal = terminal_records[0]
    expected_keys = {"purpose", "physical", "application_device"}
    if terminal.get("state") != "selected" or set(terminal) != expected_keys | {"state"}:
        raise DeviceInventoryError("malformed profiling terminal record")
    if terminal["purpose"] != "profiling":
        raise DeviceInventoryError("terminal record is not for profiling")
    physical = _canonical_int(terminal["physical"], "terminal physical device")
    if physical != selected or terminal["application_device"] != f"physical-{physical}":
        raise DeviceInventoryError("inconsistent profiling terminal selection")
    if selected not in candidates or candidates[selected] is None:
        raise DeviceInventoryError("selected profiling device is not eligible")

    eligible = tuple(
        candidate
        for candidate in sorted(
            (item for item in candidates.values() if item is not None),
            key=lambda item: item.physical_device,
        )
    )
    return ProfilingDeviceSnapshot("profiling", selected, eligible)


def _fields(line: str, marker: str) -> dict[str, str]:
    prefix = f"{marker}=" if marker == _STATE else f"{marker} "
    if not line.startswith(prefix):
        raise DeviceInventoryError(f"malformed {marker} record")
    fields: dict[str, str] = {}
    tokens = line.removeprefix(prefix).split()
    if marker == _STATE:
        if not tokens:
            raise DeviceInventoryError(f"malformed {marker} record")
        tokens[0] = f"state={tokens[0]}"
    for token in tokens:
        key, separator, value = token.partition("=")
        if separator != "=" or not key or not value or key in fields:
            raise DeviceInventoryError(f"malformed {marker} record")
        fields[key] = value
    return fields


def _candidate_device(fields: dict[str, str]) -> int:
    try:
        encoded = fields["id"]
    except KeyError as exc:
        raise DeviceInventoryError("malformed candidate record") from exc
    device = _canonical_int(encoded, "candidate device")
    if not 1 <= device <= 7:
        if device == 0:
            raise DeviceInventoryError("automatic profiling inventory excludes device 0")
        raise DeviceInventoryError("malformed candidate device")
    return device


def _eligible_device(fields: dict[str, str], device: int) -> ProfilingDevice | None:
    state = fields.get("state")
    if state == "eligible":
        required = {
            "id", "state", "health", "faults", "hbm_percent",
            "utilization_percent", "processes", "process_memory_mb",
        }
        if set(fields) != required:
            raise DeviceInventoryError("malformed eligible candidate")
        fault_count = _canonical_int(fields["faults"], "candidate fault count")
        hbm = _percent(fields["hbm_percent"], "candidate HBM usage")
        utilization = _percent(
            fields["utilization_percent"], "candidate utilization"
        )
        processes = _canonical_int(fields["processes"], "candidate process count")
        process_memory = _canonical_int(
            fields["process_memory_mb"], "candidate process memory"
        )
        if (
            fields["health"] != "OK"
            or fault_count != 0
            or hbm >= 50
            or processes != 0
            or process_memory != 0
        ):
            raise DeviceInventoryError("inconsistent eligible candidate")
        return ProfilingDevice(
            device, "OK", fault_count, hbm, utilization, processes, process_memory
        )
    if state != "excluded":
        raise DeviceInventoryError("malformed candidate state")
    _validate_excluded(fields)
    return None


def _validate_excluded(fields: dict[str, str]) -> None:
    reason = fields.get("reason")
    base = {"id", "state", "reason"}
    if reason in _UNRESPONSIVE_REASONS or (
        isinstance(reason, str)
        and reason.startswith("health-")
        and len(reason) > len("health-")
    ):
        if set(fields) == base:
            return
    elif reason == "current-fault":
        if set(fields) == base | {"count"} and fields["count"] == "unknown":
            return
        if set(fields) == base | {"count"} and _canonical_int(
            fields["count"], "fault count"
        ) > 0:
            return
    elif reason == "hbm-limit":
        if set(fields) == base | {"hbm_percent"} and _percent(
            fields["hbm_percent"], "excluded HBM usage"
        ) >= 50:
            return
    elif reason == "active-processes":
        required = base | {"hbm_percent", "processes", "process_memory_mb"}
        if set(fields) == required:
            hbm = _percent(fields["hbm_percent"], "excluded HBM usage")
            processes = _canonical_int(fields["processes"], "excluded process count")
            memory = _canonical_int(
                fields["process_memory_mb"], "excluded process memory"
            )
            if hbm < 50 and processes > 0 and memory >= 0:
                return
    raise DeviceInventoryError("malformed excluded candidate")


def _canonical_int(encoded: str, field: str) -> int:
    try:
        value = int(encoded)
    except (TypeError, ValueError) as exc:
        raise DeviceInventoryError(f"malformed {field}") from exc
    if value < 0 or str(value) != encoded:
        raise DeviceInventoryError(f"malformed {field}")
    return value


def _percent(encoded: str, field: str) -> int:
    value = _canonical_int(encoded, field)
    if value > 100:
        raise DeviceInventoryError(f"malformed {field}")
    return value
