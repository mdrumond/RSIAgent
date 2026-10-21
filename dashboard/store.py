"""Artifact indexing and a small durable snapshot cache."""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DESCRIPTION_LABELS = ("Question", "Controlled change", "Tools", "Measurements",
                       "Expected lesson")
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_NAME = re.compile(r"[a-z][a-z0-9]{0,15}\Z")
_CAPABILITIES = {"gm-ub-movement", "simd-vector", "padded-tail", "simt-reduction",
                 "simt-mask", "vector-reduction", "vector-mask", "cube-tiling",
                 "failure-repair", "timing-sweep", "profiling"}
_DTYPES = {"float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2",
           "mxfp8_e4m3fn", "mxfp8_e5m2"}
_OPS = {"copy", "add", "multiply", "relu", "mask-positive", "reduce-sum", "matmul"}
_LEARNING_FOCI = {"pipe-scalar", "pipe-vector", "pipe-cube", "pipe-mte1", "pipe-mte2",
                   "pipe-mte3", "pipe-fix", "inter-pipe-sync", "simd-barrier",
                   "simt-barrier", "cross-core-sync", "sync-correctness",
                   "sync-performance", "tool-msprof", "tool-msdebug", "tool-mssanitizer",
                   "debug-print"}
_PROJECT_FIELDS = {"project_id", "capabilities", "shape", "dtype", "block_count",
                   "padding_multiple", "failure_mode", "timing_lengths", "profile_metric",
                   "hypothesis", "graph"}


def _integer_shape(value, *, maximum=4096):
    return (isinstance(value, list) and 1 <= len(value) <= 3 and
            all(type(item) is int and 1 <= item <= maximum for item in value))


def _validated_graph(value):
    """Return graph shapes/operations, or None for a non-canonical ProjectGraph wire value."""
    if not isinstance(value, dict) or set(value) != {"inputs", "nodes", "output"}:
        return None
    inputs, nodes, output = value["inputs"], value["nodes"], value["output"]
    if (not isinstance(inputs, list) or not 1 <= len(inputs) <= 3 or
            not isinstance(nodes, list) or not 1 <= len(nodes) <= 12 or
            not isinstance(output, str)):
        return None
    shapes, input_shapes, node_names, operations = {}, [], set(), set()
    for item in inputs:
        if (not isinstance(item, dict) or set(item) != {"name", "shape"} or
                not isinstance(item["name"], str) or not _NAME.fullmatch(item["name"]) or
                not _integer_shape(item["shape"]) or math.prod(item["shape"]) > 65536 or
                item["name"] in shapes):
            return None
        shape = tuple(item["shape"]); shapes[item["name"]] = shape; input_shapes.append(shape)
    for item in nodes:
        if (not isinstance(item, dict) or set(item) != {"name", "op", "operands", "axis"} or
                not isinstance(item["name"], str) or not _NAME.fullmatch(item["name"]) or
                item["name"] in shapes or not isinstance(item["op"], str) or
                item["op"] not in _OPS or
                not isinstance(item["operands"], list) or
                any(not isinstance(name, str) or not _NAME.fullmatch(name)
                    for name in item["operands"]) or
                item["axis"] is not None and type(item["axis"]) is not int):
            return None
        try:
            operands = [shapes[name] for name in item["operands"]]
        except KeyError:
            return None
        op, axis = item["op"], item["axis"]
        if op in {"copy", "relu", "mask-positive"}:
            inferred = operands[0] if len(operands) == 1 and axis is None else None
        elif op in {"add", "multiply"}:
            inferred = (operands[0] if len(operands) == 2 and operands[0] == operands[1]
                        and axis is None else None)
        elif op == "reduce-sum":
            inferred = (operands[0][:axis] + operands[0][axis + 1:] or (1,)
                        if len(operands) == 1 and type(axis) is int and
                        0 <= axis < len(operands[0]) else None)
        else:
            inferred = ((operands[0][0], operands[1][1])
                        if len(operands) == 2 and axis is None and
                        len(operands[0]) == len(operands[1]) == 2 and
                        operands[0][1] == operands[1][0] and
                        operands[0][0] * operands[0][1] * operands[1][1] <= 2_000_000
                        else None)
        if inferred is None or math.prod(inferred) > 65536:
            return None
        shapes[item["name"]] = inferred; node_names.add(item["name"]); operations.add(op)
    if output not in node_names:
        return None
    by_name = {item["name"]: item for item in nodes}; reachable, pending = {output}, [output]
    while pending:
        for operand in by_name[pending.pop()]["operands"]:
            if operand in by_name and operand not in reachable:
                reachable.add(operand); pending.append(operand)
    if reachable != node_names:
        return None
    return tuple(input_shapes), tuple(shapes[output]), operations, nodes


def _valid_project_spec(value):
    """Local wire validator equivalent to the closed ProjectSpec admission contract."""
    if (not isinstance(value, dict) or set(value) - {"learning_focus"} != _PROJECT_FIELDS or
            not isinstance(value.get("project_id"), str) or not _ID.fullmatch(value["project_id"])):
        return False
    capabilities = value["capabilities"]
    focus = value.get("learning_focus", [])
    timing = value["timing_lengths"]
    if (not isinstance(capabilities, list) or not capabilities or
            any(not isinstance(item, str) or item not in _CAPABILITIES
                for item in capabilities) or
            len(set(capabilities)) != len(capabilities) or
            not _integer_shape(value["shape"], maximum=16384) or
            not isinstance(value["dtype"], str) or value["dtype"] not in _DTYPES or
            type(value["block_count"]) is not int or
            not 1 <= value["block_count"] <= 32 or type(value["padding_multiple"]) is not int or
            value["padding_multiple"] not in {1, 16, 32, 64, 128, 256} or
            not isinstance(value["failure_mode"], str) or
            value["failure_mode"] not in {"none", "compile", "host-mismatch"} or
            not isinstance(timing, list) or len(timing) > 8 or
            any(type(item) is not int or not 1 <= item <= 4096 for item in timing) or
            timing != sorted(set(timing)) or
            (value["profile_metric"] is not None and
             (not isinstance(value["profile_metric"], str) or
              value["profile_metric"] not in {"pipe-utilization", "memory-throughput",
                                               "timeline"})) or
            not isinstance(value["hypothesis"], str) or not value["hypothesis"].strip() or
            value["hypothesis"] != value["hypothesis"].strip() or
            not isinstance(focus, list) or len(set(focus)) != len(focus) or
            any(not isinstance(item, str) or item not in _LEARNING_FOCI for item in focus)):
        return False
    graph = _validated_graph(value["graph"])
    if graph is None:
        return False
    inputs, output, operations, nodes = graph
    selected = set(capabilities)
    if (("cube-tiling" in selected) != ("matmul" in operations) or
            ("simd-vector" in selected) != bool({"add", "multiply", "relu"} & operations) or
            len({"simt-reduction", "vector-reduction"} & selected) != int("reduce-sum" in operations) or
            len({"simt-mask", "vector-mask"} & selected) != int("mask-positive" in operations)):
        return False
    shape = tuple(value["shape"])
    if "matmul" in operations:
        if (len(shape) != 3 or inputs != ((shape[0], shape[1]), (shape[1], shape[2])) or
                output != (shape[0], shape[2])):
            return False
    elif any(item != shape for item in inputs):
        return False
    low_precision = value["dtype"].startswith("mxfp8_") or (
        value["dtype"].startswith("float8_") and "matmul" in operations)
    if low_precision and (len(nodes) != 1 or nodes[0]["op"] != "matmul" or
                          nodes[0]["operands"] != [item["name"]
                                                   for item in value["graph"]["inputs"]] or
                          "padded-tail" in selected):
        return False
    if (("failure-repair" in selected) != (value["failure_mode"] != "none") or
            (("timing-sweep" in selected) and len(timing) < 2) or
            ("timing-sweep" not in selected and timing) or
            (("profiling" in selected) != (value["profile_metric"] is not None))):
        return False
    if "padded-tail" in selected:
        if output[-1] % value["padding_multiple"] == 0:
            return False
        for item in inputs + (output,):
            padded = ((item[-1] + value["padding_multiple"] - 1) //
                      value["padding_multiple"] * value["padding_multiple"])
            if math.prod(item[:-1]) * padded > 65536:
                return False
    return True


def _description_fields(value):
    """Parse the curriculum's display convention without changing its schema."""
    text = value if isinstance(value, str) else ""
    pattern = re.compile(r"(?:^| )(" + "|".join(map(re.escape, _DESCRIPTION_LABELS)) + r"):\s*")
    matches = list(pattern.finditer(text))
    if [match.group(1) for match in matches] != list(_DESCRIPTION_LABELS):
        return [{"label": "Description", "value": text}]
    fields = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields.append({"label": match.group(1), "value": text[match.end():end].strip()})
    if matches[0].start() != 0 or any(not field["value"] for field in fields):
        return [{"label": "Description", "value": text}]
    return fields


def _description(value):
    fields = _description_fields(value)
    return {"description": value, "description_fields": fields,
            "question": fields[0]["value"] if fields[0]["label"] == "Question" else value}


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeDecodeError):
        return default


def _source(branch: Path, digest: str):
    # Artifact identifiers are canonical lowercase SHA-256, never path input.
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        return None
    path = branch / "artifacts" / "objects" / digest[:2] / digest[2:]
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _digests(value):
    if isinstance(value, dict):
        if isinstance(value.get("source_sha256"), str):
            yield value["source_sha256"]
        yield from (digest for item in value.values() for digest in _digests(item))
    elif isinstance(value, list):
        yield from (digest for item in value for digest in _digests(item))


def _writes(value):
    if isinstance(value, dict):
        write = value.get("write")
        if isinstance(write, dict) and isinstance(write.get("content"), str):
            yield write["content"]
        yield from (source for item in value.values() for source in _writes(item))
    elif isinstance(value, list):
        yield from (source for item in value for source in _writes(item))


def _memory(branch: Path):
    delta = _json(branch / "memory-delta.bin", {})
    try:
        memory = json.loads(base64.b64decode(delta["memory_base64"], validate=True))
    except (KeyError, ValueError, UnicodeDecodeError):
        memory = {}
    facts, performance = memory.get("host_facts", []), []
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("category") != "performance-study":
            continue
        try:
            performance.extend(json.loads(fact["statement"]).get("timing", []))
        except (KeyError, TypeError, ValueError):
            continue
    # Duplicate facts can cite the same timing artifact. Present measurements once.
    performance = list({json.dumps(row, sort_keys=True): row for row in performance}.values())
    return delta, memory, performance


def _percentile(values, fraction):
    """Linear sample percentile (the common R-7 definition), including singleton samples."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


_PIPE_COLUMNS = {
    "aic_mac": ("aic", "Cube / MAC"), "aic_scalar": ("aic", "Scalar"),
    "aic_mte1": ("aic", "MTE1"), "aic_mte2": ("aic", "MTE2"),
    "aic_fixpipe": ("aic", "FIX"), "aiv_vec": ("aiv", "Vector"),
    "aiv_scalar": ("aiv", "Scalar"), "aiv_mte2": ("aiv", "MTE2"),
    "aiv_mte3": ("aiv", "MTE3"),
}


def _finite_number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _logical_io_elements(spec, shape):
    """Declared input plus output traffic for shapes whose graph semantics are known."""
    graph = spec.get("graph")
    if not isinstance(graph, dict):
        return math.prod(shape) * 2 if shape == spec.get("shape") else None
    inputs, nodes = graph.get("inputs"), graph.get("nodes")
    if not isinstance(inputs, list) or not isinstance(nodes, list) or not nodes:
        return None
    validated = _validated_graph(graph)
    if validated:
        input_shapes, output_shape, _, _ = validated
        if shape == list(output_shape):
            return sum(map(math.prod, input_shapes)) + math.prod(output_shape)
    operations = {node.get("op") for node in nodes if isinstance(node, dict)}
    timing_lengths = spec.get("timing_lengths", [])
    if (len(shape) == 1 and shape[0] in timing_lengths and
            operations <= {"copy", "add", "multiply", "relu", "mask-positive"}):
        return math.prod(shape) * (len(inputs) + 1)
    return None


def _performance(project):
    """Build the bounded, presentation-only performance wire response."""
    captures, seen = [], set()
    for raw in project.get("performance", []):
        if not isinstance(raw, dict):
            continue
        samples = raw.get("samples_us")
        shape = raw.get("shape")
        if (not isinstance(samples, list) or not samples or
                not all(_finite_number(item) and item > 0 for item in samples) or
                not isinstance(shape, list) or not shape or
                not all(type(item) is int and item > 0 for item in shape)):
            continue
        try:
            identity = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            continue
        if identity in seen:
            continue
        seen.add(identity)
        values = [float(item) for item in samples]
        stats = {"count": len(values), "min_us": min(values),
                 "q1_us": _percentile(values, .25), "median_us": _percentile(values, .5),
                 "q3_us": _percentile(values, .75), "max_us": max(values)}
        physical = raw.get("physical_shape")
        physical = (physical if isinstance(physical, list) and physical and
                    all(type(item) is int and item > 0 for item in physical) else None)
        boundary = raw.get("measurement") if isinstance(raw.get("measurement"), str) else None
        bandwidth = None
        spec = project.get("spec", {})
        if boundary == "launch-plus-synchronize":
            dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4,
                           "float8_e4m3fn": 1, "float8_e5m2": 1,
                           "mxfp8_e4m3fn": 1, "mxfp8_e5m2": 1}.get(
                               spec.get("dtype"))
            logical_elements = _logical_io_elements(spec, shape)
            if dtype_bytes and logical_elements is not None:
                logical_bytes = logical_elements * dtype_bytes
                bandwidth = logical_bytes / (stats["median_us"] * 1e-6) / (1024 ** 3)
        captures.append({"attempt": len(captures) + 1, "boundary": boundary,
                         "instrumented": raw.get("instrumented") is True,
                         "logical_shape": shape, "physical_shape": physical,
                         "shape_axis": ({"kind": "log-length", "value": shape[0]}
                                        if len(shape) == 1 else
                                        {"kind": "category", "value": "×".join(map(str, shape))}),
                         "statistics": stats, "useful_logical_io_gib_s": bandwidth,
                         "evidence_sha256": hashlib.sha256(identity.encode()).hexdigest()})

    profiles, seen_profiles = [], set()
    expected = [(f"{engine}-{name.replace(' / MAC', '').lower()}", engine, label, prefix)
                for prefix, (engine, label) in _PIPE_COLUMNS.items() for name in [label]]
    for fact_index, fact in enumerate(project.get("memory", {}).get("facts", [])):
        if not isinstance(fact, dict) or fact.get("category") != "profiling":
            continue
        try:
            raw = json.loads(fact["statement"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(raw, dict) or raw.get("metric") != "PipeUtilization":
            continue
        metric_values = raw.get("metric_values")
        if not isinstance(metric_values, list) or any(not isinstance(item, list) or len(item) != 2
                                                      for item in metric_values):
            continue
        try:
            identity = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            continue
        if identity in seen_profiles:
            continue
        seen_profiles.add(identity)
        columns = {}
        for item in metric_values:
            if (isinstance(item, list) and len(item) == 2 and isinstance(item[0], str)
                    and _finite_number(item[1])):
                match = re.search(r":(aic|aiv)_(mac|scalar|mte1|mte2|mte3|vec|fixpipe)_(ratio|time\(us\)):\d+\Z",
                                  item[0])
                if match:
                    columns[f"{match.group(1)}_{match.group(2)}_{match.group(3)}"] = float(item[1])
        metrics = []
        for metric_id, engine, label, prefix in expected:
            ratio, duration = columns.get(prefix + "_ratio"), columns.get(prefix + "_time(us)")
            percentage = (ratio * 100 if ratio is not None and 0 <= ratio <= 1 else
                          ratio if ratio is not None and 1 < ratio <= 100 else None)
            metrics.append({"id": metric_id, "engine": engine.upper(), "label": label,
                            "unit": "%", "value": percentage,
                            "time_us": duration if duration is None or duration >= 0 else None,
                            "available": percentage is not None})
        profiles.append({"capture": len(profiles) + 1, "metric": "PipeUtilization",
                         "instrumented": raw.get("instrumented") is True,
                         "duration_semantics": raw.get("duration_semantics"), "metrics": metrics,
                         "evidence": {"fact_index": fact_index,
                                      "sha256": hashlib.sha256(fact["statement"].encode()).hexdigest()}})
    return {"captures": captures, "profiles": profiles,
            "mte_bandwidth": {"available": False, "value": None, "unit": "GiB/s",
                              "reason": "No retained Memory profiler capture."}}


def _revisions(branch: Path, final_digest: str):
    ordered, inline = [], {}
    for ledger in sorted(branch.glob("**/evidence.jsonl")):
        try:
            rows = ledger.read_text().splitlines()
        except OSError:
            continue
        for row in rows:
            try:
                record = json.loads(row)
                ordered.extend(_digests(record))
                payload = record.get("payload", {})
                if payload.get("role") == "provider-response":
                    response = _source(branch, payload.get("sha256"))
                    if response:
                        for source in _writes(json.loads(response)):
                            digest = hashlib.sha256(source.encode()).hexdigest()
                            inline[digest] = source
                            ordered.append(digest)
            except ValueError:
                continue
    ordered.append(final_digest)
    revisions, seen = [], set()
    for digest in ordered:
        source = inline.get(digest) or _source(branch, digest)
        if source is not None and digest not in seen:
            revisions.append({"sha256": digest, "source": source})
            seen.add(digest)
    return revisions


def _project(branch: Path, spec: dict, failed=None):
    outcome = _json(branch / "outcome.json", failed or {}) or {}
    delta, memory, performance = _memory(branch)
    revisions = _revisions(branch, memory.get("source_revision", ""))
    transitions = []
    for before, after in zip(revisions, revisions[1:]):
        transitions.append({"from": before["sha256"], "to": after["sha256"],
                            "diff": "".join(difflib.unified_diff(
                                before["source"].splitlines(True), after["source"].splitlines(True),
                                fromfile=before["sha256"][:12], tofile=after["sha256"][:12]))})
    lessons = memory.get("agent_lessons", memory.get("lessons", []))
    description = spec.get("hypothesis", "No project description retained.")
    return {"id": spec.get("project_id", branch.name), **_description(description),
            "summary": delta.get("summary", (failed or {}).get("summary", "No outcome summary retained.")),
            "verified": outcome.get("verified") is True,
            "classification": outcome.get("classification", "running" if not outcome else "unknown"),
            "spec": spec, "performance": performance,
            "memory": {"actions": memory.get("actions", []), "facts": memory.get("host_facts", []),
                       "lessons": lessons, "interpretations": memory.get("agent_interpretations", [])},
            "revisions": revisions, "source": revisions[-1]["source"] if revisions else "",
            "transitions": transitions}


def _goal_selection(root):
    """Only publish the bootstrap's public metadata, never reference artifacts."""
    value = _json(root / "goal-selection.json")
    if not isinstance(value, dict) or value.get("stage") not in {
            "selection", "qualification", "qualified", "failed"}:
        return None
    candidate = value.get("candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    return {"stage": value["stage"], "proposal_count": value.get("proposal_count"),
            "candidate": {key: candidate[key] for key in ("spec", "task_id", "submission", "rationale", "citations")
                          if key in candidate},
            "diagnostic": value.get("diagnostic"),
            "qualified_target_id": value.get("qualified_target_id")}


def _attempt_id(wave, project):
    return f"{wave}--{project}"


def _optimizer_attempt_events(store, key, result):
    selected = []
    for event in store.events(key):
        selected.append(event)
        if event["kind"] == "phase2-attempt-result":
            if event["payload"] == result:
                return selected
            selected = []
    return []


def _optimizer_projects(root, state):
    from optimizer.jobs import JobStore
    from .optimizer_results import projection
    from types import SimpleNamespace
    projects = []
    for row in state.get("attempts", []):
        task = row["task"]
        result = row.get("result", {})
        directory = root / row["execution_directory"]
        store = JobStore(directory / "jobs")
        key = result["job_id"]
        events = _optimizer_attempt_events(store, key, result)
        artifacts = [e["payload"] for e in events if e["kind"] == "artifact"]
        if result.get("candidate") and not any(
                a["sha256"] == result["candidate"]["sha256"] for a in artifacts):
            artifacts.append(result["candidate"])
        revisions = []
        for artifact in artifacts:
            if artifact["name"] not in {"candidate.cpp", "candidate.py"}:
                continue
            source = json.loads(store.read_artifact(key, artifact["sha256"]))
            revisions.append({"sha256": artifact["sha256"], "source": source})
        measured_row = {"candidate_sha256": result.get("source_sha256"),
            "content": result.get("result") or {}, "speedup": result.get("speedup"),
            "summary": result.get("summary"), "source": result.get("candidate"),
            "profiling": result.get("profiling")}
        # Reuse the optimizer projection, but bind it to this committed attempt,
        # not the latest best state of a shared task's subsequent retries.
        view = SimpleNamespace(get=store.get, read_artifact=store.read_artifact,
            artifacts=lambda _: artifacts,
            events=lambda _: [{"kind": "optimizer-state", "payload": {
                "baseline": result.get("baseline"), "history": [measured_row],
                "best": measured_row}}])
        measured = projection(view, key)
        description = task.get("rationale", "Optimizer reference task")
        projects.append({"id": row["attempt_id"], "project_id": row["task_id"],
            "attempt": row["attempt_id"], "kind": row["kind"], **_description(description),
            "summary": result.get("summary", result.get("status", "Attempt completed")),
            "verified": row.get("verified") is True,
            "classification": result.get("classification", "unknown"),
            "spec": task, "performance": [], "optimizer_results": measured,
            "memory": {"actions": [], "facts": [], "lessons": result.get("lessons", []),
                       "interpretations": []},
            "revisions": revisions, "source": revisions[-1]["source"] if revisions else "",
            "transitions": [{"from": a["sha256"], "to": b["sha256"], "diff": "".join(
                difflib.unified_diff(a["source"].splitlines(True), b["source"].splitlines(True)))}
                for a, b in zip(revisions, revisions[1:])]})
    return projects


def _load_campaign(entry):
    root = Path(entry["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"campaign root unavailable: {root}")
    state = _json(root / "state.json")
    bootstrap = _goal_selection(root) if int(entry["phase"]) == 2 else None
    if not isinstance(state, dict) and bootstrap is not None:
        state = {}
    if not isinstance(state, dict):
        raise ValueError(f"campaign state unavailable: {root / 'state.json'}")
    if state.get("phase") == "optimizer-phase2":
        projects = _optimizer_projects(root, state)
        terminal = state.get("terminal") or {}
        pending = _json(root / "pending.json", {}) or {}
        control = Path(entry["control"]) if entry.get("control") else root.parent / "control"
        worker = _json(control / "worker.json")
        return {"phase": int(entry["phase"]), "name": entry.get("name", "Phase 2"),
            "status": terminal.get("status") or (worker or {}).get("status") or "running",
            "terminal": terminal, "bootstrap": bootstrap,
            "learner_stage": pending.get("kind", state.get("next_kind")),
            "root": str(root), "available": True, "worker": worker,
            "stats": {"attempted": len(projects), "verified": sum(p["verified"] for p in projects),
                      "published": len(state.get("memory_entries", []))},
            "projects": projects, "planned": []}
    target_driven = state.get("phase") == "target-driven-phase2"
    pending = _json(root / "pending.json", {}) or {}
    specs = {}
    attempts = {row["wave_id"]: row for row in state.get("attempts", [])}
    for plan_path in sorted(root.glob("waves/*/plan.json")):
        plan = _json(plan_path, {}) or {}
        for item in plan.get("wave", {}).get("projects", []):
            spec = item.get("spec", item)
            if spec.get("project_id"):
                identity = (_attempt_id(plan_path.parent.name, spec["project_id"])
                            if target_driven else spec["project_id"])
                specs[identity] = spec
    failures = {row.get("project_id"): row for row in state.get("failed_outcomes", [])}
    branches = {}
    for branch in sorted(root.glob("waves/*/branches/*")):
        if branch.is_dir():
            identity = (_attempt_id(branch.parent.parent.name, branch.name)
                        if target_driven else branch.name)
            branches[identity] = branch
    if target_driven:
        failures = {_attempt_id(wave, row["project_id"]): row
                    for wave, row in attempts.items() if not row.get("verified")}
    executed = set(branches) | {project_id for project_id in failures if project_id}
    projects = []
    for project_id, spec in specs.items():
        if project_id in executed:
            project = _project(branches.get(project_id, root), spec, failures.get(project_id))
            if target_driven:
                wave = project_id.split("--", 1)[0]
                kind = attempts.get(wave, {}).get("kind")
                if kind is None and pending.get("wave", {}).get("wave_id") == wave:
                    kind = pending.get("kind")
                project.update(id=project_id, project_id=spec["project_id"],
                               attempt=wave, kind=kind or "pending")
            projects.append(project)
    # Retain failures even if their old plan is temporarily unavailable.
    for project_id, failure in failures.items():
        if project_id and project_id not in specs:
            project = _project(branches.get(project_id, root),
                               failure.get("spec", {"project_id": project_id}), failure)
            if target_driven:
                project.update(id=project_id, project_id=failure["project_id"],
                               attempt=failure["wave_id"], kind=failure["kind"])
            projects.append(project)
    control = Path(entry["control"]) if entry.get("control") else root.parent / "control"
    worker = _json(control / "worker.json")
    published = (len(state["prior_outcomes"]) if "prior_outcomes" in state else
                 sum(1 for project in projects if project["verified"]))
    planned = [{"id": project_id, **_description(spec.get("hypothesis", "")), "spec": spec}
               for project_id, spec in specs.items() if project_id not in executed]
    terminal = state.get("terminal") or {}
    learner_stage = ((pending.get("kind") or state.get("next_kind", "target"))
                     if target_driven and not terminal else None)
    status = terminal.get("status") or (worker or {}).get("status") or (
        bootstrap["stage"] if bootstrap and not target_driven else learner_stage or "idle")
    return {"phase": int(entry["phase"]), "name": entry.get("name", f"Phase {entry['phase']}"),
            "status": status, "terminal": terminal, "bootstrap": bootstrap,
            "learner_stage": learner_stage,
            "root": str(root), "available": root.is_dir(), "worker": worker,
            "stats": {"attempted": len(projects),
                      "verified": sum(project["verified"] for project in projects),
                      "published": published}, "projects": projects, "planned": planned}


def _campaign(entry):
    from .usefulness import enrich
    return enrich(_load_campaign(entry))


class DashboardStore:
    def __init__(self, config_path, cache_path):
        self.config_path, self.cache_path = Path(config_path), Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS payloads (sha256 TEXT PRIMARY KEY, body BLOB NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS trace_events (
                phase INTEGER NOT NULL, project_id TEXT NOT NULL, event_id TEXT NOT NULL,
                sha256 TEXT, PRIMARY KEY(phase, project_id, event_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS trace_indexes (
                phase INTEGER NOT NULL, project_id TEXT NOT NULL, body TEXT NOT NULL,
                PRIMARY KEY(phase, project_id))""")
            db.execute("PRAGMA journal_mode=WAL")

    def _db(self):
        return sqlite3.connect(self.cache_path)

    def cached(self):
        with self._db() as db:
            row = db.execute("SELECT body FROM snapshots WHERE id=1").fetchone()
        return json.loads(row[0]) if row else {"generated_at": None, "stale": True, "campaigns": []}

    def performance(self, project):
        return _performance(project)

    def _save(self, view):
        with self._db() as db:
            db.execute("INSERT INTO snapshots(id,body) VALUES(1,?) "
                       "ON CONFLICT(id) DO UPDATE SET body=excluded.body", (json.dumps(view),))

    def refresh(self):
        try:
            config = json.loads(self.config_path.read_text())
            view = {"generated_at": time.time(), "stale": False,
                    "campaigns": [_campaign(entry) for entry in config["campaigns"]]}
            self._save(view)
            return view
        except (OSError, ValueError, KeyError, TypeError):
            view = self.cached()
            view["stale"] = True
            self._save(view)
            return view

    def _project_sources(self, phase, project_id):
        try:
            config = json.loads(self.config_path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(config, dict) or not isinstance(config.get("campaigns"), list):
            return None
        entry = next((row for row in config["campaigns"]
                      if isinstance(row, dict) and row.get("phase") == phase), None)
        if entry is None:
            return None
        try:
            root = Path(entry["root"])
        except (KeyError, TypeError):
            return None
        state = _json(root / "state.json", {}) or {}
        if state.get("phase") == "optimizer-phase2":
            row = next((row for row in state.get("attempts", [])
                        if row["attempt_id"] == project_id), None)
            if row:
                return root, root / "attempts" / (project_id + ".json"), root / row[
                    "execution_directory"] / "jobs" / row["result"]["job_id"], row["task"]
            return None
        for plan_path in sorted(root.glob("waves/*/plan.json")):
            plan = _json(plan_path, {}) or {}
            for item in plan.get("wave", {}).get("projects", []):
                spec = item.get("spec", item)
                identity = spec.get("project_id")
                if state.get("phase") == "target-driven-phase2":
                    identity = _attempt_id(plan_path.parent.name, identity)
                if identity == project_id:
                    branch = plan_path.parent / "branches" / spec["project_id"]
                    if branch.is_dir():
                        return root, plan_path, branch, spec
        return None

    @staticmethod
    def _event(label, category, source=None, *, stage=None, turn=None, reason=None):
        try:
            body = source.read_bytes() if source and source.is_file() else None
        except OSError:
            body = None
        digest = hashlib.sha256(body).hexdigest() if body is not None else None
        return {"label": label, "category": category, "stage": stage, "turn": turn,
                "sha256": digest, "size": len(body) if body is not None else None,
                "available": body is not None, "reason": reason}

    @staticmethod
    def _inline_event(label, category, body):
        body = body.encode() if isinstance(body, str) else body
        return {"label": label, "category": category, "stage": None, "turn": None,
                "sha256": hashlib.sha256(body).hexdigest(), "size": len(body),
                "available": True, "reason": None, "inline": body}

    @staticmethod
    def _artifact_event(label, category, source, digest, *, stage=None, turn=None):
        try:
            size = source.stat().st_size
            available, reason = source.is_file(), None
        except (OSError, AttributeError):
            size, available, reason = None, False, "retained payload unavailable"
        return {"label": label, "category": category, "stage": stage, "turn": turn,
                "sha256": digest, "size": size, "available": available,
                "reason": reason}

    @staticmethod
    def _wave_response(content):
        if not isinstance(content, str) or not content.startswith("DECISION: WAVE"):
            return None
        try:
            return json.loads(content[content.index("{"):])
        except (ValueError, TypeError):
            return None

    @classmethod
    def _matches_plan(cls, content, expected):
        response = cls._wave_response(content)
        if not isinstance(response, dict) or not isinstance(expected, dict):
            return False
        if response == expected:
            return True
        if set(response) != set(expected) or any(
                response[key] != expected[key] for key in expected if key != "projects"):
            return False
        proposed, approved = response.get("projects"), expected.get("projects")
        if (not isinstance(proposed, list) or not isinstance(approved, list) or
                not approved or len(proposed) <= len(approved)):
            return False
        coverage = response.get("pre_wave_coverage")
        if (not isinstance(coverage, list) or
                any(not isinstance(item, str) or item not in _CAPABILITIES for item in coverage) or
                len(set(coverage)) != len(coverage)):
            return False

        def project_id(project):
            try:
                valid_spec = (isinstance(project, dict) and
                              set(project) == {"spec", "requires"} and
                              _valid_project_spec(project["spec"]))
            except (KeyError, TypeError, ValueError):
                valid_spec = False
            if (not valid_spec or
                    not isinstance(project["requires"], list) or
                    any(not isinstance(item, str) or item not in _CAPABILITIES
                        for item in project["requires"]) or
                    len(set(project["requires"])) != len(project["requires"]) or
                    not set(project["requires"]) <= set(coverage)):
                return None
            value = project["spec"].get("project_id")
            return value if isinstance(value, str) and value else None

        proposed_ids = [project_id(project) for project in proposed]
        approved_ids = [project_id(project) for project in approved]
        if (None in proposed_ids or None in approved_ids or
                len(set(proposed_ids)) != len(proposed_ids) or
                len(set(approved_ids)) != len(approved_ids)):
            return False
        selected = set(approved_ids)
        return [project for project, identity in zip(proposed, proposed_ids)
                if identity in selected] == approved

    @staticmethod
    def _authenticated_ledger(path):
        """Authenticate an EvidenceLedger-compatible JSONL without execution imports."""
        try:
            raw_rows = path.read_text().splitlines()
        except (OSError, TypeError):
            return None, "retained evidence ledger unavailable"
        rows, previous = [], "0" * 64
        for position, raw in enumerate(raw_rows):
            try:
                record = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite value: {value}")))
                if not isinstance(record, dict) or set(record) != {
                        "sequence", "kind", "payload", "previous_sha256", "entry_sha256"}:
                    raise ValueError("invalid row shape")
                if (isinstance(record["sequence"], bool) or
                        not isinstance(record["sequence"], int) or
                        record["sequence"] != position or
                        record["kind"] not in {"request", "action", "artifact", "result"} or
                        not isinstance(record["payload"], dict) or
                        record["previous_sha256"] != previous or
                        not isinstance(record["entry_sha256"], str) or
                        not _SHA256.fullmatch(record["entry_sha256"])):
                    raise ValueError("invalid row types or chain")
                body = {key: record[key] for key in (
                    "sequence", "kind", "payload", "previous_sha256")}
                canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=True, allow_nan=False).encode("utf-8")
                if hashlib.sha256(canonical).hexdigest() != record["entry_sha256"]:
                    raise ValueError("invalid row digest")
                previous = record["entry_sha256"]
                rows.append((record, raw + "\n", position))
            except (ValueError, TypeError, KeyError):
                return None, "evidence ledger integrity verification failed"
        return rows, None

    def _host_evidence(self, root, project_id):
        rows, error = self._authenticated_ledger(root / "project-evidence.jsonl")
        if error:
            return [self._event("Host verification unavailable", "verification",
                                reason="host " + error)]
        by_digest = {record.get("entry_sha256"): (record, raw, position)
                     for record, raw, position in rows
                     if isinstance(record.get("entry_sha256"), str)}
        terminals = [(record, raw, position) for record, raw, position in rows
                     if record.get("payload", {}).get("project_id") == project_id]
        if not terminals:
            return []
        included = {}
        for terminal, terminal_raw, terminal_position in terminals:
            record, raw, position = terminal, terminal_raw, terminal_position
            if not record.get("entry_sha256"):
                included[position] = (record, raw)
                continue
            seen = set()
            while record is not None and record.get("entry_sha256") not in seen:
                attributed = record.get("payload", {}).get("project_id")
                if attributed is not None and attributed != project_id:
                    break
                digest = record.get("entry_sha256")
                if digest:
                    seen.add(digest)
                included[position] = (record, raw)
                previous = record.get("previous_sha256")
                if not previous or previous == "0" * 64 or previous not in by_digest:
                    break
                record, raw, position = by_digest[previous]
        chain = [included[position] for position in sorted(included)]
        events = []
        for record, raw in chain:
            events.append(self._inline_event("Host verification ledger entry", "verification", raw))
            payload = record.get("payload", {})
            artifact, digest = payload.get("payload"), payload.get("artifact_sha256")
            if artifact is None or not isinstance(digest, str):
                continue
            body = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
            event = self._inline_event(
                f"Host verification artifact: {payload.get('artifact_kind', 'unknown')}",
                "verification", body)
            if event["sha256"] != digest:
                event.update(sha256=digest, available=False,
                             reason="referenced artifact digest mismatch")
            events.append(event)
        return events

    @staticmethod
    def _metadata(path):
        try:
            stat = path.stat()
            return [stat.st_size, stat.st_mtime_ns, stat.st_mode]
        except (OSError, TypeError, AttributeError):
            return None

    def _trace_fingerprint(self, phase, project_id):
        """Fingerprint cheap trace metadata without reading actor object bodies."""
        located = self._project_sources(phase, project_id)
        if located is None:
            return None
        root, plan_path, branch, _ = located
        if (_json(root / "state.json", {}) or {}).get("phase") == "optimizer-phase2":
            return hashlib.sha256(json.dumps({"attempt": self._metadata(plan_path),
                "events": self._metadata(branch / "events.jsonl")}, sort_keys=True).encode()).hexdigest()
        ledger = branch / "evidence.jsonl"
        manifest = {"root": str(root), "files": {
            "plan": self._metadata(plan_path),
            "state": self._metadata(root / "state.json"),
            "pending": self._metadata(root / "pending.json"),
            "actor-ledger": self._metadata(ledger),
            "host-ledger": self._metadata(root / "project-evidence.jsonl"),
            "outcome": self._metadata(branch / "outcome.json"),
            "verifier-result": self._metadata(branch / "execution-evidence.bin"),
            "memory": self._metadata(branch / "memory-delta.bin"),
        }, "objects": {}}
        rows, _ = self._authenticated_ledger(ledger)
        for record, _, _ in rows or []:
            digest = record["payload"].get("sha256")
            if isinstance(digest, str) and _SHA256.fullmatch(digest):
                source = branch / "artifacts" / "objects" / digest[:2] / digest[2:]
                manifest["objects"][digest] = self._metadata(source)
        return hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _trace_sources(self, phase, project_id):
        located = self._project_sources(phase, project_id)
        if located is None:
            return None
        root, plan_path, branch, spec = located
        if (_json(root / "state.json", {}) or {}).get("phase") == "optimizer-phase2":
            from optimizer.jobs import JobStore
            store = JobStore(branch.parent)
            result = _json(plan_path)["result"]
            retained = _optimizer_attempt_events(store, branch.name, result)
            events = [self._inline_event("Committed Phase 2 attempt", "outcome", plan_path.read_bytes())]
            for row in retained:
                events.append(self._inline_event(row["kind"], "actor", json.dumps(row)))
            artifacts = [row["payload"] for row in retained if row["kind"] == "artifact"]
            artifacts += [result[name] for name in ("candidate", "baseline", "profiling", "result_artifact")
                          if result.get(name)]
            seen = set()
            for artifact in artifacts:
                if artifact["sha256"] in seen:
                    continue
                seen.add(artifact["sha256"])
                events.append(self._inline_event(artifact["name"], "verification",
                    store.read_artifact(branch.name, artifact["sha256"])))
            for index, event in enumerate(events):
                event.update(id=f"{event['sha256']}-{index}", sequence=index)
            return events
        plan = _json(plan_path, {}) or {}
        expected_wave = {"wave_id": plan_path.parent.name, **plan.get("wave", {})}
        pending = _json(root / "pending.json", {}) or {}
        if not isinstance(pending, dict):
            pending = {}
        pending_session = pending.get("session") if pending.get("wave") == expected_wave else None
        state = _json(root / "state.json", {}) or {}
        if not isinstance(state, dict):
            state = {}
        session = (pending_session if isinstance(pending_session, dict) else
                   state.get("session", {}))
        messages = session.get("messages", []) if isinstance(session, dict) else []
        decision_index = next((index for index, message in enumerate(messages)
                               if message.get("role") == "assistant" and
                               self._matches_plan(message.get("content"), expected_wave)), None)
        if decision_index is None:
            events = [self._event("Curriculum system prompt", "curriculum", reason="not retained"),
                      self._event("Curriculum user prompt", "curriculum", reason="not retained"),
                      self._event("Retained normalized curriculum plan", "curriculum", plan_path)]
        else:
            expected_id = expected_wave.get("wave_id")
            previous_decision = max((index for index, message in enumerate(messages[:decision_index])
                                     if message.get("role") == "assistant" and
                                     (self._wave_response(message.get("content")) or {}).get(
                                         "wave_id") not in (None, expected_id)),
                                    default=-1)
            relevant = [message for index, message in enumerate(messages[:decision_index + 1])
                        if index > previous_decision or message.get("role") == "system"]
            events, attempts, users = [], 0, 0
            for message in relevant:
                role, content = message.get("role"), message.get("content", "")
                if role == "system":
                    label = "Curriculum system prompt"
                elif role == "user":
                    label = "Curriculum user prompt" if users == 0 else "Curriculum user correction"
                    users += 1
                elif message is messages[decision_index]:
                    label = "Curriculum admitted decision"
                elif role == "assistant":
                    attempts += 1
                    label = f"Curriculum assistant attempt {attempts}"
                else:
                    label = f"Curriculum {role or 'unknown'} message"
                events.append(self._inline_event(label, "curriculum", content))
        spec_body = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
        events.append(self._inline_event("Complete project specification", "project", spec_body))
        ledger = branch / "evidence.jsonl"
        rows, actor_error = self._authenticated_ledger(ledger)
        labels = {"provider-request": "Actor prompt", "provider-response": "Actor response",
                  "host-observation": "Host observation",
                  "provider-response-failure": "Actor response failure",
                  "provider-attempt-rejection": "Actor attempt rejection"}
        if actor_error:
            events.append(self._event("Actor trace unavailable", "actor",
                                      reason="actor " + actor_error))
        else:
            for record, _, _ in rows:
                payload = record["payload"]
                role, digest = payload.get("role"), payload.get("sha256")
                if role not in labels:
                    continue
                source = (branch / "artifacts" / "objects" / digest[:2] / digest[2:]
                          if isinstance(digest, str) and _SHA256.fullmatch(digest) else None)
                event = self._artifact_event(labels[role], "actor", source, digest,
                                             stage=payload.get("stage"), turn=payload.get("turn"))
                events.append(event)
        if state.get("phase") == "target-driven-phase2":
            # The root ledger is keyed by project, not attempt: repeated target
            # retries cannot safely be attributed from it. The branch verifier
            # result below remains the authoritative attempt-specific evidence.
            events.append(self._event("Shared host ledger", "verification",
                                     reason="project-wide ledger; see attempt verifier result"))
        else:
            events.extend(self._host_evidence(root, project_id))
        events.append(self._event("Verifier prompt", "verification", reason=(
            "unavailable: verification is deterministic and host-owned")))
        events.append(self._event("Verifier result", "verification", branch / "execution-evidence.bin"))
        events.append(self._event("Final outcome", "outcome", branch / "outcome.json"))
        occurrences = {}
        for sequence, event in enumerate(events):
            identity = event.get("sha256") or hashlib.sha256(json.dumps(
                {key: event.get(key) for key in ("label", "category", "stage", "turn", "reason")},
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            occurrences[identity] = occurrences.get(identity, 0) + 1
            event["id"] = f"{identity}-{occurrences[identity]}"
            event["sequence"] = sequence
        return events

    def trace_index(self, phase, project_id):
        with self._db() as db:
            retained = db.execute("SELECT body FROM trace_indexes WHERE phase=? AND project_id=?",
                                  (phase, project_id)).fetchone()
        retained_body = json.loads(retained[0]) if retained else None
        retained_catalog = (retained_body.get("events") if isinstance(retained_body, dict) and
                            retained_body.get("schema") in {
                                "content-addressed-v1", "source-fingerprint-v1"} else None)
        source_fingerprint = self._trace_fingerprint(phase, project_id)
        unchanged = (retained_catalog is not None and source_fingerprint is not None and
                     retained_body.get("source_fingerprint") == source_fingerprint)
        events = None if unchanged or source_fingerprint is None else self._trace_sources(
            phase, project_id)
        if events is None:
            if retained_catalog is None:
                return None
            catalog = retained_catalog
        else:
            catalog = [{key: value for key, value in event.items() if key != "inline"}
                       for event in events]
            with self._db() as db:
                db.execute("INSERT OR REPLACE INTO trace_indexes(phase,project_id,body) VALUES(?,?,?)",
                           (phase, project_id, json.dumps({
                               "schema": "source-fingerprint-v1",
                               "source_fingerprint": source_fingerprint,
                               "events": catalog})))
                db.executemany("INSERT OR REPLACE INTO trace_events(phase,project_id,event_id,sha256) "
                               "VALUES(?,?,?,?)", [(phase, project_id, row["id"], row.get("sha256"))
                                                   for row in catalog])
        public = []
        for item in catalog:
            item = dict(item)
            if item["available"]:
                base = f"/api/phase/{phase}/project/{project_id}/trace/{item['id']}"
                item.update(payload_url=base, download_url=base + "/download")
            public.append(item)
        return {"events": public}

    def trace_payload(self, phase, project_id, event_id):
        with self._db() as db:
            retained = db.execute("SELECT sha256 FROM trace_events WHERE phase=? AND "
                                  "project_id=? AND event_id=?",
                                  (phase, project_id, event_id)).fetchone()
            cached = (db.execute("SELECT body FROM payloads WHERE sha256=?", (retained[0],)).fetchone()
                      if retained and retained[0] else None)
        if cached is not None:
            return bytes(cached[0]), None
        events = self._trace_sources(phase, project_id)
        event = next((row for row in events or [] if row["id"] == event_id), None)
        digest = retained[0] if retained else (event.get("sha256") if event else None)
        if not digest:
            return None, "missing"
        with self._db() as db:
            cached = db.execute("SELECT body FROM payloads WHERE sha256=?", (digest,)).fetchone()
        if cached is not None:
            return bytes(cached[0]), None
        if event is None or event.get("sha256") != digest:
            return None, "unavailable"
        if not event.get("available"):
            return None, "digest" if event.get("reason") == "content digest mismatch" else "unavailable"
        if "inline" in event:
            body = event["inline"]
        else:
            located = self._project_sources(phase, project_id)
            if located is None:
                return None, "unavailable"
            branch = located[2]
            try:
                if event["label"] == "Retained normalized curriculum plan":
                    body = located[1].read_bytes()
                elif event["label"] == "Verifier result":
                    body = (branch / "execution-evidence.bin").read_bytes()
                elif event["label"] == "Final outcome":
                    body = (branch / "outcome.json").read_bytes()
                elif _SHA256.fullmatch(digest):
                    body = (branch / "artifacts" / "objects" / digest[:2] / digest[2:]).read_bytes()
                else:
                    return None, "unavailable"
            except (OSError, TypeError, AttributeError):
                return None, "unavailable"
        if hashlib.sha256(body).hexdigest() != digest:
            return None, "digest"
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO payloads(sha256,body) VALUES(?,?)", (digest, body))
        return body, None
