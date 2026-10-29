import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.a5kernels.sft_dataset import (
    REQUIRED_SOURCE_NAMES,
    SourceManifest,
    TraceRecorder,
    build_dataset,
    saturation_decision,
)


def _source_manifest(path: Path) -> Path:
    sources = [
        {
            "name": name,
            "repository": f"https://example.invalid/{name}",
            "revision": hashlib.sha1(name.encode()).hexdigest(),
            "license": "Apache-2.0",
            "include": ["**/*.py"],
        }
        for name in sorted(REQUIRED_SOURCE_NAMES)
    ]
    path.write_text(json.dumps({"schema": "a5-sft-sources-v1", "sources": sources}))
    return path


def _trace(
    root: Path,
    trace_id: str,
    family: str,
    source: str,
    *,
    failed_first: bool = False,
    passed: bool = True,
    round_id: str = "round-01",
    model: str = "deepseek-v4-flash",
    source_manifest_sha256: str = "a" * 64,
) -> Path:
    recorder = TraceRecorder(
        root / f"{trace_id}.jsonl",
        trace_id=trace_id,
        phase="phase1",
        round_id=round_id,
        operator_family=family,
        model=model,
        source_manifest_sha256=source_manifest_sha256,
    )
    recorder.message("system", "Write a Catlass kernel.")
    recorder.message("user", f"Implement {family}")
    if failed_first:
        recorder.message("assistant", "first attempt")
        recorder.tool_event("compile", {"status": "failed", "stderr": "bad type"})
        recorder.message("user", "The host compile failed: bad type")
    recorder.message("assistant", source)
    recorder.candidate(source, candidate_id=f"{trace_id}-candidate")
    recorder.validation(
        candidate_id=f"{trace_id}-candidate",
        validator="bz-a5-fixture",
        status="passed" if passed else "failed",
        evidence_sha256="b" * 64,
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    return recorder.finish()


def test_source_manifest_requires_all_pinned_sources(tmp_path):
    path = _source_manifest(tmp_path / "sources.json")
    loaded = SourceManifest.load(path)
    assert {source.name for source in loaded.sources} == REQUIRED_SOURCE_NAMES
    assert len(loaded.sha256) == 64

    raw = json.loads(path.read_text())
    raw["sources"] = raw["sources"][:-1]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="required sources"):
        SourceManifest.load(path)


def test_checked_source_manifest_is_valid():
    manifest = Path(__file__).parents[1] / "config/a5/sft-source-manifest.json"
    assert {source.name for source in SourceManifest.load(manifest).sources} == REQUIRED_SOURCE_NAMES


def test_trace_is_lossless_canonical_and_redacts_secrets(tmp_path):
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(
        path,
        trace_id="trace-1",
        phase="phase1",
        round_id="round-01",
        operator_family="vector",
        model="deepseek-v4-flash",
        source_manifest_sha256="a" * 64,
    )
    recorder.message("user", "token=supersecret AKIAABCDEFGHIJKLMNOP")
    recorder.tool_event(
        "compile",
        {"api_key": "secret", "command": ["compile", "kernel.py"], "status": "passed"},
    )
    recorder.candidate("def kernel():\n    return 1\n", candidate_id="candidate-1")
    recorder.validation(
        candidate_id="candidate-1",
        validator="fixture",
        status="passed",
        evidence_sha256="b" * 64,
        candidate_sha256=hashlib.sha256(b"def kernel():\n    return 1\n").hexdigest(),
    )
    recorder.finish()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    assert "supersecret" not in path.read_text()
    assert "AKIAABCDEFGHIJKLMNOP" not in path.read_text()
    assert rows[2]["payload"]["api_key"] == "[REDACTED]"
    assert rows[-1]["kind"] == "terminal"
    assert rows[-1]["payload"]["event_count"] == len(rows) - 1
    assert all(len(row["record_sha256"]) == 64 for row in rows)


def test_candidate_bytes_are_exact_and_actual_secrets_fail_closed(tmp_path):
    source = "def kernel(token = 0):\n    return token\n"
    path = _trace(tmp_path, "exact-source", "vector", source)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    candidate = next(row["payload"] for row in rows if row["kind"] == "candidate")
    assert candidate["source"] == source
    assert candidate["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()

    recorder = TraceRecorder(
        tmp_path / "secret.jsonl", trace_id="secret", phase="phase1",
        round_id="round-01", operator_family="vector", model="model",
        source_manifest_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="secret"):
        recorder.candidate(
            'API_KEY = "sk-live-abcdefghijklmnopqrstuvwxyz"', candidate_id="bad"
        )


def test_export_rejects_verified_candidate_digest_mismatch(tmp_path):
    path = _trace(tmp_path, "digest", "vector", "source")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    validation = next(row for row in rows if row["kind"] == "validation")
    validation["payload"]["candidate_sha256"] = "c" * 64
    previous = None
    for row in rows:
        row["previous_sha256"] = previous
        body = {key: value for key, value in row.items() if key != "record_sha256"}
        row["record_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        previous = row["record_sha256"]
    terminal = rows[-1]
    terminal["payload"]["events_sha256"] = hashlib.sha256(json.dumps(
        [row["record_sha256"] for row in rows[:-1]],
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    terminal_body = {key: value for key, value in terminal.items() if key != "record_sha256"}
    terminal["record_sha256"] = hashlib.sha256(json.dumps(
        terminal_body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    with pytest.raises(ValueError, match="verified candidate digest"):
        build_dataset(tmp_path, tmp_path / "dataset")


def test_dataset_admits_verified_finals_and_successful_repairs_only(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    _trace(traces, "plain", "attention", "final attention")
    _trace(traces, "repair", "matmul", "repaired matmul", failed_first=True)
    _trace(traces, "failed", "norm", "bad norm", passed=False)

    result = build_dataset(traces, tmp_path / "dataset", validation_fraction=0.1)
    assert result["eligible"] == 2
    assert result["rejected"] == 1
    examples = []
    for name in ("train.jsonl", "validation.jsonl"):
        examples.extend(json.loads(line) for line in (tmp_path / "dataset" / name).read_text().splitlines())
    assert {row["messages"][-1]["content"] for row in examples} == {
        "final attention", "repaired matmul"
    }
    assert all(set(row) == {"messages"} for row in examples)
    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text())
    repair = next(item for item in manifest["examples"] if item["trace_id"] == "repair")
    assert repair["repair"] is True


def test_fireworks_export_preserves_tools_and_masks_repair_history(tmp_path):
    path = tmp_path / "traces" / "tool-repair.jsonl"
    recorder = TraceRecorder(
        path,
        trace_id="tool-repair",
        phase="phase1",
        round_id="round-01",
        operator_family="attention",
        model="deepseek-v4-flash",
        source_manifest_sha256="a" * 64,
        tools=[{"type": "function", "function": {"name": "compile"}}],
    )
    recorder.message("system", "Write a kernel")
    recorder.message("user", "Implement attention")
    recorder.message(
        "assistant", "", reasoning_content="first try",
        tool_calls=[{
            "id": "call-1", "type": "function",
            "function": {"name": "compile", "arguments": "{}"},
        }],
    )
    recorder.message(
        "tool", '{"status":"failed"}', tool_call_id="call-1", name="compile"
    )
    recorder.tool_event("compile", {"status": "failed"})
    recorder.message("user", "Host compile failed")
    recorder.message("assistant", "good source", reasoning_content="repair")
    recorder.candidate("good source", candidate_id="final")
    recorder.validation(
        candidate_id="final", validator="fixture", status="passed",
        evidence_sha256="b" * 64,
        candidate_sha256=hashlib.sha256(b"good source").hexdigest(),
    )
    recorder.finish()
    build_dataset(path.parent, tmp_path / "dataset")
    row = json.loads((tmp_path / "dataset" / "train.jsonl").read_text())
    assert row["tools"][0]["function"]["name"] == "compile"
    assistant = [message for message in row["messages"] if message["role"] == "assistant"]
    assert [message["weight"] for message in assistant] == [0, 1]
    assert assistant[0]["tool_calls"][0]["id"] == "call-1"
    tool_result = next(message for message in row["messages"] if message["role"] == "tool")
    assert tool_result["tool_call_id"] == "call-1"
    assert assistant[-1]["reasoning_content"] == "repair"


def test_dataset_exact_and_near_dedup_is_deterministic_and_stratified(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    for index in range(12):
        _trace(traces, f"attn-{index:02}", "attention", f"kernel attention variant {index}")
        _trace(traces, f"mm-{index:02}", "matmul", f"kernel matmul variant {index}")
    _trace(traces, "exact", "attention", "kernel attention variant 0")
    _trace(
        traces,
        "near",
        "attention",
        "kernel   attention variant 0  # formatting-only",
    )

    first = build_dataset(traces, tmp_path / "one", validation_fraction=0.1)
    second = build_dataset(traces, tmp_path / "two", validation_fraction=0.1)
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["duplicates"] == 2
    manifest = json.loads((tmp_path / "one" / "manifest.json").read_text())
    assert set(manifest["split_counts_by_operator"]["validation"]) == {"attention", "matmul"}
    assert manifest["train_sha256"] == hashlib.sha256(
        (tmp_path / "one" / "train.jsonl").read_bytes()
    ).hexdigest()


def test_native_dedup_retains_distinct_preprocessor_values_and_identifier_case(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    template = ('#include "kernel_operator.h"\n#define TILE {tile}\n'
                'extern "C" __global__ __aicore__ void kernel(GM_ADDR output) {{\n'
                '  AscendC::LocalTensor<float> {identifier};\n'
                '  // Common implementation makes the sources near-identical.\n'
                '  AscendC::Duplicate({identifier}, 0.0f, TILE);\n}}\n')
    sources = [template.format(tile=16, identifier="tile"),
               template.format(tile=32, identifier="tile"),
               template.format(tile=16, identifier="Tile")]
    for index, source in enumerate(sources):
        _trace(traces, f"native-{index}", "native", source)
    _trace(traces, "native-exact-copy", "native", sources[0])
    manifest = build_dataset(traces, tmp_path / "dataset")
    assert manifest["eligible"] == 3
    assert manifest["duplicates"] == 1
    assert {row["candidate_sha256"] for row in manifest["examples"]} == {
        hashlib.sha256(source.encode()).hexdigest() for source in sources
    }


def test_saturation_requires_three_rounds_and_stops_below_ten_percent():
    assert saturation_decision([20, 5], [4, 2]) == {"stop": False, "reason": "minimum-rounds"}
    assert saturation_decision([20, 5, 1], [4, 2, 0]) == {"stop": False, "reason": "continue"}
    assert saturation_decision([20, 5, 0], [4, 2, 0]) == {
        "stop": True, "reason": "novelty-below-10-percent"
    }
    assert saturation_decision([20, 5, 0], [0, 0, 0]) == {
        "stop": True, "reason": "novelty-below-10-percent"
    }
    assert saturation_decision([20, 5, 4, 3, 2], [4, 3, 2, 1, 1]) == {
        "stop": True, "reason": "maximum-rounds"
    }


def test_dataset_provenance_rejects_mixed_manifests_and_persists_round_metrics(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    _trace(traces, "r1", "attention", "source one", round_id="round-01")
    _trace(
        traces, "r2", "attention", "source two", round_id="round-02",
        failed_first=True,
    )
    _trace(traces, "r3", "attention", "source three", round_id="round-03")
    manifest = build_dataset(traces, tmp_path / "dataset")
    assert manifest["phases"] == ["phase1"]
    assert manifest["models"] == ["deepseek-v4-flash"]
    assert manifest["source_manifest_sha256"] == "a" * 64
    assert manifest["rounds"] == [
        {"round_id": "round-01", "novel_verified_examples": 1, "novel_repair_patterns": 0},
        {"round_id": "round-02", "novel_verified_examples": 1, "novel_repair_patterns": 1},
        {"round_id": "round-03", "novel_verified_examples": 1, "novel_repair_patterns": 0},
    ]
    assert manifest["saturation"] == {"stop": False, "reason": "continue"}
    assert all(
        {"phase", "model", "source_manifest_sha256"} <= set(example)
        for example in manifest["examples"]
    )

    _trace(
        traces, "mixed", "attention", "source four", round_id="round-03",
        source_manifest_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="mixed source manifests"):
        build_dataset(traces, tmp_path / "mixed-dataset")
