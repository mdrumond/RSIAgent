"""Large evidence remains exact on disk and useful in bounded Actor messages."""

import base64
import hashlib
import json

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.trial import parse_trial_action
from benchmarks.ascend.agent_presentation import (
    artifact_page_presentation, compact_observation, memory_presentation, mismatch_summary, read_artifact,
)


def test_large_tensor_observation_retains_full_artifacts_and_error_samples(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    expected = list(range(32768))
    observed = expected.copy()
    observed[2048] = -2
    payload = {"event": "host-execution", "classification": "host-mismatch",
               "source_sha256": "a" * 64,
               "numerical_summary": mismatch_summary(expected, observed),
               "artifacts": {"observed": {
                   "sha256": store.put_json({"output": observed}),
                   "payload": {"output": observed}},
                   "expected": {"sha256": store.put_json({"storage": expected}),
                                "payload": {"storage": expected}}}}
    view = compact_observation(payload, store)
    assert len(canonical_bytes(view)) < 12 * 1024
    assert view["classification"] == "host-mismatch"
    assert view["numerical_summary"]["samples"][0]["index"] == 2048
    assert view["numerical_summary"]["max_finite_absolute_error"] == 2050
    assert store.read_json(view["full_observation_sha256"]) == payload
    assert view["artifacts"]["observed"]["sha256"] == payload["artifacts"]["observed"]["sha256"]


def test_large_compiler_and_profile_observations_are_hash_linked(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    diagnostic = "frontend error: invalid intrinsic\n" + "verbose\n" * 40000 + "last error: 42"
    for payload in ({"event": "host-compile", "passed": False, "diagnostic": diagnostic},
                    {"event": "profile", "rows": [{"metric": "pipe", "value": 1}] * 50000}):
        view = compact_observation(payload, store)
        assert len(canonical_bytes(view)) < 12 * 1024
        assert store.read_json(view["full_observation_sha256"]) == payload
    view = compact_observation({"event": "host-compile", "passed": False,
                                "diagnostic": diagnostic}, store)
    assert "invalid intrinsic" in view["diagnostic"]["prefix"]
    assert "last error" in view["diagnostic"]["suffix"]


def test_memory_indexes_every_original_and_shows_real_lessons(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    projects = [{"project_spec": {"project_id": f"p-{i}"},
                 "agent_lessons": [{"claim": "Use padded movement", "detail": "x" * 3000}]}
                for i in range(64)]
    encoded = lambda value: base64.b64encode(canonical_bytes(value)).decode("ascii")
    raw = canonical_bytes({"initial_memory_base64": encoded({}),
                           "project_memory_base64": [encoded(row) for row in projects]})
    view = memory_presentation(raw, store)
    assert len(view["projects"]) == 64
    assert "Use padded movement" in json.dumps(view)
    assert "base64" not in json.dumps(view)
    assert len(canonical_bytes(view)) < 24 * 1024
    assert store.read_bytes(view["sha256"]) == raw
    for row, original in zip(view["projects"], projects):
        assert store.read_json(row["sha256"]) == original


def test_phase2_reviewed_seed_also_has_readable_lessons(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    initial = {"phase1_memory": [{"project_spec": {"project_id": "seed"},
                                 "agent_lessons": [{"claim": "profile before tuning"}]}]}
    raw = canonical_bytes({"initial_memory_base64": base64.b64encode(
        canonical_bytes(initial)).decode("ascii"), "project_memory_base64": []})
    view = memory_presentation(raw, store)
    assert view["projects"][0]["origin"] == "reviewed-phase1"
    assert view["projects"][0]["lessons"][0]["claim"] == "profile before tuning"


def test_artifact_pages_exact_bytes_including_unicode_boundaries(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    raw = ("\u20ac" * 10000).encode()
    digest = store.put_bytes(raw)
    offset, recovered = 0, b""
    while offset is not None:
        action = parse_trial_action(json.dumps({"read-artifact": {"sha256": digest, "offset": offset}}))
        page = read_artifact(action.payload, store)
        part = base64.b64decode(page["base64"])
        assert len(part) <= 8192
        recovered += part
        offset = page["next_offset"]
    assert hashlib.sha256(recovered).hexdigest() == digest
    assert read_artifact({"sha256": digest, "offset": 999999}, store)["next_offset"] is None


@pytest.mark.parametrize("raw", [b"plain kernel research\n" * 1000, ("\u20ac\U0001f600" * 2000).encode()])
def test_readable_pages_preserve_text_and_raw_evidence(tmp_path, raw):
    store = ContentAddressedArtifacts(tmp_path)
    digest = store.put_bytes(raw)
    offset, text = 0, ""
    while offset is not None:
        view = artifact_page_presentation({"sha256": digest, "offset": offset}, store)
        assert "base64" not in view
        assert view["encoding"] == "utf-8"
        assert view["page_bytes"] <= 8192
        original = store.read_json(view["full_observation_sha256"])
        assert base64.b64decode(original["base64"]) == view["text"].encode()
        text += view["text"]
        offset = view["next_offset"]
    assert text.encode() == raw
    assert store.read_bytes(digest) == raw


def test_binary_and_nonboundary_offset_are_metadata_only(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    for raw, offset in [(b"\xff\x00binary", 0), ("\u20ac\u20ac".encode(), 1)]:
        digest = store.put_bytes(raw)
        view = artifact_page_presentation({"sha256": digest, "offset": offset, "limit": 4}, store)
        assert view["text_available"] is False
        assert "text" not in view and "base64" not in view
        page = store.read_json(view["full_observation_sha256"])
        assert base64.b64decode(page["base64"]) == raw[offset:offset + 4]
    with pytest.raises(ValueError, match="codepoint"):
        artifact_page_presentation({"sha256": digest, "limit": 1}, store)


@pytest.mark.parametrize("extra", [{"limit": 8193}, {"offset": -1}, {"limit": True},
                                   {"offset": 1.5}, {"path": "/tmp/file"}])
def test_artifact_parser_rejects_invalid_pages(extra):
    assert parse_trial_action(json.dumps({"read-artifact": {"sha256": "a" * 64, **extra}})) is None


def test_nonfinite_and_length_mismatch_summary():
    view = mismatch_summary([1.0, 2.0], ["NaN"])
    assert view["observed_count"] == 1
    assert view["samples"][0]["observed"] == "NaN"
    canonical_bytes(view)


@pytest.mark.parametrize("expected,observed,missing_side", [
    ([1.0, 2.0, 3.0], [1.0], "observed"),
    ([1.0], [1.0, 2.0, 3.0], "expected"),
])
def test_matching_prefix_length_mismatch_counts_and_samples_tail(expected, observed, missing_side):
    view = mismatch_summary(expected, observed)
    assert view["unequal_count"] == 2
    assert [sample["index"] for sample in view["samples"]] == [1, 2]
    for sample in view["samples"]:
        assert sample["missing_side"] == missing_side
        assert sample[missing_side] is None
        assert sample["absolute_error"] is None
    canonical_bytes(view)


def test_tail_samples_share_eight_entry_budget_with_prefix_mismatches():
    view = mismatch_summary([0] * 100, [1] * 6)
    assert view["unequal_count"] == 100
    assert len(view["samples"]) == 8
    assert [sample["index"] for sample in view["samples"]] == list(range(8))
    assert view["samples"][-1]["missing_side"] == "observed"


def test_wide_diagnostic_stays_bounded_and_plain_initial_seed_is_visible(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    payload = {"event": "host-compile", "source_sha256": "b" * 64,
               "passed": False, "diagnostic": "compiler error\n" * 100000,
               "detail": "detail\n" * 100000,
               "other": {str(i): "x" * 3000 for i in range(50)}}
    view = compact_observation(payload, store)
    assert len(canonical_bytes(view)) <= 12 * 1024
    assert view["passed"] is False
    raw = canonical_bytes({"initial_memory_base64": base64.b64encode(
        b"Use measured host results.").decode("ascii"), "project_memory_base64": []})
    assert memory_presentation(raw, store)["initial_memory_preview"] == "Use measured host results."
