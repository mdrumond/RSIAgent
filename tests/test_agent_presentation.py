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
@pytest.mark.parametrize("reviewed", [False, True])
@pytest.mark.parametrize("private", [False, True])
def test_historical_artifact_reads_follow_current_memory_without_copying(tmp_path, reviewed, private):
    import base64
    from types import SimpleNamespace
    from benchmarks.ascend.agent_presentation import MemoryArtifactReader, artifact_page_presentation
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.evidence import canonical_bytes
    from benchmarks.a5kernels.phase1_adaptive import MEMORY_SCHEMA
    from benchmarks.a5kernels.phase2_adaptive_memory import admit_adaptive_snapshot
    from tests.test_a5_phase2_adaptive_memory import _source
    root, source, review, memory = _source(tmp_path)
    initial, entries = b"", [base64.b64encode(canonical_bytes(memory)).decode()]
    if reviewed:
        root = tmp_path / "phase2"
        snapshot = admit_adaptive_snapshot(tmp_path / "adaptive", b"", review, source, root)
        initial = canonical_bytes({"schema": "ascend-phase2-curriculum-seed-v1",
                                  "snapshot_id": snapshot.snapshot_id, "phase1_memory": [memory]})
        entries = []
    current = ContentAddressedArtifacts((tmp_path / "control/bootstrap/reference" if private
                                        else root / "waves/wave-02/branches/next") / "artifacts")
    raw = canonical_bytes({"schema": MEMORY_SCHEMA,
                          "initial_memory_base64": base64.b64encode(initial).decode(),
                          "project_memory_base64": entries})
    request = SimpleNamespace(pre_wave_memory=raw, branch_root=current.root.parent,
                              memory_root=root if private else None)
    reader = MemoryArtifactReader(request, current)
    assert reader.read_bytes(memory["source_revision"]) == b"actual candidate source"
    evidence = memory["host_facts"][0]["evidence_sha256"]
    page = artifact_page_presentation({"sha256": evidence}, reader)
    assert '"host":"verified"' in page["text"]
    assert not current._path(evidence).exists()
    with pytest.raises(FileNotFoundError):
        reader.read_bytes("f" * 64)


@pytest.mark.parametrize('adaptive', [False, True])
def test_usefulness_treatment_keeps_lesson_and_prior_memories_visible(tmp_path, adaptive):
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    prior = {'memory_entries': [{'statement': 'prior optimizer insight'}]}
    if adaptive:
        prior = dict(initial_memory_base64=base64.b64encode(b'{}').decode(),
                     project_memory_base64=[])
    prior['usefulness_lesson'] = {'statement': 'new isolated insight'}
    view = memory_presentation(json.dumps(prior).encode(), ContentAddressedArtifacts(tmp_path))
    assert view['usefulness_lesson']['statement'] == 'new isolated insight'
    if not adaptive:
        assert view['memory_entries'][0]['lesson']['statement'] == 'prior optimizer insight'
def test_subject_catalog_keeps_useful_lessons_and_exact_documents_within_one_page(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    source = store.put_bytes(b'out = max(x, 0)\n')
    proposal = dict(summary='Clamp negative values', applicability='float32 vectors',
                    preconditions='Padded storage', expectations='Logical values match relu',
                    limitations='No timing claim', before=None,
                    after=dict(sha256=source, start_line=1, end_line=1,
                               text='out = max(x, 0)\n'), evidence=[source])
    record = dict(memory_id='accepted-relu', proposal=proposal,
                  provenance={'logs': 'old trace\n' * 9000})
    raw = canonical_bytes(dict(schema='subject-memory-snapshot-v1', initial_memory='Use host oracles',
                               memories=[dict(record, memory_id=f'lesson-{i}') for i in range(100)]))
    view = memory_presentation(raw, store)
    assert len(canonical_bytes(view)) <= 8192
    assert 0 < view['shown_count'] < 100
    assert view['shown_count'] + view['omitted_count'] == view['memory_count'] == 100
    assert store.read_bytes(view['sha256']) == raw
    assert store.read_bytes(view['initial_memory_sha256']) == b'Use host oracles'
    for index, row in enumerate(view['memories']):
        assert store.read_json(row['memory_document_sha256']) == dict(record, memory_id=f'lesson-{index}')
        assert row['version'] == row['memory_document_sha256']
        assert store.read_json(row['proposal_sha256']) == proposal
        assert row['summary'] == proposal['summary']
        assert row['after'] == proposal['after']


def test_subject_unicode_and_long_examples_stay_bounded_and_readable_by_reference(tmp_path):
    from types import SimpleNamespace
    from benchmarks.ascend.agent_presentation import MemoryArtifactReader
    shared = ContentAddressedArtifacts(tmp_path / 'shared')
    current = ContentAddressedArtifacts(tmp_path / 'branch' / 'artifacts')
    code = ('out = x\n' * 10000).encode()
    source = shared.put_bytes(code)
    unrelated = shared.put_bytes(b'not reachable from this memory')
    # Reproduce the production eager-copy boundary: selected source only.
    current.put_bytes(shared.read_bytes(source))
    proposal = dict(summary='修正' * 10000, applicability='适用' * 10000,
                    preconditions='前提' * 10000, expectations='结果' * 10000,
                    limitations='限制' * 10000, before=None,
                    after=dict(sha256=source, start_line=1, end_line=10000, text=code.decode()),
                    evidence=[source] * 100)
    record = dict(memory_id='unicode', proposal=proposal, provenance={})
    raw = canonical_bytes(dict(schema='subject-memory-snapshot-v1', initial_memory='种子' * 10000,
                               memories=[record]))
    reader = MemoryArtifactReader(SimpleNamespace(pre_wave_memory=raw), current)
    view = memory_presentation(raw, current)
    assert len(canonical_bytes(view)) <= 8192
    row = view['memories'][0]
    assert len(row['summary']) > 0
    assert 'text' not in row['after']
    assert row['omitted_evidence_count'] == 96
    assert json.loads(reader.read_bytes(row['proposal_sha256'])) == proposal
    assert reader.read_bytes(row['after']['sha256']) == code
    with pytest.raises(FileNotFoundError):
        reader.read_bytes(unrelated)
