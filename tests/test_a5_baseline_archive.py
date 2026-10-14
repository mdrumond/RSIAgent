import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.a5kernels.baseline_archive import (
    ArchiveError,
    archive_baselines,
    load_archive_spec,
)
from run_a5_baseline_archive import main


def write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def spec(tmp_path: Path) -> dict[str, object]:
    phase1 = tmp_path / "inputs" / "p1"
    phase2 = tmp_path / "inputs" / "p2"
    memory = {
        "memory": {
            "agent_interpretations": [],
            "host_facts": [{
                "statement": "host pass",
                "evidence_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            }],
        }
    }
    write(phase1 / "cells/a/memory.jsonl", json.dumps(memory).encode() + b"\n")
    write(
        phase1 / "cells/a/evidence.jsonl",
        b'{"kind":"action","payload":{"action":"compile"},'
        b'"entry_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n',
    )
    write(phase1 / "cells/a/agent-prompt.json", b'{"system_prompt":"write a kernel"}\n')
    write(
        phase1 / "cells/a/knowledge.jsonl",
        b'{"kind":"knowledge_query","citations":[{"path":"guide.md"}]}\n',
    )
    write(phase2 / "cells/b/candidate.py", b"def kernel():\n    pass\n")
    write(phase2 / "cells/b/profile.json", b'{"metric":"pipe"}\n')
    write(phase2 / "duplicate.txt", b"shared")
    write(phase1 / "duplicate.txt", b"shared")
    return {
        "campaigns": [
            {
                "id": "phase2-openai-kdb",
                "phase": "phase2",
                "root": str(phase2),
                "model": "openai",
                "treatments": {"kdb": "enabled", "profiling": "disabled"},
            },
            {
                "id": "phase1-deepseek-profile",
                "phase": "phase1",
                "root": str(phase1),
                "model": "deepseek",
                "treatments": {"kdb": "disabled", "profiling": "enabled"},
            },
        ],
        "known_missing": [
            {
                "scope": "phase1-deepseek-profile",
                "artifact": "provider completion text",
                "reason": "the historical runner retained only action hashes",
            }
        ],
    }


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_archive_is_deterministic_content_addressed_and_source_immutable(tmp_path):
    value = spec(tmp_path)
    source = tmp_path / "inputs"
    before = tree_digest(source)

    first = archive_baselines(load_archive_spec(value), tmp_path / "archive-a")
    second = archive_baselines(load_archive_spec(value), tmp_path / "archive-b")

    assert before == tree_digest(source)
    assert first == second
    assert (tmp_path / "archive-a/raw/manifest.json").read_bytes() == (
        tmp_path / "archive-b/raw/manifest.json"
    ).read_bytes()
    assert (tmp_path / "archive-a/derived/audit.json").read_bytes() == (
        tmp_path / "archive-b/derived/audit.json"
    ).read_bytes()
    inventory = first["inventory"]
    shared = [item for item in inventory if item["path"] == "duplicate.txt"]
    assert len(shared) == 2 and shared[0]["sha256"] == shared[1]["sha256"]
    objects = list((tmp_path / "archive-a/raw/objects/sha256").rglob("*"))
    assert len([item for item in objects if item.is_file()]) == 7


def test_review_resolves_content_addressed_roles_and_kernel_source_names(tmp_path):
    root = tmp_path / "inputs" / "p1"
    instruction = b"write a kernel"
    source = b"import catlass.tla as tla\n"
    observation = b'{"host":{"event":"submitted","passed":true}}'
    refs = {
        "instruction": hashlib.sha256(instruction).hexdigest(),
        "source": hashlib.sha256(source).hexdigest(),
        "observation": hashlib.sha256(observation).hexdigest(),
    }
    for name, content in (
        (refs["instruction"], instruction),
        (refs["source"], source),
        (refs["observation"], observation),
    ):
        write(root / "artifacts/objects" / name[:2] / name[2:], content)
    records = [
        {"kind": "request", "payload": {
            "instruction_artifact_sha256": refs["instruction"],
        }},
        {"kind": "artifact", "payload": {
            "role": "final-source", "sha256": refs["source"],
        }},
        {"kind": "artifact", "payload": {
            "role": "host-observation", "sha256": refs["observation"],
        }},
    ]
    write(root / "evidence.jsonl", b"".join(
        json.dumps(record, sort_keys=True).encode() + b"\n" for record in records
    ))
    write(root / "workspace/kernel.py", source)
    value = {
        "campaigns": [{
            "id": "phase1-cas", "phase": "phase1", "root": str(root),
            "model": "openai", "treatments": {"kdb": "disabled"},
        }],
        "known_missing": [],
    }

    archive_baselines(load_archive_spec(value), tmp_path / "archive")
    review = json.loads(
        (tmp_path / "archive/derived/review.json").read_text(encoding="utf-8")
    )["campaigns"][0]
    audit = json.loads(
        (tmp_path / "archive/derived/audit.json").read_text(encoding="utf-8")
    )["campaigns"][0]

    assert review["evidence_linkage"] == {
        "archive_resolved_count": 3,
        "reference_count": 3,
        "unresolved_count": 0,
    }
    assert review["retained_layers"]["prompts"]["record_count"] == 1
    assert review["retained_layers"]["source"]["record_count"] == 1
    assert review["retained_layers"]["host_observations"]["record_count"] == 1
    assert audit["signals"]["candidate_sources"] == 1


def test_audit_is_symmetric_and_declares_missing_historical_material(tmp_path):
    result = archive_baselines(load_archive_spec(spec(tmp_path)), tmp_path / "archive")
    audit = json.loads((tmp_path / "archive/derived/audit.json").read_text())

    assert [item["campaign_id"] for item in audit["campaigns"]] == [
        "phase1-deepseek-profile", "phase2-openai-kdb",
    ]
    assert set(audit["campaigns"][0]) == set(audit["campaigns"][1])
    assert audit["campaigns"][0]["signals"] == {
        "candidate_sources": 0,
        "evidence_records": 1,
        "interpretations": 0,
        "json_records": 4,
        "memory_records": 1,
        "profile_artifacts": 0,
    }
    assert audit["campaigns"][1]["signals"]["candidate_sources"] == 1
    assert audit["campaigns"][1]["signals"]["profile_artifacts"] == 1
    assert audit["known_missing"] == result["known_missing"]
    encoded = json.dumps(audit)
    assert str(tmp_path) not in encoded


def test_review_describes_retained_layers_links_claims_and_information_loss(tmp_path):
    value = spec(tmp_path)
    phase1 = Path(value["campaigns"][1]["root"])
    write(
        phase1 / "cells/a/memory-claims.jsonl",
        b'{"statement":"kernel output is correct"}\n'
        b'{"statement":"profile latency is lower",'
        b'"evidence_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n',
    )

    archive_baselines(load_archive_spec(value), tmp_path / "archive")
    review = json.loads((tmp_path / "archive/derived/review.json").read_text())

    assert review["schema"] == "a5-baseline-content-review-v1"
    campaigns = {item["campaign_id"]: item for item in review["campaigns"]}
    phase1_review = campaigns["phase1-deepseek-profile"]
    layers = phase1_review["retained_layers"]
    assert layers["prompts"]["file_count"] == 1
    assert layers["actions"]["record_count"] == 1
    assert layers["host_observations"]["record_count"] == 1
    assert layers["knowledge_queries"]["record_count"] == 1
    assert layers["memory"]["file_count"] == 2
    assert phase1_review["evidence_linkage"]["reference_count"] == 2
    assert phase1_review["evidence_linkage"]["archive_resolved_count"] == 1
    assert phase1_review["claims"]["with_evidence_reference"] == {
        "correctness": 1,
        "performance": 1,
    }
    assert phase1_review["claims"]["unsupported"] == {
        "correctness": 1,
        "performance": 0,
    }
    assert phase1_review["claims"]["unsupported_findings"][0]["path"].endswith(
        "memory-claims.jsonl"
    )
    assert "text" not in phase1_review["claims"]["unsupported_findings"][0]
    assert phase1_review["information_loss"]["declared"][0]["artifact"] == (
        "provider completion text"
    )
    assert "source" in phase1_review["information_loss"]["absent_layers"]


def test_content_review_is_deterministic_and_does_not_embed_source_roots(tmp_path):
    value = spec(tmp_path)
    archive_baselines(load_archive_spec(value), tmp_path / "archive-a")
    archive_baselines(load_archive_spec(value), tmp_path / "archive-b")

    first = (tmp_path / "archive-a/derived/review.json").read_bytes()
    second = (tmp_path / "archive-b/derived/review.json").read_bytes()
    assert first == second
    assert str(tmp_path).encode() not in first


@pytest.mark.parametrize(
    "name,content",
    [
        (".env", b"OPENAI_API_KEY=not-safe"),
        ("notes.txt", b"DEEPSEEK_API_KEY=not-safe"),
        ("response.json", b'{"access_token":"this-is-a-long-secret-value"}'),
        ("key.pem", b"-----BEGIN PRIVATE KEY-----\nabc"),
    ],
)
def test_secret_material_fails_before_any_archive_is_written(tmp_path, name, content):
    value = spec(tmp_path)
    write(Path(value["campaigns"][0]["root"]) / name, content)
    destination = tmp_path / "archive"

    with pytest.raises(ArchiveError, match="secret-like material"):
        archive_baselines(load_archive_spec(value), destination)

    assert not destination.exists()


def test_archive_rejects_symlinks_and_destination_inside_source(tmp_path):
    value = spec(tmp_path)
    root = Path(value["campaigns"][0]["root"])
    (root / "link").symlink_to(root / "duplicate.txt")

    with pytest.raises(ArchiveError, match="symbolic link"):
        archive_baselines(load_archive_spec(value), tmp_path / "archive")
    (root / "link").unlink()
    with pytest.raises(ArchiveError, match="inside a campaign"):
        archive_baselines(load_archive_spec(value), root / "archive")


def test_invalid_metadata_is_rejected_without_leaking_values(tmp_path):
    value = spec(tmp_path)
    value["campaigns"][0]["treatments"] = {"api_key": "secret"}
    with pytest.raises(ValueError, match="treatment name"):
        load_archive_spec(value)


@pytest.mark.parametrize(
    "location",
    ["model", "treatment", "missing-reason"],
)
def test_secret_metadata_values_fail_before_destination_creation(tmp_path, location):
    value = spec(tmp_path)
    secret = "OPENAI_API_KEY=sk-" + "x" * 32
    if location == "model":
        value["campaigns"][0]["model"] = secret
    elif location == "treatment":
        value["campaigns"][0]["treatments"]["kdb"] = secret
    else:
        value["known_missing"][0]["reason"] = secret
    destination = tmp_path / "archive"

    with pytest.raises(ValueError, match="secret-like metadata"):
        archive_baselines(load_archive_spec(value), destination)

    assert not destination.exists()


def test_malformed_jsonl_line_does_not_hide_valid_surrounding_records(tmp_path):
    value = spec(tmp_path)
    root = Path(value["campaigns"][1]["root"])
    write(
        root / "cells/a/mixed.jsonl",
        b'{"action":"compile"}\nnot-json\n'
        b'{"action":"run","summary":"kernel output passed"}\n',
    )

    archive_baselines(load_archive_spec(value), tmp_path / "archive")
    review = json.loads((tmp_path / "archive/derived/review.json").read_text())
    campaign = next(
        item for item in review["campaigns"]
        if item["campaign_id"] == "phase1-deepseek-profile"
    )

    assert campaign["retained_layers"]["actions"]["record_count"] == 3
    assert campaign["claims"]["unsupported"]["correctness"] == 1


def test_cli_counts_validated_empty_campaigns(tmp_path, capsys):
    root = tmp_path / "empty"
    root.mkdir()
    value = {
        "campaigns": [{
            "id": "empty-phase1", "phase": "phase1", "root": str(root),
            "model": "openai", "treatments": {},
        }],
        "known_missing": [],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(value))

    assert main(["--spec", str(path), "--destination", str(tmp_path / "archive")]) == 0
    assert json.loads(capsys.readouterr().out)["campaigns"] == 1


def test_cli_reads_spec_and_reports_only_archive_identity(tmp_path, capsys):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec(tmp_path)))

    assert main(["--spec", str(path), "--destination", str(tmp_path / "archive")]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["campaigns"] == 2
    assert output["files"] == 8
    assert len(output["manifest_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(output)
