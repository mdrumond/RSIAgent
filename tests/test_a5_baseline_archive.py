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
            "host_facts": [{"statement": "host pass"}],
        }
    }
    write(phase1 / "cells/a/memory.jsonl", json.dumps(memory).encode() + b"\n")
    write(phase1 / "cells/a/evidence.jsonl", b'{"kind":"result"}\n')
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
    assert len([item for item in objects if item.is_file()]) == 5


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
        "json_records": 2,
        "memory_records": 1,
        "profile_artifacts": 0,
    }
    assert audit["campaigns"][1]["signals"]["candidate_sources"] == 1
    assert audit["campaigns"][1]["signals"]["profile_artifacts"] == 1
    assert audit["known_missing"] == result["known_missing"]
    encoded = json.dumps(audit)
    assert str(tmp_path) not in encoded


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


def test_cli_reads_spec_and_reports_only_archive_identity(tmp_path, capsys):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec(tmp_path)))

    assert main(["--spec", str(path), "--destination", str(tmp_path / "archive")]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["campaigns"] == 2
    assert output["files"] == 6
    assert len(output["manifest_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(output)
