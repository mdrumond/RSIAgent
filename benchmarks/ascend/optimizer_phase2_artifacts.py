"""Authenticate committed optimizer-backed Phase 2 learning and package its evidence."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from optimizer.jobs import JobStore, canonical, sha
from optimizer.memory import verify_snapshot


def _relative(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Phase 2 evidence path must be campaign-relative")
    return root / path


def _artifact(store, key, digest):
    # The caller has authenticated this job and its event chain. Do not rehash
    # the large immutable submission once per compact profiler CSV.
    data = store.artifact_path(key, digest).read_bytes()
    if sha(data) != digest:
        raise ValueError("Phase 2 artifact digest mismatch")
    return data


def verify_attempt_result(store, result, *, attempt_id, task_id):
    """Authenticate a retained attempt verdict against native execution evidence."""
    key = result['job_id']
    store.get(key)
    events = store.events(key)
    if not any(e['kind'] == 'phase2-attempt-result' and e['payload'] == result for e in events):
        raise ValueError('Phase 2 attempt result missing from optimizer event chain')
    body = json.loads(_artifact(store, key, result['result_artifact']['sha256']))
    if body != {k: v for k, v in result.items() if k != 'result_artifact'}:
        raise ValueError('Phase 2 result artifact mismatch')
    if result['attempt_id'] != attempt_id or result['task_id'] != task_id:
        raise ValueError('Phase 2 execution identity mismatch')
    if result.get('verified'):
        native = result['result']
        source = json.loads(_artifact(store, key, result['candidate']['sha256']))
        baseline = json.loads(_artifact(store, key, result['baseline']['sha256']))
        if (not isinstance(source, str) or sha(source.encode()) != result['source_sha256']
                or native.get('candidate_sha256') != result['source_sha256']
                or native.get('status') != 'verified'
                or native.get('suite_sha256') != baseline['suite_sha256']
                or not baseline['cases']
                or sorted(c['case'] for c in native['cases']) != sorted(c['case'] for c in baseline['cases'])):
            raise ValueError('verified attempt does not match frozen suite and candidate')


def verify_campaign(root, *, require_ready=False):
    """Check seed identity, committed rows, and immutable optimizer evidence."""
    root = Path(root)
    state = json.loads((root / "state.json").read_bytes())
    if state.get("phase") != "optimizer-phase2":
        raise ValueError("not an optimizer Phase 2 campaign")
    seed = json.loads(_relative(root, state.get("seed_file", "seed-memory.json")).read_bytes())
    verify_snapshot(seed)
    if seed["snapshot_id"] != state["snapshot_id"]:
        raise ValueError("Phase 2 seed snapshot mismatch")
    attempts = state.get("attempts", [])
    for row in attempts:
        retained = json.loads(_relative(root, f"attempts/{row['attempt_id']}.json").read_bytes())
        if retained != row:
            raise ValueError("committed Phase 2 attempt mismatch")
        directory = _relative(root, row["execution_directory"])
        result = row["result"]
        store, key = JobStore(directory / "jobs"), result["job_id"]
        store.get(key)
        submission = store.submission(key)
        verify_snapshot(submission["memory_snapshot"])
        if submission["memory_snapshot"]["snapshot_id"] != seed["snapshot_id"]:
            raise ValueError("optimizer task memory differs from reviewed Phase 2 seed")
        task_submission = row["task"]["submission"]
        if (any(submission.get(k) != task_submission.get(k) for k in ("source", "entrypoint"))
                or submission["signature"] != {"signatures": task_submission["signatures"]}
                or submission["init_args"] != task_submission.get("init_args", [])
                or submission["init_kwargs"] != task_submission.get("init_kwargs", {})):
            raise ValueError("Phase 2 task differs from frozen optimizer submission")
        if row["kind"] == "target" and row["task"] != state["target"]:
            raise ValueError("Phase 2 target changed between attempts")
        for digest in {artifact["sha256"] for artifact in store.artifacts(key)}:
            _artifact(store, key, digest)
        verify_attempt_result(store, result, attempt_id=row['attempt_id'], task_id=row['task_id'])
        if row.get("verified") != result.get("verified"):
            raise ValueError("Phase 2 verified flag mismatch")
    evidence = {sha(canonical(row)) for row in attempts}
    for entry in state.get("memory_entries", []):
        if entry.get("evidence_sha256") not in evidence:
            raise ValueError("Phase 2 memory missing committed attempt evidence")
    ready = (state.get("terminal") or {}).get("status") == "READY"
    if ready or require_ready:
        if (not ready or (root / "pending.json").exists() or not attempts
                or attempts[-1].get("kind") != "target" or not attempts[-1].get("verified")
                or attempts[-1].get("task_id") != state["target"]["task_id"]):
            raise ValueError("Phase 2 has no committed verified final target")
    return state, seed


def freeze_campaign(root):
    """Return the same portable snapshot interface as optimizer.memory.freeze_source."""
    from .training_annex import _retained
    root = Path(root)
    before = (root / "state.json").read_bytes()
    state, seed = verify_campaign(root, require_ready=True)
    files = {}
    for name, path in _retained(root):
        data = path.read_bytes()
        files[name] = {"sha256": sha(data), "base64": base64.b64encode(data).decode()}
    verify_campaign(root, require_ready=True)
    if (root / "state.json").read_bytes() != before or (root / "pending.json").exists():
        raise ValueError("Phase 2 changed during snapshot")
    body = dict(schema="optimizer-memory-snapshot-v1", source_identity=seed["source_identity"],
                phase="optimizer-phase2", terminal=state["terminal"],
                phase1_snapshot_id=state["snapshot_id"],
                memory_entries=[*seed["memory_entries"], *state.get("memory_entries", [])],
                files=files)
    return {**body, "snapshot_id": sha(canonical(body))}
