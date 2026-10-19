"""Artifact indexing and a small durable snapshot cache."""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeDecodeError):
        return default


def _source(branch: Path, digest: str):
    if not isinstance(digest, str) or len(digest) != 64:
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
        if fact.get("category") != "performance-study":
            continue
        try:
            performance.extend(json.loads(fact["statement"]).get("timing", []))
        except (KeyError, TypeError, ValueError):
            continue
    # Duplicate facts can cite the same timing artifact. Present measurements once.
    performance = list({json.dumps(row, sort_keys=True): row for row in performance}.values())
    return delta, memory, performance


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
    return {"id": spec.get("project_id", branch.name),
            "description": spec.get("hypothesis", "No project description retained."),
            "summary": delta.get("summary", (failed or {}).get("summary", "No outcome summary retained.")),
            "verified": outcome.get("verified") is True,
            "classification": outcome.get("classification", "running" if not outcome else "unknown"),
            "spec": spec, "performance": performance,
            "memory": {"actions": memory.get("actions", []), "facts": memory.get("host_facts", []),
                       "lessons": lessons, "interpretations": memory.get("agent_interpretations", [])},
            "revisions": revisions, "source": revisions[-1]["source"] if revisions else "",
            "transitions": transitions}


def _campaign(entry):
    root = Path(entry["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"campaign root unavailable: {root}")
    state = _json(root / "state.json")
    if not isinstance(state, dict):
        raise ValueError(f"campaign state unavailable: {root / 'state.json'}")
    specs = {}
    for plan_path in sorted(root.glob("waves/*/plan.json")):
        plan = _json(plan_path, {}) or {}
        for item in plan.get("wave", {}).get("projects", []):
            spec = item.get("spec", item)
            if spec.get("project_id"):
                specs[spec["project_id"]] = spec
    failures = {row.get("project_id"): row for row in state.get("failed_outcomes", [])}
    branches = {}
    for branch in sorted(root.glob("waves/*/branches/*")):
        if branch.is_dir():
            branches[branch.name] = branch
    executed = set(branches) | {project_id for project_id in failures if project_id}
    projects = []
    for project_id, spec in specs.items():
        if project_id in executed:
            projects.append(_project(branches.get(project_id, root), spec, failures.get(project_id)))
    # Retain failures even if their old plan is temporarily unavailable.
    for project_id, failure in failures.items():
        if project_id and project_id not in specs:
            projects.append(_project(branches.get(project_id, root),
                                     failure.get("spec", {"project_id": project_id}), failure))
    control = Path(entry["control"]) if entry.get("control") else root.parent / "control"
    worker = _json(control / "worker.json")
    published = (len(state["prior_outcomes"]) if "prior_outcomes" in state else
                 sum(1 for project in projects if project["verified"]))
    planned = [{"id": project_id, "description": spec.get("hypothesis", ""), "spec": spec}
               for project_id, spec in specs.items() if project_id not in executed]
    return {"phase": int(entry["phase"]), "name": entry.get("name", f"Phase {entry['phase']}"),
            "root": str(root), "available": root.is_dir(), "worker": worker,
            "stats": {"attempted": len(projects),
                      "verified": sum(project["verified"] for project in projects),
                      "published": published}, "projects": projects, "planned": planned}


class DashboardStore:
    def __init__(self, config_path, cache_path):
        self.config_path, self.cache_path = Path(config_path), Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL)")
            db.execute("PRAGMA journal_mode=WAL")

    def _db(self):
        return sqlite3.connect(self.cache_path)

    def cached(self):
        with self._db() as db:
            row = db.execute("SELECT body FROM snapshots WHERE id=1").fetchone()
        return json.loads(row[0]) if row else {"generated_at": None, "stale": True, "campaigns": []}

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
