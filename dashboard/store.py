"""Artifact indexing and a small durable snapshot cache."""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DESCRIPTION_LABELS = ("Question", "Controlled change", "Tools", "Measurements",
                       "Expected lesson")


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
    planned = [{"id": project_id, **_description(spec.get("hypothesis", "")), "spec": spec}
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
            db.execute("CREATE TABLE IF NOT EXISTS payloads (sha256 TEXT PRIMARY KEY, body BLOB NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS trace_events (
                phase INTEGER NOT NULL, project_id TEXT NOT NULL, event_id TEXT NOT NULL,
                sha256 TEXT, PRIMARY KEY(phase, project_id, event_id))""")
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

    def _project_sources(self, phase, project_id):
        try:
            config = json.loads(self.config_path.read_text())
        except (OSError, ValueError):
            return None
        entry = next((row for row in config.get("campaigns", [])
                      if row.get("phase") == phase), None)
        if entry is None:
            return None
        root = Path(entry["root"])
        for plan_path in sorted(root.glob("waves/*/plan.json")):
            plan = _json(plan_path, {}) or {}
            for item in plan.get("wave", {}).get("projects", []):
                spec = item.get("spec", item)
                if spec.get("project_id") == project_id:
                    branch = plan_path.parent / "branches" / project_id
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

    def _trace_sources(self, phase, project_id):
        located = self._project_sources(phase, project_id)
        if located is None:
            return None
        root, plan_path, branch, spec = located
        wave_id = plan_path.parent.name
        session = (_json(root / "state.json", {}) or {}).get("session", {})
        messages = session.get("messages", []) if isinstance(session, dict) else []
        decision_index = next((index for index, message in enumerate(messages)
                               if message.get("role") == "assistant" and
                               wave_id in str(message.get("content", ""))), None)
        if decision_index is None:
            events = [self._event("Curriculum system prompt", "curriculum", reason="not retained"),
                      self._event("Curriculum user prompt", "curriculum", reason="not retained"),
                      self._event("Curriculum decision", "curriculum", plan_path)]
        else:
            system = next((message.get("content", "") for message in reversed(messages[:decision_index])
                           if message.get("role") == "system"), None)
            user = next((message.get("content", "") for message in reversed(messages[:decision_index])
                         if message.get("role") == "user"), None)
            events = [(self._inline_event("Curriculum system prompt", "curriculum", system)
                       if system is not None else self._event(
                           "Curriculum system prompt", "curriculum", reason="not retained")),
                      (self._inline_event("Curriculum user prompt", "curriculum", user)
                       if user is not None else self._event(
                           "Curriculum user prompt", "curriculum", reason="not retained")),
                      self._inline_event("Curriculum decision", "curriculum",
                                         messages[decision_index].get("content", ""))]
        spec_body = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
        events.append(self._inline_event("Complete project specification", "project", spec_body))
        ledger = branch / "evidence.jsonl"
        try:
            rows = ledger.read_text().splitlines()
        except OSError:
            rows = []
        labels = {"provider-request": "Actor prompt", "provider-response": "Actor response",
                  "host-observation": "Host observation",
                  "provider-response-failure": "Actor response failure",
                  "provider-attempt-rejection": "Actor attempt rejection"}
        for raw in rows:
            try:
                payload = json.loads(raw).get("payload", {})
            except ValueError:
                continue
            role, digest = payload.get("role"), payload.get("sha256")
            if role not in labels:
                continue
            source = (branch / "artifacts" / "objects" / digest[:2] / digest[2:]
                      if isinstance(digest, str) and _SHA256.fullmatch(digest) else None)
            event = self._event(labels[role], "actor", source, stage=payload.get("stage"),
                                turn=payload.get("turn"), reason="retained payload unavailable")
            # The ledger digest is authoritative; a mismatched object remains discoverable but invalid.
            actual_digest = event["sha256"]
            event["sha256"] = digest if isinstance(digest, str) else None
            if actual_digest is not None and event["sha256"] != actual_digest:
                event["available"], event["reason"] = False, "content digest mismatch"
            events.append(event)
        try:
            host_rows = (root / "project-evidence.jsonl").read_text().splitlines()
        except OSError:
            host_rows = []
        for raw in host_rows:
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if record.get("payload", {}).get("project_id") == project_id:
                events.append(self._inline_event("Host verifier evidence", "verification",
                                                 raw + "\n"))
        events.append(self._event("Verifier prompt", "verification", reason=(
            "unavailable: verification is deterministic and host-owned")))
        events.append(self._event("Verifier result", "verification", branch / "execution-evidence.bin"))
        events.append(self._event("Final outcome", "outcome", branch / "outcome.json"))
        for sequence, event in enumerate(events):
            event["id"] = f"event-{sequence}"
            event["sequence"] = sequence
        return events

    def trace_index(self, phase, project_id):
        events = self._trace_sources(phase, project_id)
        if events is None:
            return None
        public = []
        for event in events:
            item = {key: value for key, value in event.items() if key != "inline"}
            if item["available"]:
                base = f"/api/phase/{phase}/project/{project_id}/trace/{item['id']}"
                item.update(payload_url=base, download_url=base + "/download")
            public.append(item)
        with self._db() as db:
            db.executemany("INSERT OR REPLACE INTO trace_events(phase,project_id,event_id,sha256) "
                           "VALUES(?,?,?,?)", [(phase, project_id, row["id"], row.get("sha256"))
                                               for row in public])
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
        digest = event.get("sha256") if event else (retained[0] if retained else None)
        if not digest:
            return None, "missing"
        with self._db() as db:
            cached = db.execute("SELECT body FROM payloads WHERE sha256=?", (digest,)).fetchone()
        if cached is not None:
            return bytes(cached[0]), None
        if event is None:
            return None, "unavailable"
        if not event.get("available"):
            return None, "digest" if event.get("reason") == "content digest mismatch" else "unavailable"
        if "inline" in event:
            body = event["inline"]
        else:
            located = self._project_sources(phase, project_id)
            branch = located[2]
            if event["label"] == "Curriculum decision":
                body = located[1].read_bytes()
            elif event["label"] == "Verifier result":
                body = (branch / "execution-evidence.bin").read_bytes()
            elif event["label"] == "Final outcome":
                body = (branch / "outcome.json").read_bytes()
            else:
                body = (branch / "artifacts" / "objects" / digest[:2] / digest[2:]).read_bytes()
        if hashlib.sha256(body).hexdigest() != digest:
            return None, "digest"
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO payloads(sha256,body) VALUES(?,?)", (digest, body))
        return body, None
