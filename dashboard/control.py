"""Constrained asynchronous campaign controls for the loopback dashboard."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import uuid


class ControlError(ValueError):
    """A safe request conflict that can be returned to the dashboard user."""


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, UnicodeDecodeError):
        return default


def _safe_error(error):
    message = re.sub(r"\b\S*(?:SECRET|TOKEN|PASSWORD|API_KEY)\S*\b", "[redacted]",
                     str(error), flags=re.IGNORECASE)
    return (message or type(error).__name__)[:500]


def _information(spec):
    return {key: spec.get(key) for key in (
        "capabilities", "shape", "dtype", "failure_mode", "profile_metric",
        "learning_focus", "graph",
    )}


class ControlManager:
    """Validate commands, persist their lifecycle and execute one at a time."""

    ACTIONS = {"start", "sample", "approve", "reject", "run-approved"}

    def __init__(self, config_path, database, runner=subprocess.run):
        self.config_path, self.database = Path(config_path), Path(database)
        self.runner = runner
        config = _read(self.config_path, {}) or {}
        self.settings = config.get("controls", {})
        self.enabled = self.settings.get("enabled") is True
        self.campaigns = {str(row.get("id", f"phase{row.get('phase')}")): row
                          for row in config.get("campaigns", [])}
        self.presets = {str(row.get("id")): row for row in self.settings.get("presets", [])
                        if row.get("id")}
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY, action TEXT NOT NULL, campaign_id TEXT NOT NULL,
                status TEXT NOT NULL, request TEXT NOT NULL, result TEXT, error TEXT,
                created REAL NOT NULL, started REAL, finished REAL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS idempotency (
                key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                operation_id TEXT NOT NULL REFERENCES operations(id))""")
            db.execute("UPDATE operations SET status='failed', error=?, finished=? "
                       "WHERE status IN ('queued','running')",
                       ("operation interrupted by dashboard restart", time.time()))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rsi-dashboard")

    def _db(self):
        return sqlite3.connect(self.database, timeout=10)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _campaign(self, campaign_id):
        try:
            campaign = self.campaigns[campaign_id]
        except KeyError as error:
            raise ControlError("unknown campaign") from error
        if campaign.get("target") != "a3" or campaign.get("phase") not in (1, 2):
            raise ControlError("only A3 Phase 1 and Phase 2 campaigns are allowed")
        return campaign

    def proposal(self, campaign_id):
        campaign = self._campaign(campaign_id)
        record = _read(Path(campaign["control"]) / "curriculum-approval.json")
        if not isinstance(record, dict):
            return None
        proposal = record.get("proposal", {})
        state = _read(Path(campaign["root"]) / "state.json", {}) or {}
        covered = set(state.get("coverage", []))
        prior = {json.dumps(_information(row.get("spec", {})), sort_keys=True)
                 for row in state.get("prior_outcomes", [])}
        projects = []
        for item in proposal.get("wave", {}).get("projects", []):
            spec = item.get("spec", item)
            capabilities = list(spec.get("capabilities", []))
            novel = json.dumps(_information(spec), sort_keys=True) not in prior
            projects.append({"project_id": spec.get("project_id"),
                             "hypothesis": spec.get("hypothesis", ""),
                             "new_coverage": sorted(set(capabilities) - covered),
                             "novelty_summary": ("New project information" if novel
                                                 else "Repeats prior project information"),
                             "learning_focus": list(spec.get("learning_focus", []))})
        return {"campaign_id": campaign_id, "phase": campaign["phase"],
                "proposal_id": record.get("proposal_id"), "status": record.get("status"),
                "selected_project_ids": record.get("selected_project_ids", []),
                "projects": projects}

    def proposals(self):
        return [proposal for campaign_id in self.campaigns
                if (proposal := self.proposal(campaign_id)) is not None]

    def _validate(self, action, payload):
        if not self.enabled:
            raise PermissionError("dashboard controls are disabled")
        if action not in self.ACTIONS:
            raise ControlError("command is not allowlisted")
        identifier = payload.get("preset_id") if action == "start" else payload.get("campaign_id")
        if not isinstance(identifier, str):
            raise ControlError("campaign identifier is required")
        subject = identifier if action in {"start", "sample"} else payload.get("proposal_id")
        if payload.get("confirmation") != f"{action}:{subject}":
            raise ControlError("confirmation does not match the requested action")
        if action == "start":
            try:
                preset = self.presets[identifier]
            except KeyError as error:
                raise ControlError("unknown start preset") from error
            if preset.get("target") != "a3" or preset.get("phase") not in (1, 2):
                raise ControlError("only A3 Phase 1 and Phase 2 presets are allowed")
            return identifier
        campaign = self._campaign(identifier)
        if action == "sample":
            return identifier
        proposal = self.proposal(identifier)
        if proposal is None or proposal["proposal_id"] != payload.get("proposal_id"):
            raise ControlError("proposal does not match the retained campaign proposal")
        expected_status = "proposed" if action in {"approve", "reject"} else "approved"
        if proposal["status"] != expected_status:
            raise ControlError(f"proposal must be {expected_status}")
        retained = _read(Path(campaign["control"]) / "curriculum-approval.json", {})
        state = _read(Path(campaign["root"]) / "state.json", {}) or {}
        if retained.get("proposal", {}).get("base_wave_index") != state.get("wave_index"):
            raise ControlError("proposal is stale relative to campaign state")
        if action == "approve":
            selected = payload.get("project_ids")
            if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
                raise ControlError("project selection must be non-empty and unique")
            available = {row["project_id"] for row in proposal["projects"]}
            if any(not isinstance(item, str) for item in selected) or not set(selected) <= available:
                raise ControlError("selection contains a project outside the proposal")
            if campaign["phase"] == 2 and len(selected) != 1:
                raise ControlError("Phase 2 requires exactly one selected project")
        return identifier

    def _argv(self, action, payload):
        runner = self.settings.get("runner")
        if not isinstance(runner, str) or not runner:
            raise ControlError("control runner is not configured")
        if action == "start":
            preset = self.presets[payload["preset_id"]]
            argv = [sys.executable, runner, "start", "--control", str(preset["control"]),
                    "--root", str(preset["root"]), "--target", "a3", "--phase",
                    str(preset["phase"]), "--objective", str(preset["objective"]),
                    "--initial-memory", str(preset["initial_memory"]), "--config",
                    str(preset["config"]), "--project-target", str(preset["project_target"]),
                    "--approval-mode"]
            if preset["phase"] == 2:
                argv += ["--phase1-root", str(preset["phase1_root"]),
                         "--review", str(preset["review"])]
            return argv
        campaign = self.campaigns[payload["campaign_id"]]
        argv = [sys.executable, runner, action, "--control", str(campaign["control"])]
        if action in {"approve", "reject", "run-approved"}:
            argv += ["--proposal-id", payload["proposal_id"]]
        if action == "approve":
            for project_id in payload["project_ids"]:
                argv += ["--project-id", project_id]
        return argv

    def enqueue(self, action, payload, key):
        if not isinstance(key, str) or not key.strip() or len(key) > 200:
            raise ControlError("Idempotency-Key is required")
        fingerprint = hashlib.sha256(json.dumps(
            {"action": action, "payload": payload}, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        # A completed mutation may have changed retained state. Honor an exact
        # replay before validating that now-changed state, while still rejecting
        # reuse of the key for a different request.
        with self._db() as db:
            old = db.execute("SELECT fingerprint, operation_id FROM idempotency WHERE key=?",
                             (key,)).fetchone()
        if old:
            if old[0] != fingerprint:
                raise ControlError("idempotency key was already used for a different request")
            return old[1]
        campaign_id = self._validate(action, payload)
        operation_id = uuid.uuid4().hex
        now = time.time()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT fingerprint, operation_id FROM idempotency WHERE key=?",
                             (key,)).fetchone()
            if old:
                if old[0] != fingerprint:
                    raise ControlError("idempotency key was already used for a different request")
                return old[1]
            db.execute("INSERT INTO operations(id,action,campaign_id,status,request,created) "
                       "VALUES(?,?,?,?,?,?)", (operation_id, action, campaign_id, "queued",
                                               json.dumps(payload, sort_keys=True), now))
            db.execute("INSERT INTO idempotency(key,fingerprint,operation_id) VALUES(?,?,?)",
                       (key, fingerprint, operation_id))
        self.executor.submit(self._execute, operation_id, action, payload)
        return operation_id

    def _execute(self, operation_id, action, payload):
        with self._db() as db:
            db.execute("UPDATE operations SET status='running', started=? WHERE id=?",
                       (time.time(), operation_id))
        try:
            completed = self.runner(self._argv(action, payload), capture_output=True, text=True,
                                    check=False, shell=False)
            if completed.returncode:
                raise RuntimeError((completed.stderr or completed.stdout or
                                    f"command exited {completed.returncode}").strip())
            try:
                result = json.loads(completed.stdout)
            except (TypeError, ValueError):
                result = {"detail": "command completed"}
            with self._db() as db:
                db.execute("UPDATE operations SET status='succeeded', result=?, finished=? WHERE id=?",
                           (json.dumps(result), time.time(), operation_id))
        except Exception as error:  # background boundary: retain safe failure for the UI
            with self._db() as db:
                db.execute("UPDATE operations SET status='failed', error=?, finished=? WHERE id=?",
                           (_safe_error(error), time.time(), operation_id))

    def operation(self, operation_id):
        with self._db() as db:
            row = db.execute("SELECT id,action,campaign_id,status,result,error,created,started,finished "
                             "FROM operations WHERE id=?", (operation_id,)).fetchone()
        if row is None:
            return None
        keys = ("id", "action", "campaign_id", "status", "result", "error",
                "created", "started", "finished")
        value = dict(zip(keys, row))
        value["result"] = json.loads(value["result"]) if value["result"] else None
        return value

    def recent(self, limit=20):
        with self._db() as db:
            ids = [row[0] for row in db.execute(
                "SELECT id FROM operations ORDER BY created DESC LIMIT ?", (limit,))]
        return [self.operation(identifier) for identifier in ids]

    def _insert_for_test(self, action, campaign_id, status):
        identifier = uuid.uuid4().hex
        with self._db() as db:
            db.execute("INSERT INTO operations(id,action,campaign_id,status,request,created) "
                       "VALUES(?,?,?,?,?,?)", (identifier, action, campaign_id, status, "{}", time.time()))
        return identifier
