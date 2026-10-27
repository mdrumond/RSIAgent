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

from .store import _description_fields


class ControlError(ValueError):
    """A safe request conflict that can be returned to the dashboard user."""


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, UnicodeDecodeError):
        return default


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


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

    ACTIONS = {"start", "extend", "sample", "approve", "reject", "run-approved",
               "import-feedback", "investigate", "retry-feedback"}
    EXTENSION_WAIT_SECONDS = 300
    EXTENSION_POLL_SECONDS = .25

    def __init__(self, config_path, database, runner=subprocess.run):
        self.config_path, self.database = Path(config_path), Path(database)
        self.runner = runner
        self.handlers = {}
        config = _read(self.config_path, {}) or {}
        self.settings = config.get("controls", {})
        from .feedback import FeedbackView
        self.feedback = FeedbackView(config.get("feedback_root"))
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
                             "description_fields": _description_fields(spec.get("hypothesis", "")),
                             "new_coverage": sorted(set(capabilities) - covered),
                             "novelty_summary": ("New project information" if novel
                                                 else "Repeats prior project information"),
                             "learning_focus": list(spec.get("learning_focus", []))})
        terminal = proposal.get("terminal")
        safe_terminal = ({"status": terminal.get("status"), "reason": terminal.get("reason")}
                         if isinstance(terminal, dict) else None)
        return {"campaign_id": campaign_id, "phase": campaign["phase"],
                "proposal_id": record.get("proposal_id"), "status": record.get("status"),
                "selected_project_ids": record.get("selected_project_ids", []),
                "projects": projects, "terminal": safe_terminal,
                "controls_available": campaign["phase"] == 1}

    def proposals(self):
        return [proposal for campaign_id in self.campaigns
                if (proposal := self.proposal(campaign_id)) is not None]

    def _validate(self, action, payload):
        if not self.enabled:
            raise PermissionError("dashboard controls are disabled")
        if action in self.handlers:
            return self.handlers[action][0](payload)
        if action not in self.ACTIONS:
            raise ControlError("command is not allowlisted")
        if action in {"import-feedback", "investigate", "retry-feedback"}:
            return self._validate_feedback(action, payload)
        identifier = payload.get("preset_id") if action == "start" else payload.get("campaign_id")
        if not isinstance(identifier, str):
            raise ControlError("campaign identifier is required")
        subject = identifier if action in {"start", "extend", "sample"} else payload.get("proposal_id")
        if payload.get("confirmation") != f"{action}:{subject}":
            raise ControlError("confirmation does not match the requested action")
        if action == "start":
            try:
                preset = self.presets[identifier]
            except KeyError as error:
                raise ControlError("unknown start preset") from error
            if preset.get("target") != "a3" or preset.get("phase") not in (1, 2):
                raise ControlError("only A3 Phase 1 and Phase 2 presets are allowed")
            if preset["phase"] == 2:
                raise ControlError("new Phase 2 launches are retired; use subject campaigns")
            return identifier
        campaign = self._campaign(identifier)
        if campaign["phase"] == 2:
            raise ControlError("Phase 2 execution is retired; historical evidence remains readable")
        if action == "extend":
            step = self.settings.get("extension_step")
            if type(step) is not int or step != 8:
                raise ControlError("campaign extension step must be exactly 8")
            root, control = Path(campaign["root"]), Path(campaign["control"])
            state = _read(root / "state.json", {}) or {}
            job = _read(control / "job.json", {}) or {}
            worker = _read(control / "worker.json", {}) or {}
            approval = _read(control / "curriculum-approval.json")
            target = state.get("project_target")
            adoption = _read(control / "adoption.json", {}) or {}
            intent = _read(control / "extension-intent.json", {}) or {}
            old_target = adoption.get("project_target", job.get("project_target"))
            targets = [value for value in (
                target, job.get("project_target"), job.get("continue_target"),
                intent.get("project_target"))
                if type(value) is int]
            partial_target = max(targets) if targets else None
            partial = (job.get("adopted") is True and type(old_target) is int
                       and partial_target > old_target
                       and len(state.get("prior_outcomes", [])) == old_target
                       and (worker.get("status") in {"failed", "queued", "running"}
                            or intent.get("project_target") == partial_target))
            clean = (type(target) is int and target == job.get("project_target")
                     and len(state.get("prior_outcomes", [])) == target
                     and worker.get("status") == "waiting-approval"
                     and not intent
                     and worker.get("result", {}).get("status") == "TARGET_REACHED")
            if (campaign["phase"] != 1 or job.get("approval_mode") is not True
                    or not (clean or partial)
                    or state.get("terminal") is not None or (root / "pending.json").exists()
                    or isinstance(approval, dict) and approval.get("status") in {
                        "proposed", "approved", "running"}):
                raise ControlError("campaign must be at a clean TARGET_REACHED boundary")
            payload["_validated_project_target"] = partial_target if partial else target + step
            return identifier
        if action == "sample":
            return identifier
        proposal = self.proposal(identifier)
        if proposal is None or proposal["proposal_id"] != payload.get("proposal_id"):
            raise ControlError("proposal does not match the retained campaign proposal")
        allowed_statuses = ({"proposed"} if action in {"approve", "reject"}
                            else {"approved", "running"})
        if proposal["status"] not in allowed_statuses:
            raise ControlError(f"proposal must be {' or '.join(sorted(allowed_statuses))}")
        retained = _read(Path(campaign["control"]) / "curriculum-approval.json", {})
        state = _read(Path(campaign["root"]) / "state.json", {}) or {}
        base_wave = retained.get("proposal", {}).get("base_wave_index")
        current_wave = state.get("wave_index")
        running_wave_retry = (action == "run-approved" and proposal["status"] == "running"
                              and proposal["terminal"] is None
                              and isinstance(base_wave, int)
                              and current_wave in {base_wave, base_wave + 1})
        if base_wave != current_wave and not running_wave_retry:
            raise ControlError("proposal is stale relative to campaign state")
        if action == "approve":
            selected = payload.get("project_ids")
            if not isinstance(selected, list) or len(set(selected)) != len(selected):
                raise ControlError("project selection must be unique")
            if proposal["terminal"] is not None:
                if selected:
                    raise ControlError("terminal approval requires zero selected projects")
                return identifier
            if not selected:
                raise ControlError("wave approval requires at least one selected project")
            available = {row["project_id"] for row in proposal["projects"]}
            if any(not isinstance(item, str) for item in selected) or not set(selected) <= available:
                raise ControlError("selection contains a project outside the proposal")
            if campaign["phase"] == 2 and len(selected) != 1:
                raise ControlError("Phase 2 requires exactly one selected project")
        return identifier

    def _validate_feedback(self, action, payload):
        if action == 'retry-feedback':
            raise ControlError('Phase 2 retries are retired; request a project from the subject curriculum')
        if self.feedback.root is None:
            raise ControlError("configure feedback_root first")
        subject = 'phase2' if action == 'import-feedback' else payload.get('preset_id')
        if payload.get('confirmation') != f'{action}:{subject}':
            raise ControlError("confirmation does not match the requested action")
        if action == 'import-feedback':
            return 'feedback-import'
        preset = self.presets.get(subject, {})
        phase = 1 if action == 'investigate' else 2
        if preset.get('target') != 'a3' or preset.get('phase') != phase:
            raise ControlError(f"select an A3 Phase {phase} destination preset")
        identifiers = payload.get('feedback_ids')
        if (not isinstance(identifiers, list) or not identifiers
                or any(not isinstance(key, str) for key in identifiers)
                or len(set(identifiers)) != len(identifiers)):
            raise ControlError('select unique failure IDs')
        if action == 'retry-feedback' and len(identifiers) != 1:
            raise ControlError('retry requires exactly one original target')
        for identifier in identifiers:
            row = self.feedback.get(identifier)
            if row is None:
                raise ControlError('unknown failure ID')
            outcome, state = row['outcome'], row['investigation']
            if (outcome['target'] != preset['target']
                    or outcome['language'] != preset.get('language', 'ascend-c')):
                raise ControlError('failure target/language differs from destination preset')
            if (state['status'] in {'resolved', 'deferred'}
                    or outcome['category'] in {'verified-success', 'verified-timing-unknown', 'interrupted'}):
                raise ControlError('select an unresolved, non-infrastructure failure')
            if action == 'investigate' and state.get('classification') in {'infrastructure', 'provider'}:
                raise ControlError('infrastructure/provider observations are not kernel investigations')
            if action == 'retry-feedback' and not outcome.get('task'):
                raise ControlError('retry requires the retained original task')
        if action == 'investigate':
            if type(payload.get('project_target')) is not int or payload['project_target'] < 1:
                raise ControlError('an explicit positive project budget is required')
        return subject

    def _argv(self, action, payload):
        runner = self.settings.get("runner")
        if not isinstance(runner, str) or not runner:
            raise ControlError("control runner is not configured")
        if action == 'import-feedback':
            return [sys.executable, runner, 'feedback-phase2', '--feedback-root',
                    str(self.feedback.root), '--dashboard-config', str(self.config_path)]
        if action in {"start", "investigate", "retry-feedback"}:
            preset = self.presets[payload["preset_id"]]
            argv = [sys.executable, runner, "start", "--control", str(preset["control"]),
                    "--root", str(preset["root"]), "--target", "a3", "--phase",
                    str(preset["phase"]), "--objective", str(preset["objective"]),
                    "--initial-memory", str(preset["initial_memory"]), "--config",
                    str(preset["config"]), "--project-target",
                    str(payload['project_target'] if action == 'investigate' else
                        1 if action == 'retry-feedback' else preset["project_target"])]
            if preset['phase'] == 1:
                argv += ['--approval-mode']
            if preset["phase"] == 2:
                argv += ["--phase1-root", str(preset["phase1_root"]),
                         "--review", str(preset["review"])]
            if action in {'investigate', 'retry-feedback'}:
                argv += ['--feedback-root', str(self.feedback.root)]
                flag = '--failure-id' if action == 'investigate' else '--retry-feedback-id'
                for identifier in payload['feedback_ids']:
                    argv += [flag, identifier]
            return argv
        campaign = self.campaigns[payload["campaign_id"]]
        if action == "extend":
            target = payload["_validated_project_target"]
            return [sys.executable, runner, "continue", "--control", str(campaign["control"]),
                    "--project-target", str(target)]
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
            active = db.execute("SELECT 1 FROM operations WHERE campaign_id=? "
                                "AND status IN ('queued','running') LIMIT 1",
                                (campaign_id,)).fetchone()
            if active:
                raise ControlError("another campaign operation is already queued or running")
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
            if action == "extend":
                campaign = self.campaigns[payload["campaign_id"]]
                _write(Path(campaign["control"]) / "extension-intent.json", {
                    "schema": "ascend-dashboard-extension-intent-v1",
                    "project_target": payload["_validated_project_target"],
                    "operation_id": operation_id, "created": time.time()})
            if action in self.handlers:
                result = self.handlers[action][1](payload)
                with self._db() as db:
                    db.execute("UPDATE operations SET status='succeeded', result=?, finished=? WHERE id=?",
                               (json.dumps(result), time.time(), operation_id))
                return
            completed = self.runner(self._argv(action, payload), capture_output=True, text=True,
                                    check=False, shell=False)
            if completed.returncode:
                raise RuntimeError((completed.stderr or completed.stdout or
                                    f"command exited {completed.returncode}").strip())
            if action == "extend":
                self._wait_for_extension(payload)
                campaign = self.campaigns[payload["campaign_id"]]
                (Path(campaign["control"]) / "extension-intent.json").unlink(missing_ok=True)
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

    def _wait_for_extension(self, payload):
        campaign = self.campaigns[payload["campaign_id"]]
        root, control = Path(campaign["root"]), Path(campaign["control"])
        target = payload["_validated_project_target"]
        deadline = time.monotonic() + self.EXTENSION_WAIT_SECONDS
        while time.monotonic() < deadline:
            worker = _read(control / "worker.json", {}) or {}
            if worker.get("status") == "failed":
                raise RuntimeError(worker.get("error") or "campaign extension worker failed")
            state = _read(root / "state.json", {}) or {}
            job = _read(control / "job.json", {}) or {}
            adoption = _read(control / "adoption.json", {}) or {}
            if (worker.get("status") == "waiting-approval"
                    and state.get("project_target") == target
                    and job.get("project_target") == target
                    and (not job.get("adopted") or adoption.get("project_target") == target)):
                return
            time.sleep(self.EXTENSION_POLL_SECONDS)
        raise RuntimeError("campaign extension did not reach waiting-approval before timeout")

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
