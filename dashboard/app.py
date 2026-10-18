"""Flask application factory for cached read-only campaign views."""

from __future__ import annotations

import secrets
import time

from flask import Flask, abort, jsonify, render_template, request
from markupsafe import Markup
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import CppLexer, DiffLexer

from .store import DashboardStore
from .control import ControlError, ControlManager


def create_app(config_path, cache_path, *, refresh_seconds=30, command_runner=None):
    app = Flask(__name__)
    store = DashboardStore(config_path, cache_path)
    controls = ControlManager(config_path, cache_path, **(
        {"runner": command_runner} if command_runner is not None else {}))
    app.extensions["dashboard_controls"] = controls
    csrf_token = secrets.token_urlsafe(32)
    last = [0.0]

    def view():
        if time.monotonic() - last[0] >= refresh_seconds:
            current = store.refresh()
            last[0] = time.monotonic()
            return current
        return store.cached()

    def phase(number):
        return next((row for row in view()["campaigns"] if row["phase"] == number), None)

    def code(value, lexer):
        return Markup(highlight(value or "// Source unavailable\n", lexer, HtmlFormatter()))

    @app.get("/")
    def index():
        return render_template("index.html", view=view(), controls=controls,
                               proposals=controls.proposals() if controls.enabled else [],
                               operations=controls.recent(), csrf_token=csrf_token)

    @app.get("/phase/<int:number>")
    def phase_page(number):
        campaign = phase(number)
        return render_template("phase.html", campaign=campaign) if campaign else abort(404)

    @app.get("/phase/<int:number>/project/<project_id>")
    def project_page(number, project_id):
        campaign = phase(number)
        project = next((row for row in campaign["projects"] if row["id"] == project_id), None) if campaign else None
        if project is None:
            abort(404)
        return render_template("project.html", campaign=campaign, project=project,
                               source=code(project["source"], CppLexer()),
                               diff=code(project["diff"] or "No earlier source revision retained.\n", DiffLexer()))

    @app.get("/memory")
    def memory_page():
        return render_template("memory.html", view=view())

    @app.get("/api/status")
    def api_status():
        result = view()
        if controls.enabled:
            result = {**result, "controls": {"proposals": controls.proposals(),
                                              "operations": controls.recent()}}
        return jsonify(result)

    @app.get("/api/csrf")
    def api_csrf():
        return jsonify({"token": csrf_token})

    @app.get("/api/operations/<operation_id>")
    def api_operation(operation_id):
        operation = controls.operation(operation_id)
        return jsonify(operation) if operation else abort(404)

    @app.post("/api/control/<action>")
    def api_control(action):
        if not controls.enabled:
            abort(403)
        if not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), csrf_token):
            abort(403)
        try:
            operation_id = controls.enqueue(action, request.get_json(silent=True) or {},
                                             request.headers.get("Idempotency-Key"))
        except PermissionError:
            abort(403)
        except (ControlError, KeyError, TypeError) as error:
            return jsonify({"error": str(error)}), 409
        return jsonify({"operation_id": operation_id}), 202

    return app
