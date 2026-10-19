"""Flask application factory for cached read-only campaign views."""

from __future__ import annotations

import hashlib
import secrets
import time
from io import BytesIO

from flask import Flask, abort, jsonify, render_template, request, send_file
from markupsafe import Markup
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import CppLexer, DiffLexer

from .store import DashboardStore
from .control import ControlError, ControlManager


def create_app(config_path, cache_path, *, refresh_seconds=30, command_runner=None,
               optimizer_launcher=None, dashboard_store=None):
    app = Flask(__name__)
    store = dashboard_store if dashboard_store is not None else DashboardStore(config_path, cache_path)
    controls = ControlManager(config_path, cache_path, **(
        {"runner": command_runner} if command_runner is not None else {}))
    app.extensions["dashboard_controls"] = controls
    csrf_token = secrets.token_urlsafe(32)
    from .optimizer import register_optimizer
    register_optimizer(app, config_path, csrf_token, optimizer_launcher)
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
        transitions = [{**row, "html": code(row["diff"], DiffLexer())}
                       for row in project["transitions"]]
        trace = store.trace_index(number, project_id)
        return render_template("project.html", campaign=campaign, project=project,
                               source=code(project["source"], CppLexer()),
                               transitions=transitions, trace=trace)

    @app.get("/api/phase/<int:number>/project/<project_id>/trace")
    def api_trace(number, project_id):
        trace = store.trace_index(number, project_id)
        return jsonify(trace) if trace is not None else abort(404)

    @app.get("/api/phase/<int:number>/project/<project_id>/performance")
    def api_performance(number, project_id):
        campaign = phase(number)
        project = next((row for row in campaign["projects"] if row["id"] == project_id),
                       None) if campaign else None
        return jsonify(store.performance(project)) if project is not None else abort(404)

    @app.get("/api/phase/<int:number>/project/<project_id>/trace/<event_id>")
    def api_trace_payload(number, project_id, event_id):
        body, error = store.trace_payload(number, project_id, event_id)
        if error == "digest":
            abort(409)
        if body is None:
            abort(404)
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            abort(415)
        return jsonify({"text": text, "size": len(body),
                        "sha256": hashlib.sha256(body).hexdigest()})

    @app.get("/api/phase/<int:number>/project/<project_id>/trace/<event_id>/download")
    def api_trace_download(number, project_id, event_id):
        body, error = store.trace_payload(number, project_id, event_id)
        if error == "digest":
            abort(409)
        if body is None:
            abort(404)
        return send_file(BytesIO(body), mimetype="application/octet-stream", as_attachment=True,
                         download_name=f"{event_id}.bin")

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
        if not controls.enabled:
            abort(403)
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
