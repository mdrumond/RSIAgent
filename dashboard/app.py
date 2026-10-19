"""Flask application factory for cached read-only campaign views."""

from __future__ import annotations

import time

from flask import Flask, abort, jsonify, render_template
from markupsafe import Markup
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import CppLexer, DiffLexer

from .store import DashboardStore


def create_app(config_path, cache_path, *, refresh_seconds=30):
    app = Flask(__name__)
    store = DashboardStore(config_path, cache_path)
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
        return render_template("index.html", view=view())

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
        return render_template("project.html", campaign=campaign, project=project,
                               source=code(project["source"], CppLexer()),
                               transitions=transitions)

    @app.get("/memory")
    def memory_page():
        return render_template("memory.html", view=view())

    @app.get("/api/status")
    def api_status():
        return jsonify(view())

    return app
