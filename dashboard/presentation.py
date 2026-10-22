"""Additive descriptions for dashboard views; retained evidence stays untouched."""
import json

from flask import request
from benchmarks.ascend.status_descriptions import describe_statuses


def register_presentation(app):
    app.jinja_env.globals['status_details'] = lambda row: describe_statuses(row).get('status_explanations', [])

    @app.after_request
    def describe_response(response):
        # Artifact routes serve evidence. Never decorate their contents or hashes.
        if (response.is_json and '/artifacts/' not in request.path
                and '/failure-artifacts/' not in request.path):
            body = response.get_json()
            if isinstance(body, dict) and isinstance(body.get('text'), str):
                try:
                    body = dict(body, presentation=describe_statuses(json.loads(body['text'])))
                except (ValueError, TypeError):
                    pass
            if isinstance(body, dict) and 'payload' in body:
                body = dict(body, presentation=describe_statuses(body['payload']))
            response.set_data(app.json.dumps(describe_statuses(body)))
        return response
