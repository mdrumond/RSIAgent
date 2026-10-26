"""Immediate actor inbox submission and retained continuation status."""
import secrets

from flask import abort, jsonify, request

from benchmarks.ascend import subject_interaction as interaction


def register_actor_controls(app, config_path, csrf_token, launcher=None):
    app.extensions['dashboard_actor_controls'] = interaction

    @app.context_processor
    def actor_context():
        return dict(actor_controls_enabled=app.extensions['dashboard_controls'].enabled)

    def guarded():
        if not secrets.compare_digest(request.headers.get('X-CSRF-Token', ''), csrf_token):
            abort(403)

    @app.post('/api/campaigns/<campaign_id>/projects/<project_id>/feedback')
    def subject_actor_feedback(campaign_id, project_id):
        guarded()
        payload = request.get_json(silent=True) or {}
        try:
            result = interaction.submit_feedback(config_path, campaign_id, project_id,
                payload.get('text'), request.headers.get('Idempotency-Key'),
                payload.get('additional_seconds'), launcher=launcher)
            return jsonify(result), 202
        except PermissionError:
            abort(403)
        except (KeyError, StopIteration, FileNotFoundError):
            abort(404)
        except (ValueError, OSError) as error:
            return jsonify(error=str(error)), 409

    @app.get('/api/campaigns/<campaign_id>/projects/<project_id>/feedback')
    def subject_actor_receipts(campaign_id, project_id):
        try:
            _, root = interaction.binding(config_path, campaign_id)
            _, record, directory = interaction._project(root, project_id)
            state = interaction._inbox(directory, record).status()
            return jsonify(messages=state['messages'], stop_requested=state['stop_requested'])
        except PermissionError:
            abort(403)
        except (KeyError, StopIteration, FileNotFoundError):
            abort(404)
        except (ValueError, OSError) as error:
            return jsonify(error=str(error)), 409

    @app.get('/api/campaigns/<campaign_id>/manual-requests/<identifier>')
    def subject_actor_request(campaign_id, identifier):
        try:
            result = interaction.request_status(config_path, campaign_id, identifier)
            return jsonify(result) if result else abort(404)
        except PermissionError:
            abort(403)
        except (KeyError, StopIteration, FileNotFoundError):
            abort(404)
        except ValueError as error:
            return jsonify(error=str(error)), 409
