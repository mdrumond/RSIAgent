"""Small, read-only views of Phase 2 observations and investigation lineage."""
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from flask import abort, jsonify, render_template, request, send_file

from benchmarks.ascend.phase2_feedback import FeedbackStore


class FeedbackView:
    def __init__(self, root):
        self.root = Path(root) if root else None

    def records(self):
        # Viewing an unconfigured/missing feedback root must not create it.
        return FeedbackStore(self.root).records() if self.root and self.root.is_dir() else []

    def get(self, identifier):
        return next((row for row in self.records() if row['feedback_id'] == identifier), None)

    def projects(self, record, campaigns):
        # The legacy project route selects the first configured campaign per phase.
        campaign = next((row for row in campaigns.values() if row.get('phase') == 1), {})
        rows = []
        for item in record['investigation']['projects']:
            link = None
            if (isinstance(item, dict) and item.get('campaign_root') and campaign.get('root')
                    and Path(item['campaign_root']).resolve() == Path(campaign['root']).resolve()
                    and item.get('project_id')):
                link = '/phase/1/project/' + quote(item['project_id'], safe='')
            rows.append(dict(record=item, url=link))
        return rows

    def evidence(self, identifier, role):
        record = self.get(identifier)
        ref = record['outcome']['evidence'].get(role) if record else None
        if not ref:
            raise KeyError(role)
        return FeedbackStore(self.root).read_evidence(ref['sha256'])


def register_feedback(app, controls, csrf_token):
    @app.get('/feedback')
    def feedback_page():
        return render_template('feedback.html', controls=controls,
                               records=controls.feedback.records(),
                               operations=controls.recent() if controls.enabled else [],
                               csrf_token=csrf_token)

    @app.get('/api/feedback')
    def feedback_records():
        return jsonify(controls.feedback.records())

    @app.get('/api/feedback/<identifier>/evidence/<role>')
    def feedback_evidence(identifier, role):
        try:
            return jsonify(controls.feedback.evidence(identifier, role))
        except (KeyError, FileNotFoundError):
            abort(404)
        except (ValueError, OSError):
            abort(409)

    @app.get('/api/feedback/<identifier>/failure-cases')
    def failure_cases(identifier):
        if controls.feedback.get(identifier) is None:
            abort(404)
        return jsonify(FeedbackStore(controls.feedback.root).failure_cases(identifier))

    @app.get('/api/feedback/<identifier>/failure-artifacts/<digest>')
    @app.get('/api/feedback/<identifier>/failure-artifacts/<digest>/download')
    def failure_artifact(identifier, digest):
        if controls.feedback.get(identifier) is None:
            abort(404)
        store = FeedbackStore(controls.feedback.root)
        try:
            if request.path.endswith('/download'):
                return send_file(BytesIO(store.failure_artifact_bytes(identifier, digest)),
                                 mimetype='application/octet-stream', as_attachment=True,
                                 download_name=f'{digest}.bin')
            offset = int(request.args.get('offset', 0))
            if offset < 0:
                raise ValueError('offset must be nonnegative')
            return jsonify(store.read_failure_artifact(identifier, digest, offset=offset, limit=8192))
        except FileNotFoundError:
            abort(404, description='This failing-example artifact is unavailable or is not linked to this failure.')
        except (ValueError, OSError):
            abort(409, description='The retained artifact could not be verified or the requested offset is invalid.')
