"""Manual usefulness overrides over project-reviewed, verified memory proposals."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import secrets
import time

from flask import abort, jsonify, request
from .control import ControlError

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.project_verifier import ProjectReview, ProjectVerifierAgent
from benchmarks.ascend.subject_memory import digest


def reviewed_record(root, project_id):
    root = Path(root)
    state = json.loads((root / 'subject-campaign.json').read_bytes())
    record = state['projects'].get(project_id)
    if record is None:
        raise ValueError('Project has not completed execution and review yet.')
    outcome, review = record.get('outcome') or {}, record.get('review') or {}
    if outcome.get('host_verified') is not True:
        raise ValueError('Host verification is required before memory publication.')
    if review.get('decision') != 'accept' or review.get('memory') is None:
        raise ValueError('A project verifier must approve a reusable memory proposal first.')
    store = ContentAddressedArtifacts(root / 'artifacts')
    retained = json.loads(store.read_bytes(record['review_sha256']))
    if retained != review:
        raise ValueError('Project review does not match its retained evidence.')
    request = ProjectReview(**record['review_request'])
    if not request.host_verified or request.source_sha256 != outcome['source_sha256']:
        raise ValueError('Project review differs from the host-verified outcome.')
    path = root / 'memory-reviews' / digest(project_id) / 'job.json'
    job = json.loads(path.read_bytes()) if path.exists() else {}
    if job.get('project_repair'):
        raise ValueError('Project repair requires another project review before publication.')
    proposal = review['memory']
    refs = set(request.artifacts.values()) | set(review['evidence']) | set(proposal['evidence'])
    refs.add(request.source_sha256)
    refs.update(proposal[role]['sha256'] for role in ('before', 'after') if proposal[role])
    observed = {sha: [(0, len(store.read_bytes(sha)))] for sha in refs}
    # The immutable stored decision proves the original independent review;
    # repeat its structural and exact-snippet gates before bypassing usefulness.
    validator = ProjectVerifierAgent(None, lambda sha, **kw: store.read_bytes(sha))
    validator._validate_decision(review, request, refs, refs, True, True,
        {(row['memory_id'], row['version']) for row in proposal['base_versions']},
        proposal['citations'], lambda: 60, observed)
    return record, request, job


def force_merge(services, project_id, reason, key):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Explain why this memory should be admitted.')
    if not isinstance(key, str) or not key.strip() or len(key) > 200:
        raise ValueError('Idempotency-Key is required.')
    memory = services.knowledge.memory_getter()
    fingerprint = digest(dict(project_id=project_id, reason=reason))
    with memory.transaction() as state:
        old = state.get('manual_overrides', {}).get(project_id)
        if old is not None and old['key'] == key:
            if old['fingerprint'] != fingerprint:
                raise ValueError('Idempotency key was used for different feedback.')
            return old['admission']
        if project_id in state['admissions']:
            raise ValueError('Project has already been merged into memory.')
    record, reviewed, job = reviewed_record(services.root, project_id)
    manual = dict(reason=reason.strip(), key=key, fingerprint=fingerprint, created=time.time(),
        original_admission_status=job.get('status', 'not-submitted'),
        original_evaluations=job.get('evaluations', []), review_sha256=record['review_sha256'])
    provenance = dict(project_id=project_id,
        subject_id=record.get('subject_id'), wave_index=record.get('wave_index'),
        review_sha256=record['review_sha256'], request=asdict(reviewed), manual_override=manual)
    if provenance['subject_id'] is None:
        subjects = json.loads((services.root / 'subjects.json').read_bytes())
        waves = list(json.loads((services.root / 'subject-campaign.json').read_bytes()).get('waves', []))
        pending = json.loads((services.root / 'subject-campaign.json').read_bytes()).get('pending') or {}
        wave = pending.get('wave_id') if project_id in pending.get('projects', []) else (
            waves[record['wave_index'] - 1] if record.get('wave_index', 0) <= len(waves) else None)
        provenance['subject_id'] = subjects.get('waves', {}).get(wave, {}).get('subject_id')
    memory.index = services.knowledge.index
    return memory.accept(project_id, record['review']['memory'], provenance, manual=manual)


def register_memory_controls(app, config_path, csrf_token):
    views = app.extensions['dashboard_subjects']
    controls = app.extensions['dashboard_controls']
    def eligibility(campaign_id, project_id):
        try:
            config = views.campaigns[campaign_id]
            if not controls.enabled:
                raise ValueError('Dashboard controls are disabled.')
            if not config.get('config') or not config.get('initial_memory'):
                raise ValueError('Configure this campaign runtime before publishing memory.')
            reviewed_record(Path(config['root']), project_id)
            return dict(eligible=True, reason='Override usefulness admission while preserving project verification.')
        except (ValueError, KeyError, OSError, TypeError) as error:
            return dict(eligible=False, reason=str(error))
    @app.context_processor
    def memory_context():
        return dict(memory_eligibility=eligibility, memory_csrf=csrf_token)
    def validate(payload):
        campaign_id, project_id = payload['campaign_id'], payload['project_id']
        if payload.get('confirmation') != 'force-memory:' + project_id:
            raise ControlError('Confirmation does not match this project.')
        if campaign_id not in views.campaigns:
            raise ControlError('Unknown subject campaign.')
        if not isinstance(payload.get('reason'), str) or not payload['reason'].strip():
            raise ControlError('Explain why this memory should be admitted.')
        return 'subject:' + campaign_id
    def execute(payload):
        from .subject_services import bind_subject
        with bind_subject(config_path, payload['campaign_id'], max_seconds=60) as services:
            receipt = force_merge(services, payload['project_id'], payload['reason'], payload['_manual_key'])
        return dict(admission=receipt)
    controls.handlers['force-memory'] = (validate, execute)
    @app.post('/api/campaigns/<campaign_id>/projects/<project_id>/memory/force')
    def force_memory(campaign_id, project_id):
        if not controls.enabled or not secrets.compare_digest(request.headers.get('X-CSRF-Token', ''), csrf_token):
            abort(403)
        payload = dict(request.get_json(silent=True) or {}, campaign_id=campaign_id, project_id=project_id,
                       _manual_key=request.headers.get('Idempotency-Key'))
        try:
            operation = controls.enqueue('force-memory', payload, request.headers.get('Idempotency-Key'))
            return jsonify(operation_id=operation), 202
        except (ValueError, KeyError, OSError, TypeError) as error:
            return jsonify(error=str(error)), 409
