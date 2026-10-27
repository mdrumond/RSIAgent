"""Curriculum-authored previews and explicit approval through retained operations."""
import json
import hashlib
import math
from pathlib import Path
import re
import time

from flask import abort, jsonify

from .control import ControlError
from .subject_services import bind_subject, configured_subject, subject_editor


class CurriculumControls:
    def __init__(self, config_path, controls, *, binder=bind_subject, restart=None, status=None):
        self.config_path, self.controls = Path(config_path), controls
        self.binder, self.restart, self.status = binder, restart, status
        controls.handlers['curriculum-preview'] = self.validate_preview, self.preview
        controls.handlers['curriculum-approve'] = self.validate_approval, self.approve

    @property
    def enabled(self):
        return self.controls.enabled

    def available(self, campaign_id):
        try:
            entry, _ = configured_subject(self.config_path, campaign_id)
            return self.enabled and bool(entry.get('config') and entry.get('initial_memory'))
        except (ValueError, OSError):
            return False

    @staticmethod
    def seconds(payload, field):
        seconds = payload.get(field)
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ControlError('an explicit positive elapsed allowance is required')
        return seconds

    def entry(self, payload):
        try:
            entry, root = configured_subject(self.config_path, payload.get('campaign_id'))
        except (ValueError, OSError) as error:
            raise ControlError(str(error)) from error
        if not self.available(entry['id']):
            raise ControlError('configure subject runtime config and initial_memory before editing')
        return entry, root

    def proposal(self, root, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r'[0-9a-f]{32}', identifier):
            raise ControlError('invalid curriculum proposal ID')
        try:
            return json.loads((root / 'curriculum-edits' / (identifier + '.json')).read_bytes())
        except (OSError, ValueError) as error:
            raise ControlError('curriculum proposal is unavailable') from error

    def proposals(self, campaign_id):
        if not self.available(campaign_id):
            return []
        _, root = configured_subject(self.config_path, campaign_id)
        return [json.loads(path.read_bytes()) for path in sorted(
            (root / 'curriculum-edits').glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)]

    def restart_request(self, campaign_id, identifier):
        _, root = configured_subject(self.config_path, campaign_id)
        proposal = self.proposal(root, identifier)
        if proposal['kind'] != 'restart':
            raise ControlError('proposal is not a restart')
        request_id = hashlib.sha256(('curriculum-restart:' + identifier).encode()).hexdigest()[:32]
        path = root / 'dashboard-requests' / (request_id + '.json')
        if not path.exists():
            return None
        accessor = self.status
        if accessor is None:
            from benchmarks.ascend.subject_interaction import request_status
            accessor = request_status
        request = accessor(self.config_path, campaign_id, request_id)
        if request is None:
            return None
        return {field: request[field] for field in ('id', 'status', 'error', 'additional_seconds',
                'project_id', 'result') if field in request}

    @staticmethod
    def unmerged(root, project):
        state = json.loads((root / 'subject-campaign.json').read_bytes())
        pending = (state.get('pending') or {}).get('items', [])
        record = state['projects'].get(project)
        if record is None and not any(item['project_id'] == project for item in pending):
            raise ControlError('unknown restart project')
        active = json.loads((root / 'memory/active.json').read_bytes())
        admitted = {project, (record or {}).get('manual_review_id', project)}
        if admitted & active.get('admissions', {}).keys():
            raise ControlError('project has already been merged into memory')
        for version in active.get('active', {}).values():
            memory = json.loads((root / 'memory/versions' / (version + '.json')).read_bytes())
            if memory.get('provenance', {}).get('project_id') == project:
                raise ControlError('project has already been merged into memory')

    def validate_preview(self, payload):
        entry, root = self.entry(payload)
        if payload.get('confirmation') != 'curriculum-preview:' + entry['id']:
            raise ControlError('confirmation does not match the requested preview')
        if payload.get('kind') not in {'subjects', 'projects', 'request-project', 'restart'}:
            raise ControlError('unknown curriculum edit')
        if not isinstance(payload.get('instructions'), str) or not payload['instructions'].strip():
            raise ControlError('describe the requested change')
        self.seconds(payload, 'format_seconds')
        if payload['kind'] == 'restart':
            self.unmerged(root, payload.get('project_id'))
        return 'subject:' + entry['id']

    def validate_approval(self, payload):
        entry, root = self.entry(payload)
        if payload.get('confirmation') != 'curriculum-approve:' + str(payload.get('proposal_id')):
            raise ControlError('confirmation does not match the requested proposal')
        proposal = self.proposal(root, payload.get('proposal_id'))
        if proposal['kind'] == 'restart':
            self.unmerged(root, proposal['project'])
            self.seconds(payload, 'additional_seconds')
        return 'subject:' + entry['id']

    def preview(self, payload):
        started = time.monotonic()
        allowance = self.seconds(payload, 'format_seconds')
        with self.binder(self.config_path, payload['campaign_id'], max_seconds=allowance) as service:
            remaining = allowance - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('curriculum composition exhausted formatting allowance')
            from benchmarks.ascend.subject_editing import validate_project
            from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
            def validate(item):
                if not isinstance(item, dict):
                    raise ValueError('project must be an object')
                actor = service.bound.get('actor')
                runtime = getattr(getattr(actor, 'executor', None), 'runtime_identity', None)
                task = item.get('task') or {}
                if isinstance(task, dict) and task.get('kind') == 'target' and runtime is None:
                    raise ValueError('qualified target requires the observed composed runtime')
                validate_project(item, runtime=runtime, policy=service.policy)
                if task.get('kind') == 'optimizer':
                    if service.entry['target'] == 'a5' or service.settings.get('optimizer_worker_config') is None:
                        raise ValueError('optimizer projects are unavailable for this campaign runtime')
                else:
                    service.policy.admit(ProjectSpec.from_mapping(item['spec']))
                return item
            editor = subject_editor(service.root, lambda messages, **options:
                service.complete('curriculum', messages, **options), validate_item=validate, policy=service.policy)
            instructions = payload['instructions']
            if payload['kind'] == 'request-project':
                instructions = 'Add the requested project to the pending queue; preserve existing entries.\n' + instructions
            return editor.preview('projects' if payload['kind'] == 'request-project' else payload['kind'],
                instructions, remaining_seconds=remaining, project=payload.get('project_id'))

    def approve(self, payload):
        _, root = self.entry(payload)
        proposal = subject_editor(root).approve(payload['proposal_id'])
        if proposal['kind'] == 'restart':
            self.unmerged(root, proposal['project'])
            restart = self.restart
            if restart is None:
                from benchmarks.ascend.subject_interaction import request_restart
                restart = request_restart
            request = restart(self.config_path, payload['campaign_id'], proposal,
                self.seconds(payload, 'additional_seconds'), 'curriculum-restart:' + proposal['id'])
            return dict(proposal=proposal, restart=request)
        return dict(proposal=proposal)


def register_curriculum_controls(app, config_path, controls, csrf_token, *, binder=None, restart=None, status=None):
    options = {} if binder is None else dict(binder=binder)
    service = CurriculumControls(config_path, controls, restart=restart, status=status, **options)
    app.extensions['dashboard_curriculum_controls'] = service

    @app.context_processor
    def curriculum_context():
        return dict(curriculum_controls=service, csrf_token=csrf_token)

    @app.get('/api/curriculum/<campaign_id>/proposals')
    def curriculum_proposals(campaign_id):
        if not service.enabled:
            abort(403)
        try:
            configured_subject(config_path, campaign_id)
            return jsonify(proposals=service.proposals(campaign_id))
        except ValueError:
            abort(404)

    @app.get('/api/curriculum/<campaign_id>/proposals/<identifier>/restart')
    def curriculum_restart_status(campaign_id, identifier):
        if not service.enabled:
            abort(403)
        try:
            return jsonify(request=service.restart_request(campaign_id, identifier))
        except (ControlError, ValueError, OSError):
            abort(404)
