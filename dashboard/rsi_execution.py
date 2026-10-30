"""Instance execution controls: argv to the CLI, never a second scheduler."""
import ipaddress
import json
import secrets
import subprocess
import tempfile
from pathlib import Path

from flask import abort, jsonify, render_template, request

from .control import ControlError


CAPABILITIES = {'manual_tuning': False, 'skill_content_editing': False}
UNAVAILABLE = 'This instance interface has no registered CLI command for that capability.'


def _local():
    try:
        return ipaddress.ip_address(request.remote_addr).is_loopback
    except (ValueError, TypeError):
        return False


def _text(payload, key, required=True):
    value = payload.get(key, '')
    if not isinstance(value, str) or (required and not value.strip()) or '\x00' in value:
        raise ControlError(f'{key} must be a nonempty text value')
    return value


def _public_job(row):
    """Do not reflect provider config, prompts, exception strings or worker logs."""
    request_data = row.get('request', {})
    return {**{key: row.get(key) for key in ('job_id', 'state')},
            'mode': request_data.get('mode'), 'cycle_id': request_data.get('cycle_id'),
            'action': request_data.get('action'),
            'configuration_revision': row.get('configuration', {}).get('revision'),
            'result_state': (row.get('result') or {}).get('state')}


def _inspection(value):
    # Only an authoritative CLI inspection is displayed. No invented zero costs,
    # wall estimates, category sums, remote timestamp subtraction or CAS reads.
    inspection = value.get('inspection')
    if not isinstance(inspection, dict):
        return {'available': False, 'reason': 'The CLI has not supplied a task time/skill inspection.'}
    return {'available': True, 'tasks': [
        {key: task[key] for key in ('task_id', 'charged_seconds', 'excluded_seconds',
         'actor_union_seconds', 'remaining', 'accounting_state', 'skills_ref', 'candidate_ref')
         if key in task}
        for task in inspection.get('tasks', []) if isinstance(task, dict)]}


def register_rsi_execution(app, controls, csrf_token, cli):
    def validate(payload):
        if not _local():
            raise PermissionError('Instance execution requires loopback.')
        if not isinstance(payload, dict):
            raise ControlError('A structured command is required.')
        action = payload.get('action')
        if action in ('manual-tuning', 'skill-content'):
            raise ControlError(UNAVAILABLE)
        if action == 'phase':
            if payload.get('mode') not in ('phase1', 'phase2', 'inference'):
                raise ControlError('Unknown phase.')
            if payload.get('operation') not in ('start', 'stop', 'resume', 'revisit'):
                raise ControlError('Unknown phase operation.')
            if payload['operation'] != 'start':
                _text(payload, 'job_id')
            if payload['operation'] == 'revisit':
                _text(payload, 'cycle_id')
            if payload['operation'] == 'start':
                _text(payload, 'cycle_id', required=False)
                if payload['mode'] == 'inference':
                    if not isinstance(payload.get('project'), dict) or type(payload.get('recover')) is not bool:
                        raise ControlError('Inference requires a project and an explicit recovery choice.')
        elif action == 'add-repo':
            for key in ('source', 'database', 'embedding_cache', 'collection', 'repository',
                        'negative_records', 'evidence_store'):
                _text(payload, key)
            if payload.get('database_kind') not in ('a3', 'generic'):
                raise ControlError('Choose the existing database schema.')
        elif action == 'export':
            _text(payload, 'task'); _text(payload, 'destination')
        else:
            raise ControlError('Unknown instance action.')
        return 'rsi-instance-execution'

    def execute(payload):
        action = payload['action']
        if action == 'phase':
            operation = payload['operation']
            args = [payload['mode'], operation]
            if operation != 'start':
                args += ['--job', payload['job_id']]
            if operation in ('start', 'revisit') and payload.get('cycle_id'):
                args += ['--cycle', payload['cycle_id']]
            if payload['mode'] == 'inference' and operation == 'start':
                with tempfile.TemporaryDirectory(prefix='rsi-inference-') as directory:
                    project = Path(directory) / 'project.json'
                    project.write_text(json.dumps(payload['project'], allow_nan=False))
                    args += ['--project', str(project)]
                    if not payload['recover']:
                        args += ['--no-recovery']
                    return _public_job(cli(*args))
            return _public_job(cli(*args))
        if action == 'export':
            cli('evidence-export', payload['task'], payload['destination'], timeout=300)
            return {'exported': True, 'destination': payload['destination']}
        if action == 'add-repo':
            args = ['kdb-add-repo', payload['source']]
            for key in ('database', 'database_kind', 'embedding_cache', 'collection',
                        'repository', 'negative_records', 'evidence_store'):
                args += ['--' + key.replace('_', '-'), payload[key]]
            receipt = cli(*args, timeout=1800)
            return {key: receipt[key] for key in ('reference', 'collection', 'revision', 'documents')
                    if key in receipt}
        raise ControlError(UNAVAILABLE)

    controls.handlers['rsi-execute'] = (validate, execute)

    @app.get('/rsi/execution')
    def rsi_execution_page():
        return render_template('rsi_execution.html', csrf_token=csrf_token,
                               enabled=controls.enabled, capabilities=CAPABILITIES)

    @app.get('/api/rsi/jobs')
    def rsi_jobs():
        if not _local():
            abort(403)
        try:
            return jsonify(jobs=[_public_job(row) for row in cli('job', 'list')],
                           capabilities=CAPABILITIES)
        except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError):
            return jsonify(error='Instance job CLI unavailable; check the pinned instance release.'), 409

    @app.get('/api/rsi/jobs/<job_id>')
    def rsi_job(job_id):
        if not _local():
            abort(403)
        try:
            evidence = cli('job', 'evidence', job_id)
            return jsonify(job=_public_job(evidence['job']), inspection=_inspection(evidence),
                           event_count=len(evidence.get('events', [])))
        except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError):
            return jsonify(error='Retained job evidence is unavailable.'), 409

    @app.post('/api/rsi/execute')
    def rsi_execute():
        if not controls.enabled or not _local() or not secrets.compare_digest(
                request.headers.get('X-CSRF-Token', ''), csrf_token):
            abort(403)
        try:
            operation = controls.enqueue('rsi-execute', request.get_json(silent=True),
                                         request.headers.get('Idempotency-Key'))
            return jsonify(operation_id=operation), 202
        except PermissionError:
            abort(403)
        except (ValueError, TypeError, KeyError):
            return jsonify(error='Invalid or unavailable instance command.'), 409
