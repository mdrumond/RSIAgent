"""Instance execution controls: argv to the CLI, never a second scheduler."""
import ipaddress
import json
import secrets
import subprocess
import tempfile
from pathlib import Path

from flask import abort, jsonify, render_template, request

from .control import ControlError
from .rsi_ledger import inspection as _inspection


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
    result = row.get('result')
    result = result if isinstance(result, dict) else {}
    service_status = result.get('status', result.get('state'))
    if not isinstance(service_status, str) or service_status not in {'complete', 'completed', 'wave_limit', 'suspended', 'stopped',
                              'tuned', 'adopted', 'awaiting_tuning', 'blocked', 'budget_exhausted',
                              'failed', 'running', 'launching'}:
        service_status = None
    return {**{key: row.get(key) for key in ('job_id', 'state')},
            'mode': request_data.get('mode'), 'cycle_id': request_data.get('cycle_id'),
            'action': request_data.get('action'),
            'configuration_revision': row.get('configuration', {}).get('revision'),
            'service_status': service_status, 'result_state': service_status}


def register_rsi_execution(app, controls, csrf_token, cli):
    def capabilities():
        try:
            value = cli('capabilities')
            if not isinstance(value, dict):
                return dict(CAPABILITIES)
            return {key: value.get(key) is True for key in CAPABILITIES}
        except (ValueError, TypeError, OSError, subprocess.SubprocessError):
            return dict(CAPABILITIES)

    def validate(payload):
        if not _local():
            raise PermissionError('Instance execution requires loopback.')
        if not isinstance(payload, dict):
            raise ControlError('A structured command is required.')
        action = payload.get('action')
        if action in ('manual-tuning', 'skill-content', 'skill-map'):
            capability = 'manual_tuning' if action == 'manual-tuning' else 'skill_content_editing'
            if not capabilities()[capability]:
                raise ControlError(UNAVAILABLE)
            if action == 'manual-tuning':
                for key in ('project_id', 'guidance'):
                    _text(payload, key)
            else:
                _text(payload, 'revision')
                if action == 'skill-content':
                    _text(payload, 'name'); _text(payload, 'content')
                elif not isinstance(payload.get('step_map'), dict):
                    raise ControlError('A complete workflow mapping is required.')
            return 'rsi-instance-execution'
        if action == 'phase':
            if payload.get('mode') not in ('phase1', 'phase2', 'inference'):
                raise ControlError('Unknown phase.')
            if payload.get('operation') not in ('start', 'stop', 'resume', 'revisit', 'adopt'):
                raise ControlError('Unknown phase operation.')
            if payload['operation'] != 'start':
                _text(payload, 'job_id')
            if payload['operation'] == 'revisit':
                _text(payload, 'cycle_id')
            policy = payload.get('inheritance', '')
            parent = _text(payload, 'from_cycle', required=False)
            overrides = payload.get('input_overrides', [])
            if (policy not in ('', 'auto', 'parent', 'baseline') or not isinstance(overrides, list)
                    or any(value not in ('memory', 'skills', 'kdb') for value in overrides)):
                raise ControlError('Invalid inheritance selection.')
            if payload['operation'] not in ('start', 'revisit') and (policy or parent or overrides):
                raise ControlError('Inheritance inputs apply only to start or revisit.')
            if policy == 'baseline' and parent:
                raise ControlError('Baseline inheritance cannot select a parent cycle.')
            if payload['operation'] == 'start':
                _text(payload, 'cycle_id', required=False)
                if payload['mode'] == 'inference':
                    if not isinstance(payload.get('project'), dict) or type(payload.get('recover')) is not bool:
                        raise ControlError('Inference requires a project and an explicit recovery choice.')
        elif action == 'add-repo':
            for key in ('source', 'collection', 'repository'):
                _text(payload, key)
            for key in ('negative_records', 'revision'):
                _text(payload, key, required=False)
            if any(key in payload for key in ('database', 'database_kind', 'embedding_cache', 'evidence_store')):
                raise ControlError('Instance ingestion uses its configured KDB, embeddings and evidence store.')
        elif action == 'skill-export':
            _text(payload, 'destination')
            if payload.get('selection') == 'cycle':
                _text(payload, 'cycle_id')
            elif payload.get('selection') != 'builtin':
                raise ControlError('Select accepted cycle skills or builtin templates.')
        elif action == 'export':
            _text(payload, 'task'); _text(payload, 'destination')
        else:
            raise ControlError('Unknown instance action.')
        return 'rsi-instance-execution'

    def execute(payload):
        action = payload['action']
        if action in ('manual-tuning', 'skill-content', 'skill-map'):
            with tempfile.TemporaryDirectory(prefix='rsi-skills-') as directory:
                path = Path(directory) / 'input'
                if action == 'manual-tuning':
                    return _public_job(cli('skills', 'tune', '--project', payload['project_id'],
                                           '--guidance', payload['guidance']))
                if action == 'skill-content':
                    path.write_text(payload['content'], encoding='utf-8')
                    args = ['skills', 'set', payload['name']]
                else:
                    path.write_text(json.dumps(payload['step_map'], allow_nan=False), encoding='utf-8')
                    args = ['skills', 'map', 'set']
                result = cli(*args, '--file', str(path), '--expect-bundle', payload['revision'])
                return {key: result[key] for key in ('bundle_ref', 'configuration_revision') if key in result}
        if action == 'phase':
            operation = payload['operation']
            args = [payload['mode'], operation]
            if operation != 'start':
                args += ['--job', payload['job_id']]
            if operation in ('start', 'revisit') and payload.get('cycle_id'):
                args += ['--cycle', payload['cycle_id']]
            if operation in ('start', 'revisit'):
                if payload.get('from_cycle'):
                    args += ['--from-cycle', payload['from_cycle']]
                if payload.get('inheritance'):
                    args += ['--inheritance', payload['inheritance']]
                for value in dict.fromkeys(payload.get('input_overrides', [])):
                    args += ['--input-override', value]
            if payload['mode'] == 'inference' and operation == 'start':
                with tempfile.TemporaryDirectory(prefix='rsi-inference-') as directory:
                    project = Path(directory) / 'project.json'
                    project.write_text(json.dumps(payload['project'], allow_nan=False))
                    args += ['--project', str(project)]
                    if not payload['recover']:
                        args += ['--no-recovery']
                    return _public_job(cli(*args))
            return _public_job(cli(*args))
        if action == 'skill-export':
            selection = ['--cycle', payload['cycle_id']] if payload['selection'] == 'cycle' else ['--builtin']
            result = cli('skills', 'export', *selection, '--destination', payload['destination'])
            return {key: result[key] for key in ('destination', 'baseline_skills', 'skill_map', 'bundle_ref', 'bundle_hash') if key in result}
        if action == 'export':
            cli('evidence-export', payload['task'], payload['destination'], timeout=300)
            return {'exported': True, 'destination': payload['destination']}
        if action == 'add-repo':
            args = ['kdb-add-repo', payload['source']]
            for key in ('collection', 'repository'):
                args += ['--' + key, payload[key]]
            if payload.get('negative_records'):
                args += ['--negative-records', payload['negative_records']]
            if payload.get('revision'):
                args += ['--expect-revision', payload['revision']]
            receipt = cli(*args, timeout=1800)
            return {key: receipt[key] for key in ('reference', 'collection', 'revision', 'baseline_kdb', 'kdb_sha256',
                         'configuration_revision', 'negative_records')
                    if key in receipt}
        raise ControlError(UNAVAILABLE)

    controls.handlers['rsi-execute'] = (validate, execute)

    @app.get('/rsi/execution')
    def rsi_execution_page():
        return render_template('rsi_execution.html', csrf_token=csrf_token,
                               enabled=controls.enabled, capabilities=capabilities())

    @app.get('/api/rsi/capabilities')
    def rsi_capabilities():
        if not _local():
            abort(403)
        return jsonify(capabilities())

    @app.get('/api/rsi/skills')
    def rsi_skills():
        if not _local():
            abort(403)
        try:
            if not capabilities()['skill_content_editing']:
                raise ControlError(UNAVAILABLE)
            value = cli('skills', 'get')
            return jsonify(revision=value['bundle_ref'],
                           skills={row['name']: row['content'] for row in value['documents']},
                           step_map=value['mapping'])
        except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError):
            return jsonify(error='Immutable baseline skill editing is unavailable in this pinned CLI.'), 409

    @app.get('/api/rsi/published-skills')
    def rsi_published_skills():
        if not _local():
            abort(403)
        try:
            cycle = _text({'cycle_id': request.args.get('cycle', '')}, 'cycle_id')
            value = cli('skills', 'get', '--cycle', cycle)
            return jsonify({key: value[key] for key in ('bundle_ref', 'bundle_hash', 'cycle_id',
                            'provenance_ref', 'documents', 'mapping') if key in value})
        except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError):
            return jsonify(error='Accepted published skills are unavailable for this cycle.'), 409

    @app.get('/api/rsi/jobs')
    def rsi_jobs():
        if not _local():
            abort(403)
        try:
            return jsonify(jobs=[_public_job(row) for row in cli('job', 'list')],
                           capabilities=capabilities())
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
