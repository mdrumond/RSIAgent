"""Optional instance input editors backed exclusively by the RSI CLI."""
import ipaddress
import json
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

from flask import abort, jsonify, render_template, request

from .control import ControlError
from .store import dashboard_configuration


def register_rsi_inputs(app, config_path, csrf_token, runner=None):
    configured = dashboard_configuration(config_path).get('rsi_instance')
    if not configured:
        return
    instance = Path(configured).expanduser()
    if not instance.is_absolute():
        instance = Path(config_path).resolve().parent / instance
    instance = instance.resolve()
    controls = app.extensions['dashboard_controls']
    runner = runner or subprocess.run
    app.config['RSI_INPUTS_ENABLED'] = True

    def cli(*arguments, timeout=30):
        result = runner([sys.executable, '-m', 'rsi', '--instance', str(instance), *arguments],
                        capture_output=True, text=True, check=False, shell=False, timeout=timeout)
        if result.returncode:
            # Only known validation messages are public; never reflect provider/env content.
            if 'configuration revision conflict' in result.stderr:
                raise ControlError('configuration revision conflict; reload before saving')
            raise ControlError('Settings or folder could not be read. Check the references and values.')
        return json.loads(result.stdout)

    app.extensions['rsi_cli'] = cli
    from .rsi_execution import register_rsi_execution
    register_rsi_execution(app, controls, csrf_token, cli)

    def local_request():
        try:
            return ipaddress.ip_address(request.remote_addr).is_loopback
        except (ValueError, TypeError):
            return False

    def validate(payload):
        # Also guard the existing generic /api/control/<action> entry point.
        if not local_request():
            raise PermissionError('Input editing requires a loopback connection.')
        if (not isinstance(payload, dict) or not isinstance(payload.get('payload'), dict)
                or not isinstance(payload.get('revision'), str) or not payload['revision']):
            raise ControlError('Settings and their current revision are required.')
        return 'rsi-inputs:' + str(instance)

    def save(payload):
        with tempfile.TemporaryDirectory(prefix='rsi-inputs-') as directory:
            path = Path(directory) / 'settings.json'
            path.write_text(json.dumps(payload['payload'], allow_nan=False), encoding='utf-8')
            return cli('config', 'set', '--file', str(path), '--expect-revision', payload['revision'])

    controls.handlers['rsi-config-set'] = (validate, save)

    @app.get('/rsi')
    def rsi_inputs_page():
        return render_template('rsi_inputs.html', csrf_token=csrf_token, enabled=controls.enabled)

    @app.get('/api/rsi/config')
    def rsi_inputs_config():
        try:
            return jsonify(cli('config', 'show'))
        except (ValueError, OSError, subprocess.SubprocessError):
            return jsonify(error='Settings are unavailable. Check the instance configuration.'), 400

    @app.get('/api/rsi/folders')
    def rsi_inputs_folders():
        try:
            path = request.args.get('path')
            return jsonify(cli('folders', 'list', *([path] if path is not None else [])))
        except (ValueError, OSError, subprocess.SubprocessError):
            return jsonify(error='Folder is unavailable or outside the allowed roots.'), 400

    @app.post('/api/rsi/config')
    def rsi_inputs_save():
        if (not controls.enabled or not local_request() or not secrets.compare_digest(
                request.headers.get('X-CSRF-Token', ''), csrf_token)):
            abort(403)
        try:
            operation = controls.enqueue('rsi-config-set', request.get_json(silent=True),
                                         request.headers.get('Idempotency-Key'))
            return jsonify(operation_id=operation), 202
        except PermissionError:
            abort(403)
        except (ValueError, TypeError, KeyError) as error:
            return jsonify(error=str(error)), 409
