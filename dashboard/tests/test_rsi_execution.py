"""Behavioral argv/queue scenarios; execution requires the main serial token."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from dashboard.app import create_app


@pytest.fixture
def execution(tmp_path):
    calls = []
    retained = {'job_id': 'one', 'state': 'running',
                'request': {'mode': 'phase2', 'cycle_id': 'cycle-one', 'actor_prompt': 'private'},
                'configuration': {'revision': 'rev1', 'payload': {'API_KEY': 'private'}},
                'error': 'credential private'}
    def runner(argv, **options):
        args = argv[5:]
        calls.append((args, options))
        if '--job' in args and args[args.index('--job') + 1] == 'missing':
            return SimpleNamespace(returncode=2, stdout='', stderr='API_KEY=private')
        if args[:2] == ['inference', 'start'] and '--project' in args:
            calls[-1] += (json.loads(Path(args[args.index('--project') + 1]).read_text()),)
        if '--guidance-file' in args:
            calls[-1] += (Path(args[args.index('--guidance-file') + 1]).read_text(),)
        if '--file' in args:
            content = Path(args[args.index('--file') + 1]).read_text()
            calls[-1] += (json.loads(content) if args[:3] == ['skills', 'map', 'set'] else content,)
        if args == ['capabilities']:
            result = app.config.get('TEST_CAPABILITIES', {})
        elif args == ['skills', 'get']:
            result = {'bundle_ref': 'baseline1', 'documents': [{'name': 'test', 'content': 'original content'}],
                      'mapping': {'test': ['test']}}
        elif args[:2] == ['skills', 'set'] or args[:3] == ['skills', 'map', 'set']:
            result = {'bundle_ref': 'baseline2', 'configuration_revision': 'config2'}
        elif args == ['job', 'list']:
            result = [retained]
        elif args[:2] == ['job', 'evidence']:
            result = {'job': retained, 'events': [{'prompt': 'private'}], 'log': 'private'}
        else:
            result = {'job_id': 'new-job', 'state': 'launching'}
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr='')
    config = tmp_path / 'dashboard.json'
    config.write_text(json.dumps({'rsi_instance': str(tmp_path / 'instance ; spaces'),
                                 'controls': {'enabled': True}}))
    app = create_app(config, tmp_path / 'cache.sqlite', rsi_runner=runner)
    client = app.test_client()
    headers = {'X-CSRF-Token': client.get('/api/csrf').json['token']}
    yield app, client, calls, headers
    app.extensions['dashboard_controls'].close()


def settle(app, response):
    assert response.status_code == 202, response.data
    for _ in range(500):
        operation = app.extensions['dashboard_controls'].operation(response.json['operation_id'])
        if operation['status'] in ('succeeded', 'failed'):
            return operation
        time.sleep(.01)
    pytest.fail('CLI queue did not settle')


@pytest.mark.parametrize('mode', ['phase1', 'phase2'])
@pytest.mark.parametrize('operation', ['start', 'stop', 'resume', 'revisit'])
def test_phase_argv_and_idempotency(execution, mode, operation):
    app, client, calls, headers = execution
    payload = dict(action='phase', mode=mode, operation=operation,
                   job_id='retained ; spaces', cycle_id='new cycle')
    headers = {**headers, 'Idempotency-Key': f'{mode}-{operation}'}
    response = client.post('/api/rsi/execute', json=payload, headers=headers)
    assert settle(app, response)['status'] == 'succeeded'
    assert client.post('/api/rsi/execute', json=payload, headers=headers).json == response.json
    args, options = calls[-1]
    assert args[:2] == [mode, operation]
    assert options['shell'] is False
    if operation != 'start':
        assert args[args.index('--job') + 1] == 'retained ; spaces'
    if operation in ('start', 'revisit'):
        assert args[args.index('--cycle') + 1] == 'new cycle'
    assert len(calls) == 1


@pytest.mark.parametrize('recover', [True, False])
def test_inference_project_is_real_temp_file_and_recovery_is_explicit(execution, recover):
    app, client, calls, headers = execution
    project = {'project_id': 'sum', 'goal': 'numeric', 'command': ['python', 'verify.py']}
    response = client.post('/api/rsi/execute', headers=headers,
        json=dict(action='phase', mode='inference', operation='start', project=project,
                  recover=recover))
    assert settle(app, response)['status'] == 'succeeded'
    args, _, captured = calls[-1]
    assert captured == project
    assert ('--no-recovery' in args) is (not recover)
    assert not Path(args[args.index('--project') + 1]).exists()


def test_read_projection_does_not_display_provider_or_logs(execution):
    _, client, _, _ = execution
    jobs = client.get('/api/rsi/jobs')
    assert jobs.json['jobs'][0]['cycle_id'] == 'cycle-one'
    assert 'configuration' not in jobs.json['jobs'][0]
    evidence = client.get('/api/rsi/jobs/one').json
    assert evidence['event_count'] == 1
    assert evidence['inspection']['available'] is False
    assert set(evidence) == {'job', 'event_count', 'inspection'}
    assert client.get('/api/rsi/jobs', environ_base={'REMOTE_ADDR': '192.0.2.1'}).status_code == 403


def test_guards_apply_to_direct_and_generic_queue(execution):
    app, client, calls, headers = execution
    payload = dict(action='phase', mode='phase1', operation='start')
    assert client.post('/api/rsi/execute', json=payload).status_code == 403
    assert client.post('/api/control/rsi-execute', json=payload, headers=headers,
                       environ_base={'REMOTE_ADDR': '192.0.2.1'}).status_code == 403
    assert client.post('/api/rsi/execute', json={'action': 'manual-tuning'}, headers=headers).status_code == 409
    assert client.post('/api/rsi/execute', json=dict(payload, mode='shell'), headers=headers).status_code == 409
    app.extensions['dashboard_controls'].enabled = False
    assert client.post('/api/rsi/execute', json=payload, headers=headers).status_code == 403
    assert client.get('/rsi/execution').status_code == 200
    assert all(call[0] == ['capabilities'] for call in calls)


def test_real_ingestion_and_export_arguments_are_forwarded(execution):
    app, client, calls, headers = execution
    payload = dict(action='add-repo', source='/node/repo with spaces', collection='new-code',
        repository='local-repo', negative_records='/node/negatives.json', revision='config-before-ingestion')
    assert settle(app, client.post('/api/rsi/execute', json=payload, headers=headers))['status'] == 'succeeded'
    args, options = calls[-1]
    assert args[:2] == ['kdb-add-repo', payload['source']]
    assert args[args.index('--negative-records') + 1] == payload['negative_records']
    assert args[args.index('--expect-revision') + 1] == 'config-before-ingestion'
    assert not set(args) & {'--database', '--database-kind', '--embedding-cache', '--evidence-store'}
    assert options['timeout'] == 1800
    assert settle(app, client.post('/api/rsi/execute', headers=headers,
        json=dict(action='export', task='/node/task', destination='/node/export')))['status'] == 'succeeded'
    assert calls[-1][0] == ['evidence-export', '/node/task', '/node/export']


def test_unavailable_cli_fails_queue_without_echoing_stderr(execution):
    app, client, _, headers = execution
    result = settle(app, client.post('/api/rsi/execute', headers=headers,
        json=dict(action='phase', mode='phase2', operation='stop', job_id='missing')))
    assert result['status'] == 'failed'
    assert result['result'] is None
    assert 'private' not in result['error']


def test_inspection_preserves_host_union_and_unknown_budgets():
    from dashboard.rsi_execution import _inspection
    task = {'task_id': 'sum', 'charged_seconds': {'thinking': 9, 'execution': 4},
            'excluded_seconds': {'setup': 2, 'transfer': 3}, 'actor_union_seconds': 10,
            'remaining': {'actor_seconds': None, 'wall_seconds': 12},
            'accounting_state': 'reconciled', 'skills_ref': 'sha256:accepted',
            'candidate_ref': 'sha256:pending', 'environment': {'API_KEY': 'private'}}
    observed = _inspection({'inspection': {'tasks': [task]}})
    assert observed['available'] is True
    assert observed['tasks'][0]['actor_union_seconds'] == 10
    assert observed['tasks'][0]['charged_seconds']['thinking'] == 9
    assert observed['tasks'][0]['excluded_seconds'] == {'setup': 2, 'transfer': 3}
    assert observed['tasks'][0]['remaining']['actor_seconds'] is None
    assert 'environment' not in observed['tasks'][0]



def test_capability_gated_manual_tuning_freezes_user_guidance(execution):
    app, client, calls, headers = execution
    app.config['TEST_CAPABILITIES'] = {'manual_tuning': True}
    response = client.post('/api/rsi/execute', headers=headers,
        json=dict(action='manual-tuning', cycle_id='cycle-one', project_id='sha256:project',
                  guidance='Keep numerical error detection; investigate prior failures.'))
    assert settle(app, response)['status'] == 'succeeded'
    args, _ = calls[-1]
    assert args[:4] == ['skills', 'tune', '--project', 'sha256:project']
    assert args[args.index('--guidance') + 1] == 'Keep numerical error detection; investigate prior failures.'


def test_immutable_skill_editor_revision_and_mapping_boundary(execution):
    app, client, calls, headers = execution
    app.config['TEST_CAPABILITIES'] = {'skill_content_editing': True}
    baseline = client.get('/api/rsi/skills').json
    assert baseline['revision'] == 'baseline1'
    draft = dict(action='skill-content', revision=baseline['revision'],
                 name='test', content='revised complete SKILL.md')
    operation = settle(app, client.post('/api/rsi/execute', json=draft, headers=headers))
    assert operation['status'] == 'succeeded'
    assert operation['result']['bundle_ref'] == 'baseline2'
    args, _, captured = calls[-1]
    assert args[:3] == ['skills', 'set', 'test']
    assert args[args.index('--expect-bundle') + 1] == 'baseline1'
    assert captured == draft['content']
    assert not Path(args[args.index('--file') + 1]).exists()
    mapping = {'test': ['test']}
    operation = settle(app, client.post('/api/rsi/execute', headers=headers,
        json=dict(action='skill-map', revision='baseline2', step_map=mapping)))
    assert operation['status'] == 'succeeded'
    args, _, captured = calls[-1]
    assert args[:3] == ['skills', 'map', 'set']
    assert args[args.index('--expect-bundle') + 1] == 'baseline2'
    assert captured == mapping


def test_ledger_closed_intervals_union_latest_snapshot_and_unresolved():
    from dashboard.rsi_ledger import inspection
    snapshot = dict(type='workflow-accounting', task_id='sum', clock_domain='controller',
        intervals={'a': ['thinking', 10, 20], 'b': ['execution', 15, 23],
                   'c': ['setup', 5, 9], 'd': ['transfer', 6, 8]},
        remaining={'actor_seconds': 7, 'wall_seconds': 60}, skills_ref='sha256:baseline')
    value = inspection({'events': [dict(snapshot, intervals={'a': ['thinking', 10, 15]}), snapshot]})
    task = value['tasks'][0]
    assert task['charged_seconds'] == {'thinking': 10, 'execution': 8}
    assert task['excluded_seconds'] == {'setup': 4, 'transfer': 2}
    assert task['actor_union_seconds'] == 13
    assert len(value['tasks']) == 1
    snapshot['intervals']['e'] = ['profiling', 24, None]
    task = inspection({'events': [snapshot]})['tasks'][0]
    assert task['accounting_state'] == 'unresolved'
    assert task['actor_union_seconds'] is None
    snapshot['clock_domain'] = 'remote-epoch'
    assert inspection({'events': [snapshot]})['tasks'][0]['accounting_state'] == 'invalid-clock-domain'



def test_instance_ingestion_retains_defaults_and_rejects_standalone_overrides(execution):
    app, client, calls, headers = execution
    payload = dict(action='add-repo', source='/node/repo', collection='new-code', repository='local')
    assert settle(app, client.post('/api/rsi/execute', json=payload, headers=headers))['status'] == 'succeeded'
    assert calls[-1][0] == ['kdb-add-repo', '/node/repo', '--collection', 'new-code', '--repository', 'local']
    assert client.post('/api/rsi/execute', headers=headers,
        json=dict(payload, database='/old/baseline.sqlite3')).status_code == 409
    assert len(calls) == 1
