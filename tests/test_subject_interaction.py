import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.ascend import subject_interaction as interaction
from benchmarks.ascend.actor_control import ActorControl
from benchmarks.ascend.subjects import Subject, SubjectScheduler
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from optimizer.jobs import atomic, canonical


def setup(tmp_path):
    root = tmp_path / 'campaign'
    directory = root / 'waves/wave-1/one'
    directory.mkdir(parents=True)
    plan = dict(project_id='one', spec={'project_id': 'one'})
    record = dict(plan=plan, objective='study', wave_index=1,
                  outcome={'host_verified': False}, review={'decision': 'repair'})
    state = dict(projects={'one': record}, waves=['wave-1'], pending=None)
    atomic(root / 'subject-campaign.json', canonical(state))
    artifacts = ContentAddressedArtifacts(root / 'artifacts')
    snapshot = artifacts.put_bytes(b'frozen memory')
    scheduler = SubjectScheduler(root, [Subject('study', 'study')])
    scheduler.start_wave('wave-1', snapshot)
    scheduler.complete_wave('wave-1')
    atomic(directory / 'subject-execution.json', canonical(dict(identity=dict(memory_sha256=snapshot),
                                                              outcome={'old': True})))
    (directory / 'sft-traces').mkdir()
    (directory / 'sft-traces/old.jsonl').write_text('old trace\n')
    ActorControl(directory / 'actor').send('original', 'old advice')
    for name in ('settings.json', 'initial-memory.txt'):
        (tmp_path / name).write_text('{}')
    config = tmp_path / 'dashboard.json'
    row = dict(id='campaign', root=str(root), target='a3', config=str(tmp_path/'settings.json'),
               initial_memory=str(tmp_path/'initial-memory.txt'), max_seconds=20)
    config.write_text(json.dumps(dict(controls={'enabled': True}, subject_campaigns=[row], campaigns=[])))
    return config, root, directory, scheduler, artifacts


def campaign_builder(root, scheduler, artifacts, calls):
    class Campaign:
        def __init__(self):
            self.root, self.scheduler, self.artifacts = root, scheduler, artifacts
            self.state = json.loads((root/'subject-campaign.json').read_text())
        def _save(self): atomic(root/'subject-campaign.json', canonical(self.state))
        def _remaining(self): return 10
        def execute(self, item, snapshot, directory, **options):
            calls.append((item, snapshot, directory))
            assert snapshot == b'frozen memory'
            (directory / 'sft-traces').mkdir(exist_ok=True)
            (directory / 'sft-traces/new.jsonl').write_text('new trace\n')
            atomic(directory/'subject-execution.json', canonical(dict(identity=dict(memory_sha256=artifacts.put_bytes(snapshot)), outcome={'new': True})))
            return SimpleNamespace(as_dict=lambda: dict(host_verified=True, source_sha256='b'*64))
        def _review(self, pid, record):
            record.update(review={'decision': 'accept', 'memory': None}, review_sha256='c'*64)
            self._save()
    return lambda *args: Campaign()


def test_stopped_feedback_runs_once_archives_history_and_preserves_snapshot(tmp_path):
    config, root, directory, scheduler, artifacts = setup(tmp_path)
    launches, calls = [], []
    result = interaction.submit_feedback(config, 'campaign', 'one', 'repair masks', 'user-key', 15,
        launcher=lambda *args: launches.append(args))
    identifier = result['request']['id']
    assert len(launches) == 1
    replay = interaction.submit_feedback(config, 'campaign', 'one', 'repair masks', 'user-key', 15,
        launcher=lambda *args: pytest.fail('duplicate launch'))
    assert replay['request']['id'] == identifier
    builder = campaign_builder(root, scheduler, artifacts, calls)
    done = interaction.run_request(config, 'campaign', identifier, builder=builder)
    assert done['status'] == 'succeeded' and len(calls) == 1
    history = directory / 'manual-history' / identifier
    assert json.loads((history/'project.json').read_text())['outcome'] == {'host_verified': False}
    assert (history/'sft-traces/old.jsonl').read_text() == 'old trace\n'
    assert (directory/'sft-traces/new.jsonl').exists()
    assert json.loads((root/'elapsed.json').read_text())['grants'] == {identifier: 15}
    assert len(ActorControl(directory/'actor').status()['messages']) == 2
    interaction.run_request(config, 'campaign', identifier, builder=builder)
    assert len(calls) == 1
    assert not (root/'manual-pause.json').exists()


def test_live_feedback_bypasses_worker_and_requires_no_new_allocation(tmp_path):
    config, root, directory, _, _ = setup(tmp_path)
    state = json.loads((root/'subject-campaign.json').read_text())
    state['projects']['one'].pop('outcome')
    atomic(root/'subject-campaign.json', canonical(state))
    with ActorControl(directory/'actor').lease():
        result = interaction.submit_feedback(config, 'campaign', 'one', 'redirect now', 'live',
            launcher=lambda *args: pytest.fail('live feedback must not launch'))
    assert result['request'] is None
    assert ActorControl(directory/'actor').status()['messages'][-1]['text'] == 'redirect now'


def test_restart_requires_retained_approval_and_links_fresh_project(tmp_path):
    config, root, directory, scheduler, artifacts = setup(tmp_path)
    proposal = dict(id='a'*32, status='approved', kind='restart', project='one', after=dict(
        item={'project_id': 'two', 'spec': {'project_id': 'two'}}, instruction='reviewed direction'))
    with pytest.raises(ValueError, match='retained'):
        interaction.request_restart(config, 'campaign', proposal, 20, 'restart')
    (root/'curriculum-edits').mkdir()
    atomic(root/'curriculum-edits'/(proposal['id']+'.json'), canonical(proposal))
    request = interaction.request_restart(config, 'campaign', proposal, 20, 'restart', launcher=lambda *args: None)
    assert ActorControl(directory/'actor').status()['stop_requested']
    calls = []
    done = interaction.run_request(config, 'campaign', request['id'],
        builder=campaign_builder(root, scheduler, artifacts, calls))
    assert done['status'] == 'succeeded'
    state = json.loads((root/'subject-campaign.json').read_text())
    assert state['projects']['one']['outcome'] == {'host_verified': False}
    assert state['projects']['two']['restart_of'] == 'one'
    new_directory = root/state['projects']['two']['directory']
    assert ActorControl(new_directory/'actor').status()['messages'][0]['text'] == 'reviewed direction'


def test_detached_launcher_uses_fixed_module_and_server_paths(tmp_path, monkeypatch):
    config, root, *_ = setup(tmp_path)
    calls = []
    monkeypatch.setattr(interaction.subprocess, 'Popen', lambda argv, **kwargs: calls.append((argv, kwargs)))
    result = interaction.submit_feedback(config, 'campaign', 'one', 'continue', 'key', 5)
    argv, options = calls[0]
    assert argv[1:3] == ['-m', 'benchmarks.ascend.subject_interaction']
    assert argv[-1] == result['request']['id']
    assert options['start_new_session'] is True and 'shell' not in options


def test_feedback_http_guards_and_receipts(tmp_path):
    from dashboard.app import create_app
    config, root, *_ = setup(tmp_path)
    app = create_app(config, tmp_path/'cache.db', actor_launcher=lambda *args: None)
    client = app.test_client()
    url = '/api/campaigns/campaign/projects/one/feedback'
    assert client.post(url, json={'text': 'continue'}).status_code == 403
    csrf = client.get('/api/csrf').json['token']
    headers = {'X-CSRF-Token': csrf, 'Idempotency-Key': 'key'}
    assert client.post(url, json={'text': 'continue'}, headers=headers).status_code == 409
    assert client.post(url, json={'text': 'continue', 'additional_seconds': 10}, headers=headers).status_code == 202
    assert client.get(url).json['messages'][-1]['text'] == 'continue'


def test_optimizer_restart_instruction_reaches_production_job_context(tmp_path):
    from test_optimizer_task_execution import executor, candidate
    ActorControl(tmp_path/'task/actor').send('approved', 'Use the reviewed repair direction')
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    execution.prepare()
    execution.attempt('reviewed-restart')
    assert all(context['user_feedback'][0]['content'] == 'Use the reviewed repair direction'
               for _, context in agent.contexts)
    manifest = execution.job_directory/'manifest.json'
    assert manifest.exists()
    assert interaction._inbox(tmp_path/'task', {'plan': {'task': {'kind': 'optimizer'}}}).root == manifest.parent


def test_completed_optimizer_uses_original_wave_snapshot_and_interrupted_request_does_not_replay(tmp_path):
    config, root, directory, scheduler, artifacts = setup(tmp_path)
    (directory/'subject-execution.json').unlink()
    state = json.loads((root/'subject-campaign.json').read_text())
    state['projects']['one']['plan'] = {'project_id': 'one', 'task': {'kind': 'optimizer'}}
    atomic(root/'subject-campaign.json', canonical(state))
    request = interaction.submit_feedback(config, 'campaign', 'one', 'repair', 'optimizer', 10,
        launcher=lambda *args: None)['request']
    calls = []
    done = interaction.run_request(config, 'campaign', request['id'],
        builder=campaign_builder(root, scheduler, artifacts, calls))
    assert done['status'] == 'succeeded' and calls[0][1] == b'frozen memory'
    path = root/'dashboard-requests'/(request['id']+'.json')
    done['status'] = 'running'
    atomic(path, canonical(done))
    blocked = interaction.run_request(config, 'campaign', request['id'],
        builder=lambda *args: pytest.fail('interrupted execution must be observed first'))
    assert blocked['status'] == 'observation-required'
    assert len(calls) == 1
