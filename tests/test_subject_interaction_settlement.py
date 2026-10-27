import json
import hashlib
from types import SimpleNamespace

import pytest

from benchmarks.ascend import subject_interaction as interaction
from benchmarks.ascend.actor_control import ActorControl
from benchmarks.ascend.subject_campaign import SubjectCampaign
from benchmarks.ascend.subject_execution import SubjectOutcome
from benchmarks.ascend.subjects import Subject, SubjectScheduler
from benchmarks.ascend.subject_memory import digest
from optimizer.jobs import atomic, canonical
from test_subject_interaction import setup


def real_builder(root, scheduler, artifacts, *, decision='accept', reviewed=None, execution=None):
    state = json.loads((root/'subject-campaign.json').read_bytes())
    for row in state['projects'].values():
        if 'outcome' in row:
            row['outcome'].setdefault('artifacts', {})
            row['outcome'].setdefault('observations', {})
    atomic(root/'subject-campaign.json', canonical(state))
    def build(*args):
        state = json.loads((root/'subject-campaign.json').read_bytes())
        state.update(schema='ascend-subject-campaign-v1', needs_decision=True, terminal=None)
        atomic(root/'subject-campaign.json', canonical(state))
        def execute(item, snapshot, directory, **options):
            source = artifacts.put_bytes(item['project_id'].encode())
            return SubjectOutcome(True, source, {'source': source}, {}, artifacts.root)
        def review(request, **options):
            if reviewed is not None:
                reviewed(request)
            return dict(decision=decision, reason='observed result', evidence=['a'*64], memory=None)
        def complete(*args, **kwargs):
            action = 'advance' if scheduler.context()['current_subject'] is not None else 'complete'
            return json.dumps(dict(action=action, reason='settled', evidence=['retained']))
        return SubjectCampaign(root, scheduler.subjects, b'', select_wave=None,
            execute=execution or execute, verifier=SimpleNamespace(review=review, last_trace=[]),
            complete=complete, max_seconds=20)
    return build


def request(config, text='continue', key='manual'):
    return interaction.submit_feedback(config, 'campaign', 'one', text, key, 15,
        launcher=lambda *args: None)['request']


def test_superseded_pending_review_settles_and_campaign_can_complete(tmp_path):
    config, root, _, scheduler, artifacts = setup(tmp_path)
    snapshot = scheduler.state()['waves']['wave-1']['memory_snapshot_sha256']
    for number in (2, 3):
        scheduler.start_wave('wave-'+str(number), snapshot)
        scheduler.complete_wave('wave-'+str(number))
    scheduler.register_manual_review('one', 'study')
    scheduler.decide(dict(action='advance', reason='minimum complete', evidence=['wave-3']))
    assert not scheduler.is_complete()
    job = root/'memory-reviews/old/job.json'
    job.parent.mkdir(parents=True)
    atomic(job, canonical(dict(review_id='one', provenance={'project_id': 'one'},
        status='pending', evaluations=[{'decision': 'inconclusive'}])))
    operation = request(config)
    builder = real_builder(root, scheduler, artifacts)
    assert interaction.run_request(config, 'campaign', operation['id'], builder=builder)['status'] == 'succeeded'
    settled = scheduler.state()['reviews']['one']
    assert settled['status'] == 'rejected' and settled['original_status'] == 'pending'
    assert settled['supersession']['request_id'] == operation['id']
    old_job = json.loads(job.read_bytes())
    assert old_job['status'] == 'superseded' and old_job['evaluations'] == [{'decision': 'inconclusive'}]
    assert scheduler.is_complete()
    # Exercise the ordinary campaign owner after the manual worker has settled.
    campaign = builder()
    assert campaign.run()['terminal'] == 'COMPLETE'


def test_manual_repair_creates_obligation_and_success_without_memory_resolves_it(tmp_path):
    config, root, _, scheduler, artifacts = setup(tmp_path)
    first = request(config)
    assert interaction.run_request(config, 'campaign', first['id'],
        builder=real_builder(root, scheduler, artifacts, decision='repair'))['status'] == 'succeeded'
    obligations = scheduler.state()['repairs']
    obligation = next(iter(obligations))
    assert obligations[obligation]['project_id'] == 'one' and obligations[obligation]['status'] == 'open'
    second = request(config, 'repair the observed problem', 'repair')
    assert interaction.run_request(config, 'campaign', second['id'],
        builder=real_builder(root, scheduler, artifacts))['status'] == 'succeeded'
    repaired = scheduler.state()['repairs'][obligation]
    assert repaired['status'] == 'resolved' and repaired['resolution_evidence']
    assert json.loads((root/'memory/active.json').read_bytes())['active'] == {}


@pytest.mark.parametrize('newer', [False, True])
def test_failed_launch_clears_only_its_pause_marker(tmp_path, newer):
    config, root, *_ = setup(tmp_path)
    def fail(*args):
        if newer:
            atomic(root/'manual-pause.json', canonical({'request': 'newer'}))
        raise OSError('launch failed')
    with pytest.raises(OSError):
        interaction.submit_feedback(config, 'campaign', 'one', 'continue', 'failed', 15, launcher=fail)
    assert (root/'manual-pause.json').exists() is newer
    if newer:
        assert json.loads((root/'manual-pause.json').read_bytes()) == {'request': 'newer'}


def test_earlier_project_keeps_subject_with_an_unrelated_pending_subject(tmp_path):
    config, root, _, scheduler, artifacts = setup(tmp_path)
    # Extend the fixture's unused future subject through its normal retained API.
    state = scheduler.state()
    subjects = [*state['subjects'], Subject('later', 'Later').as_dict()]
    scheduler.edit('subjects', subjects, expected=digest(state), proposal_id='future')
    scheduler = SubjectScheduler(root, [Subject.from_mapping(item) for item in subjects])
    state = json.loads((root/'subject-campaign.json').read_bytes())
    state['pending'] = dict(wave_id='later-wave', subject={'subject_id': 'later', 'objective': 'Later'}, items=[], projects=[])
    atomic(root/'subject-campaign.json', canonical(state))
    operation = request(config)
    assert interaction.run_request(config, 'campaign', operation['id'],
        builder=real_builder(root, scheduler, artifacts))['status'] == 'succeeded'
    assert json.loads((root/'subject-campaign.json').read_bytes())['projects']['one']['subject_id'] == 'study'


def test_stopped_project_queues_even_when_another_actor_is_live(tmp_path):
    config, root, directory, *_ = setup(tmp_path)
    state = json.loads((root/'subject-campaign.json').read_bytes())
    state['projects']['one'].pop('outcome')
    atomic(root/'subject-campaign.json', canonical(state))
    with ActorControl(root/'waves/wave-1/sibling/actor').lease():
        operation = request(config)
    assert operation['status'] == 'queued'
    assert not ActorControl(directory/'actor').live()


@pytest.mark.parametrize('sibling_interrupted', [False, True])
def test_manual_review_waits_for_linked_sibling_execution_and_preserves_obligation(tmp_path, sibling_interrupted):
    config, root, _, scheduler, artifacts = setup(tmp_path)
    scheduler.add_repair('paired', 'study', 'one', 'compare both probes', ['a'*64])
    state = json.loads((root/'subject-campaign.json').read_bytes())
    state['projects']['one']['repairs'] = ['paired']
    state['projects']['one']['review'].update(reason='old failure')
    for number in (2, 3):
        scheduler.start_wave('wave-'+str(number), scheduler.state()['waves']['wave-1']['memory_snapshot_sha256'])
        scheduler.complete_wave('wave-'+str(number))
    state['waves'] = ['wave-1', 'wave-2', 'wave-3']
    snapshot = scheduler.state()['waves']['wave-1']['memory_snapshot_sha256']
    scheduler.start_wave('paired-wave', snapshot)
    state['pending'] = dict(wave_id='paired-wave', subject={'subject_id': 'study', 'objective': 'Study'},
        projects=['one', 'control'], memory_sha256=snapshot, items=[
            {'project_id': 'one', 'repairs': ['paired'], 'spec': {'project_id': 'one'}},
            {'project_id': 'control', 'repairs': ['paired'], 'spec': {'project_id': 'control'}}])
    atomic(root/'subject-campaign.json', canonical(state))
    def verify(request):
        observed = json.loads((root/'subject-campaign.json').read_bytes())
        assert all('outcome' in observed['projects'][pid] for pid in ('one', 'control'))
    operation = request(config)
    def execute(item, snapshot, directory, **options):
        from optimizer.control import ObservationUnavailable
        if item['project_id'] == 'control':
            raise ObservationUnavailable('retained sibling operation requires observation')
        source = artifacts.put_bytes(b'selected source')
        return SubjectOutcome(True, source, {'source': source}, {}, artifacts.root)
    result = interaction.run_request(config, 'campaign', operation['id'],
        builder=real_builder(root, scheduler, artifacts, reviewed=verify,
                             execution=execute if sibling_interrupted else None))
    if sibling_interrupted:
        assert result['status'] == 'observation-required'
        assert scheduler.state()['repairs']['paired']['status'] == 'open'
        return
    assert result['status'] == 'succeeded'
    assert scheduler.state()['repairs']['paired']['status'] == 'resolved'


def test_unfinished_optimizer_attempt_retains_its_identity_in_manual_worker(tmp_path):
    from test_phase2_optimizer_execution import executor, candidate, snapshot
    from benchmarks.ascend.phase2_optimizer_execution import OptimizerTaskExecutor
    from optimizer.control import ObservationUnavailable
    config, root, directory, scheduler, artifacts = setup(tmp_path)
    template, agent, backend = executor(tmp_path/'template', [candidate('good')])
    durable = OptimizerTaskExecutor(directory, template.task, snapshot(), template.config,
                                   agent=agent, backend=backend)
    durable.prepare()
    backend.interrupt = True
    with pytest.raises(ObservationUnavailable):
        durable.attempt('original-unfinished')
    state = json.loads((root/'subject-campaign.json').read_bytes())
    state['projects']['one']['plan'] = dict(project_id='one', task={'kind': 'optimizer', 'task': template.task})
    atomic(root/'subject-campaign.json', canonical(state))
    (directory/'subject-execution.json').unlink()
    def execute(item, snapshot_bytes, workspace, **options):
        attempt = json.loads((workspace/'manual-execution.json').read_bytes())['attempt_id']
        assert attempt == 'original-unfinished'
        result = durable.attempt(attempt)
        assert result['verified']
        source = artifacts.put_bytes(b'verified durable candidate')
        return SubjectOutcome(True, source, {'source': source}, {}, artifacts.root)
    operation = request(config)
    result = interaction.run_request(config, 'campaign', operation['id'],
        builder=real_builder(root, scheduler, artifacts, execution=execute))
    assert result['status'] == 'succeeded'
    assert len(backend.evaluations) == 1
    assert durable.store.get(durable.job_id)['attempts'] == 1


def test_successful_probe_keeps_repair_open_until_linked_control_accepts(tmp_path):
    config, root, _, scheduler, artifacts = setup(tmp_path)
    scheduler.add_repair('paired', 'study', 'one', 'compare both', ['a'*64])
    state = json.loads((root/'subject-campaign.json').read_bytes())
    state['projects']['one']['repairs'] = ['paired']
    state['projects']['one']['review']['reason'] = 'old failure'
    state['projects']['control'] = dict(plan={'project_id': 'control', 'repairs': ['paired']},
        repairs=['paired'], wave_index=1, review={'decision': 'repair', 'reason': 'control fails'},
        review_sha256='b'*64, outcome={'host_verified': False, 'artifacts': {}, 'observations': {}})
    atomic(root/'subject-campaign.json', canonical(state))
    first = request(config)
    assert interaction.run_request(config, 'campaign', first['id'],
        builder=real_builder(root, scheduler, artifacts))['status'] == 'succeeded'
    assert scheduler.state()['repairs']['paired']['status'] == 'open'


@pytest.mark.parametrize('cached_reviews', [False, True])
@pytest.mark.parametrize('paired', [False, True])
def test_reserved_restart_cancels_old_execution_and_normal_resume_preserves_repairs(tmp_path, cached_reviews, paired):
    config, root, _, scheduler, artifacts = setup(tmp_path)
    snapshot = scheduler.state()['waves']['wave-1']['memory_snapshot_sha256']
    scheduler.start_wave('wave-2', snapshot)
    scheduler.complete_wave('wave-2')
    scheduler.start_wave('reserved', snapshot)
    scheduler.add_repair('obligation', 'study', 'failed-origin', 'repair failure', ['a'*64])
    item = {'project_id': 'one', 'repairs': ['obligation'], 'spec': {'project_id': 'one'}}
    state = json.loads((root/'subject-campaign.json').read_bytes())
    state['projects'] = {'failed-origin': dict(objective='original failure', wave_index=1,
        outcome={'host_verified': False, 'artifacts': {}, 'observations': {}},
        review={'decision': 'repair', 'reason': 'old failure'})}
    state['waves'] = ['wave-1', 'wave-2']
    state['pending'] = dict(wave_id='reserved', subject={'subject_id': 'study', 'objective': 'Study'},
        memory_sha256=snapshot, projects=['one'], items=[item])
    if paired:
        state['pending']['projects'].append('control')
        state['pending']['items'].append(dict(project_id='control', repairs=['obligation'],
                                             spec={'project_id': 'control'}))
    if cached_reviews:
        state['pending']['review_ids'] = ['one']
    atomic(root/'subject-campaign.json', canonical(state))
    proposal = dict(id='f'*32, kind='restart', status='approved', project='one', after=dict(
        item={'project_id': 'replacement', 'repairs': ['obligation'], 'spec': {'project_id': 'replacement'}},
        instruction='reviewed repair'))
    (root/'curriculum-edits').mkdir()
    atomic(root/'curriculum-edits'/(proposal['id']+'.json'), canonical(proposal))
    operation = interaction.request_restart(config, 'campaign', proposal, 15, 'reserved-restart', launcher=lambda *args: None)
    executed, reviewed = [], {}
    def execute(plan, frozen, directory, **options):
        executed.append(plan['project_id'])
        assert plan['project_id'] != 'one'
        source = artifacts.put_bytes(plan['project_id'].encode())
        return SubjectOutcome(True, source, {'source': source},
                              {'comparison': plan['project_id']}, artifacts.root)
    def verify(request):
        reviewed[request.project_id] = request
    builder = real_builder(root, scheduler, artifacts, execution=execute, reviewed=verify)
    result = interaction.run_request(config, 'campaign', operation['id'], builder=builder)
    assert result['status'] == 'succeeded'
    state = json.loads((root/'subject-campaign.json').read_bytes())
    assert state['projects']['one']['superseded_by'] == 'replacement'
    assert 'outcome' not in state['projects']['one']
    assert state['pending']['items'][0] == item
    assert state['projects']['replacement']['subject_id'] == 'study'
    assert scheduler.state()['repairs']['obligation']['status'] == 'resolved'
    assert builder().run()['terminal'] == 'COMPLETE'
    assert executed == (['replacement', 'control'] if paired else ['replacement'])
    if paired:
        for project, sibling in [('control', 'replacement'), ('replacement', 'control')]:
            request = reviewed[project]
            assert json.loads(request.objective)['sibling_evidence'] == {
                sibling: {'comparison': sibling}}
            assert request.artifacts[f'sibling:{sibling}:source'] == artifacts.put_bytes(sibling.encode())
        assert all(not role.startswith('sibling:one:')
                   for request in reviewed.values() for role in request.artifacts)
    if cached_reviews:
        superseded = scheduler.state()['reviews']['one']
        assert superseded['status'] == 'rejected'
        assert superseded['supersession']['replacement_project'] == 'replacement'


def test_default_curriculum_status_reports_orphaned_actor_worker(tmp_path, monkeypatch):
    from dashboard.app import create_app
    config, root, *_ = setup(tmp_path)
    identifier = 'd'*32
    (root/'curriculum-edits').mkdir()
    atomic(root/'curriculum-edits'/(identifier+'.json'), canonical(dict(id=identifier, kind='restart', status='approved')))
    request_id = hashlib.sha256(('curriculum-restart:'+identifier).encode()).hexdigest()[:32]
    (root/'dashboard-requests').mkdir()
    path = root/'dashboard-requests'/(request_id+'.json')
    atomic(path, canonical(dict(id=request_id, status='running', project_id='one', additional_seconds=15)))
    monkeypatch.setattr(interaction, '_launch', lambda *args: pytest.fail('status must not relaunch work'))
    app = create_app(config, tmp_path/'cache.db')
    response = app.test_client().get('/api/curriculum/campaign/proposals/'+identifier+'/restart')
    assert response.status_code == 200
    assert response.json['request']['status'] == 'observation-required'
    assert json.loads(path.read_bytes())['status'] == 'observation-required'


def test_new_feedback_key_cannot_bypass_unsettled_request(tmp_path):
    config, root, *_ = setup(tmp_path)
    first = request(config)
    path = root/'dashboard-requests'/(first['id']+'.json')
    first['status'] = 'running'
    atomic(path, canonical(first))
    assert interaction.request_status(config, 'campaign', first['id'])['status'] == 'observation-required'
    with pytest.raises(ValueError, match='settle prior'):
        interaction.submit_feedback(config, 'campaign', 'one', 'new instructions', 'different-key', 30,
            launcher=lambda *args: pytest.fail('unsettled execution cannot launch again'))
    assert len(list((root/'dashboard-requests').glob('*.json'))) == 1
