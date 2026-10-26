import json
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
        return SubjectCampaign(root, scheduler.subjects, b'', select_wave=None,
            execute=execution or execute, verifier=SimpleNamespace(review=review, last_trace=[]),
            complete=lambda *args, **kwargs: json.dumps(dict(action='complete', reason='settled', evidence=['retained'])))
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


def test_manual_review_waits_for_linked_sibling_execution_and_preserves_obligation(tmp_path):
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
    result = interaction.run_request(config, 'campaign', operation['id'],
        builder=real_builder(root, scheduler, artifacts, reviewed=verify))
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
