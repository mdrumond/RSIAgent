import json

import pytest

from benchmarks.ascend.subject_editing import SubjectEdits, validate_project
from benchmarks.ascend.subjects import Subject, SubjectScheduler


def setup(tmp_path, responses):
    subjects = [Subject('current', 'Current experiments'), Subject('future', 'Future experiments')]
    scheduler = SubjectScheduler(tmp_path, subjects)
    calls = []
    responses = iter(responses)
    def complete(messages, **options):
        calls.append((messages.copy(), options))
        return json.dumps(next(responses))
    return scheduler, SubjectEdits(tmp_path, scheduler, complete), calls, subjects


def item(identifier):
    return dict(project_id=identifier, repairs=[], spec=dict(
        project_id=identifier, capabilities=['gm-ub-movement'], shape=[16], dtype='float32',
        block_count=1, padding_multiple=1, failure_mode='none', timing_lengths=[],
        profile_metric=None, hypothesis='Question: Can a copy preserve all values?',
        graph=dict(inputs=[dict(name='x', shape=[16])],
                   nodes=[dict(name='y', op='copy', operands=['x'], axis=None)], output='y')))


def test_preview_is_read_only_then_approved_subject_edit_survives_resume(tmp_path):
    rows = [Subject('current', 'Current experiments').as_dict(),
            Subject('new', 'Reviewed replacement objective').as_dict()]
    scheduler, service, calls, original = setup(tmp_path, [rows])
    before = scheduler.state()
    proposal = service.preview('subjects', 'Replace future with new', remaining_seconds=10)
    assert scheduler.state() == before
    assert proposal['before'] == before['subjects'] and proposal['after'] == rows
    receipt = service.approve(proposal['id'])['receipt']
    assert receipt['revision'] == 1
    assert service.approve(proposal['id'])['receipt'] == receipt
    resumed = SubjectScheduler(tmp_path, original)
    assert resumed.context()['subjects'] == rows
    assert len(calls) == 1
    with pytest.raises(ValueError, match='configuration changed'):
        SubjectScheduler(tmp_path, [Subject('changed', 'Unapproved configuration')])


def test_invalid_curriculum_response_gets_feedback_before_preview(tmp_path):
    changed = [Subject('current', 'Silently change active objective').as_dict()]
    good = [Subject('current', 'Current experiments').as_dict(), Subject('new', 'New').as_dict()]
    scheduler, service, calls, _ = setup(tmp_path, [changed, good])
    proposal = service.preview('subjects', 'Improve future', remaining_seconds=10)
    assert proposal['after'] == good and len(calls) == 2
    assert scheduler.context()['subjects'][1]['subject_id'] == 'future'
    assert calls[1][1]['remaining_seconds'] <= calls[0][1]['remaining_seconds']


def test_queue_claim_is_durable_and_not_editable_after_start(tmp_path):
    rows = [dict(subject_id='current', item=item('one')), dict(subject_id='current', item=item('two'))]
    scheduler, service, _, original = setup(tmp_path, [rows, []])
    proposal = service.preview('projects', 'Request two projects', remaining_seconds=10)
    service.approve(proposal['id'])
    pending_edit = service.preview('projects', 'Remove pending work', remaining_seconds=10)
    assert scheduler.claim_projects('wave-one') == [item('one'), item('two')]
    assert SubjectScheduler(tmp_path, original).claim_projects('wave-one') == [item('one'), item('two')]
    with pytest.raises(ValueError, match='stale'):
        service.approve(pending_edit['id'])
    assert scheduler.state()['project_queue'] == []


def test_approval_rejects_campaign_progress_and_started_project_ids(tmp_path):
    scheduler, service, _, _ = setup(tmp_path, [[dict(subject_id='current', item=item('fresh'))]])
    proposal = service.preview('projects', 'New project', remaining_seconds=10)
    path = tmp_path / 'subject-campaign.json'
    path.write_text(json.dumps(dict(projects={'started': dict(plan=item('started'))}, pending=None)))
    with pytest.raises(ValueError, match='stale'):
        service.approve(proposal['id'])
    assert scheduler.state().get('project_queue', []) == []
    with pytest.raises(ValueError, match='already started'):
        service._validate('projects', [dict(subject_id='current', item=item('started'))],
                          scheduler.state(), service._campaign())


def test_restart_is_previewed_and_approval_returns_plan_without_execution(tmp_path):
    restart = dict(item=item('retry'), instruction='Check physical padding and preserve logical values.')
    scheduler, service, _, _ = setup(tmp_path, [restart])
    path = tmp_path / 'subject-campaign.json'
    campaign = dict(projects={'original': dict(plan=item('original'))}, pending=None)
    path.write_text(json.dumps(campaign))
    proposal = service.preview('restart', 'Fix the experiment prompt', project='original', remaining_seconds=10)
    approved = service.approve(proposal['id'])
    assert approved['after'] == restart and approved['project'] == 'original'
    assert json.loads(path.read_bytes()) == campaign
    assert scheduler.state().get('project_queue', []) == []


def test_queue_validation_preserves_contract_and_open_repair_references(tmp_path):
    scheduler, service, _, _ = setup(tmp_path, [])
    plan = item('bad')
    plan['spec']['project_id'] = 'different'
    with pytest.raises(ValueError, match='identifiers differ'):
        validate_project(plan)
    plan = item('good')
    plan['repairs'] = ['missing']
    with pytest.raises(ValueError, match='existing open repairs'):
        service._validate('projects', [dict(subject_id='current', item=plan)], scheduler.state(), service._campaign())


def test_existing_scheduler_instance_refreshes_approved_subject_revision(tmp_path):
    replacement = [Subject('current', 'Current experiments').as_dict(), Subject('new', 'New').as_dict()]
    scheduler, service, _, original = setup(tmp_path, [replacement])
    live_scheduler = SubjectScheduler(tmp_path, original)
    service.approve(service.preview('subjects', 'Replace future', remaining_seconds=10)['id'])
    assert live_scheduler.context()['subjects'] == replacement
    assert live_scheduler.context()['current_subject']['subject_id'] == 'current'


def test_campaign_executes_approved_queue_before_automatic_generation(tmp_path):
    from benchmarks.ascend.subject_campaign import SubjectCampaign
    from benchmarks.ascend.subject_execution import SubjectOutcome
    calls = []
    class Verifier:
        last_trace = []
        def review(self, request, **options):
            return dict(decision='accept', reason='Known result', evidence=[request.source_sha256],
                        repair=None, memory=None)
    def execute(plan, snapshot, directory, **options):
        calls.append(plan)
        source = campaign.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True, source, {'source': source}, {}, campaign.artifacts.root)
    def select(*args, **options):
        raise AssertionError('automatic generation must not replace approved queue')
    campaign = SubjectCampaign(tmp_path, [Subject('current', 'Current', 1)], b'initial',
        select_wave=select, execute=execute, verifier=Verifier(),
        complete=lambda *args, **kwargs: json.dumps(dict(action='advance', reason='Done', evidence=['one'])))
    rows = [dict(subject_id='current', item=item('one'))]
    service = SubjectEdits(tmp_path, campaign.scheduler, lambda *args, **kwargs: json.dumps(rows))
    service.approve(service.preview('projects', 'Copy project', remaining_seconds=10)['id'])
    assert campaign.run()['terminal'] == 'COMPLETE'
    assert calls == [item('one')]
    assert campaign.scheduler.state()['project_queue'] == []
