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


def test_approved_queue_blocks_advance_and_completion_and_is_visible_to_policy(tmp_path):
    from benchmarks.ascend.subject_memory import digest
    scheduler = SubjectScheduler(tmp_path, [Subject('current', 'Current', 1)])
    scheduler.start_wave('first', 'a' * 64)
    scheduler.complete_wave('first')
    queue = [dict(subject_id='current', item=item('requested'))]
    scheduler.edit('projects', queue, expected=digest(scheduler.state()), proposal_id='edit')
    assert scheduler.context()['queued_projects'] == [dict(subject_id='current', project_id='requested')]
    decision = dict(action='advance', reason='Minimum reached', evidence=['first'])
    with pytest.raises(ValueError, match='queued projects'):
        scheduler.decide(decision)
    with pytest.raises(ValueError):
        scheduler.decide(dict(decision, action='complete'))
    assert not scheduler.is_complete()
    assert scheduler.context()['current_subject']['subject_id'] == 'current'


@pytest.mark.parametrize('when', ['last-wave', 'earlier-subject', 'completed'])
def test_campaign_services_approved_work_at_boundary_and_preserves_frontier(tmp_path, when):
    from benchmarks.ascend.subject_campaign import SubjectCampaign
    from benchmarks.ascend.subject_execution import SubjectOutcome
    subjects = [Subject('current', 'Current', 1)]
    if when == 'earlier-subject':
        subjects.append(Subject('later', 'Later', 1))
    executed, before = [], {}
    def approve():
        before.update(campaign.scheduler.state())
        service = SubjectEdits(tmp_path, campaign.scheduler, lambda *a, **k: json.dumps([
            dict(subject_id='current', item=item('requested'))]))
        service.approve(service.preview('projects', 'One more current-subject project', remaining_seconds=10)['id'])
    def execute(plan, snapshot, directory, **options):
        executed.append(plan['project_id'])
        if ((when == 'last-wave' and plan['project_id'] == 'auto-current')
                or (when == 'earlier-subject' and plan['project_id'] == 'auto-later')):
            approve()  # A running wave remains frozen until the next boundary.
            assert campaign.scheduler.context()['current_subject']['subject_id'] == (
                'later' if when == 'earlier-subject' else 'current')
        source = campaign.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True, source, {'source': source}, {}, campaign.artifacts.root)
    class NoMemory:
        last_trace = []
        def review(self, request, **options):
            return dict(decision='accept', reason='Known result', evidence=[request.source_sha256],
                        repair=None, memory=None)
    def decide(messages, **options):
        context = json.loads(messages[1]['content'])
        return json.dumps(dict(action='complete' if context['current_subject'] is None else 'advance',
                               reason='Minimum complete', evidence=['retained-wave']))
    campaign = SubjectCampaign(tmp_path, subjects, b'initial', execute=execute, verifier=NoMemory(),
        select_wave=lambda context, *a, **k: [item('auto-' + context['current_subject']['subject_id'])],
        complete=decide)
    if when == 'completed':
        assert campaign.run()['terminal'] == 'COMPLETE'
        approve()
    result = campaign.run()
    assert result['terminal'] == 'COMPLETE'
    assert executed == (['auto-current', 'auto-later', 'requested'] if when == 'earlier-subject'
                        else ['auto-current', 'requested'])
    final = campaign.scheduler.state()
    assert final['frontier'] == len(subjects) and final['current'] is None
    assert final['project_queue'] == []
    assert final['subjects'] == before['subjects']
    for wave_id, record in before['waves'].items():
        assert final['waves'][wave_id]['memory_snapshot_sha256'] == record['memory_snapshot_sha256']


@pytest.mark.parametrize('when', ['earlier-subject', 'completed'])
def test_interrupted_queue_claim_resumes_under_its_original_subject(tmp_path, when, monkeypatch):
    from benchmarks.ascend.subject_campaign import SubjectCampaign
    from benchmarks.ascend.subject_execution import SubjectOutcome
    subjects = [Subject('first', 'Original subject', 1)]
    if when == 'earlier-subject':
        subjects.append(Subject('later', 'Later subject', 1))
    executed, reviewed, selected = [], [], []

    def approve():
        rows = [dict(subject_id='first', item=item('requested'))]
        service = SubjectEdits(tmp_path, campaign.scheduler, lambda *a, **k: json.dumps(rows))
        service.approve(service.preview('projects', 'Revisit first', remaining_seconds=10)['id'])

    def execute(plan, snapshot, directory, **options):
        executed.append(plan['project_id'])
        if when == 'earlier-subject' and plan['project_id'] == 'auto-later':
            approve()
        source = campaign.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True, source, {'source': source}, {}, campaign.artifacts.root)

    class NoMemory:
        last_trace = []
        def review(self, request, **options):
            reviewed.append(request.objective)
            return dict(decision='accept', reason='Known result', evidence=[request.source_sha256],
                        repair=None, memory=None)

    def select(context, *args, **options):
        selected.append(context['current_subject']['subject_id'])
        return [item('auto-' + selected[-1])]

    def decide(messages, **options):
        context = json.loads(messages[1]['content'])
        return json.dumps(dict(action='complete' if context['current_subject'] is None else 'advance',
                               reason='Covered', evidence=['retained-wave']))

    def build():
        return SubjectCampaign(tmp_path, subjects, b'initial', execute=execute, verifier=NoMemory(),
                               select_wave=select, complete=decide)

    campaign = build()
    if when == 'completed':
        assert campaign.run()['terminal'] == 'COMPLETE'
        approve()
    claim = campaign.scheduler.claim_projects
    def interrupted(wave_id):
        plans = claim(wave_id)
        if plans:
            raise OSError('interrupted after durable claim, before pending save')
        return plans
    monkeypatch.setattr(campaign.scheduler, 'claim_projects', interrupted)
    with pytest.raises(OSError, match='after durable claim'):
        campaign.run()
    assert campaign.state['pending'] is None
    retained = campaign.scheduler.state()
    assert retained['project_queue'] == []
    assert retained['queue_return'] == (1 if when == 'earlier-subject' else None)
    reservation = next(key for key in retained['claimed_projects'] if key not in retained['waves'])

    # A fresh controller must execute the reservation, rather than finish early
    # or attach it to the later subject after restoring queue_return.
    campaign = build()
    assert campaign.run()['terminal'] == 'COMPLETE'
    assert executed == (['auto-first', 'auto-later', 'requested'] if when == 'earlier-subject'
                        else ['auto-first', 'requested'])
    assert selected == (['first', 'later'] if when == 'earlier-subject' else ['first'])
    final = campaign.scheduler.state()
    assert final['waves'][reservation]['subject_id'] == 'first'
    assert final['waves'][reservation]['status'] == 'completed'
    assert final['frontier'] == len(subjects) and final['current'] is None
    assert 'queue_return' not in final
    assert campaign.state['projects']['requested']['objective'] == 'Original subject'
    assert json.loads(reviewed[-1])['subject'] == 'Original subject'


def test_unstarted_claim_blocks_completion_and_advance(tmp_path):
    from benchmarks.ascend.subject_memory import digest
    scheduler = SubjectScheduler(tmp_path, [Subject('first', 'First', 1)])
    scheduler.start_wave('completed', 'a' * 64)
    scheduler.complete_wave('completed')
    scheduler.decide(dict(action='advance', reason='Covered', evidence=['completed']))
    scheduler.edit('projects', [dict(subject_id='first', item=item('requested'))],
                   expected=digest(scheduler.state()), proposal_id='requested')
    assert scheduler.prioritize_projects()
    scheduler.claim_projects('reserved')
    state = scheduler.state()
    state['current'] = None
    assert not scheduler.is_complete(state)
    with pytest.raises(ValueError, match='reserved projects'):
        scheduler.decide(dict(action='advance', reason='Covered', evidence=['completed']))
    assert scheduler.prioritize_projects()
    assert scheduler.context()['current_subject']['subject_id'] == 'first'


def test_target_preview_rejects_missing_or_corrupt_qualification_before_mutation(tmp_path):
    from copy import deepcopy
    from test_ascend_phase2_targets import qualified, RUNTIME
    from benchmarks.ascend.targets import a3_curriculum_policy
    target, artifacts, backend = qualified(tmp_path / 'qualification')
    good = dict(project_id=target.spec.project_id, spec=target.spec.as_dict(), repairs=[],
                task=dict(kind='target', contract=target.as_dict(), artifact_root=str(artifacts.root)))
    bad = []
    missing = deepcopy(good)
    missing['task'].pop('artifact_root')
    bad.append(missing)
    unavailable = deepcopy(good)
    unavailable['task']['artifact_root'] = str(tmp_path / 'unavailable')
    bad.append(unavailable)
    corrupt = deepcopy(good)
    corrupt['task']['contract']['reference_sha256'] = artifacts.put_json(dict(schema='invalid'))
    bad.append(corrupt)
    scheduler = SubjectScheduler(tmp_path / 'campaign', [Subject('current', 'Current')])
    before = scheduler.state()
    calls, responses = [], iter(bad + [good])
    def complete(messages, **options):
        calls.append(messages.copy())
        return json.dumps([dict(subject_id='current', item=next(responses))])
    service = SubjectEdits(tmp_path / 'campaign', scheduler, complete,
        validate_item=lambda plan: validate_project(plan, runtime=RUNTIME, policy=a3_curriculum_policy()))
    counts = len(backend.compile_requests), len(backend.run_requests)
    proposal = service.preview('projects', 'Use the qualified target', remaining_seconds=10)
    assert len(calls) == 4 and scheduler.state() == before
    assert (len(backend.compile_requests), len(backend.run_requests)) == counts
    service.approve(proposal['id'])
    assert scheduler.state()['project_queue'][0]['item'] == good
    with pytest.raises(ValueError, match='runtime changed'):
        validate_project(good, runtime=dict(RUNTIME, device='other'), policy=a3_curriculum_policy())


def test_production_editor_checks_actual_runtime_and_rechecks_qualification_on_approval(tmp_path):
    from test_ascend_phase2_targets import qualified, RUNTIME
    from test_subject_runtime import bound
    from benchmarks.ascend.subject_runtime import build_subject_campaign
    from benchmarks.ascend.targets import a3_curriculum_policy
    target, artifacts, _ = qualified(tmp_path / 'qualification')
    plan = dict(project_id=target.spec.project_id, spec=target.spec.as_dict(), repairs=[],
                task=dict(kind='target', contract=target.as_dict(), artifact_root=str(artifacts.root)))
    composed = bound(tmp_path)
    composed['actor'].executor.runtime_identity = dict(RUNTIME)
    composed['coordinator'].curriculum.target_policy = a3_curriculum_policy()
    campaign = build_subject_campaign({}, target='a3', root=tmp_path / 'campaign',
        subjects=[Subject('current', 'Current')], initial_memory=b'', compose=lambda *a, **k: composed)
    editor = campaign.curriculum_edits
    editor.complete = lambda *a, **k: json.dumps([dict(subject_id='current', item=plan)])
    proposal = editor.preview('projects', 'Qualified target', remaining_seconds=10)
    composed['actor'].executor.runtime_identity = dict(RUNTIME, device='other')
    with pytest.raises(ValueError, match='runtime changed'):
        editor.approve(proposal['id'])
    assert campaign.scheduler.state().get('project_queue', []) == []
    composed['actor'].executor.runtime_identity = dict(RUNTIME)
    reference = target.reference_sha256
    (artifacts.objects / reference[:2] / reference[2:]).unlink()
    with pytest.raises(ValueError, match='qualification evidence'):
        editor.approve(proposal['id'])
    assert campaign.scheduler.state().get('project_queue', []) == []


def test_pending_project_can_receive_reviewed_restart_without_execution(tmp_path):
    replacement = dict(item=item('fresh'), instruction='Review physical tail handling.')
    scheduler, service, calls, _ = setup(tmp_path, [replacement])
    pending = dict(wave_id='queued-wave', subject=dict(subject_id='current', objective='Queued work'),
                   projects=['queued'], items=[item('queued')])
    path = tmp_path / 'subject-campaign.json'
    path.write_text(json.dumps(dict(projects={}, pending=pending)))
    original = path.read_bytes()
    proposal = service.preview('restart', 'Revise this pending project', project='queued', remaining_seconds=10)
    assert proposal['before']['plan'] == item('queued')
    assert service.approve(proposal['id'])['after'] == replacement
    assert path.read_bytes() == original and not (tmp_path / 'waves').exists()
    assert len(calls) == 1


def test_qualified_target_restart_uses_fresh_scheduling_id_without_requalification(tmp_path):
    from copy import deepcopy
    from test_ascend_phase2_targets import qualified, RUNTIME
    from benchmarks.ascend.targets import a3_curriculum_policy
    target, artifacts, backend = qualified(tmp_path / 'qualification')
    original = dict(project_id='original-attempt', spec=target.spec.as_dict(), repairs=[],
        task=dict(kind='target', contract=target.as_dict(), artifact_root=str(artifacts.root)))
    restarted = dict(deepcopy(original), project_id='fresh-attempt')
    root = tmp_path / 'campaign'
    scheduler = SubjectScheduler(root, [Subject('current', 'Current')])
    path = root / 'subject-campaign.json'
    path.write_text(json.dumps(dict(projects={'original-attempt': dict(plan=original)}, pending=None)))
    original_campaign = path.read_bytes()
    qualification = {p.relative_to(artifacts.root): p.read_bytes() for p in artifacts.root.rglob('*') if p.is_file()}
    calls = len(backend.compile_requests), len(backend.run_requests)
    replacement = dict(item=restarted, instruction='Inspect the retained target contract and improve execution.')
    service = SubjectEdits(root, scheduler, lambda *a, **k: json.dumps(replacement),
        validate_item=lambda plan: validate_project(plan, runtime=RUNTIME, policy=a3_curriculum_policy()))
    preview = service.preview('restart', 'Review this qualified-target prompt',
                              project='original-attempt', remaining_seconds=10)
    approved = service.approve(preview['id'])
    assert approved['after']['item']['project_id'] == 'fresh-attempt'
    assert approved['after']['item']['spec'] == target.spec.as_dict()
    assert approved['after']['item']['task']['contract'] == target.as_dict()
    assert path.read_bytes() == original_campaign
    assert {p.relative_to(artifacts.root): p.read_bytes() for p in artifacts.root.rglob('*') if p.is_file()} == qualification
    assert (len(backend.compile_requests), len(backend.run_requests)) == calls
    target.verify(artifacts, RUNTIME, a3_curriculum_policy())
