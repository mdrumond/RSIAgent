"""Production timing depends on actor work, never preparation duration."""
import json
from contextvars import copy_context
from threading import Thread

import pytest
from benchmarks.ascend.actor_time import ActorWorkBudget, preparation, live_deadline
from benchmarks.ascend.actor_control import grant_elapsed


def budget(tmp_path, seconds=10):
    now, wall = [0.0], [100.0]
    value = ActorWorkBudget(tmp_path/'elapsed.json', seconds,
        clock=lambda: now[0], wall_clock=lambda: wall[0])
    return value, now, wall


def test_preparation_and_migration_do_not_spend_actor_allocation(tmp_path):
    b, now, _ = budget(tmp_path)
    with b.bind(), b.actor('writer', 10) as actor:
        now[0] += 10000  # Inputs and native probe are prepared before reasoning.
        with actor.scope(work=True):
            now[0] += 3
            deadline = live_deadline(10)
            with preparation('migration'):
                now[0] += 10000
                with preparation('candidate upload'):
                    now[0] += 1000
            assert deadline-now[0] == 7
            now[0] += 7
            assert now[0] >= deadline
    assert b.report()['elapsed_seconds'] == 10
    assert b.report()['preparation_seconds'] == 21000
    assert b.report()['actors']['writer']['work_seconds'] == 10


def test_parallel_setup_cannot_pause_other_work_and_overlap_is_not_summed(tmp_path):
    b, now, _ = budget(tmp_path, 20)
    with b.actor('one', 20) as one:
        with one.scope(work=True):
            now[0] += 2
            with b.actor('two', 20) as two:
                now[0] += 3  # one working while two prepares
                with two.scope(work=True):
                    now[0] += 4  # overlap charged once
                    with one.scope(reason='health'):
                        now[0] += 2
    stats = b.report()
    assert stats['elapsed_seconds'] == 11
    assert stats['actors']['one']['work_seconds'] == 9
    assert stats['actors']['two']['work_seconds'] == 6


def test_crashed_setup_resume_preserves_remaining_and_explicit_grants(tmp_path):
    b, now, wall = budget(tmp_path)
    with b.actor('writer', 10) as actor:
        with actor.scope(work=True): now[0] += 3
        with preparation('native staging'):
            b.report()  # persisted uncharged phase, then emulate process loss
            state = json.loads(b.path.read_bytes())
    b.path.write_text(json.dumps(state))
    wall[0] += 100000
    grant_elapsed(b.path, 'user-retry', 5)
    resumed = ActorWorkBudget(b.path, 10, clock=lambda: now[0], wall_clock=lambda: wall[0])
    assert resumed.remaining() == 12
    assert resumed.grants == {'user-retry': 5}


def test_historical_elapsed_usage_is_not_refunded(tmp_path):
    path = tmp_path/'elapsed.json'
    path.write_text(json.dumps(dict(max_seconds=10, elapsed_seconds=4, active_since=None)))
    b = ActorWorkBudget(path, 10)
    assert b.remaining() == 6


def test_expired_live_deadline_stays_expired_as_real_clock_advances(tmp_path):
    b, now, _ = budget(tmp_path, 1)
    with b.actor('writer', 1) as actor, actor.scope(work=True):
        now[0] += 1
        for _ in range(3):
            assert now[0] >= actor.deadline
            now[0] += 100


def test_initial_controller_preflight_is_visible_and_uncharged(tmp_path):
    b, now, _ = budget(tmp_path)
    with b.bind(), preparation('known-good probe'):
        now[0] += 1000
        report = b.report()
        assert report['elapsed_seconds'] == 0
        assert report['preparation_seconds'] == 1000
        assert next(iter(report['actors'].values()))['reason'] == 'known-good probe'
    assert b.remaining() == 10


def test_same_actor_resume_does_not_reset_local_allocation(tmp_path):
    b, now, _ = budget(tmp_path, 100)
    with b.actor('writer', 10) as actor, actor.scope(work=True): now[0] += 6
    with b.actor('writer', 10) as actor:
        assert actor.remaining() == 4


def test_transient_preflight_retries_but_configuration_mismatch_does_not(tmp_path):
    from benchmarks.ascend.actor_time import prepare_until_ready
    from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
    b, now, _ = budget(tmp_path)
    calls = []
    def check():
        calls.append(1)
        if len(calls) < 4: raise ProjectInfrastructureError('device is not idle')
        return 'ready'
    with b.bind(), preparation('preflight'):
        assert prepare_until_ready(check, sleep=lambda _: now.__setitem__(0, now[0]+100)) == 'ready'
    assert len(calls) == 4 and b.remaining() == 10
    with pytest.raises(ProjectInfrastructureError, match='pinned'):
        prepare_until_ready(lambda: (_ for _ in ()).throw(ProjectInfrastructureError('pinned runtime mismatch')))
    with pytest.raises(TimeoutError, match='operator'):
        prepare_until_ready(check, stopped=lambda: True)


@pytest.mark.parametrize('engine', ['direct', 'pi'])
def test_actor_provider_deadlines_survive_mid_action_staging(tmp_path, monkeypatch, engine):
    from types import SimpleNamespace
    from benchmarks.a5kernels.phase1_provider import DirectActorDriver, DirectCompletionProvider, DirectCompletion, direct_profile
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action
    b, now, _ = budget(tmp_path)
    turns = iter(['compile', 'submit'])
    def transport(**kwargs):
        now[0] += 1
        return DirectCompletion(json.dumps({next(turns): {}}), 1)
    provider = DirectCompletionProvider({'OPENAI_API_KEY':'fake','DEEPSEEK_API_KEY':'fake'}, transport=transport, clock=lambda: now[0])
    if engine == 'pi':
        import coding.pi.actor as module
        monkeypatch.setattr(module, 'runtime_preflight', lambda: None)
        class Session:
            completion_tokens = 0
            deadline = 10
            gateway = SimpleNamespace(deadline=10)
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def next_action(self, *args, **kwargs):
                assert now[0] < self.deadline and now[0] < self.gateway.deadline
                now[0] += 1
                return next(turns), {}
        monkeypatch.setattr(module.PiActorDriver, '_new_session', lambda *args: Session())
        driver = module.PiActorDriver(provider, max_seconds=10, clock=lambda: now[0])
    else:
        driver = DirectActorDriver(provider, max_seconds=10, clock=lambda: now[0])
    seen = []
    def execute(action):
        seen.append(action.kind)
        if action.kind == 'compile':
            with preparation('transfer and health'):
                now[0] += 1000
            now[0] += 2  # actual candidate compile remains charged
        else:
            now[0] += 6
        return ActionOutcome('{}', action.kind == 'submit', 'host-verified')
    with b.bind(), b.actor('writer', 10):
        outcome = driver('task', profile=direct_profile(MODEL_IDENTITIES[0]), workspace=tmp_path/'actor',
            context=None, turn_parser=parse_trial_action, action_executor=execute)
    assert outcome.status == 'host-verified' and seen == ['compile', 'submit']
    assert b.report()['elapsed_seconds'] == 10


def test_phase1_memory_writer_has_own_allowance_after_actor_exhaustion(tmp_path):
    from benchmarks.a5kernels.phase1_provider import DirectActorDriver, DirectCompletionProvider, DirectCompletion, direct_profile
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action
    b, now, _ = budget(tmp_path)
    responses = iter(['{"submit":{}}', '{"distillation":{"no_new_knowledge":true,"lessons":[]}}'])
    provider = DirectCompletionProvider({'OPENAI_API_KEY':'fake','DEEPSEEK_API_KEY':'fake'},
        transport=lambda **kw: DirectCompletion(next(responses), 1), clock=lambda: now[0])
    driver = DirectActorDriver(provider, max_seconds=10, clock=lambda: now[0])
    def execute(action):
        now[0] += 10
        return ActionOutcome('{}', True, 'host-verified')
    with b.bind(), b.actor('writer', 10):
        outcome = driver('task', profile=direct_profile(MODEL_IDENTITIES[0]), workspace=tmp_path/'actor',
            context=None, turn_parser=parse_trial_action, action_executor=execute)
        assert b.remaining() == 0
        learned = driver.distill(outcome)
    assert learned.distillation is not None
    assert b.report()['elapsed_seconds'] == 10


def test_actor_adapter_resume_keeps_original_local_limit(tmp_path):
    from types import SimpleNamespace
    from benchmarks.ascend.actor_time import actor_attempt, actor_work
    b, now, _ = budget(tmp_path)
    class Executor:
        max_seconds = 10
        @actor_attempt
        def __call__(self, request):
            return self.run()
        @actor_work
        def run(self):
            remaining = live_deadline(0)-now[0]
            now[0] += 6 if now[0] == 0 else 0
            return remaining
    with b.bind():
        assert Executor()(SimpleNamespace(branch_root=tmp_path/'actor',remaining_seconds=10)) == 10
        assert Executor()(SimpleNamespace(branch_root=tmp_path/'actor',remaining_seconds=4)) == 4


def test_cpl_upload_and_remote_staging_are_free_but_compile_wait_is_work(tmp_path):
    from tests.test_cpl_native import backend, Remote
    from benchmarks.a3kernels.native_smoke import smoke_plan
    from benchmarks.ascend.actor_time import preparation_active
    b, now, _ = budget(tmp_path, 30)
    remote = Remote()
    def runner(argv, **kwargs):
        now[0] += 20 if preparation_active() else 1
        return remote(argv, **kwargs)
    candidate = backend(tmp_path, runner, clock=lambda: now[0], wall_clock=lambda: now[0],
        remaining_seconds=lambda: float('inf') if preparation_active() else b.remaining())
    with b.bind(), b.actor('writer', 30) as actor, actor.scope(work=True):
        compilation = candidate.compile(smoke_plan(target='discovered-a3',arch='discovered-arch'), tmp_path/'bundle')
    assert compilation.library_sha256 == 'b'*64
    assert 0 < b.report()['elapsed_seconds'] < 30
    assert b.report()['preparation_seconds'] > 30
    assert len([call for call in remote.calls if call[2] == 'run']) == 2


def test_subject_acceptance_finishes_after_last_actor_spends_all_work_time(tmp_path):
    from benchmarks.ascend.subject_campaign import SubjectCampaign
    from benchmarks.ascend.subjects import Subject
    from benchmarks.ascend.subject_execution import SubjectOutcome
    now = [0.]
    reviewed = []
    class Verifier:
        last_trace = []
        def review(self, request, **kwargs):
            reviewed.append(request.project_id)
            now[0] += 1000
            return dict(decision='accept', reason='verified', evidence=[request.source_sha256], repair=None, memory=None)
    def execute(*args, **kwargs):
        now[0] += 10
        sha = campaign.artifacts.put_bytes(b'source')
        return SubjectOutcome(True, sha, {'source':sha}, {}, campaign.artifacts.root)
    campaign = SubjectCampaign(tmp_path, [Subject('s','learn',1)], b'', max_seconds=10,clock=lambda: now[0],
        select_wave=lambda *a,**kw:[dict(project_id='one')],execute=execute,verifier=Verifier(),
        complete=lambda *a,**kw:json.dumps(dict(action='advance',reason='complete',evidence=['one'])))
    result = campaign.run()
    assert result['terminal'] == 'COMPLETE' and reviewed == ['one']
    assert result['timing']['elapsed_seconds'] == 10
    assert result['timing']['wall_seconds'] == 1010


def test_repair_budget_tracks_only_its_actors_and_excludes_preflight(tmp_path):
    from benchmarks.ascend.automatic_remediation import FailureTimeBudget
    b, now, _ = budget(tmp_path, 100)
    failure = FailureTimeBudget(tmp_path/'failure.json', 20, clock=lambda: now[0])
    with b.bind(), failure.active():
        with preparation('original retry known-good probe'):
            now[0] += 10000
        assert failure.remaining_seconds == 20
        with b.actor('repair', 100) as repair, repair.scope(work=True):
            now[0] += 3
            with preparation('migration'):
                now[0] += 10000
            now[0] += 2
        assert failure.elapsed_seconds == 5
    assert b.remaining() == 95
    state = json.loads(failure.path.read_bytes())
    assert state['elapsed_seconds'] == 5 and state['active_since'] is None


def test_terminal_candidate_evidence_is_collectable_after_actor_time_reaches_zero(tmp_path):
    from tests.test_cpl_native import Remote, backend
    from benchmarks.ascend.actor_time import preparation_active
    b, now, _ = budget(tmp_path, 10)
    remote = Remote()
    def runner(argv, **kwargs):
        if argv[2] == 'observe': now[0] += 10
        return remote(argv, **kwargs)
    candidate = backend(tmp_path, runner, clock=lambda: now[0], wall_clock=lambda: now[0],
        remaining_seconds=lambda: float('inf') if preparation_active() else b.remaining())
    with b.bind(), b.actor('writer', 10) as writer, writer.scope(work=True):
        result = candidate.run_remote(('true',), key='compile')
    assert b.remaining() == 0 and result.returncode == 0
    assert result.stdout and 'remote:fixture:job:' in result.job_handle
    assert len([call for call in remote.calls if call[2] == 'run']) == 1


def test_repair_budget_multiple_scopes_keep_charged_time(tmp_path):
    from benchmarks.ascend.automatic_remediation import FailureTimeBudget
    b, now, _ = budget(tmp_path, 100)
    failure = FailureTimeBudget(tmp_path/'failure.json', 20, clock=lambda: now[0])
    with b.bind():
        for name in ['first','second']:
            with failure.active(), b.actor(name, 100) as actor, actor.scope(work=True):
                now[0] += 3
        assert failure.elapsed_seconds == 6
    assert json.loads(failure.path.read_bytes())['elapsed_seconds'] == 6
