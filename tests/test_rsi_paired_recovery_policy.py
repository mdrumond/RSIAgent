"""Independent elapsed allocations preserve canonical paired trials and jobs."""
from dataclasses import asdict
import time

import pytest

from core.artifacts import ContentAddressedArtifacts
from rsi.configuration import set_configuration, show_configuration
from rsi.contracts import BudgetLimits, CycleSpec, Mode, ProjectSpec
from rsi.cycles import CycleScheduler, HostTaskResult, SnapshotUpdate
from rsi.project_runner import ProjectRunner
from rsi.recovery import CycleController, PairedRecovery
from test_rsi_project_runner import Session
from test_rsi_recovery import LearningStub, run_pair, setup_pair
from test_rsi_services import instance


def test_each_real_arm_gets_full_elapsed_limits_with_zero_parent_pools(tmp_path):
    pair, original, _, calls, database = setup_pair(tmp_path, repairs=(True, True),
        independent=None, limits=BudgetLimits(actor_seconds=30, wall_seconds=60),
        remaining=dict(actor_seconds=0, recovery_seconds=0, wall_seconds=0))
    try:
        original_state = (original.root / 'workflow.json').read_bytes()
        requests = [pair.request(arm) for arm in ('incumbent', 'candidate')]
        for request in requests:
            assert request.spec.goal == original.spec.goal
            assert request.spec.budgets.actor_seconds == 30
            assert request.spec.budgets.wall_seconds == 60
            assert request.spec.budgets.execution_tries is None
        assert requests[0].config.commands == requests[1].config.commands
        assert requests[0].memory_ref == requests[1].memory_ref
        assert requests[0].kdb_ref == requests[1].kdb_ref
        result = run_pair(pair)
        assert all(row['outcome'] == 'accepted' and row['correctness'] is True
                   for row in result['arms'].values())
        assert result['actor_seconds'] == sum(row['actor_seconds'] for row in result['arms'].values())
        assert [arm for arm, reopen, _ in calls if not reopen] == ['incumbent', 'candidate']
        assert (original.root / 'workflow.json').read_bytes() == original_state
    finally:
        database.close()


def test_second_arm_has_own_wall_origin_after_first_expires_and_resume(tmp_path):
    class Clock:
        value = 1000.
        started = time.monotonic()
        def __call__(self):
            return self.value + time.monotonic() - self.started
    clock = Clock()
    prepared = []
    class SlowFirst(Session):
        def prepare(self, timeout, *, schemas):
            prepared.append(timeout)
            clock.value += 50
            return super().prepare(timeout, schemas=schemas)
    def factory(request, reopen, injections):
        injections['session_factory'] = SlowFirst if request.arm == 'incumbent' else Session
        return ProjectRunner(request.root, **injections) if reopen else ProjectRunner.create(
            request.root, request.spec, request.bundle, asdict(request.config), **injections)
    pair, _, store, calls, database = setup_pair(tmp_path, independent=True, clock=clock,
        wrap=factory, limits=BudgetLimits(actor_seconds=30, wall_seconds=40),
        remaining=dict(actor_seconds=0, recovery_seconds=0, wall_seconds=0))
    try:
        first = pair.step()
        assert first['wall_limited'] and first['outcome'] == 'failed'
        deadline = pair.state['arms']['incumbent']['wall_deadline']
        clock.value += 100
        reopened = PairedRecovery(pair.path, artifacts=store, runner_factory=pair.factory, clock=clock)
        started = clock()
        second = reopened.step()
        assert second['outcome'] == 'accepted'
        assert second['wall_deadline'] == pytest.approx(started + 40, abs=.2)
        assert reopened.state['arms']['incumbent']['wall_deadline'] == deadline
        assert reopened.result()['decision'] == 'inconclusive'
        assert len(prepared) == 1 and len(calls) == 2
        assert reopened.step() == reopened.result()
    finally:
        database.close()


def test_controller_preserves_memory_writer_and_validated_promotion_at_zero_pools(tmp_path):
    calls = []
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    project = ProjectSpec('root', 'Frozen original oracle', ('python', 'verify.py'),
                          budgets=BudgetLimits(actor_seconds=30))
    spec = CycleSpec('paired', Mode.PHASE2, (project,), 'memory', 'incumbent')
    def run(project, memory, skills, context):
        calls.append(('original', project.budgets.actor_seconds))
        return HostTaskResult('failed', 'original-evidence', 31)
    def memory(*args):
        calls.append(('memory', args[1][0]['artifact_ref']))
        return SnapshotUpdate('updated-memory', 2)
    def writer(*args):
        calls.append(('writer', args[1]))
        return SnapshotUpdate('candidate', 3)
    scheduler = CycleScheduler(tmp_path / 'cycle.json', spec, run, memory,
        skill_update=writer, independent_projects=True, actor_seconds=0,
        recovery_seconds=0, wall_seconds=0)
    learning = LearningStub('incumbent')
    class Pair:
        def result(self):
            return dict(decision='benefit', actor_seconds=45,
                        evidence_ref=store.put_json({'actual': 'paired evidence'}))
    def pair(event, operation, remaining):
        assert remaining == dict(actor_seconds=None, recovery_seconds=None, wall_seconds=None)
        calls.append(('pair', event['memory']))
        return Pair()
    controller = CycleController(tmp_path / 'controller.json', scheduler, learning,
                                 artifacts=store, pair_factory=pair)
    try:
        for _ in range(4):
            controller.step()
        assert calls == [('original', 30), ('memory', 'original-evidence'),
                         ('writer', 'updated-memory'), ('pair', 'updated-memory')]
        assert scheduler.state['skills'] == 'candidate'
        assert len(learning.calls) == 1
        assert scheduler.remaining() == dict(actor_seconds=0, recovery_seconds=0, wall_seconds=0)
        assert scheduler.state['spent'] == 81
        assert len(scheduler.state['extension_receipts']) == 1
    finally:
        scheduler.close()


@pytest.mark.parametrize('mode', ['phase1', 'phase2', 'inference'])
def test_owned_job_ignores_shared_wall_but_keeps_real_project_limits(instance, monkeypatch, mode):
    import rsi.service
    settings = show_configuration(instance.root)['payload']
    settings['cycle_limits'] = dict(actor_seconds=0, recovery_seconds=0, wall_seconds=0)
    settings['tuning'] = dict(at_wave_end=False)
    set_configuration(instance.root, settings)
    monkeypatch.setattr(rsi.service, 'InstanceServices', lambda root: instance)
    events = []
    result = rsi.service.run_job(instance.root,
        dict(action='start', cycle_id='independent-' + mode, mode=mode, recover=False),
        show_configuration(instance.root), should_stop=lambda: False, emit=events.append)
    assert result['status'] == 'complete'
    cycle = instance.open_cycle('independent-' + mode)
    try:
        assert cycle.scheduler.independent_projects
        assert cycle.scheduler.state['originals'][0]['outcome'] == 'accepted'
        assert cycle.scheduler.spec.projects[0].budgets.actor_seconds == 30
        assert cycle.scheduler.remaining()['wall_seconds'] == 0
        assert not any(row['kind'] == 'paired_recovery_due'
                       for row in cycle.scheduler.extension_events())
        assert not cycle.scheduler.state.get('root_retries')
    finally:
        cycle.close()
