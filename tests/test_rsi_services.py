"""Actual CPU controllers with explicitly injected model decisions only."""
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from benchmarks.a5kernels.knowledge import KnowledgeDB
from rsi.configuration import set_configuration, show_configuration
from rsi.contracts import BudgetLimits, ProjectSpec, Stage
from rsi.roles import CurriculumProposal, ProjectIdea, SkillProposal
from rsi.services import InstanceServices, TestBindings as Bindings
from rsi.skills import builtin_bundle


class Embedding:
    model, revision = 'TEST', '1'
    def embed(self, texts):
        return [[1., 0.] for _ in texts]


class Decisions:
    calls = 0
    tasks = []
    def prepare(self, timeout, *, schemas):
        pass
    def next_action(self, context, allowed, **kwargs):
        type(self).calls += 1
        type(self).tasks.append(context['task'])
        for skill in context['skills']:
            if skill not in {r['skill'] for r in context['read_skills']}:
                return 'read_skill', {'name': skill}
        evidence = {r['name']: r['result'] for r in context['evidence']}
        if Stage(context['stage']) in (Stage.DESIGN, Stage.OPTIMISATION):
            if 'query_kdb' not in evidence:
                return 'query_kdb', {'query': 'safe numeric computation'}
            if 'retrospect' not in evidence:
                return 'retrospect', {'supplement': 'Use retained independent correctness checks'}
            if 'select_strategy' not in evidence:
                return 'select_strategy', dict(proposal='Preserve verified result',
                    query_ref=evidence['query_kdb']['reference'],
                    retrospective_ref=evidence['retrospect']['reference'],
                    skill_refs=[r['receipt_id'] for r in context['read_skills']], decisions=[])
            if context['stage'] == 'optimisation' and 'optimisation_decision' not in evidence:
                return 'optimisation_decision', dict(action='noop', reason='Correctness before speed')
        return 'perform', {'proposal': 'Preserve verified result'}
    def close(self):
        pass


class Roles:
    prompts = []
    def __init__(self, observer):
        self.observer = observer
    def skillwriter(self, prompt, evidence, *, deadline):
        type(self).prompts.append(prompt)
        self.observer('role_finished', dict(role='skillwriter', success=True, elapsed_seconds=0.25))
        return SkillProposal('No supported changes', (evidence[0].evidence_id,), ())
    def curriculum(self, prompt, evidence, *, deadline):
        type(self).prompts.append(prompt)
        self.observer('role_finished', dict(role='curriculum', success=True, elapsed_seconds=0.125))
        return CurriculumProposal('Retain configured host cases', (evidence[0].evidence_id,),
                                  (ProjectIdea('answer', 'Compute answer', 'Independent verifier passes'),))


@pytest.fixture
def instance(tmp_path):
    root = tmp_path / 'instance'
    for name in ('config', 'skills', 'memory', 'source', 'runs'):
        (root / name).mkdir(parents=True)
    (root / 'source/solution.py').write_text('answer = 42\n')
    (root / 'verify.py').write_text("import runpy\nassert runpy.run_path('solution.py')['answer'] == 42\n")
    db = KnowledgeDB(root / 'memory/knowledge.sqlite3', Embedding())
    db.close()
    bundle = builtin_bundle()
    for doc in bundle.documents:
        path = root / 'skills' / doc.name / 'SKILL.md'
        path.parent.mkdir()
        path.write_text(doc.content)
    (root / 'config/skill-map.json').write_text(json.dumps(
        {stage.value: list(names) for stage, names in bundle.mapping}))
    config = dict(kdb='memory/knowledge.sqlite3', skills='skills', skill_map='config/skill-map.json',
                  machinery='machinery', dashboard={})
    (root / 'config/instance.json').write_text(json.dumps(config))
    checksum = hashlib.sha256((root / config['kdb']).read_bytes()).hexdigest()
    (root / 'instance.json').write_text(json.dumps({'files': {config['kdb']: checksum}}))
    command = (sys.executable, '{verifier}')
    project = ProjectSpec('answer', 'Compute correct answer', command,
                          budgets=BudgetLimits(actor_seconds=30, wall_seconds=60))
    config.update(phase1_projects=[project.to_dict()], phase2_projects=[project.to_dict()],
        project_configs={'answer': dict(source='source', verifier='verify.py',
            commands=dict(compile=[sys.executable, '-m', 'py_compile', 'solution.py'],
                          test=list(command), validation=list(command)))},
        prompts=dict(actor='actor-steering', curriculum='curriculum-steering',
                     waves='waves-steering', skillwriter='writer-steering'),
        recovery={'enabled': False})
    set_configuration(root, config)
    services = InstanceServices(root, environment={}, test_bindings=Bindings(
        lambda _: dict(results=[], negative_records=[], provenance={'test': True}),
        Decisions, Roles, 'explicit-test-binding'))
    Decisions.calls, Decisions.tasks, Roles.prompts = 0, [], []
    return services


def test_phase1_runs_real_host_validation_retains_usage_and_freezes_config(instance):
    cycle = instance.create_cycle('first', 'phase1')
    frozen = cycle.spec.configuration_ref
    edited = show_configuration(instance.root)['payload']
    edited['prompts']['actor'] = 'later-steering'
    set_configuration(instance.root, edited)
    try:
        result = cycle.controller.step()
        assert result.outcome == 'accepted'
        row = cycle.scheduler.state['originals'][0]
        packet = cycle.load_project(row)
        assert packet['outcome'] == 'accepted' and packet['queries']
        assert any(r['kind'] == 'read' for r in packet['usage'])
        assert all(task['configuration_ref'] == frozen for task in Decisions.tasks)
        assert all(task['user_guidance']['actor'] == 'actor-steering' for task in Decisions.tasks)
        cycle.controller.step()  # Phase1 wave memory only.
        assert cycle.memory.records()[0]['proposal']['kind'] == 'correctness'
        assert not Roles.prompts
    finally:
        cycle.close()
    later = instance.create_cycle('later', 'phase1')
    try:
        assert later.config['prompts']['actor'] == 'later-steering'
    finally:
        later.close()


def test_phase2_writer_uses_real_observer_elapsed_and_keeps_incumbent(instance):
    cycle = instance.create_cycle('second', 'phase2')
    try:
        incumbent = cycle.scheduler.state['skills']
        cycle.controller.step()
        cycle.controller.step()  # Memory.
        cycle.controller.step()  # Typed writer.
        assert cycle.scheduler.state['spent'] >= .25
        assert cycle.role_elapsed(cycle.scheduler.state['candidates'][0]['operation_id']) == .25
        assert Roles.prompts and cycle.config['prompts']['skillwriter'] in Roles.prompts[-1]
        assert cycle.config['prompts']['curriculum'] in Roles.prompts[-1]
        assert cycle.config['prompts']['waves'] in Roles.prompts[-1]
        cycle.controller.step()  # No independent success cases => no fabricated promotion.
        assert cycle.scheduler.state['skills'] == incumbent
        assert cycle.learning.active == incumbent
    finally:
        cycle.close()


def test_inference_mutable_heads_are_separate_and_reopen_uses_frozen_settings(instance):
    cycle = instance.create_cycle('inference', 'inference')
    try:
        parent = cycle.memory.root
        receipt = cycle.controller.step()
        receipt.verify()
        assert cycle.memory.root != parent
        cycle.controller.step()
        cycle.controller.step()
        assert cycle.memory.records()
        assert not json.loads((parent / 'active.json').read_text())['active']
    finally:
        cycle.close()
    reopened = instance.open_cycle('inference')
    try:
        assert reopened.memory.root == Path(receipt.branch_root) / 'memory'
    finally:
        reopened.close()


def test_baseline_corruption_and_backend_substitution_are_rejected(instance):
    cycle = instance.create_cycle('safe', 'phase1')
    cycle.close()
    with pytest.raises(ValueError, match='backend binding'):
        InstanceServices(instance.root, environment={}).open_cycle('safe')
    with (instance.root / 'memory/knowledge.sqlite3').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        instance.create_cycle('bad', 'phase1')
    assert not (instance.root / 'runs/bad').exists()


def test_unknown_work_is_not_replayed_and_status_does_not_dispatch(instance):
    cycle = instance.create_cycle('pending', 'phase1')
    def fail(*args):
        raise RuntimeError('provider work unknown')
    cycle.scheduler.runner = fail
    with pytest.raises(RuntimeError):
        cycle.controller.step()
    pending = cycle.scheduler.state['pending']['id']
    cycle.close()
    before = Decisions.calls
    status = instance.status('pending')
    assert status['cycle.json']['pending']['id'] == pending
    reopened = instance.open_cycle('pending')
    try:
        with pytest.raises(Exception, match='reconciliation'):
            reopened.controller.step()
        assert Decisions.calls == before
    finally:
        reopened.close()


def test_role_curriculum_receives_frozen_guidance_and_no_setup_wallcharge(instance):
    cycle = instance.create_cycle('ideas', 'phase1')
    try:
        proposal = cycle.propose_curriculum(operation_id='ideas-op', deadline=time.monotonic() + 10)
        assert proposal.evidence_ids == (cycle.spec.configuration_ref,)
        assert json.loads(Roles.prompts[-1]) == {
            'curriculum': cycle.config['prompts']['curriculum'],
            'waves': cycle.config['prompts']['waves']}
        assert cycle.role_elapsed('ideas-op') == .125
        assert cycle.scheduler.state['spent'] == .125
        assert cycle.propose_curriculum(operation_id='ideas-op', deadline=time.monotonic() + 10) == proposal
        assert cycle.scheduler.state['spent'] == .125
        with pytest.raises(ValueError, match='finished'):
            cycle.role_elapsed('missing')
    finally:
        cycle.close()


def test_production_embedding_load_failure_never_becomes_empty_success(instance, monkeypatch):
    from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
    instance.test = None
    cycle = instance.create_cycle('production-kdb', 'phase1')
    def unavailable(self):
        raise RuntimeError('pinned embedding cache unavailable')
    monkeypatch.setattr(PinnedBGEEmbeddings, '_load', unavailable)
    try:
        with pytest.raises(RuntimeError, match='cache unavailable'):
            cycle.query(cycle.spec.baseline_kdb)
        assert cycle.database is None
    finally:
        cycle.close()


def test_global_caps_and_configured_public_command_are_enforced(instance):
    config = show_configuration(instance.root)['payload']
    config['budgets']['actor_seconds'] = 5
    set_configuration(instance.root, config)
    cycle = instance.create_cycle('bounded', 'phase1')
    try:
        assert cycle.spec.projects[0].budgets.actor_seconds == 5
    finally:
        cycle.close()
    config['project_configs']['answer']['commands']['test'] = ['unrelated-command']
    set_configuration(instance.root, config)
    with pytest.raises(ValueError, match='command must equal'):
        instance.create_cycle('mismatched', 'phase1')


def two_projects(instance):
    config = show_configuration(instance.root)['payload']
    second = dict(config['phase2_projects'][0], project_id='second')
    config['phase2_projects'].append(second)
    config['project_configs']['second'] = dict(config['project_configs']['answer'])
    set_configuration(instance.root, config)
    return instance.create_cycle('steered', 'phase2')


def task_boundary(cycle):
    cycle.step()  # Original.
    cycle.step()  # Memory.
    cycle.step()  # Writer.
    cycle.step()  # Candidate validation remains pending without independent cases.
    assert cycle.scheduler.state['phase'] == 'original'


def test_upcoming_revision_adopts_transactionally_without_changing_history_or_budget(instance):
    import copy
    cycle = two_projects(instance)
    try:
        task_boundary(cycle)
        before = copy.deepcopy(cycle.scheduler.state)
        old_ref = cycle.spec.configuration_ref
        first_op = before['originals'][0]['operation_id']
        config = show_configuration(instance.root)['payload']
        config['phase2_projects'][1]['goal'] = 'Steered second goal'
        config['prompts'] = {key: 'new-' + value for key, value in config['prompts'].items()}
        config['budgets']['actor_seconds'] = 4
        snapshot = set_configuration(instance.root, config)
        result = cycle.adopt_revision(snapshot)
        assert result['cursor'] == 1 and result['configuration'] != old_ref
        for field in ('spent', 'recovery_spent', 'created', 'originals', 'completed_operations', 'memory', 'skills'):
            assert cycle.scheduler.state[field] == before[field]
        assert cycle.spec.projects[0].to_dict() == before['config']['spec']['projects'][0]
        assert cycle.spec.projects[1].goal == config['phase2_projects'][1]['goal']
        assert cycle.spec.projects[1].budgets.actor_seconds == 4
        assert cycle.open_project(first_op).context()['user_guidance']['actor'] == 'actor-steering'
        # Folder contents are frozen when NEXT task is admitted, not at revision editing.
        (instance.root / 'source/solution.py').write_text('answer = 42\n# changed before admission\n')
        cycle.step()
        second = cycle.open_project(cycle.scheduler.state['originals'][1]['operation_id'])
        assert second.backend.read('solution.py')['content'] == (instance.root / 'source/solution.py').read_text()
        assert second.context()['user_guidance'] == config['prompts']
        cycle.step()
        cycle.step()
        assert config['prompts']['skillwriter'] in Roles.prompts[-1]
        assert list((cycle.root / 'revisions').glob('*.json'))
    finally:
        cycle.close()


def test_adoption_refuses_pending_and_historical_edits(instance):
    cycle = two_projects(instance)
    try:
        task_boundary(cycle)
        config = show_configuration(instance.root)['payload']
        config['phase2_projects'][0]['goal'] = 'Rewrite completed history'
        with pytest.raises(ValueError, match='admitted project contracts'):
            cycle.adopt_revision(set_configuration(instance.root, config))
        cycle.scheduler.state['pending'] = {'id': 'unknown', 'kind': 'original'}
        cycle.scheduler._save()
        with pytest.raises(ValueError, match='idle task/wave boundary'):
            cycle.adopt_revision(show_configuration(instance.root))
    finally:
        cycle.close()


def test_partial_adoption_recovers_exact_transaction_without_dispatch_or_refund(instance, monkeypatch):
    import copy
    from rsi import revisions
    cycle = two_projects(instance)
    task_boundary(cycle)
    before = copy.deepcopy(cycle.scheduler.state)
    config = show_configuration(instance.root)['payload']
    config['prompts']['actor'] = 'recovered-adoption'
    snapshot = set_configuration(instance.root, config)
    publish = revisions.atomic_json
    def fail_service(path, value):
        if path.name == 'service.json':
            raise OSError('controller interrupted after cycle pointer')
        return publish(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(revisions, 'atomic_json', fail_service)
        with pytest.raises(OSError):
            cycle.adopt_revision(snapshot)
    cycle.close()
    calls = Decisions.calls
    reopened = instance.open_cycle('steered')
    try:
        assert Decisions.calls == calls
        assert reopened.config['prompts']['actor'] == config['prompts']['actor']
        assert reopened.scheduler.state['spent'] == before['spent']
        assert reopened.scheduler.state['created'] == before['created']
        assert reopened.scheduler.state['originals'] == before['originals']
        assert not (reopened.root / 'revision.pending.json').exists()
    finally:
        reopened.close()


def test_dispatcher_snapshot_and_recovery_override_are_frozen_at_create(instance):
    frozen = show_configuration(instance.root)
    edited = json.loads(json.dumps(frozen['payload']))
    edited['prompts']['actor'] = 'later-editor-value'
    set_configuration(instance.root, edited)
    cycle = instance.create_cycle('launch-snapshot', 'phase1', configuration=frozen, recover=False)
    try:
        assert cycle.config['prompts']['actor'] == frozen['payload']['prompts']['actor']
        assert cycle.scheduler.config['recover'] is False
    finally:
        cycle.close()


def test_owned_job_stop_reaps_canonical_python_child_and_retains_pending(instance, tmp_path):
    import os
    import subprocess
    # Run signal handling in its OWN worker, never pytest's main interpreter.
    verifier = instance.root / 'verify.py'
    verifier.write_text('import time\ntime.sleep(60)\n')
    result_path = tmp_path / 'worker-result.json'
    program = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from test_rsi_services import Bindings, Decisions, Roles
from rsi.services import InstanceServices
import rsi.service as adapter
root, output = Path(sys.argv[2]), Path(sys.argv[3])
services = InstanceServices(root, environment={}, test_bindings=Bindings(
    lambda _: dict(results=[], negative_records=[], provenance={'test': True}),
    Decisions, Roles, 'explicit-test-binding'))
adapter.InstanceServices = lambda _: services
from rsi.configuration import show_configuration
child = None
def stop():
    global child
    for path in (root / 'runs/worker/projects').glob('*/receipts/*.json'):
        row = json.loads(path.read_text())
        if row.get('command') == 'test' and row.get('pid'):
            child = row['pid']
            return True
    return False
result = adapter.run_job(root, dict(action='start', cycle_id='worker', mode='phase1'),
                         show_configuration(root), should_stop=stop, emit=lambda _: None)
project = next((root / 'runs/worker/projects').iterdir())
workflow = json.loads((project / 'workflow.json').read_text())
output.write_text(json.dumps(dict(result=result, child=child,
    child_alive=Path('/proc/' + str(child)).exists(), workflow=workflow)))
'''
    completed = subprocess.run([sys.executable, '-c', program, str(Path(__file__).parent),
        str(instance.root), str(result_path)], capture_output=True, text=True, timeout=20,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1])))
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(result_path.read_text())
    assert receipt['result']['status'] == 'stopped'
    assert receipt['child'] and not receipt['child_alive']
    assert receipt['workflow']['stopped'] and receipt['workflow']['pending']
    assert all(row[2] is not None for row in receipt['workflow']['intervals'].values())


def configure_heldout(instance):
    source = instance.root / 'heldout'
    source.mkdir()
    (source / 'solution.py').write_text('answer = sum([1, 2, 3])\n')
    (instance.root / 'heldout-verify.py').write_text(
        "import runpy\nassert runpy.run_path('solution.py')['answer'] == 6\n")
    config = show_configuration(instance.root)['payload']
    project = dict(config['phase2_projects'][0], project_id='independent-sum', goal='Sum integers correctly')
    execution = json.loads(json.dumps(config['project_configs']['answer']))
    execution.update(source='heldout', verifier='heldout-verify.py')
    config['tuning'].update(heldout_projects=1, runs_per_arm=1, every_projects=1,
                           heldout_cases=[dict(project=project, execution=execution)])
    set_configuration(instance.root, config)


def test_configured_heldout_tuning_uses_real_pairs_and_exact_validation_scope(instance):
    configure_heldout(instance)
    cycle = instance.create_cycle('tuned', 'phase2')
    try:
        incumbent = cycle.scheduler.state['skills']
        for _ in range(4):
            cycle.step()  # Original, memory, writer, successful-original validation.
        assert len([e for e in cycle.scheduler.extension_events() if e['kind'] == 'tuning_due']) == 1
        cycle.step()  # First real arm.
        assert cycle.tuning.state['pending']
        cycle.step()  # Second real arm and conservative tie decision.
        assert not cycle.tuning.state['pending']
        path = next((cycle.root / 'evaluations').glob('*/evaluation.json'))
        from rsi.evaluation import SkillEvaluation
        evaluation = SkillEvaluation(path, artifacts=cycle.store, runner_factory=cycle.trial)
        result = evaluation.result()
        assert evaluation.plan['case_count'] == 1 and evaluation.plan['repetitions'] == 1
        assert result['decision'] != 'benefit'
        assert cycle.learning.active == incumbent
        verdict = cycle.validate(incumbent, evaluation.plan['candidate'], [result['evidence_ref']], 'exact')
        assert verdict.workflow is False and verdict.correctness is False
        with pytest.raises(ValueError, match='scope'):
            cycle.validate('different-incumbent', evaluation.plan['candidate'], [result['evidence_ref']], 'foreign')
        assert result['actor_seconds'] > 0
    finally:
        cycle.close()


def test_manual_tuning_guidance_reaches_real_writer_and_frozen_suite(instance):
    configure_heldout(instance)
    config = show_configuration(instance.root)['payload']
    config['tuning'].update(every_projects=None, at_wave_end=False)
    set_configuration(instance.root, config)
    cycle = instance.create_cycle('manual', 'phase2')
    try:
        for _ in range(5):
            cycle.step()  # Includes explicit wave-end-disabled receipt.
        steering = dict(project_ref='answer', user_guidance='Detect numerical mistakes before tuning speed')
        cycle.request_tuning(steering)
        cycle.step()
        cycle.step()
        assert any(steering['user_guidance'] in prompt for prompt in Roles.prompts)
        path = next((cycle.root / 'evaluations').glob('*/evaluation.json'))
        from rsi.evaluation import SkillEvaluation
        suite = SkillEvaluation(path, artifacts=cycle.store, runner_factory=cycle.trial)
        assert suite.plan['steering'] == steering
        assert not cycle.tuning.state['pending']
    finally:
        cycle.close()


def test_setup_telemetry_is_retained_without_charging_actor_setup(instance):
    class SlowReady(Decisions):
        def prepare(self, timeout, *, schemas):
            time.sleep(.05)
    instance.test = Bindings(instance.test.query, SlowReady, Roles, instance.test.identity)
    before = time.time()
    cycle = instance.create_cycle('telemetry', 'phase1')
    try:
        assert cycle.scheduler.state['created'] >= before
        assert cycle.scheduler.state['created'] == cycle.binding['created']
        cycle.step()
        runner = cycle.open_project(cycle.scheduler.state['originals'][0]['operation_id'])
        rows = [json.loads(line) for line in (instance.root / 'artifacts/setup.jsonl').read_text().splitlines()]
        readiness = [r for r in rows if r['category'] == 'actor-process-readiness']
        assert readiness and readiness[-1]['elapsed_seconds'] >= .05
        assert runner.workflow.accounting.seconds('setup') >= .05
        assert all(r['actor_billable'] is False for r in rows)
        assert any(r['category'] == 'baseline-kdb-copy' for r in rows)
        assert any(r['category'] == 'project-snapshot-copy-and-bind' for r in rows)
        for row in rows:
            assert row['elapsed_seconds'] >= 0 and row['reference']
            assert cycle.store.read_json(row['reference'])['elapsed_seconds'] == row['elapsed_seconds']
    finally:
        cycle.close()


def test_explicit_project_continue_resumes_only_after_authoritative_pending_reconciliation(instance, monkeypatch):
    from rsi.project_runner import ProjectRunner
    from rsi.runtime import WorkflowError
    cycle = instance.create_cycle('continue', 'phase1')
    def stop_before_dispatch(runner):
        try:
            runner.actor.step(runner.context())  # Real host preflight.
            runner.actor.step(runner.context())  # Real skill-read/model task construction.
            runner.workflow.begin()  # Design reservation; NO provider or host dispatch.
            runner.workflow.stop()
            raise KeyboardInterrupt('stopped at known undispatched boundary')
        finally:
            runner.actor.close()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(ProjectRunner, 'run', stop_before_dispatch)
            with pytest.raises(KeyboardInterrupt):
                cycle.step()
        operation = cycle.scheduler.state['pending']['id']
        runner = cycle.open_project(operation)
        pending = runner.workflow.pending['id']
        assert runner.workflow.pending['handle'] is None and runner.workflow.state['stopped']
        created = runner.workflow.state['created']
        elapsed = runner.workflow.accounting.seconds()
        allocation = runner.workflow.limits.to_dict()
        calls = Decisions.calls
        with pytest.raises(WorkflowError, match='no stage receipt'):
            cycle.run_project(operation)
        assert Decisions.calls == calls
        assert cycle.open_project(operation).workflow.state['stopped']
        # Trusted host knows this exact reservation was never dispatched above.
        runner.workflow.reconcile(pending, not_dispatched=True)
        assert runner.workflow.state['stopped'] and not runner.workflow.pending
        result = cycle.run_project(operation)
        assert result.outcome == 'accepted'
        resumed = cycle.open_project(operation)
        assert resumed.workflow.state['created'] == created
        assert resumed.workflow.limits.to_dict() == allocation
        assert resumed.workflow.accounting.seconds() >= elapsed
        assert not resumed.workflow.state['stopped']
        assert any(h.get('not_dispatched') for h in resumed.workflow.state['history'])
        cycle.scheduler.reconcile(operation, result)
        assert cycle.scheduler.state['originals'][0]['outcome'] == 'accepted'
    finally:
        cycle.close()
