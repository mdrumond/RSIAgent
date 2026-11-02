"""Real materialization/runner bindings with an explicitly scripted research fixture."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from rsi.configuration import set_configuration, show_configuration, validate_configuration
from rsi.contracts import Stage
from rsi.curriculum_research import DiagnosticProposal, ResearchResult
from test_rsi_services import instance
from test_rsi_curriculum_materialization import cpu


@pytest.fixture
def autonomous(instance, cpu, request):
    parent, config, _ = cpu
    archive = 'a' * 64  # Deliberately absent: historical SFT is not a child dependency.
    observations = instance.store.put_json(dict(applicability=dict(project_id=parent.project_id,
        final_source_hash='c'*64, project_contract_ref='b'*64), artifact_refs=[]))
    facts = instance.store.put_json(dict(project_id=parent.project_id, outcome='budget_exhausted',
        original_actor_seconds_total=50., source_hash='c'*64, observations_ref=observations, artifact_refs=[observations],
        error='scripted historical fixture, not a production failure witness',
        proof_scope=dict(actor_closed=True, provider_settled=True, accounting_unknown=False),
        origin_task_archive=dict(scope='origin-task-archive', service_result=archive)))
    admission = instance.store.put_json(dict(project=parent.to_dict(), host_failure_ref=facts,
        parent_config=asdict(config), original_id=parent.project_id, operation_id='historical-operation'))
    options = dict(getattr(request, 'param', {}))
    other_cap = options.pop('other_parent_cap', None)
    admissions, parents = [admission], [parent]
    if other_cap is not None:
        other = replace(parent, project_id='other-parent', budgets=replace(parent.budgets, actor_seconds=other_cap))
        other_observations = instance.store.read_json(observations)
        other_observations['applicability']['project_id'] = other.project_id
        other_facts = dict(instance.store.read_json(facts), project_id=other.project_id,
            observations_ref=instance.store.put_json(other_observations))
        admissions.append(instance.store.put_json(dict(project=other.to_dict(),
            host_failure_ref=instance.store.put_json(other_facts), parent_config=asdict(config),
            original_id=other.project_id, operation_id='other-historical-operation')))
        parents.append(other)
    settings = show_configuration(instance.root)['payload']
    settings.update(phase1_projects=[parent.to_dict()], project_configs={parent.project_id: asdict(config)},
        curriculum=dict(enabled=True, only_derived=True, project_actor_seconds=7, failed_parent_refs=admissions),
        cycle_limits=dict(actor_seconds=90, recovery_seconds=90, recovery_limit=None))
    if other_cap is not None:
        override = options.pop('project_actor_seconds')
        if override is None:
            settings['curriculum'].pop('project_actor_seconds')
        else:
            settings['curriculum']['project_actor_seconds'] = override
    settings['cycle_limits'].update(options)
    set_configuration(instance.root, settings)
    calls = []
    class ScriptedResearch:
        def actor_context(self, receipt):
            return dict(query='independent sum oracle', evidence_refs=[facts], rationale='Study one frozen case')
        def research(self, spec, failure, context):
            calls.append(context)
            assert failure['host_failure_ref'] == facts and 'result' not in failure
            assert context['failed_parents'][0]['failure']['origin_task_archive']['service_result'] == archive
            receipt = instance.store.put_json(dict(kind='explicit-scripted-research-fixture', artifact_refs=[facts]))
            return ResearchResult(DiagnosticProposal('one case', 'Investigate one frozen case',
                'Independent oracle passes', 'cpu.case-subset', dict(case_indices=[1]), parents[-1].project_id),
                (facts,), .5, receipt)
    instance.curriculum_researcher = ScriptedResearch()
    cycle = instance.create_cycle('autonomous', 'phase1')
    try:
        yield cycle, parent, config, calls, archive, facts
    finally:
        cycle.close()


@pytest.mark.parametrize('autonomous', [{}, dict(actor_seconds=0, recovery_seconds=0, wall_seconds=0)], indirect=True)
def test_derived_config_executes_canonical_stages_and_feeds_memory_without_root_retry(autonomous):
    cycle, parent, parent_config, calls, archive, facts = autonomous
    source = Path(parent_config.source)
    original = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}
    baseline = cycle.scheduler.state['memory']
    revised = cycle.step()
    assert revised.spec.project_id != parent.project_id and revised.spec.budgets.actor_seconds == 7
    assert len(calls) == 1 and cycle.scheduler.state['originals'] == []
    assert cycle.scheduler.config['recover'] is True  # Legacy operator config had recovery disabled.
    assert tuple(cycle.project_config(revised.spec).commands['test']) == revised.spec.command
    child_configuration = cycle.project_configuration(revised.spec)
    assert child_configuration != cycle.spec.configuration_ref
    result = cycle.step()
    while result is None:
        result = cycle.step()
    assert result.outcome == 'accepted'
    row = cycle.scheduler.state['recoveries'][0]
    runner = cycle.open_project(row['operation_id'])
    assert runner.spec.project_id == revised.spec.project_id
    assert {Stage(item['evidence']['stage']) for item in runner.workflow.state['history']} == set(Stage)
    assert runner.store.read_json(facts)['original_actor_seconds_total'] == 50
    with pytest.raises(FileNotFoundError):
        runner.store.read_bytes(archive)
    child_result = cycle.load_project(row)
    assert child_result['project_id'] == parent.project_id  # Existing family identity contract remains intact.
    scope = child_result['observations']['applicability']
    assert scope['project_id'] == revised.spec.project_id
    assert scope['project_contract_ref'] == runner.contract_ref
    assert scope['final_source_hash'] == runner.backend.project_hash()
    memory = cycle.step()
    assert cycle.scheduler.state['memory'] != baseline
    assert cycle.scheduler.state['recovery_spent'] == pytest.approx(.5 + result.actor_seconds + memory.actor_seconds)
    assert cycle.scheduler.state['originals'] == []
    positive = cycle.memory.records()[-1]['proposal']
    assert positive['kind'] == 'correctness'
    assert positive['observations']['applicability']['project_id'] == revised.spec.project_id
    assert cycle.scheduler.state['recovery_queue'][0]['status'] == 'queued'
    retried = cycle.step()
    while retried is None or not hasattr(retried, 'outcome'):
        retried = cycle.step()
    assert retried.outcome == 'accepted'
    assert cycle.scheduler.state['root_retries'][0]['project']['project_id'] == parent.project_id
    assert cycle.scheduler.state['originals'] == []
    assert cycle.scheduler.state['recovery_queue'][0]['status'] == 'accepted'
    assert original == {path: path.read_bytes() for path in original}


def test_completed_simplification_receipt_reopens_without_new_research(autonomous):
    cycle, parent, config, calls, archive, facts = autonomous
    result = cycle.step()
    event = next(row for row in cycle.scheduler.state['events'] if row['kind'] == 'project_revised')
    retained = cycle.curriculum.retained(event['operation_id'])
    assert retained == result and len(calls) == 1
    state = cycle.scheduler.state.copy()
    assert cycle.curriculum.project_config(result.spec) == cycle.project_config(result.spec)
    assert cycle.scheduler.state == state


@pytest.mark.parametrize('autonomous', [
    dict(other_parent_cap=30, project_actor_seconds=None),
    dict(other_parent_cap=120, project_actor_seconds=None),
    dict(other_parent_cap=30, project_actor_seconds=7),
], indirect=True)
def test_selected_parent_owns_child_default_cap_while_research_keeps_its_allocation(autonomous):
    cycle, parent, _, calls, _, _ = autonomous
    selected = cycle.scheduler.state['recovery_queue'][1]
    expected = cycle.config['curriculum'].get('project_actor_seconds',
        selected['root_spec']['budgets']['actor_seconds'])
    result = cycle.step()
    assert result.parent_project_id == 'other-parent'
    assert result.spec.budgets.actor_seconds == expected
    assert calls[0]['remaining']['actor_seconds'] == cycle.config['curriculum'].get(
        'project_actor_seconds', parent.budgets.actor_seconds)
    entry = cycle.scheduler.state['derived_projects'][result.spec.project_id]
    allocation = cycle.store.read_json(entry['allocation_ref'])
    assert allocation['parent_project_id'] == 'other-parent'
    assert allocation['actor_seconds'] == expected
    assert cycle.scheduler.state['recovery_queue'][0]['original_id'] == 'other-parent'


@pytest.mark.parametrize('cap', [0, -1, True, float('inf')])
def test_invalid_child_caps_cannot_admit_configuration(cap):
    with pytest.raises(ValueError):
        validate_configuration(dict(curriculum=dict(project_actor_seconds=cap)))


@pytest.mark.parametrize('pool', [None, 0])
def test_autonomy_does_not_require_shared_semantic_pool(pool):
    validate_configuration(dict(curriculum=dict(enabled=True, project_actor_seconds=7),
        cycle_limits=dict(actor_seconds=7200, recovery_seconds=pool)))


@pytest.mark.parametrize('prior_seconds', [0., .5])
def test_exhausted_host_catalog_defers_without_research_or_setup(autonomous, monkeypatch, prior_seconds):
    from rsi.cycles import SimplificationDeferred
    cycle, parent, config, calls, archive, facts = autonomous
    cases = Path(parent.correctness_ref)
    payload = json.loads(cases.read_text())
    payload['cases'] = payload['cases'][:1]
    cases.write_text(json.dumps(payload))
    def forbidden(*args, **kwargs):
        pytest.fail('Exhausted templates must not load research dependencies or dispatch a model')
    monkeypatch.setattr(cycle, 'query', forbidden)
    original_simplify = cycle.scheduler.simplify
    def resumed(spec, failure, context):
        from rsi.python_execution import atomic_json
        root = cycle.root / 'curriculum' / context['operation_id']
        root.mkdir(parents=True)
        receipt = cycle.store.put_json(dict(kind='known-prior-research-fixture', actor_seconds=prior_seconds))
        atomic_json(root / 'integration.json', dict(history=[dict(receipt_ref=receipt,
            actor_seconds=prior_seconds)], rejections=[]))
        return original_simplify(spec, failure, context)
    monkeypatch.setattr(cycle.scheduler, 'simplify', resumed)
    result = cycle.step()
    assert isinstance(result, SimplificationDeferred)
    assert result.actor_seconds == prior_seconds and calls == []
    assert cycle.scheduler.state['recovery_spent'] == prior_seconds
    receipt = cycle.store.read_json(result.receipt_ref)
    assert receipt['reason'] == 'no-admitted-simpler-template'
    assert cycle.scheduler.state['recovery_queue']
    assert not cycle.scheduler.state.get('derived_projects')


def deferred_wave_query(payload):
    return dict(results=[],negative_records=[],provenance={'fixture':True})


class DeferredWaveDecisions:
    """Explicit CPU test actor: the first root provider closes; retries run normally."""
    def __init__(self):
        from test_rsi_services import Decisions
        self.delegate = Decisions()
    def prepare(self, *args, **kwargs):
        return self.delegate.prepare(*args, **kwargs)
    def next_action(self, context, allowed, **kwargs):
        from coding.pi.rpc import PiProviderError
        if (context['task']['project']['project_id'] == '0'
                and not context['task'].get('curriculum_context', {}).get('retry_original')):
            raise PiProviderError('explicit closed provider-failure fixture for original 0')
        return self.delegate.next_action(context, allowed, **kwargs)
    def close(self):
        self.delegate.close()


def test_actual_cpu_wave_drains_ordinary_siblings_then_retries_original_in_later_wave(instance, cpu):
    from dataclasses import replace
    from rsi.services import TestBindings
    from test_rsi_services import Roles
    parent, config, _ = cpu
    settings=show_configuration(instance.root)['payload']
    specs=[replace(parent, project_id=str(i)) for i in range(3)]
    settings.update(phase1_projects=[p.to_dict() for p in specs],
        project_configs={p.project_id:asdict(config) for p in specs},
        curriculum=dict(enabled=True, project_actor_seconds=15),
        recovery=dict(enabled=True, max_failed_attempts=2),
        cycle_limits=dict(wave_size=2, actor_seconds=300, recovery_seconds=100),
        project_wave=dict(max_parallel=1, wave_size=2, devices=[dict(device='cpu0')]))
    set_configuration(instance.root,settings)
    instance.test=TestBindings(deferred_wave_query,
        DeferredWaveDecisions,Roles,'explicit-test-binding')
    observed=[]
    class Research:
        def research(self, spec, failure, context):
            snapshot=json.loads((instance.root/'runs/next-wave/cycle.json').read_text())
            assert [row['outcome'] for row in snapshot['originals']]==['failed','accepted']
            assert context['wave']==2
            observed.append(context)
            receipt=instance.store.put_json(dict(kind='scripted-wave-research'))
            return ResearchResult(DiagnosticProposal('one frozen case','Study one input','Frozen oracle passes',
                'cpu.case-subset',dict(case_indices=[0]),spec.project_id),(receipt,),.1,receipt)
        def actor_context(self, receipt):
            return dict(rationale='explicit scripted investigation fixture',evidence_refs=[receipt])
    instance.curriculum_researcher=Research()
    cycle=instance.create_cycle('next-wave','phase1')
    try:
        for _ in range(40):
            if cycle.scheduler.state['phase']=='complete':break
            cycle.step()
        assert cycle.scheduler.state['phase']=='complete'
        assert len(observed)==1
        state=cycle.scheduler.state
        assert [r['outcome'] for r in state['originals']]==['failed','accepted','accepted']
        assert len(state['recoveries'])==1 and state['recoveries'][0]['outcome']=='accepted'
        assert len(state['root_retries'])==1 and state['root_retries'][0]['outcome']=='accepted'
        assert state['recovery_queue'][0]['status']=='accepted'
        waves=state['wave_history']
        assert [op['kind'] for op in waves[0]['operations']]==['original','original']
        mixed=next(w for w in waves if {op['kind'] for op in w['operations']}=={'original','recovery'})
        assert mixed['memory']!=waves[0]['memory']
        retry=next(w for w in waves if w['operations'][0]['kind']=='root_retry')
        assert retry['memory']!=mixed['memory']
        records=cycle.memory.records()
        assert any(r['proposal']['kind']=='failure' for r in records)
        assert sum(r['proposal']['kind']=='correctness' for r in records)==4
    finally:
        cycle.close()
