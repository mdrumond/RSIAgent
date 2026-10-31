"""Real materialization/runner bindings with an explicitly scripted research fixture."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from rsi.configuration import set_configuration, show_configuration, validate_configuration
from rsi.contracts import Stage
from rsi.curriculum_research import DiagnosticProposal, ResearchResult
from test_rsi_services import instance
from test_rsi_curriculum_materialization import cpu


@pytest.fixture
def autonomous(instance, cpu):
    parent, config, _ = cpu
    archive = 'a' * 64  # Deliberately absent: historical SFT is not a child dependency.
    observations = instance.store.put_json(dict(project_id=parent.project_id, artifact_refs=[]))
    facts = instance.store.put_json(dict(project_id=parent.project_id, outcome='budget_exhausted',
        original_actor_seconds_total=7200., observations_ref=observations, artifact_refs=[observations],
        error='scripted historical fixture, not a production failure witness',
        proof_scope=dict(actor_closed=True, provider_settled=True, accounting_unknown=False),
        origin_task_archive=dict(scope='origin-task-archive', service_result=archive)))
    admission = instance.store.put_json(dict(project=parent.to_dict(), host_failure_ref=facts,
        parent_config=asdict(config), original_id=parent.project_id, operation_id='historical-operation'))
    settings = show_configuration(instance.root)['payload']
    settings.update(phase1_projects=[parent.to_dict()], project_configs={parent.project_id: asdict(config)},
        curriculum=dict(enabled=True, only_derived=True, project_actor_seconds=7, failed_parent_refs=[admission]),
        cycle_limits=dict(actor_seconds=90, recovery_seconds=90, recovery_limit=None))
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
                'Independent oracle passes', 'cpu.case-subset', dict(case_indices=[1]), parent.project_id),
                (facts,), .5, receipt)
    instance.curriculum_researcher = ScriptedResearch()
    cycle = instance.create_cycle('autonomous', 'phase1')
    try:
        yield cycle, parent, config, calls, archive, facts
    finally:
        cycle.close()


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
    assert result.outcome == 'accepted'
    row = cycle.scheduler.state['recoveries'][0]
    runner = cycle.open_project(row['operation_id'])
    assert runner.spec.project_id == revised.spec.project_id
    assert {Stage(item['evidence']['stage']) for item in runner.workflow.state['history']} == set(Stage)
    assert runner.store.read_json(facts)['original_actor_seconds_total'] == 7200
    with pytest.raises(FileNotFoundError):
        runner.store.read_bytes(archive)
    child_result = cycle.load_project(row)
    assert child_result['project_id'] == revised.spec.project_id
    assert child_result['parent_project_id'] == parent.project_id
    memory = cycle.step()
    assert cycle.scheduler.state['memory'] != baseline
    assert cycle.scheduler.state['recovery_spent'] == pytest.approx(.5 + result.actor_seconds + memory.actor_seconds)
    assert cycle.scheduler.state['originals'] == []
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


@pytest.mark.parametrize('cap', [0, -1, True, float('inf')])
def test_invalid_child_caps_cannot_admit_configuration(cap):
    with pytest.raises(ValueError):
        validate_configuration(dict(curriculum=dict(project_actor_seconds=cap)))


@pytest.mark.parametrize('pool', [None, 0, float('inf')])
def test_autonomy_requires_explicit_finite_semantic_pool(pool):
    with pytest.raises(ValueError):
        validate_configuration(dict(curriculum=dict(enabled=True),
            cycle_limits=dict(actor_seconds=7200, recovery_seconds=pool)))
