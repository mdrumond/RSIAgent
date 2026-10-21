import json

from benchmarks.ascend.automatic_remediation import AutomaticRemediation
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from test_a5_phase1_adaptive import _project, _wave, _result


def test_failed_project_remedial_wave_then_original_retry_beyond_target(tmp_path):
    calls, executions = [], []
    runner = None
    automatic = None
    def complete(messages):
        calls.append(messages)
        prompt = messages[-1]['content']
        if len(calls) == 1:
            return _wave(prompt, [_project('original', 'gm-ub-movement')])
        failure = automatic.feedback.selection['selected'][0]['feedback_id']
        row = dict(feedback_id=failure, status='investigating', classification='learnable',
                   reason='isolate movement', project_id='probe', control_project_id='control',
                   hypothesis='alignment', observation='compare oracle')
        if len(calls) == 2:
            response = _wave(prompt, [_project('probe', 'padded-tail'),
                                      _project('control', 'gm-ub-movement')])
            payload = json.loads(response.split('\n', 1)[1])
            payload['failure_investigations'] = [row]
            return 'DECISION: WAVE\n' + json.dumps(payload)
        row.update(status='ready-to-retry', reason='host control and probe isolated movement')
        return 'DECISION: SATURATED\n' + json.dumps(dict(reason='ready', failure_investigations=[row]))
    def execute(request):
        executions.append(request)
        passed = request.spec.project_id != 'original' or len(executions) > 1
        return _result(request.spec, passed, 'host outcome')
    runner = AdaptivePhase1Coordinator(tmp_path / 'run', 'Repair failures.', b'{}',
        CurriculumAgent(complete), execute, max_parallel=2, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3')
    result = runner.run()
    original = next(r for r in automatic.store.records() if
                    r['outcome']['source_root'] == str(runner.root.resolve()) and
                    r['outcome']['attempt_id'] == 'original')
    assert original['lifecycle'] == 'fixed'
    assert [r.spec.project_id for r in executions].count('original') == 2
    retries = [r for r in executions if 'original-retries' in str(r.branch_root)]
    assert retries[0].spec == executions[0].spec
    assert retries[0].pre_wave_memory != executions[0].pre_wave_memory
    assert result['project_target'] == 1
    before = len(executions)
    runner.resume()
    assert len(executions) == before


def test_expired_failure_budget_does_not_extend_campaign(tmp_path):
    from test_generic_failure_feedback import phase1
    phase1(tmp_path / 'source')
    runner = AdaptivePhase1Coordinator(tmp_path / 'run', 'Bounded.', b'{}',
        CurriculumAgent(lambda _: 'DECISION: SATURATED\n{"reason":"done"}'),
        lambda request: _result(request.spec, True, 'pass'), max_parallel=1, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3', sources=[dict(type='phase1', root=str(tmp_path/'source'))])
    automatic.refresh(runner._load_state())
    key = next(iter(automatic.budgets))
    path = runner.root / 'failure-budgets' / (key + '.json')
    path.write_text(json.dumps(dict(schema='active-time-v1', seconds=1800, elapsed_seconds=1800)))
    assert not automatic.refresh(runner._load_state())
    assert automatic.store.records()[0]['lifecycle'] == 'failed'


def test_committed_memory_export_authenticates_without_review(tmp_path):
    import pytest
    from tests.test_a3_reviewed_phase2 import native_source
    from optimizer.memory import freeze_committed_phase1, verify_snapshot
    root, review, _ = native_source(tmp_path)
    state = json.loads((root / 'state.json').read_bytes())
    state['prior_outcomes'][0]['summary'] = 'host verified'
    (root / 'state.json').write_text(json.dumps(state))
    review.unlink()
    snapshot = freeze_committed_phase1(root, b'')
    assert snapshot['phase'] == 'committed-phase1'
    assert not any('review' in path for path in snapshot['files'])
    assert len(snapshot['memory_entries']) == 1
    verify_snapshot(snapshot)
    next(root.glob('waves/*/branches/*/artifacts/objects/*/*')).write_bytes(b'changed')
    with pytest.raises(ValueError):
        freeze_committed_phase1(root, b'')


def test_unrelated_projects_do_not_consume_failure_budget(tmp_path):
    from types import SimpleNamespace
    from test_generic_failure_feedback import phase1
    phase1(tmp_path/'source')
    runner = AdaptivePhase1Coordinator(tmp_path/'run', 'Bounded.', b'{}',
        CurriculumAgent(lambda _: ''), lambda _: None, max_parallel=1, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3', sources=[dict(type='phase1', root=str(tmp_path/'source'))])
    automatic.refresh(runner._load_state())
    budget = next(iter(automatic.budgets.values()))
    now = [1.]
    budget.clock = lambda: now[0]
    request = SimpleNamespace(branch_root=runner.root/'waves/wave-1/branches/new',
                              assignment=SimpleNamespace(project_id='new'))
    with automatic.project_active(request):
        now[0] += 60
    assert budget.elapsed_seconds == 0
