import json

import pytest

from benchmarks.a3_model_profiles import A3Completion
from optimizer.agent import OptimizerAgent


@pytest.mark.parametrize('remaining', [0, 2])
def test_cases_stage_directive_and_actual_recorded_request(remaining):
    requests, records = [], []
    def complete(system, user, **kwargs):
        requests.append((system, user))
        return A3Completion('{"action":"cases","signatures":[],"summary":"bounded"}', 1, {})
    agent = OptimizerAgent(guide='native guide', complete=complete)
    result = agent.complete('cases', {'max_cases': 3, 'remaining_signatures': remaining},
                            lambda kind, value: records.append((kind, value)))
    system, user = requests[0]
    assert system.startswith('CURRENT STAGE: cases.')
    assert 'Return ONLY {"action":"cases"' in system
    assert 'max_cases is 3' in system and f'remaining_signatures is {remaining}' in system
    assert 'If it is zero, return signatures: []' in system
    assert json.loads(user)['stage'] == 'cases'
    assert json.loads(user)['remaining_signatures'] == remaining
    assert records[0][1]['messages'] == [
        {'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    assert result['action'] == 'cases'


def test_optimize_request_explicitly_freezes_suite_and_forbids_case_proposals():
    requests, records = [], []
    def complete(system, user, **kwargs):
        requests.append((system, user))
        return A3Completion('{"action":"finish","summary":"not implemented"}', 1, {})
    agent = OptimizerAgent(guide='native guide', complete=complete)
    agent.complete('optimize', {'stage': 'cases', 'baseline': {'cases': ['frozen-case']}},
                   lambda kind, value: records.append((kind, value)))
    system, user = requests[0]
    assert system.startswith('CURRENT STAGE: optimize.')
    assert 'ALREADY PREPARED AND FROZEN' in system
    assert 'Do NOT return action:"cases"' in system
    assert 'candidate, compile, query-knowledge, read-memory, read-artifact, or finish' in system
    assert 'First stage' not in system
    assert json.loads(user)['stage'] == 'optimize'
    assert json.loads(user)['current_stage_instruction'].startswith('CURRENT STAGE: optimize.')
    assert records[0][1]['messages'][0]['content'] == system


def test_unknown_stage_does_not_call_provider():
    agent = OptimizerAgent(guide='', complete=lambda *args, **kwargs: pytest.fail('provider called'))
    with pytest.raises(ValueError, match='unknown optimizer model stage'):
        agent.complete('unknown', {}, lambda *args: None)
