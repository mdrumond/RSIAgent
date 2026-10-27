import pytest
from optimizer.agent import OptimizerAgent


class Session:
    def __init__(self, **kwargs):
        self.contexts = []
    def next_action(self, context, allowed, **kwargs):
        self.contexts.append((context, allowed))
        if context['stage'] == 'cases':
            return 'cases', {'signatures': [], 'summary': 'bounded'}
        return 'finish', {'summary': 'done'}
    def close(self): pass


@pytest.mark.parametrize('remaining', [0, 2])
def test_cases_stage_declares_only_cases_and_preserves_remaining_limit(remaining):
    agent = OptimizerAgent(guide='native guide', session_factory=Session)
    result = agent.complete('cases', {'max_cases': 3, 'remaining_signatures': remaining}, lambda *args: None)
    context, allowed = agent.session.contexts[0]
    assert allowed == ['cases']
    assert context['remaining_signatures'] == remaining
    assert result['action'] == 'cases'
    assert agent.identity['coding_engine'] == 'pi'
    assert agent.identity['model'] == 'deepseek-flash'


def test_optimizer_preserves_session_and_frozen_suite():
    agent = OptimizerAgent(guide='guide', session_factory=Session)
    agent.complete('cases', {'max_cases':3, 'remaining_signatures':0}, lambda *args: None)
    first = agent.session
    agent.complete('optimize', {'stage':'cases','baseline': {'cases':['frozen']}}, lambda *args: None)
    assert first is agent.session
    context, allowed = first.contexts[1]
    assert 'cases' not in allowed and 'compile' in allowed
    assert context['stage'] == 'optimize'
    assert context['baseline'] == {'cases':['frozen']}


def test_unknown_stage_does_not_start_pi():
    agent = OptimizerAgent(guide='', session_factory=lambda **kwargs: pytest.fail('started Pi'))
    with pytest.raises(ValueError, match='unknown optimizer model stage'):
        agent.complete('unknown', {}, lambda *args: None)
