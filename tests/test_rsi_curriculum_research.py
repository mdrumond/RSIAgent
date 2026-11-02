"""Typed research contracts preserve the public role wire format."""
import pytest

from rsi.curriculum_research import parse_proposal
from rsi.roles import _proposal


def test_typed_role_wire_is_compatible_and_negative_reuse_needs_redirection():
    legacy = _proposal('curriculum', dict(rationale='Observed', evidence_ids=['failure'],
        projects=[dict(name='old', goal='goal', acceptance='oracle')]), {'failure'})
    assert legacy.projects[0].name == 'old'
    data = dict(rationale='Read docs', evidence_ids=['query'], proposal=dict(name='child', goal='goal',
        acceptance='oracle', template_id='cpu.case-subset', parameters={'case_indices': [0]}),
        negative_decisions=[dict(query_ref='query', index=0, action='used', reason='Change applicability')])
    with pytest.raises(ValueError):
        parse_proposal('diagnostic', data, {'query'})
    data['negative_decisions'][0]['redirection'] = 'Use a smaller isolated shape'
    assert _proposal('diagnostic', data, {'query'}).proposal.parameters == {'case_indices': [0]}
