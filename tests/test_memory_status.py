import json

import pytest

from benchmarks.ascend.subject_campaign import SubjectCampaign
from benchmarks.ascend.subject_memory import SubjectMemory
from benchmarks.ascend.subjects import Subject


@pytest.mark.parametrize('repair', [False, True])
def test_source_audit_failure_and_semantic_repair_stay_distinct(tmp_path, repair):
    class Verifier:
        last_trace = []
        def review(self, *args, **kwargs):
            if not repair:
                raise TimeoutError('source review allowance expired')
            return dict(decision='repair', reason='Unsupported input domain', evidence=[],
                        repair={'instructions': 'Restrict the claim', 'expected_evidence': 'Boundary cases'},
                        memory=None)

    campaign = SubjectCampaign(tmp_path, [Subject('s', 'subject', 1)], b'',
        select_wave=None, execute=None, verifier=Verifier(), complete=None)
    campaign._remaining = lambda: 60
    sha = campaign.artifacts.put_bytes(b'kernel source')
    record = dict(objective='subject', outcome=dict(source_sha256=sha, host_verified=True,
                  artifacts={'source': sha}, observations={}))
    campaign.state['projects']['p'] = record
    if repair:
        campaign._review('p', record)
    else:
        with pytest.raises(TimeoutError):
            campaign._review('p', record)
    saved = json.loads(campaign.path.read_bytes())['projects']['p']
    shown = campaign.report()['projects']['p']
    expected = 'needs-repair' if repair else 'audit-incomplete'
    assert saved['assessment']['status'] == shown['assessment']['status'] == expected
    assert campaign.memory.records() == []
    assert not repair or saved['review']['decision'] == 'repair'
    assert repair or saved.get('review') is None
