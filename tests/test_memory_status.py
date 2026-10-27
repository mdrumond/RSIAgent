import json

import pytest

from benchmarks.ascend.memory_admission import MemoryAdmission
from benchmarks.ascend.subject_campaign import SubjectCampaign
from benchmarks.ascend.subject_memory import SubjectMemory
from benchmarks.ascend.subjects import Subject
from test_memory_admission import Evaluation, draft, request


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


def test_campaign_reports_held_review_without_settling_or_publishing(tmp_path):
    memory = SubjectMemory(tmp_path/'memory', b'')
    evaluator = Evaluation(['inconclusive'])
    gate = MemoryAdmission(tmp_path/'gate', memory, evaluator, lambda *a, **k: pytest.fail('rewrote'))
    campaign = SubjectCampaign(tmp_path/'campaign', [Subject('s', 'subject', 1)], b'',
        select_wave=None, execute=None, verifier=None, complete=None, memory_gate=gate)
    try:
        gate.submit('p', draft(), {}, request(), remaining_seconds=100)
        gate.wait(remaining_seconds=5)
        before = campaign.scheduler.state()
        shown = campaign.report()['memory_verification']['p']
        assert shown['status'] == 'inconclusive'
        assert shown['assessment']['status'] == 'benefit-unproven'
        assert campaign.scheduler.state() == before
        assert gate.poll(remaining_seconds=5) == []
        assert memory.records() == campaign.memory.records() == []
        assert len(evaluator.calls) == 1
    finally:
        gate.close()


def test_source_repair_from_rewrite_is_not_a_quality_rejection(tmp_path):
    memory = SubjectMemory(tmp_path/'memory', b'')
    gate = MemoryAdmission(tmp_path/'gate', memory, Evaluation(['rewrite']),
        lambda *a, **k: dict(decision='repair', memory=None,
                            repair={'instructions': 'Fix the source'}, reason='Incorrect boundary'))
    try:
        gate.submit('p', draft(), {}, request(), remaining_seconds=100)
        gate.wait(remaining_seconds=5)
        result, = gate.poll(remaining_seconds=5)
        assert result['outcome'] == 'rejected'  # scheduler compatibility
        assert result['assessment']['status'] == 'needs-repair'
        assert result['repair']['reason'] == 'Incorrect boundary'
        assert memory.records() == []
    finally:
        gate.close()
