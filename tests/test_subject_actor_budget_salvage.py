"""Finished independent projects remain admissible after actor work expires."""
import json

from benchmarks.ascend.actor_control import grant_elapsed
from benchmarks.ascend.subject_campaign import SubjectCampaign
from benchmarks.ascend.subject_execution import SubjectOutcome
from benchmarks.ascend.subjects import Subject


def campaign_at_deadline(root, *, linked=False, reject=False):
    now, executed, reviewed = [0.0], [], []

    class Verifier:
        last_trace = []

        def review(self, request, **kwargs):
            reviewed.append(request.project_id)
            now[0] += 1000  # Source verification is outside the actor clock.
            memory = dict(operation='create', base_versions=[], summary='tail zeros',
                applicability='tail kernels', preconditions='padding present',
                expectations='padding zero', limitations='one dtype', before=None,
                before_unavailable_reason='first source',
                after=dict(sha256=request.source_sha256, start_line=1, end_line=1, text='source\n'),
                evidence=[request.source_sha256], citations=[])
            return dict(decision='repair' if reject else 'accept', reason='source evidence',
                evidence=[request.source_sha256], memory=None if reject else memory,
                repair=dict(instructions='fix source', expected_evidence='correct kernel') if reject else None)

    def execute(item, *args, **kwargs):
        executed.append(item['project_id'])
        now[0] += 10 if len(executed) == 1 else 1
        sha = campaign.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True, sha, {'source': sha}, {}, campaign.artifacts.root)

    campaign = SubjectCampaign(root, [Subject('s', 'learn tails', 1)], b'',
        select_wave=lambda *a, **k: [dict(project_id=pid, repairs=['failed'] if linked else [])
                                    for pid in ['one', 'two']],
        execute=execute, verifier=Verifier(), max_seconds=10, clock=lambda: now[0],
        complete=lambda *a, **k: json.dumps(dict(action='advance', reason='finished', evidence=['one', 'two'])))
    if linked:
        campaign.scheduler.add_repair('failed', 's', 'failed', 'requires both controls', ['original'])
    return campaign, executed, reviewed


def test_expired_wave_publishes_independent_result_and_resumes_without_duplicates(tmp_path):
    campaign, executed, reviewed = campaign_at_deadline(tmp_path)
    report = campaign.run()
    assert report['terminal'] == 'TIME_BUDGET_EXHAUSTED'
    assert executed == reviewed == ['one']
    assert len(report['accepted_memory']) == 1
    assert campaign.scheduler.state()['waves']['wave-00001']['status'] == 'running'
    assert report['timing']['elapsed_seconds'] == 10
    campaign.resume()
    assert executed == reviewed == ['one']
    grant_elapsed(tmp_path / 'elapsed.json', 'finish-second', 1)
    report = campaign.resume()
    assert report['terminal'] == 'COMPLETE'
    assert executed == reviewed == ['one', 'two']
    assert len(report['accepted_memory']) == 2
    assert report['timing']['elapsed_seconds'] == 11
    assert all(row['status'] == 'accepted' for row in campaign.scheduler.state()['reviews'].values())


def test_expired_wave_does_not_review_a_probe_without_its_linked_control(tmp_path):
    campaign, executed, reviewed = campaign_at_deadline(tmp_path, linked=True)
    report = campaign.run()
    assert report['terminal'] == 'TIME_BUDGET_EXHAUSTED'
    assert executed == ['one'] and reviewed == []
    assert report['accepted_memory'] == []
    assert campaign.scheduler.state()['repairs']['failed']['status'] == 'open'


def test_partial_source_rejection_still_imposes_repair_at_actor_deadline(tmp_path):
    campaign, executed, reviewed = campaign_at_deadline(tmp_path, reject=True)
    report = campaign.run()
    assert report['terminal'] == 'TIME_BUDGET_EXHAUSTED'
    assert executed == reviewed == ['one']
    assert report['accepted_memory'] == []
    assert campaign.scheduler.state()['repairs']['one']['status'] == 'open'


def test_partial_salvage_does_not_reregister_already_published_manual_review(tmp_path):
    campaign, executed, reviewed = campaign_at_deadline(tmp_path)
    report = campaign.run()
    record = campaign.state['projects']['one']
    record.pop('partial_published')
    prior = record.pop('partial_review_id')
    record.update(manual_review_id=prior, manual_request='retained-manual')
    before = campaign.scheduler.state()['reviews']
    campaign.resume()
    assert campaign.scheduler.state()['reviews'] == before
    assert reviewed == ['one'] and len(campaign.memory.records()) == 1
