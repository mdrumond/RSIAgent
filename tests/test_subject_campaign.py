from dataclasses import dataclass
import hashlib
import json

import pytest

from benchmarks.ascend.subjects import Subject
from benchmarks.ascend.subject_campaign import SubjectCampaign
from benchmarks.ascend.subject_memory import SubjectMemory, StaleMemory
from benchmarks.ascend.subject_execution import SubjectOutcome


def proposal(source):
    return dict(operation='create', base_versions=[], summary='tail zeros',
        applicability='tail kernels', preconditions='padding present', expectations='padding zero',
        limitations='one dtype', before=None, before_unavailable_reason='first source',
        after=dict(sha256=source, start_line=1, end_line=1,text='source\n'),
        evidence=[source], citations=[])


class Verifier:
    last_trace = []
    def __init__(self):
        self.calls = []
    def review(self, request, *, max_seconds):
        self.calls.append(request)
        repair = len(self.calls) == 1
        return dict(decision='repair' if repair else 'accept', reason='needs control' if repair else 'control passed',
            evidence=[request.source_sha256], repair=dict(instructions='add control',expected_evidence='control result') if repair else None,
            memory=None if repair else proposal(request.source_sha256))


def test_repair_then_advance_and_resume_without_reexecution(tmp_path):
    calls, snapshots = [], []
    verifier = Verifier()
    def select(context, memory, history, **kwargs):
        pid = 'p' + str(len(calls))
        return [dict(project_id=pid, repairs=[r['repair_id'] for r in context['repairs']])]
    def execute(item, memory, directory, **kwargs):
        calls.append(item['project_id']); snapshots.append(memory)
        sha = campaign.artifacts.put_bytes(b'source\n')
        result = campaign.artifacts.put_json(dict(passed=True))
        return SubjectOutcome(True, sha, {'source':sha,'result':result}, {},campaign.artifacts.root)
    decisions = iter([dict(action='repair',repair_id='p0',reason='need control',evidence=['p0']),
                      dict(action='advance',reason='repaired',evidence=['p1'])])
    campaign = SubjectCampaign(tmp_path, [Subject('basics','learn tails',1)], b'initial',
        select_wave=select,execute=execute,verifier=verifier,
        complete=lambda *a,**kw:json.dumps(next(decisions)))
    result = campaign.run()
    assert result['terminal']=='COMPLETE'
    assert result['subjects']['repairs']==[]
    assert len(result['accepted_memory'])==1
    assert snapshots[0]==snapshots[1]  # rejected project never enters memory
    assert campaign.resume()['terminal']=='COMPLETE'
    assert calls==['p0','p1']


def test_catalog_keeps_prior_version_until_accept_and_refuses_stale_merge(tmp_path):
    store = SubjectMemory(tmp_path,b'initial')
    source = 'a'*64
    first = store.accept('p0',proposal(source),{})
    before = store.snapshot()
    replacement = dict(proposal(source),operation='append',base_versions=[first],summary='new control')
    assert store.snapshot()==before  # constructing a draft cannot publish it
    second = store.accept('p1',replacement,{})
    assert second!=first
    assert store.read(first['version'])['proposal']['summary']=='tail zeros'
    with pytest.raises(StaleMemory):
        store.accept('p2',replacement,{})
    assert len(store.records())==1
    assert store.search('control')[0]['version']==second['version']


def test_no_new_knowledge_can_accept_without_publishing(tmp_path):
    class NoMemory:
        last_trace=[]
        def review(self, request, **kw):
            return dict(decision='accept',reason='already known',evidence=[request.source_sha256],repair=None,memory=None)
    def execute(*args, **kwargs):
        sha=c.artifacts.put_bytes(b'source\n')
        return SubjectOutcome(True,sha,{'source':sha},{},c.artifacts.root)
    c=SubjectCampaign(tmp_path,[Subject('s','subject',1)],b'',
        select_wave=lambda *a,**k:[dict(project_id='p')],execute=execute,verifier=NoMemory(),
        complete=lambda *a,**k:json.dumps(dict(action='advance',reason='covered',evidence=['p'])))
    assert c.run()['terminal']=='COMPLETE'
    assert c.memory.records()==[]
