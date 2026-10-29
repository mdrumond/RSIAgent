"""Native adaptive references remain usable without a curriculum handshake."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json

from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_provider import direct_profile, DirectCompletion
from benchmarks.ascend import subject_runtime
from benchmarks.ascend.project_verifier import ProjectReview, ProjectVerifierAgent
from benchmarks.ascend.production import build_native_adaptive_phase1
from benchmarks.ascend.subject_runtime import build_subject_campaign
from benchmarks.ascend.subjects import Subject
from test_phase1_adaptive_kdb import config, completion, actor_context, Embeddings, NativeTransport
from test_a5_phase1_adaptive import _project, _wave
from test_a5_phase1_curriculum_selector import context as selection_context
from test_subject_runtime import selection_context as subject_selection_context


def composed_campaign(tmp_path, *, transport=None):
    cfg, calls, bindings = config(tmp_path), [], []
    profile = direct_profile(MODEL_IDENTITIES[0])
    def compose(settings, **options):
        native = build_native_adaptive_phase1(config=replace(cfg, root=options['root']/'binding'),
            native_transport=NativeTransport(), profile=profile, guide_bytes=b'Native guide',
            objective='Native reference binding', initial_memory=b'',
            environment={'OPENAI_API_KEY':'fixture', 'DEEPSEEK_API_KEY':'fixture'},
            provider_transport=transport or (lambda **kw: DirectCompletion('{}', 1)),
            embeddings=Embeddings(), librarian_complete=completion(calls))
        bindings.append(native)
        return dict(coordinator=native, actor=native.execute_project,
            provider=native.execute_project.actor.provider, profile=profile,
            paths=replace(cfg, root=options['root']), embeddings=Embeddings())
    worker=tmp_path/'worker.json'
    worker.write_text(json.dumps(dict(execution='local', workspace=str(tmp_path/'device'), physical_device=0)))
    campaign=build_subject_campaign(dict(verifier_engine='completion',
        phase1_arguments=['--worker-config',str(worker)]), target='a3', root=tmp_path/'subject',
        subjects=[Subject('copy','Find useful native copy evidence')], initial_memory=b'', compose=compose)
    return campaign, calls, bindings


def packets(calls):
    return [json.loads(messages[1]['content']) for messages,_ in calls]


def test_fresh_child_actor_queries_before_any_curriculum_selection(tmp_path, monkeypatch):
    class Execution:
        def __init__(self, actor, *args):self.actor=actor
        def graph(self, request, **kwargs):
            action=actor_context(request.branch_root)
            action=replace(action,request=request)
            return self.actor.query_knowledge({'query':'pipeline','limit':1},action)
    monkeypatch.setattr(subject_runtime,'SubjectExecutor',Execution)
    campaign,calls,bindings=composed_campaign(tmp_path)
    item=dict(project_id='transfer',spec=_project('transfer','gm-ub-movement'))
    first=campaign.execute(item,campaign.memory.snapshot(),tmp_path/'first-project',remaining_seconds=30)
    second=campaign.execute(item,campaign.memory.snapshot(),tmp_path/'second-project',remaining_seconds=30)
    assert first.citations[0].collection==second.citations[0].collection=='refs'
    assert [p['role'] for p in packets(calls)]==['writer','writer']
    assert all(p['context']['project']['project_id']=='transfer' for p in packets(calls))
    assert all(seconds<=30 for _,seconds in calls)
    roots=list(tmp_path.glob('*/reference-librarian/librarian-budget.json'))
    assert len(roots)==2 and roots[0]!=roots[1]
    assert len(bindings)==1


def test_source_and_rewrite_reviews_bind_independent_request_contexts(tmp_path, monkeypatch):
    campaign,calls,bindings=composed_campaign(tmp_path)
    # A previous curriculum selection must not become either verifier's task.
    bindings[0].curriculum.query_knowledge.begin_selection(selection_context())
    def review(self,request,**options):
        result=self.query_knowledge({'query':'pipeline','limit':1},remaining_seconds=20)
        return dict(project=request.project_id,citations=result['citations'])
    monkeypatch.setattr(ProjectVerifierAgent,'review',review)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(campaign.verifier.review,ProjectReview('source',None,False,{}),max_seconds=20)
        b=pool.submit(campaign.verifier.review,ProjectReview('rewrite',None,False,{}),feedback={},max_seconds=20)
        assert a.result()['citations'] and b.result()['citations']
    received=packets(calls)
    assert {p['context']['project_id'] for p in received}=={'source','rewrite'}
    assert all(p['role']=='verifier' and p['task'].endswith(p['context']['project_id']) for p in received)
    assert len(list((campaign.root/'verifier-librarian').glob('*/librarian-budget.json')))==2


def test_subject_curriculum_uses_its_current_objective_not_native_selection_state(tmp_path):
    decisions=[]
    def transport(**request):
        messages=request['request']['messages'];decisions.append(messages)
        if not any(m['role']=='assistant' for m in messages):
            return DirectCompletion('{"knowledge_query":{"query":"pipeline","limit":1}}',1)
        return DirectCompletion(_wave(json.dumps(subject_selection_context(messages)),[_project('fresh','gm-ub-movement')]),1)
    campaign,calls,bindings=composed_campaign(tmp_path,transport=transport)
    bindings[0].curriculum.query_knowledge.begin_selection(selection_context())
    for objective in ['Copy odd tails','Copy padded rows']:
        context=campaign.scheduler.context()
        context['current_subject']['objective']=objective
        assert campaign.select_wave(context,campaign.memory.snapshot(),{},remaining_seconds=30)[0]['project_id']=='fresh'
    received=packets(calls)
    assert [p['role'] for p in received]==['curriculum','curriculum']
    assert received[0]['task'].startswith('Copy odd tails') and received[1]['task'].startswith('Copy padded rows')
    assert len(list((campaign.root/'curriculum-librarian').glob('*/librarian-budget.json')))==2


def test_contextual_references_preserve_budget_and_pinned_collection_guards(tmp_path):
    import pytest
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    from benchmarks.a5kernels.knowledge_agent import KnowledgeQuery
    campaign,calls,bindings=composed_campaign(tmp_path)
    query=bindings[0].contextual_reference_query
    options=dict(root=tmp_path/'review-context',role='verifier',task='Review exact copy source',context={})
    assert query(KnowledgeQuery('pipeline',1),check_remaining=lambda:10,**options)
    path=options['root']/'librarian-budget.json'
    budget=json.loads(path.read_text());budget['elapsed_seconds']=90;path.write_text(json.dumps(budget))
    assert not query(KnowledgeQuery('pipeline',1),check_remaining=lambda:10,**options)
    assert len(calls)==1
    def expired():raise TimeoutError('caller expired')
    with pytest.raises(TimeoutError,match='caller expired'):
        query(KnowledgeQuery('pipeline',1),check_remaining=expired,**dict(options,root=tmp_path/'other'))
    assert len(calls)==1
    changed=tmp_path/'changed.cpp';changed.write_text('new pipeline source\n')
    (tmp_path/'references.sqlite3').unlink()
    with KnowledgeDB(tmp_path/'references.sqlite3',Embeddings()) as database:
        database.index(tmp_path,[changed],collection='refs',language='ascend-c')
    with pytest.raises(ValueError,match='identity changed'):
        query(KnowledgeQuery('pipeline',1),check_remaining=lambda:10,**dict(options,root=tmp_path/'fresh'))
    assert len(calls)==1
