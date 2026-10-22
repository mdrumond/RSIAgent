import base64
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from benchmarks.ascend.context_files import ContextFiles
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent, CurriculumContext
from optimizer.jobs import canonical, sha
from test_a5_phase1_adaptive import _project, _memory, _wave


def read_all(reader, ref):
    chunks, offset = [], 0
    while True:
        page = reader(sha256=ref['sha256'], offset=offset, limit=8192)
        chunks.append(page['text'])
        offset = page['next_offset']
        if offset is None:
            return json.loads(''.join(chunks))


def test_context_file_paging_integrity_and_recreated_reader(tmp_path):
    value = {'measurements': [{'index': i, 'detail': 'x'*1000} for i in range(25)]}
    files = ContextFiles(tmp_path/'context')
    ref = files.put('failure-case-details', value)
    assert (files.root/ref['filename']).read_bytes() == canonical(value)
    assert read_all(files.read, ref) == value
    resumed = ContextFiles(files.root)
    assert resumed.put('failure-case-details', value) == ref
    assert read_all(resumed.read, ref) == value
    private = resumed.artifacts.put_json({'private_reference': 'not advertised'})
    with pytest.raises(ValueError):
        resumed.read(sha256=private)
    resumed.artifacts._path(ref['sha256']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        resumed.read(**ref['read_artifact'])


def test_coordinator_context_exposes_full_memory_without_feedback(tmp_path):
    runner = AdaptivePhase1Coordinator(tmp_path/'run', 'Learn.', b'{}',
        CurriculumAgent(lambda _: ''), lambda _: None, max_parallel=1, project_target=2)
    state = runner._load_state()
    spec = ProjectSpec.from_mapping(_project('prior', 'gm-ub-movement'))
    memory = _memory(spec).as_dict()
    memory['host_facts'][0]['statement'] = 'recorded fact '*2000
    state.update(prior_outcomes=[dict(project_id='prior', spec=spec.as_dict(), verified=True, summary='passed')],
                 memory_bytes_base64=[base64.b64encode(canonical(memory)).decode()])
    context = runner._context(state, runner._memory_snapshot(state))
    assert runner.failure_feedback is None
    assert read_all(context.artifact_reader, context.context_files[0]) == memory
    calls = []
    def complete(messages):
        calls.append(messages)
        if len(calls) == 1:
            return json.dumps({'read_artifact': context.context_files[0]['read_artifact']})
        assert json.loads(messages[-1]['content'])['text'] == canonical(memory)[:8192].decode()
        return 'DECISION: SATURATED\n{"reason":"learned from existing work"}'
    _, session = CurriculumAgent(complete).select(context)
    assert session.turns == 2 and len(calls) == 2


def test_failure_context_preserves_every_row_in_readable_file(tmp_path):
    from test_failure_curriculum import setup
    _, _, feedback = setup(tmp_path)
    cases = [dict(case_id=f'case-{i}', classification='host-mismatch', detail='y'*1000,
                  artifacts={'inputs': {'sha256': sha(str(i).encode()), 'retained_path': f'/retained/{i}'}})
             for i in range(30)]
    feedback.store.failure_cases = lambda _: cases
    presentation = feedback.context({})
    ref = presentation[-1]['selected'][0]['failure_cases_file']
    assert read_all(feedback.read, ref) == cases
    assert len(presentation[-1]['selected']) == 1


def test_multiple_reads_keep_full_audit_but_only_latest_page_on_wire(tmp_path):
    files = ContextFiles(tmp_path/'files')
    ref = files.put('large-observation', {'rows':['z'*1000 for _ in range(25)]})
    requests, pages = [], []
    def complete(messages):
        requests.append(messages)
        if len(requests) <= 2:
            return json.dumps({'read_artifact': dict(ref['read_artifact'], offset=(len(requests)-1)*8192)})
        observations = [json.loads(row['content']) for row in messages if row['role']=='user' and row['content'].startswith('{')]
        pages.extend(row for row in observations if row.get('event')=='artifact-page')
        return _wave(messages[1]['content'], [_project('new', 'gm-ub-movement')])
    context = CurriculumContext('Learn.', 1, 'a'*64, (), (), context_files=(ref,), artifact_reader=files.read)
    _, session = CurriculumAgent(complete).select(context)
    assert len(pages) == 2 and 'text' not in pages[0] and pages[1]['text']
    audit = [json.loads(row['content']) for row in session.messages
             if row['role']=='user' and row['content'].startswith('{')]
    assert sum(bool(row.get('text')) for row in audit) == 2
    assert session.turns == 3


def test_native_phase2_reads_and_more_than_three_corrections_share_time(tmp_path):
    from benchmarks.ascend.phase2_target_selector import TargetCurriculumAgent
    from test_ascend_phase2_targets import qualified, context as target_context
    target, _, _ = qualified(tmp_path/'reference')
    files = ContextFiles(tmp_path/'context')
    ref = files.put('observed-failure', {'actual':'mismatch', 'details':['a','b']})
    context = replace(target_context(target), remaining_seconds=20, context_files=(ref,), artifact_reader=files.read)
    calls, now = [], [0]
    def complete(messages):
        calls.append(messages)
        now[0] += 1
        if len(calls)==1:
            return json.dumps({'read_artifact':ref['read_artifact']})
        if len(calls)<6:
            return 'DECISION: READY\n{"reason":"incorrect proposal"}'
        return 'DECISION: RETRY_TARGET\n{"reason":"apply observed repair"}'
    decision, session = TargetCurriculumAgent(complete, clock=lambda:now[0]).select(context)
    assert decision.action=='retry-target' and len(calls)==6 and session.turns==6
    assert json.loads(calls[1][-1]['content'])['text'] == canonical({'actual':'mismatch','details':['a','b']}).decode()
    now[0]=0
    with pytest.raises(ValueError, match='elapsed-time budget exhausted'):
        TargetCurriculumAgent(complete, clock=lambda:now[0]).select(replace(context, remaining_seconds=.5))


@pytest.mark.parametrize('native', [False, True])
def test_read_audit_survives_provider_interruption(tmp_path, native):
    files = ContextFiles(tmp_path/'context')
    ref = files.put('observed-case', {'expected': [3, 7], 'observed': [0, 0]})
    calls = []
    def complete(messages):
        calls.append(messages)
        if len(calls) == 1:
            return json.dumps({'read_artifact': ref['read_artifact']})
        raise OSError('provider disconnected')
    if native:
        from benchmarks.ascend.phase2_target_selector import TargetCurriculumAgent
        from test_ascend_phase2_targets import qualified, context as target_context
        target, _, _ = qualified(tmp_path/'reference')
        context = replace(target_context(target), context_files=(ref,), artifact_reader=files.read)
        agent = TargetCurriculumAgent(complete)
    else:
        context = CurriculumContext('Learn.', 1, 'a'*64, (), (),
            context_files=(ref,), artifact_reader=files.read)
        agent = CurriculumAgent(complete)
    with pytest.raises(OSError, match='provider disconnected'):
        agent.select(context)
    pages = [json.loads(line) for line in (files.root/'read-pages.jsonl').read_bytes().splitlines()]
    assert len(pages) == 1
    assert json.loads(pages[0]['text']) == {'expected': [3, 7], 'observed': [0, 0]}
    now = [0]
    def read_then_expire(messages):
        return json.dumps({'read_artifact': ref['read_artifact']})
    def read_and_expire(**payload):
        page = files.read(**payload)
        now[0] += 1
        return page
    agent = (TargetCurriculumAgent if native else CurriculumAgent)(
        read_then_expire, clock=lambda: now[0])
    with pytest.raises(ValueError, match='budget exhausted'):
        agent.select(replace(context, remaining_seconds=.5, artifact_reader=read_and_expire))
    pages = [json.loads(line) for line in (files.root/'read-pages.jsonl').read_bytes().splitlines()]
    assert len(pages) == 2 and pages[1] == pages[0]


def test_optimizer_context_pages_resume_without_resending_prior_page_bodies(tmp_path):
    from test_ascend_phase2_optimizer_tasks import seed, proposal
    from benchmarks.ascend.phase2_optimizer_tasks import select_optimizer_task
    memory = {'task_id':'prior-knowledge', 'statement':'m'*24000, 'interpretation':'recorded'}
    snapshot = seed([memory])
    calls = []
    def interrupted(messages):
        calls.append(messages)
        if len(calls)==3:
            raise OSError('provider unavailable')
        presentation = json.loads(messages[1]['content'])['memory']
        ref = presentation['projects'][0]
        return json.dumps({'read_artifact':dict(sha256=ref['sha256'], offset=(len(calls)-1)*8192, limit=8192)})
    options = dict(snapshot=snapshot, objective='Learn.', root=tmp_path/'selection', remaining_seconds=lambda:100)
    with pytest.raises(OSError):
        select_optimizer_task(complete=interrupted, **options)
    audit = json.loads((options['root']/'selection.json').read_bytes())['messages']
    assert sum('text' in json.loads(row['content']) for row in audit[2:]
               if row['role']=='user')==2
    resumed = []
    selected = select_optimizer_task(complete=lambda messages:(resumed.append(messages), json.dumps(proposal()))[1], **options)
    assert selected['task_id']==proposal()['task_id']
    pages = [json.loads(row['content']) for row in resumed[0][2:] if row['role']=='user']
    assert len(pages)==2 and 'text' not in pages[0] and pages[1]['text']


def test_large_context_indexes_every_memory_without_inlining_all_references(tmp_path):
    runner = AdaptivePhase1Coordinator(tmp_path/'run', 'Learn.', b'{}',
        CurriculumAgent(lambda _: ''), lambda _: None, max_parallel=1, project_target=20)
    entries = [dict(project_spec={'project_id':f'project-{i}'}, observed={'index':i}) for i in range(20)]
    state = runner._load_state()
    state['memory_bytes_base64'] = [base64.b64encode(canonical(row)).decode() for row in entries]
    refs, reader = runner._context_files(state)
    assert len(refs) == 5
    index = read_all(reader, refs[0])
    assert len(index['files']) == 20
    assert read_all(reader, index['files'][0]) == entries[0]
    assert read_all(reader, index['files'][-1]) == entries[-1]


def test_context_files_uses_authenticated_snapshot_without_seed_directory_lookup(tmp_path):
    runner = AdaptivePhase1Coordinator(tmp_path/'run', 'Learn.', b'{}',
        CurriculumAgent(lambda _: ''), lambda _: None, max_parallel=1, project_target=1)
    entry = dict(project_spec={'project_id':'reviewed-seed'}, host_facts=[{'observed':True}])
    runner.snapshot = SimpleNamespace(verify=lambda:None, memory_entries=lambda:[entry])
    refs, reader = runner._context_files(runner._load_state())
    assert read_all(reader, refs[0]) == entry
