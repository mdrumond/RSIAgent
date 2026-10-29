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


def test_partial_named_context_copy_is_atomically_healed_from_authenticated_value(tmp_path):
    files = ContextFiles(tmp_path/'context')
    value = dict(observed=[1, 2, 3])
    ref = files.put('observed', value)
    (files.root/ref['filename']).write_bytes(b'{"observed":')
    resumed = ContextFiles(files.root)
    assert resumed.put('observed', value) == ref
    assert (files.root/ref['filename']).read_bytes() == canonical(value)
    assert read_all(resumed.read, ref) == value
    resumed.artifacts._path(ref['sha256']).write_bytes(b'corrupt authoritative data')
    with pytest.raises(RuntimeError, match='collision'):
        resumed.put('observed', value)
