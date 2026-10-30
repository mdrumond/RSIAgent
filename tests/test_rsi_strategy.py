"""Behavioral strategy evidence tests; no live provider or device required."""
import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.knowledge import KnowledgeDB
from rsi.contracts import Stage
from rsi.runtime import WorkflowError
from rsi.skills import SkillUsage, builtin_bundle
from rsi.strategy import QueryUnavailable, StrategyEvidence, knowledge_db_query


class TestEmbedding:
    model, revision = 'TEST', '1'

    def embed(self, texts):
        return [[1., float('stable' in text)] for text in texts]


@pytest.fixture
def setup(tmp_path):
    bundle = builtin_bundle()
    state = dict(stage='design', source='v1', pending={'id': 'op1'}, history=[
        dict(passed=False, evidence={'stage': 'test', 'artifact_refs': ['numeric-error']}),
        dict(passed=False, redesign=True, new_source_hash='v1')])
    log = []
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    database = KnowledgeDB(tmp_path / 'knowledge.sqlite3', TestEmbedding())
    source = tmp_path / 'stable.py'
    source.write_text('def stable_sum(values):\n    return sum(values)\n')
    database.index(tmp_path, [source], collection='examples', language='python', revision='abc')
    negative = dict(record_id='failure1', outcome='numeric-failure',
                    reason='unstable accumulation', provenance={'artifact': 'retained-failure'})
    callback = knowledge_db_query(database, ('examples',), negative_query=lambda p: [negative])

    def make(query=callback, events=None, selected_bundle=bundle):
        return StrategyEvidence(store, log.append, lambda: state, selected_bundle, query,
                                workflow_id='project1', backend_identity='TEST@1',
                                test_backend=True, events=log if events is None else events)

    service = make()
    usage = SkillUsage(bundle, 'op1', Stage.DESIGN, service.record_skill_event)
    usage.expose()
    exposure = log[-1]['reference']
    usage.read('design')
    skill = service.read_receipt(usage.last_read_receipt['receipt_id'])
    usage.read('kdb')
    yield service, make, state, log, store, skill, exposure
    database.close()


def prepare(service, skill):
    query = service.query({'query': 'stable numeric sum'})
    retro = service.retrospect(supplement='Try a more stable algorithm')
    return query, retro


def decisions(service, query, *, reuse=False):
    rows = service.artifacts.read_json(query)['payload']['response']
    result = [dict(group='results', index=i, action='used', reason='reference implementation')
              for i in range(len(rows['results']))]
    result += [dict(group='negative_records', index=i,
                   action='used' if reuse else 'rejected', reason='observed numeric failure',
                   **({'redirection': 'Use compensated accumulation instead'} if reuse else {}))
               for i in range(len(rows['negative_records']))]
    return result


def select(service, skill, query, retro, choices=None):
    return service.select('Use compensated sum', query_ref=query, retrospective_ref=retro,
                          skill_refs=list(dict.fromkeys([skill, *service._current_reads(service._context())])), decisions=choices if choices is not None
                          else decisions(service, query))


def test_real_query_retains_provenance_and_actual_failure_history(setup):
    service, _, state, _, store, skill, _ = setup
    query, retro = prepare(service, skill)
    payload = store.read_json(query)['payload']
    assert payload['response']['results'][0]['path'] == 'stable.py'
    assert payload['response']['provenance']['collections']['examples']['revision'] == 'abc'
    actual = store.read_json(retro)['payload']
    assert actual['history'] == state['history']
    assert actual['failures'] == [0, 1]
    assert actual['redirections'] == [1]
    assert not service.resolve('retrieval', query, Stage.DESIGN, 'op1')
    selection = select(service, skill, query, retro)
    assert store.read_json(selection)['payload']['decisions'][-1]['action'] == 'rejected'
    assert service.resolve('retrieval', query, Stage.DESIGN, 'op1')
    assert service.resolve('retrospective', retro, Stage.DESIGN, 'op1')


def test_negative_reuse_requires_changed_applicability(setup):
    service, _, _, _, store, skill, _ = setup
    query, retro = prepare(service, skill)
    choices = decisions(service, query, reuse=True)
    del choices[-1]['redirection']
    with pytest.raises(WorkflowError):
        select(service, skill, query, retro, choices)
    choices[-1]['redirection'] = 'Replace the failed accumulation with compensation'
    ref = select(service, skill, query, retro, choices)
    assert store.read_json(ref)['payload']['decisions'][-1]['action'] == 'used'


def test_restart_requires_host_log_and_rejects_cas_only_forgery(setup):
    service, make, _, log, store, skill, _ = setup
    query, retro = prepare(service, skill)
    select(service, skill, query, retro)
    restarted = make()
    assert restarted.resolve('skill', skill, Stage.DESIGN, 'op1')
    assert restarted.resolve('retrieval', query, Stage.DESIGN, 'op1')
    forged = store.read_json(query)
    forged['payload']['request']['query'] = 'invented query'
    fake_ref = store.put_json(forged)
    assert not restarted.resolve('retrieval', fake_ref, Stage.DESIGN, 'op1')
    assert not make(events=[]).resolve('skill', skill, Stage.DESIGN, 'op1')
    assert len(log) > 0


@pytest.mark.parametrize('change', ['source', 'history', 'operation', 'stage', 'bundle'])
def test_receipts_bound_to_frozen_context(setup, change):
    service, make, state, _, _, skill, _ = setup
    query, retro = prepare(service, skill)
    select(service, skill, query, retro)
    if change == 'source':
        state['source'] = 'v2'
    elif change == 'history':
        state['history'].append({'passed': True})
    elif change == 'operation':
        state['pending']['id'] = 'op2'
    elif change == 'stage':
        state['stage'] = 'optimisation'
    else:
        from dataclasses import replace
        bundle = service.bundle
        doc = bundle.documents[0]
        bundle = replace(bundle, documents=(replace(doc, content=doc.content + '\nUpdated'),
                                           *bundle.documents[1:]))
        service = make(selected_bundle=bundle)
    assert not service.resolve('retrieval', query, state['stage'], state['pending']['id'])
    assert not service.resolve('skill', skill, state['stage'], state['pending']['id'])


def test_exposure_and_omitted_negative_cannot_authorize_selection(setup):
    service, _, _, _, _, skill, exposure = setup
    query, retro = prepare(service, skill)
    with pytest.raises(WorkflowError):
        select(service, exposure, query, retro)
    with pytest.raises(WorkflowError):
        select(service, skill, query, retro, decisions(service, query)[:-1])
    assert not service.resolve('skill', exposure, Stage.DESIGN, 'op1')


def test_empty_query_valid_unavailability_suspends(setup):
    _, make, _, log, _, skill, _ = setup
    empty = make(query=lambda p: dict(results=[], negative_records=[], provenance={'snapshot': 'empty'}))
    query, retro = prepare(empty, skill)
    select(empty, skill, query, retro, [])
    assert empty.resolve('retrieval', query, Stage.DESIGN, 'op1')

    def unavailable(payload):
        raise TimeoutError('backend timed out')

    failed = make(query=unavailable)
    before = len(log)
    with pytest.raises(QueryUnavailable):
        failed.query({'query': 'stable'})
    assert log[before:][0]['kind'] == 'query-unavailable'
    assert not failed.resolve('retrieval', log[-1]['reference'], Stage.DESIGN, 'op1')


def test_selection_requires_real_retrospective_and_query(setup):
    service, _, _, _, _, skill, _ = setup
    query, retro = prepare(service, skill)
    with pytest.raises(WorkflowError):
        select(service, skill, '0' * 64, retro, decisions(service, query))
    with pytest.raises(WorkflowError):
        select(service, skill, query, '0' * 64)


def test_failed_log_append_grants_no_authority(setup):
    service, _, _, _, store, skill, _ = setup
    query, retro = prepare(service, skill)
    def fail(event):
        raise OSError('log unavailable')
    service.emit = fail
    count = len(service.receipts)
    with pytest.raises(OSError):
        select(service, skill, query, retro)
    assert len(service.receipts) == count
    assert not service.resolve('retrieval', query, Stage.DESIGN, 'op1')


@pytest.mark.parametrize('test_backend', [False, True])
def test_backend_identity_not_interpreted_and_explicit_kind_retained(setup, test_backend):
    service, _, state, log, store, _, _ = setup
    configured = StrategyEvidence(store, log.append, lambda: state, service.bundle,
                                  service.query_backend, workflow_id='project1',
                                  backend_identity='latest', test_backend=test_backend)
    ref = configured.query({'query': 'stable sum'})
    receipt = store.read_json(ref)['payload']
    assert receipt['backend'] == 'latest'
    assert receipt['test_backend'] is test_backend


def test_kdb_alone_cannot_complete_or_select_and_restart_preserves_gate(setup):
    service, make, state, _, _, _, _ = setup
    state['pending']['id'] = 'op2'
    usage = SkillUsage(service.bundle, 'op2', Stage.DESIGN, service.record_skill_event)
    usage.expose()
    usage.read('kdb')
    kdb = service.read_receipt(usage.last_read_receipt['receipt_id'])
    query, retro = prepare(service, kdb)
    assert not service.resolve('skill', kdb, Stage.DESIGN, 'op2')
    assert not make().resolve('skill', kdb, Stage.DESIGN, 'op2')
    with pytest.raises(WorkflowError):
        service.select('Try stable accumulation', query_ref=query, retrospective_ref=retro,
                       skill_refs=[kdb], decisions=decisions(service, query))
    usage.read('design')
    design = service.read_receipt(usage.last_read_receipt['receipt_id'])
    assert service.resolve('skill', kdb, Stage.DESIGN, 'op2')
    assert make().resolve('skill', design, Stage.DESIGN, 'op2')
    # Host has both reads, but selection must explicitly cite the complete set.
    with pytest.raises(WorkflowError):
        service.select('Try stable accumulation', query_ref=query, retrospective_ref=retro,
                       skill_refs=[kdb], decisions=decisions(service, query))
    service.select('Try stable accumulation', query_ref=query, retrospective_ref=retro,
                   skill_refs=[kdb, design], decisions=decisions(service, query))
    assert service.resolve('retrieval', query, Stage.DESIGN, 'op2')


def test_runtime_completion_requires_all_mapped_reads(setup, tmp_path):
    from rsi.contracts import StepEvidence
    from rsi.runtime import Workflow
    service, _, state, _, _, _, _ = setup
    workflow = Workflow(tmp_path / 'workflow.json', service.resolve, source_hash='v1')
    workflow.state.update(state, pending={'id': 'completion-op'})
    service.snapshot = lambda: workflow.state
    usage = SkillUsage(service.bundle, 'completion-op', Stage.DESIGN, service.record_skill_event)
    usage.expose()
    usage.read('kdb')
    kdb = service.read_receipt(usage.last_read_receipt['receipt_id'])
    query, retro = prepare(service, kdb)
    evidence = StepEvidence(Stage.DESIGN, 'completion-op', skill_refs=(kdb,),
                            retrieval_refs=(query,), retrospective_ref=retro, source_hash='v1')
    with pytest.raises(WorkflowError):
        workflow.complete(evidence)
    assert workflow.stage == Stage.DESIGN
    usage.read('design')
    design = service.read_receipt(usage.last_read_receipt['receipt_id'])
    service.select('Stable accumulation', query_ref=query, retrospective_ref=retro,
                   skill_refs=[design, kdb], decisions=decisions(service, query))
    workflow.complete(evidence)
    assert workflow.stage == Stage.IMPLEMENTATION
