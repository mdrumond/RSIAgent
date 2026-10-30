"""Default public constructors, actual canonical stages, CPU-only host runtimes."""
import json
from types import SimpleNamespace

import pytest

from rsi.contracts import Stage
import rsi.domain_bindings as bindings


@pytest.fixture(autouse=True)
def declared_cpu_fixture_backend(monkeypatch):
    # These unit models/hosts are scripted CPU fixtures, never production evidence.
    from rsi.domain_execution import DomainRunner
    original = DomainRunner.create.__func__
    original_init = DomainRunner.__init__
    def init(self, *args, **kwargs):
        return original_init(self, *args, **dict(kwargs, test_backend=True))
    monkeypatch.setattr(DomainRunner, '__init__', init)
    def create(cls, *args, **kwargs):
        return original(cls, *args, **dict(kwargs, test_backend=True))
    monkeypatch.setattr(DomainRunner, 'create', classmethod(create))


def empty_cpu_knowledge(*_):
    from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActionResult
    return AdaptiveActionResult(dict(available=True, results=[]))
empty_cpu_knowledge.canonical_identity = dict(test_collection='CPU', manifest='empty-v1')


class CanonicalSession:
    def __init__(self, *, initial='bad', repaired='fixed source'):
        self.initial, self.repaired, self.contexts = initial, repaired, []
    def prepare(self, timeout, *, schemas):
        self.schemas = schemas
    def next_action(self, context, allowed, **kwargs):
        self.contexts.append(context)
        if context['unread_skills']:
            return 'read_skill', dict(name=context['unread_skills'][0])
        stage = Stage(context['stage'])
        evidence = {row['name']: row['result'] for row in context['evidence']}
        if stage in (Stage.DESIGN, Stage.OPTIMISATION):
            if 'query_kdb' not in evidence:
                return 'query_kdb', dict(query='bounded verified source and known failures')
            if 'retrospect' not in evidence:
                return 'retrospect', dict(supplement='Inspect actual earlier host failures')
            if 'select_strategy' not in evidence:
                response = evidence['query_kdb']['evidence']['payload']['response']
                decisions = [dict(group=group, index=i, action='rejected' if group == 'negative_records' else 'used',
                                  reason='Keep host provenance and avoid observed failure')
                             for group in ('results', 'negative_records') for i in range(len(response[group]))]
                return 'select_strategy', dict(proposal='Use host-verified source', decisions=decisions,
                    query_ref=evidence['query_kdb']['reference'], retrospective_ref=evidence['retrospect']['reference'],
                    skill_refs=[row['receipt_id'] for row in context['read_skills']])
            if stage == Stage.OPTIMISATION and 'optimisation_decision' not in evidence:
                return 'optimisation_decision', dict(action='noop', reason='Preserve numeric correctness')
        args = dict(proposal='Use the verified candidate for this frozen fixture')
        if (stage == Stage.DEBUG or stage == Stage.IMPLEMENTATION and
                context['source']['text'] in ('', json.dumps(dict(source='', block_count=1)))):
            args['edit'] = dict(path=context['source']['path'], sha256=context['source']['sha256'],
                                source=self.initial if stage == Stage.IMPLEMENTATION else self.repaired)
        return 'perform', args
    def close(self):
        pass


def test_default_shared_kernel_binding_runs_real_adaptive_host_and_numeric_repair(tmp_path, monkeypatch):
    from test_a5_phase1_adaptive_actor import SourceBackend, _request
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from benchmarks.a5kernels.phase1_adaptive_actor import _ProjectActions, AdaptiveActionResult
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import DirectActorDriver, DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.trial import parse_trial_action
    backend = SourceBackend()
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / 'host.jsonl'))
    request = _request(tmp_path, executor)
    store = ContentAddressedArtifacts(request.branch_root / 'artifacts')
    queries = []
    def query(payload, context):
        queries.append(payload)
        return AdaptiveActionResult(dict(available=True, results=[]))
    query.canonical_identity = dict(test_collection='CPU', manifest='empty-v1')
    actions = _ProjectActions(executor, request, executor.prepare(request.spec), store,
        max_actions=40, max_seconds=500, max_source_bytes=128000, query_knowledge=query,
        request_profile=lambda *_: None, request_timing=lambda *_: None)
    session = CanonicalSession()
    monkeypatch.setattr(bindings, 'PiSessionAdapter', lambda **_: session)
    driver = DirectActorDriver(DirectCompletionProvider({'OPENAI_API_KEY':'TEST', 'DEEPSEEK_API_KEY':'TEST'}))
    outcome = driver('Repair the retained numerical mismatch', profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=tmp_path / 'actor', context=None, turn_parser=parse_trial_action, action_executor=actions)
    assert outcome.status == 'host-verified'
    assert actions.latest.evidence.verified and actions.source == 'fixed source'
    assert len(queries) == 2
    assert len(backend.run_requests) >= 2
    runner = outcome.continuation.runner
    assert {row['evidence']['stage'] for row in runner.workflow.state['history']} == {stage.value for stage in Stage}
    assert any(row['passed'] is False for row in runner.workflow.state['history'])
    assert actions.admitted[outcome.continuation.runner.backend.host.actions.latest.artifact_sha256['classification']] == 'host-verification'


def test_default_optimizer_freezes_reference_and_evaluates_without_old_search(tmp_path, monkeypatch):
    from test_optimizer_loop import fixture, Backend
    from optimizer.loop import OptimizationLoop
    store, key = fixture(tmp_path)
    backend = Backend()
    session = CanonicalSession(initial=json.dumps(dict(source='fast', block_count=1)),
                               repaired=json.dumps(dict(source='fast', block_count=1)))
    monkeypatch.setattr(bindings, 'PiSessionAdapter', lambda **_: session)
    owner = OptimizationLoop(store, key, SimpleNamespace(identity={'model':'TEST'}, model='TEST'), backend,
                             knowledge=lambda _: [])
    owner._run_owned = lambda **_: pytest.fail('old optimizer search ran')
    result = owner.run(single_evaluation=True)
    assert result['status'] == 'running'
    assert backend.prepares == 1 and backend.evaluations == ['fast']
    assert owner.state['best']['speedup'] == 2
    assert owner.state['candidate']
    assert owner.state['canonical_active']['status'] == 'accepted'
    assert store.get(key)['canonical_elapsed']


def test_default_optimizer_prepare_only_preserves_frozen_baseline_for_project(tmp_path, monkeypatch):
    from test_optimizer_loop import fixture, Backend
    from optimizer.loop import OptimizationLoop
    store, key = fixture(tmp_path)
    backend = Backend()
    monkeypatch.setattr(bindings, 'PiSessionAdapter', lambda **_: CanonicalSession(
        initial=json.dumps(dict(source='fast', block_count=1)), repaired=json.dumps(dict(source='fast', block_count=1))))
    agent = SimpleNamespace(identity={'model':'TEST'}, model='TEST')
    first = OptimizationLoop(store, key, agent, backend, knowledge=lambda _: [])
    assert first.run(prepare_only=True)['status'] == 'running'
    assert backend.prepares == 1 and not backend.evaluations
    assert first.state['canonical_active']['status'] == 'prepared'
    second = OptimizationLoop(store, key, agent, backend, knowledge=lambda _: [])
    assert second.run(single_evaluation=True)['status'] == 'running'
    assert backend.prepares == 1 and backend.evaluations == ['fast']


def test_cumulative_optimizer_elapsed_is_idempotent_and_preserves_parent_allocation(tmp_path):
    from test_optimizer_loop import fixture
    store, key = fixture(tmp_path)
    store.transition(key, 'running')
    store.account_canonical_elapsed(key, 'project-one', 4)
    store.account_canonical_elapsed(key, 'project-one', 4)
    store.account_canonical_elapsed(key, 'project-one', 7)
    store.account_canonical_elapsed(key, 'project-two', 2)
    assert store.get(key)['elapsed_seconds'] == 9
    with pytest.raises(ValueError):
        store.account_canonical_elapsed(key, 'project-one', 6)


def test_existing_transfer_scope_excludes_actual_copy_without_remote_clock_math(tmp_path):
    from test_rsi_domain_execution import make
    from rsi.domain_execution import DomainActivity
    from rsi.domain_timing import transfer_phase, canonical_allowance
    runner, host, _, _, _ = make(tmp_path)
    operation = runner.workflow.begin()
    runner.workflow.dispatched(operation, 'CPU-STUB')
    @transfer_phase
    def copy():
        assert canonical_allowance() is None
        host.advance(30)
    def command(seconds):
        before = canonical_allowance()
        host.advance(3)
        copy()
        assert canonical_allowance() == before - 3
        host.advance(2)
    DomainActivity(runner, operation).run('compile', 'compilation', command, execution=True)
    assert runner.workflow.accounting.seconds() == 5
    assert runner.workflow.accounting.seconds('transfer') == 30


def canonical_a3_fixture(tmp_path, monkeypatch, *, prepare_failure=False):
    from test_a3_trial_loop import (_cell, FakeCandidate, FakeKnowledge, FakeProfiler, Resolver, SOURCE)
    from benchmarks.a3_experiments import KnowledgeMode
    from benchmarks.a3_model_profiles import load_a3_model_profile
    from benchmarks.a3kernels.phase1_evidence import EvidenceLedger
    from benchmarks.a3kernels.phase1_memory import Phase1LearningJournal
    from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
    from benchmarks.a3kernels.trial import A3TrialLoop, trial_protocol_sha256
    session = CanonicalSession(initial=SOURCE, repaired=SOURCE)
    class Adapter:
        @classmethod
        def from_existing_session(cls, session):
            result = cls()
            result.session = session
            return result
        def prepare(self, timeout, *, schemas):
            self.session.prepare(timeout, schemas=schemas)
        def next_action(self, context, allowed, **kwargs):
            return self.session.next_action(context, allowed, **kwargs)
        def close(self):
            self.session.close()
    monkeypatch.setattr(bindings, 'PiSessionAdapter', Adapter)
    cell = _cell(knowledge=KnowledgeMode.WITH_KDB)
    resolver = Resolver()
    memory = Phase1LearningJournal(tmp_path / 'memory.jsonl', (DEFAULT_PROPOSALS[0],),
        cell_id=cell.cell_id, lineage_id='canonical-test', evidence_resolver=resolver,
        trial_protocol_sha256=trial_protocol_sha256())
    class Candidate(FakeCandidate):
        compile_calls = 0
        def compile(self, source, workdir, **kw):
            self.compile_calls += 1
            if prepare_failure:
                from benchmarks.a3kernels.phase1_protocol import FailedEvidence
                plan = self._plan(source, kw['request_id'], kw['attempt_id'], kw['length'], kw['project_id'],
                    kw['padded_length'], kw['block_count'], kw['execution_profile'])
                return FailedEvidence.create(plan, stage='prepare', error_type='RuntimeError', detail='CPU TEST unavailable route')
            return super().compile(source, workdir, **kw)
    actor = SimpleNamespace(session=session, close=lambda: None)
    knowledge = FakeKnowledge(True)
    knowledge.canonical_identity = dict(test_collection='CPU', manifest='a3-v1')
    trial = A3TrialLoop(cell=cell, proposal=DEFAULT_PROPOSALS[0],
        profile=load_a3_model_profile(cell.backend_model), actor=actor,
        candidate=Candidate(resolver), knowledge=knowledge, profiler=FakeProfiler(resolver),
        evidence=EvidenceLedger(tmp_path / 'evidence.jsonl'), memory=memory, workdir=tmp_path / 'work')
    trial._run = lambda: pytest.fail('old A3 trial loop ran')
    result = trial.run()
    return result, session, trial


def test_default_a3_trial_preserves_typed_host_verification_and_memory(tmp_path, monkeypatch):
    result, session, trial = canonical_a3_fixture(tmp_path, monkeypatch)
    assert result.status == 'passed' and result.verified.passed
    assert result.memory_entry_sha256
    assert result.knowledge_queries == 2
    from benchmarks.a3kernels.candidate import CANDIDATE_SOURCE_CONTRACT
    presented = [json.loads(context['task']['domain']['instruction']) for context in session.contexts]
    assert all(row['source_contract'] == CANDIDATE_SOURCE_CONTRACT.as_dict() for row in presented)
    assert {context['stage'] for context in session.contexts} == {s.value for s in Stage if s != Stage.PREFLIGHT}


def test_native_checkpoint_codec_preserves_derived_graph_and_typed_compile_result(tmp_path):
    from test_a5_phase1_adaptive_actor import SourceBackend, _request
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from rsi.domain_checkpoint import encode, decode
    executor=ProjectExecutor(SourceBackend(),EvidenceLedger(tmp_path/'host.jsonl'))
    request=_request(tmp_path,executor)
    compiled=executor.compile(request.spec,'fixed source')
    restored=decode(encode(compiled))
    assert type(restored) is type(compiled)
    assert restored.prepared.spec.graph.output_shape == compiled.prepared.spec.graph.output_shape
    assert restored == compiled


def test_fresh_a3_trial_returns_retained_native_result_without_target_or_duplicate_memory(tmp_path,monkeypatch):
    from benchmarks.a3kernels.trial import A3TrialLoop
    result, session, original=canonical_a3_fixture(tmp_path,monkeypatch)
    before=original.memory.read()
    class ForbiddenCandidate:
        def compile(self,*args,**kwargs):
            pytest.fail('accepted native compilation reran')
        def run(self,*args,**kwargs):
            pytest.fail('accepted native verification reran')
    class ForbiddenSession:
        completion_tokens=0
        def prepare(self,*args,**kwargs):
            pytest.fail('accepted native provider setup reran')
        def close(self):
            pass
    fresh=A3TrialLoop(cell=original.cell,proposal=original.proposal,profile=original.profile,
        actor=SimpleNamespace(session=ForbiddenSession(),close=lambda:None),candidate=ForbiddenCandidate(),
        knowledge=original.knowledge,profiler=original.profiler,evidence=original.evidence,
        memory=original.memory,workdir=original.workdir)
    resumed=fresh.run()
    assert resumed == result and resumed.verified.passed
    assert original.memory.read() == before


def test_fresh_kernel_actions_restore_verified_native_cas_without_reexecution(tmp_path,monkeypatch):
    from test_a5_phase1_adaptive_actor import SourceBackend, _request
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from benchmarks.a5kernels.phase1_adaptive_actor import _ProjectActions, AdaptiveActionResult
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import DirectActorDriver, DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.trial import parse_trial_action
    backend=SourceBackend()
    executor=ProjectExecutor(backend,EvidenceLedger(tmp_path/'host.jsonl'))
    request=_request(tmp_path,executor)
    store=ContentAddressedArtifacts(request.branch_root/'artifacts')
    def knowledge(*_):
        return AdaptiveActionResult(dict(available=True,results=[]))
    knowledge.canonical_identity = dict(test_collection='CPU', manifest='empty-v1')
    def actions():
        return _ProjectActions(executor,request,executor.prepare(request.spec),store,
            max_actions=40,max_seconds=500,max_source_bytes=128000,query_knowledge=knowledge,
            request_profile=lambda *_:None,request_timing=lambda *_:None)
    monkeypatch.setattr(bindings,'PiSessionAdapter',lambda **_:CanonicalSession())
    driver=DirectActorDriver(DirectCompletionProvider({'OPENAI_API_KEY':'TEST','DEEPSEEK_API_KEY':'TEST'}))
    inputs=dict(profile=direct_profile(MODEL_IDENTITIES[0]),workspace=tmp_path/'actor',context=None,turn_parser=parse_trial_action)
    first=driver('frozen native task',action_executor=actions(),**inputs)
    assert first.status=='host-verified'
    before=(len(backend.requests),len(backend.run_requests))
    fresh_actions=actions()
    assert fresh_actions.latest is None and fresh_actions.source is None
    monkeypatch.setattr(bindings,'PiSessionAdapter',lambda **_:pytest.fail('accepted actor reran'))
    second=driver('frozen native task',action_executor=fresh_actions,**inputs)
    assert second.status=='host-verified' and fresh_actions.latest.evidence.verified
    assert fresh_actions.source=='fixed source'
    assert before==(len(backend.requests),len(backend.run_requests))
    knowledge.canonical_identity['manifest'] = 'same-label-different-content'
    with pytest.raises(bindings.CanonicalConfigurationError, match='changed on resume'):
        driver('frozen native task', action_executor=actions(), **inputs)


def test_failed_native_kernel_exports_actual_domain_receipts_with_pending_stage_intact(tmp_path,monkeypatch):
    from test_a5_phase1_adaptive_actor import SourceBackend, _request
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from benchmarks.a5kernels.phase1_adaptive_actor import _ProjectActions, AdaptiveActionResult
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import DirectActorDriver, DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.trial import parse_trial_action
    from rsi.evidence_io import export_task
    executor=ProjectExecutor(SourceBackend(),EvidenceLedger(tmp_path/'host.jsonl'))
    request=_request(tmp_path,executor)
    actions=_ProjectActions(executor,request,executor.prepare(request.spec),
        ContentAddressedArtifacts(request.branch_root/'artifacts'),max_actions=40,max_seconds=500,
        max_source_bytes=128000,query_knowledge=empty_cpu_knowledge,
        request_profile=lambda *_:None,request_timing=lambda *_:None)
    class FailedSession(CanonicalSession):
        def next_action(self,context,allowed,**kwargs):
            if context['stage']=='debug':
                raise ValueError('CPU fixture terminal malformed repair proposal')
            return super().next_action(context,allowed,**kwargs)
    monkeypatch.setattr(bindings,'PiSessionAdapter',lambda **_:FailedSession())
    driver=DirectActorDriver(DirectCompletionProvider({'OPENAI_API_KEY':'TEST','DEEPSEEK_API_KEY':'TEST'}))
    failed=driver('frozen native negative task',profile=direct_profile(MODEL_IDENTITIES[0]),
        workspace=tmp_path/'actor',context=None,turn_parser=parse_trial_action,action_executor=actions)
    runner=failed.continuation.runner
    before=dict(runner.workflow.pending)
    proof=runner.terminal_failure_proof()
    assert proof['failed_stage']=='debug' and len(proof['command_receipts'])==2
    assert proof['pending']==before and proof['correctness'] is None
    export=export_task(runner.root,tmp_path/'native-dataset')
    assert export['task_contract']==runner.contract_ref
    assert runner.workflow.pending==before and actions.latest.evidence.verified is False


def test_native_terminal_status_inspection_does_not_reclassify_diagnostic_history():
    from rsi.domain_checkpoint import has_pending_result
    assert has_pending_result({'result':{'status':'running','handle':'native-owned'}})
    assert not has_pending_result({'status':'verified','history':[{'status':'running'}],
                                  'diagnostics':{'status':'queued'}})


def test_local_native_source_rejection_has_observed_failure_receipt_and_export(tmp_path,monkeypatch):
    from test_optimizer_loop import fixture, Backend
    from optimizer.loop import OptimizationLoop
    from rsi.evidence_io import export_task
    store,key=fixture(tmp_path)
    class BadEdit(CanonicalSession):
        def next_action(self,context,allowed,**kwargs):
            if context['stage']=='implementation' and not context['unread_skills']:
                return 'perform',dict(proposal='invalid local candidate document',edit=dict(
                    path='candidate.json',sha256=context['source']['sha256'],source='{'))
            return super().next_action(context,allowed,**kwargs)
    monkeypatch.setattr(bindings,'PiSessionAdapter',lambda **_:BadEdit())
    owner=OptimizationLoop(store,key,SimpleNamespace(identity={'model':'TEST'},model='TEST'),Backend(),knowledge=lambda _:[])
    runner=owner.canonical.build(owner,prepare_only=False,single_evaluation=False,memory_snapshot=None,context=None)
    result=runner.run()
    assert result['status']=='suspended' and result['stage']=='implementation'
    proof=runner.terminal_failure_proof()
    assert proof['pending']==runner.workflow.pending
    rows=[runner.store.read_json(e['reference']) for e in runner.events if e['type']=='domain-result']
    assert any(row['status']=='failed' and row['observed_terminal'] is True for row in rows)
    assert export_task(runner.root,tmp_path/'local-native-failure')['task_contract']==runner.contract_ref


def test_existing_pi_session_adoption_transfers_controller_route_and_cleanup(monkeypatch):
    import time
    import coding.pi.session as native
    from coding.pi.controller import ControllerSession
    from rsi.actor import PiSessionAdapter
    from pathlib import Path
    monkeypatch.setattr(native, 'runtime_preflight', lambda: Path('/unused-cpu-fixture'))
    session = native.PiCodingSession(model='deepseek-flash', schemas={},
        environ={'DEEPSEEK_API_KEY': 'CPU-fixture-secret'}, own_process_group=False)
    # A3's actor request deadline may expire during excluded setup/transfer. It
    # must survive as provenance, not become an inferred enclosing wall limit.
    session.deadline = session.gateway.deadline = time.monotonic() - 1
    provenance = dict(session=session.deadline, gateway=session.gateway.deadline)
    profile, gateway, original_stack = session.profile, session.gateway, session.stack
    cleaned = []
    original_stack.callback(lambda: cleaned.append('original transport cleanup'))
    adapter = PiSessionAdapter.from_existing_session(session)
    controller = adapter.session
    try:
        assert isinstance(controller, ControllerSession) and controller is not session
        assert controller.model == session.model and controller.profile is profile
        assert controller.gateway is gateway and controller.stack is original_stack
        assert controller._provider_secret == 'CPU-fixture-secret'
        assert controller.own_process_group is False and controller.rpc is None
        assert controller.request_deadline_provenance == provenance
        assert controller.enclosing_deadline is None
        assert session.closed and not controller.closed
        session.close()
        assert not controller.closed and cleaned == []
        delivered = []
        adapter.delivery.begin('operation')
        adapter.expect_read('receipt', 'observed skill bytes')
        adapter.delivery.observe(dict(messages=[dict(content='observed skill bytes')]),
            lambda kind, data: delivered.append((kind, data)))
        assert adapter.missing_reads(['receipt']) == []
        assert any(kind == 'skill-delivered' for kind, _ in delivered)
        controller.schemas['perform'] = dict(type='object', properties=dict(proposal=dict(type='string')))
        request = dict(tools=[dict(function=dict(name='perform', parameters={}))])
        assert gateway.request_projector(request)['tools'][0]['function']['parameters'] == controller.schemas['perform']
    finally:
        adapter.close()
    assert controller.closed and cleaned == ['original transport cleanup']
    adapter.close()
    assert cleaned == ['original transport cleanup']


@pytest.mark.parametrize('changed', ['memory', 'collection', 'negative_snapshot'])
def test_native_reopen_refuses_same_labels_with_changed_admitted_content(tmp_path, monkeypatch, changed):
    _, _, owner = canonical_a3_fixture(tmp_path, monkeypatch)
    if changed == 'memory':
        original = owner.memory.project_context
        def changed_context():
            value = json.loads(original())
            value['projects'].append(dict(negative_result='newly admitted failure'))
            value['completed_projects'] += 1
            return json.dumps(value)
        monkeypatch.setattr(owner.memory, 'project_context', changed_context)
    else:
        before = dict(owner.knowledge.canonical_identity)
        owner.knowledge.canonical_identity = dict(before, **{changed: 'different-content'})
    from rsi.domain_execution import CanonicalConfigurationError
    with pytest.raises(CanonicalConfigurationError):
        bindings.a3_adoption().build(owner)


def test_collection_identity_includes_actual_sources_and_negative_provenance():
    from dataclasses import replace
    from benchmarks.a5kernels.knowledge import CollectionManifest, SourceFingerprint
    manifest = CollectionManifest('same-name', 'python', 'TEST', 'TEST',
        (SourceFingerprint('kernel.py', 'a'*64, 1),), 'b'*64)
    knowledge = SimpleNamespace(enabled=True, collection='same-name', manifest=manifest,
        snapshots={'same-name': SimpleNamespace(manifest=dict(fingerprint='negative-v1'))})
    original = bindings.fingerprint(bindings.knowledge_identity(knowledge))
    knowledge.manifest = replace(manifest, sources=(SourceFingerprint('kernel.py', 'c'*64, 1),))
    assert bindings.fingerprint(bindings.knowledge_identity(knowledge)) != original
    knowledge.manifest = manifest
    knowledge.snapshots['same-name'].manifest['fingerprint'] = 'negative-v2'
    assert bindings.fingerprint(bindings.knowledge_identity(knowledge)) != original
    with pytest.raises(bindings.CanonicalConfigurationError):
        bindings.knowledge_identity(lambda *_: [])


def _optimizer_read_case(tmp_path, monkeypatch, *, memory=False, corrupt=None, strict=False):
    import base64
    from optimizer.jobs import JobStore, canonical, sha
    from optimizer.loop import OptimizationLoop
    from test_optimizer_loop import Backend
    data=b'actual frozen memory document'
    digest=sha(data)
    files={'docs/contract.txt':dict(base64=base64.b64encode(data).decode(),sha256=digest)} if memory else {}
    body=dict(files=files,memory_entries=[])
    snapshot=dict(body,snapshot_id=sha(canonical(body)))
    store=JobStore(tmp_path/'jobs')
    key=store.create('def forward(x): return x','forward',dict(args=[dict(shape=[5],dtype='float32')]),
        snapshot,max_seconds=120)['job_id']
    jobref=store.put_artifact(key,'frozen-contract.txt',b'actual admitted job document')['sha256']
    reads=[]
    original=store.read_artifact
    def read(job,reference):
        reads.append(reference)
        return original(job,reference)
    monkeypatch.setattr(store,'read_artifact',read)
    class ScopeSession(CanonicalSession):
        requested=False
        valid=False
        unknown=None
        denied=None
        payload=None
        def next_action(self,context,allowed,**kwargs):
            if context['unread_skills']:return super().next_action(context,allowed,**kwargs)
            if context['stage']=='task_construction':
                if not self.requested:
                    if not corrupt:
                        queries=[row for row in context['evidence'] if row['name']=='query_kdb']
                        if not queries:return 'query_kdb',dict(query='frozen admitted artifact scope')
                        self.unknown=queries[-1]['result']['reference']
                    self.requested=True
                    return 'read_artifact',dict(sha256=(digest if memory else jobref) if corrupt else self.unknown)
                if not corrupt and not self.valid:
                    self.denied=context['observation']
                    assert reads==[]  # Scope check did not touch any host artifact.
                    self.valid=True
                    return 'read_artifact',dict(sha256=digest if memory else jobref)
                material=next(row['result'] for row in context['evidence'] if row['name']=='read_artifact')
                self.payload=material['text']
            return super().next_action(context,allowed,**kwargs)
    session=ScopeSession(initial=json.dumps(dict(source='fast',block_count=1)),
                         repaired=json.dumps(dict(source='fast',block_count=1)))
    monkeypatch.setattr(bindings,'PiSessionAdapter',lambda **_:session)
    backend=Backend()
    owner=OptimizationLoop(store,key,SimpleNamespace(identity={'model':'TEST'},model='TEST'),backend,knowledge=lambda _:[])
    runner=owner.canonical.build(owner,prepare_only=False,single_evaluation=False,memory_snapshot=None,context=None)
    # The session deliberately requests its actual delivered canonical KDB receipt
    # through the wrong artifact endpoint. It must not expose global CAS.
    if corrupt=='missing':store.artifact_path(key,jobref).unlink()
    elif corrupt=='job':store.artifact_path(key,jobref).write_bytes(b'corrupt stored evidence')
    elif corrupt=='memory':
        runner.backend.host.submission['memory_snapshot']['files']['docs/contract.txt']['base64']=base64.b64encode(b'changed memory bytes').decode()
    runner.actor.recover_invalid_actions=not strict
    return runner,session,backend,reads,jobref,digest


def test_optimizer_unknown_artifact_feedback_then_admitted_job_read_and_real_stages(tmp_path,monkeypatch):
    runner,session,backend,reads,ref,_=_optimizer_read_case(tmp_path,monkeypatch)
    result=runner.run()
    assert result['status']=='accepted' and result['correctness'] is True
    assert session.denied['accepted'] is False and session.denied['executed'] is False
    assert session.payload=='actual admitted job document' and ref in reads
    assert session.unknown not in reads
    assert backend.prepares==1 and backend.evaluations==['fast']
    assert {h['evidence']['stage'] for h in result['history']}=={stage.value for stage in Stage}
    assert any(e['type']=='actor-invalid-action' for e in runner.events)


def test_optimizer_path_keyed_frozen_memory_is_readable_by_actual_content_digest(tmp_path,monkeypatch):
    runner,session,backend,reads,_,digest=_optimizer_read_case(tmp_path,monkeypatch,memory=True)
    assert 'docs/contract.txt' in runner.backend.host.submission['memory_snapshot']['files']
    assert digest not in runner.backend.host.submission['memory_snapshot']['files']
    result=runner.run()
    assert result['status']=='accepted' and session.payload=='actual frozen memory document'
    assert digest not in reads  # Memory stays in authenticated snapshot namespace.
    assert backend.prepares==1


@pytest.mark.parametrize('corrupt',['missing','job','memory'])
def test_optimizer_admitted_evidence_loss_and_corruption_remain_fatal(tmp_path,monkeypatch,corrupt):
    runner,session,backend,reads,ref,_=_optimizer_read_case(tmp_path,monkeypatch,
        memory=corrupt=='memory',corrupt=corrupt)
    result=runner.run()
    assert result['status']=='suspended' and runner.workflow.pending['stage']=='task_construction'
    assert not any(e['type']=='actor-invalid-action' for e in runner.events)
    assert backend.prepares==0 and backend.evaluations==[]
    assert (ref in reads) is (corrupt!='memory')


def test_optimizer_unknown_artifact_strict_mode_does_not_recover(tmp_path,monkeypatch):
    runner,_,backend,reads,_,_=_optimizer_read_case(tmp_path,monkeypatch,strict=True)
    result=runner.run()
    assert result['status']=='suspended' and runner.workflow.pending
    assert reads==[] and backend.prepares==0
    assert not any(e['type']=='actor-invalid-action' for e in runner.events)


def test_single_evaluation_settles_failed_candidate_before_debug_and_reopens_with_current_memory(tmp_path):
    from optimizer.jobs import canonical, sha
    from optimizer.loop import OptimizationLoop
    from test_optimizer_loop import fixture, Backend
    store, key = fixture(tmp_path)
    backend = Backend()
    class NextAttempt(CanonicalSession):
        def next_action(self, context, allowed, **kwargs):
            if not context['unread_skills'] and context['stage'] == Stage.IMPLEMENTATION.value:
                return 'perform', dict(proposal='Repair the prior failed candidate using current attempt memory',
                    edit=dict(path=context['source']['path'], sha256=context['source']['sha256'], source=self.initial))
            return super().next_action(context, allowed, **kwargs)
    sessions = iter([CanonicalSession(initial=json.dumps(dict(source='bad', block_count=1)),
                                      repaired=json.dumps(dict(source='fast', block_count=1))),
                     NextAttempt(initial=json.dumps(dict(source='fast', block_count=1)),
                                      repaired=json.dumps(dict(source='fast', block_count=1)))])
    adoption = bindings.optimizer_adoption(session_factory=lambda: next(sessions), test_backend=True)
    agent = SimpleNamespace(identity={'test_decisions':'canonical'})
    first = OptimizationLoop(store, key, agent, backend, knowledge=lambda _: [], canonical=adoption)
    first.run(single_evaluation=True)
    assert backend.evaluations == ['bad']
    assert first.state['history'][-1]['content']['status'] == 'incorrect'
    assert first.state['canonical_active']['status'] == 'evaluated'
    root = first.directory/'canonical-1'
    failed = json.loads((root/'workflow.json').read_text())
    assert failed['pending'] is None and failed['domain_boundary'] == 'test'
    assert not any(row['evidence']['stage']=='debug' for row in failed['history'])
    body = dict(files={}, memory_entries=[dict(note='new attempt feedback')])
    current = dict(body, snapshot_id=sha(canonical(body)))
    second = OptimizationLoop(store, key, agent, backend, knowledge=lambda _: [], canonical=adoption)
    second.run(single_evaluation=True, memory_snapshot=current)
    assert backend.prepares == 1 and backend.evaluations == ['bad','fast']
    assert len([row for row in second.state['history'] if 'candidate_sha256' in row]) == 2
    assert second.state['canonical_active']['index'] == 2
    contract = json.loads((second.directory/'canonical-2'/'domain.json').read_text())['contract']
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    retained = ContentAddressedArtifacts(second.directory/'canonical-artifacts').read_json(contract)
    assert retained['identity']['memory'] == bindings.fingerprint(current)
    assert json.loads((root/'workflow.json').read_text()) == failed


def test_frozen_context_directory_hashes_all_named_bytes_and_rejects_missing_context(tmp_path):
    from rsi.domain_execution import CanonicalConfigurationError
    context = tmp_path/'context'; context.mkdir()
    empty = bindings.frozen_context_bytes(context)
    (context/'memory.json').write_bytes(b'actual admitted memory')
    first = bindings.frozen_context_bytes(context)
    assert first != empty
    (context/'guide.txt').write_bytes(b'actual admitted guide')
    second = bindings.frozen_context_bytes(context)
    assert first != second
    (context/'memory.json').write_bytes(b'changed memory')
    assert bindings.frozen_context_bytes(context) != second
    with pytest.raises(CanonicalConfigurationError):
        bindings.frozen_context_bytes(tmp_path/'missing')


def test_source_context_refresh_consumes_new_receipt_without_mutating_delivered_snapshot(tmp_path):
    from benchmarks.ascend.actor_control import ActorControl
    from rsi.domain_bindings import SourceHost
    host = SourceHost()
    host.control_root, host.feedback, host.observations = tmp_path, [], []
    host.instruction, host.identity = 'actual host task', {'baseline':'frozen'}
    host.workflow = SimpleNamespace(stop=lambda: pytest.fail('unexpected stop'))
    delivered = host.context()
    assert host.context_refresh(delivered) is None
    control = ActorControl(tmp_path)
    receipt = control.send('steer-1', 'review numeric evidence before running')
    fresh = host.context_refresh(delivered)
    assert delivered['feedback'] == []
    assert fresh['feedback'][0]['feedback_id'] == receipt['id']
    assert fresh['feedback'][0]['content'] == 'review numeric evidence before running'
    assert control.status()['messages'][0]['consumed'] is not None
    assert host.context_refresh(fresh) is None
    fresh['observations'].append({'untrusted':'caller mutation'})
    assert host.observations == []


def test_source_context_refresh_stop_preserves_unconsumed_feedback_and_stops_same_workflow(tmp_path):
    from benchmarks.ascend.actor_control import ActorControl
    from rsi.domain_bindings import SourceHost
    from rsi.runtime import WorkflowError
    stopped = []
    host = SourceHost()
    host.control_root, host.feedback, host.observations = tmp_path, [], []
    host.instruction, host.identity = 'actual host task', {'baseline':'frozen'}
    host.workflow = SimpleNamespace(stop=lambda: stopped.append(True))
    delivered = host.context()
    control = ActorControl(tmp_path)
    control.send('steer-1', 'retain this feedback across stop')
    control.stop()
    with pytest.raises(WorkflowError):
        host.context_refresh(delivered)
    assert stopped == [True] and delivered['feedback'] == []
    assert control.status()['messages'][0]['consumed'] is None


def test_optimizer_repeated_same_candidate_compile_probes_have_distinct_admitted_invocations(tmp_path):
    from optimizer.loop import OptimizationLoop
    from test_optimizer_loop import fixture, Backend
    class Compiler(Backend):
        def __init__(self):
            super().__init__(); self.compilations = []
        def compile(self, source, directory, *, block_count):
            self.compilations.append(source)
            try:
                compile(source, 'cpu-diagnostic.py', 'exec')
            except SyntaxError as error:
                return dict(status='compile-error', diagnostics=str(error))
            return dict(status='compiled')
    class Probes(CanonicalSession):
        def next_action(self, context, allowed, **kwargs):
            if context['stage']=='implementation' and not context['unread_skills']:
                observed = [row for row in context['evidence'] if row['name']=='compile']
                if len(observed)<3:
                    return 'compile', dict(source='not valid Python' if len(observed)<2 else 'pass',
                                           block_count=1, summary='Actual CPU diagnostic probe')
            return super().next_action(context, allowed, **kwargs)
    store,key=fixture(tmp_path); backend=Compiler()
    model=Probes(initial=json.dumps(dict(source='fast',block_count=1)),repaired=json.dumps(dict(source='fast',block_count=1)))
    adoption=bindings.optimizer_adoption(session_factory=lambda:model,test_backend=True)
    owner=OptimizationLoop(store,key,SimpleNamespace(identity={'model':'CPU decisions'}),backend,
                           knowledge=lambda _:[],canonical=adoption)
    owner.run(single_evaluation=True)
    assert backend.compilations==['not valid Python','not valid Python','pass']
    assert backend.evaluations==['fast'] and store.get(key)['attempts']==1
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    cas=ContentAddressedArtifacts(owner.directory/'canonical-artifacts')
    events=[cas.read_json(ref) for ref in json.loads((owner.directory/'canonical-1/events.json').read_text())]
    probes=[cas.read_json(row['reference']) for row in events if row['type']=='domain-dispatch']
    probes=[row for row in probes if row['name']=='compile-diagnostic']
    assert len({row['reservation'] for row in probes})==3
    assert len({row['invocation_ref'] for row in probes})==3
    assert len({row['operation_id'] for row in probes})==1
    assert owner.state['canonical_active']['status']=='accepted'


def test_optimizer_recovers_observed_evaluation_before_history_without_model_host_or_attempt_replay(tmp_path,monkeypatch):
    from optimizer.loop import OptimizationLoop
    from test_optimizer_loop import fixture, Backend
    from rsi.domain_execution import DomainRunner
    store,key=fixture(tmp_path);backend=Backend()
    model=CanonicalSession(initial=json.dumps(dict(source='fast',block_count=1)),repaired=json.dumps(dict(source='fast',block_count=1)))
    admission=bindings.optimizer_adoption(session_factory=lambda:model,test_backend=True)
    agent=SimpleNamespace(identity={'model':'CPU decisions'})
    original=DomainRunner.append
    def crash(runner,event):
        original(runner,event)
        if event['type']=='domain-result':
            receipt=runner.store.read_json(event['reference'])
            if runner.store.read_json(receipt['dispatch_ref'])['name']=='evaluate':
                raise RuntimeError('after authenticated result, before history')
    monkeypatch.setattr(DomainRunner,'append',crash)
    first=OptimizationLoop(store,key,agent,backend,knowledge=lambda _:[],canonical=admission)
    with pytest.raises(RuntimeError): first.run(single_evaluation=True)
    before=store.get(key);calls=(backend.prepares,list(backend.evaluations),len(model.contexts))
    assert first.state['history']==[] and backend.evaluations==['fast']
    monkeypatch.setattr(DomainRunner,'append',original)
    no_replay=bindings.optimizer_adoption(session_factory=lambda:pytest.fail('actor replay'),test_backend=True)
    second=OptimizationLoop(store,key,agent,backend,knowledge=lambda _:pytest.fail('retrieval replay'),canonical=no_replay)
    result=second.run(single_evaluation=True)
    assert result['status']=='running'
    assert (backend.prepares,backend.evaluations,len(model.contexts))==calls
    assert second.state['history'][-1]['content']['status']=='verified'
    assert second.state['canonical_active']['status']=='evaluated'
    after=store.get(key)
    assert after['attempts']==before['attempts']==1
    assert after['max_seconds']==before['max_seconds'] and after['elapsed_seconds']>=before['elapsed_seconds']
    workflow=json.loads((second.directory/'canonical-1/workflow.json').read_text())
    assert workflow['pending'] is None and workflow['domain_boundary']=='test'
    assert workflow['history'][-1]['evidence']['stage']=='test'


def test_optimizer_cached_accepted_stop_and_resume_use_current_control_status_without_replay(tmp_path):
    from optimizer.loop import OptimizationLoop
    from test_optimizer_loop import fixture, Backend
    store,key=fixture(tmp_path);backend=Backend();agent=SimpleNamespace(identity={'model':'CPU decisions'})
    model=CanonicalSession(initial=json.dumps(dict(source='fast',block_count=1)),repaired=json.dumps(dict(source='fast',block_count=1)))
    initial=bindings.optimizer_adoption(session_factory=lambda:model,test_backend=True)
    first=OptimizationLoop(store,key,agent,backend,knowledge=lambda _:[],canonical=initial)
    first.run(single_evaluation=True)
    calls=(backend.prepares,list(backend.evaluations),len(model.contexts))
    no_replay=bindings.optimizer_adoption(session_factory=lambda:pytest.fail('cached actor dispatch'),test_backend=True)
    store.stop(key)
    second=OptimizationLoop(store,key,agent,backend,knowledge=lambda _:pytest.fail('cached retrieval'),canonical=no_replay)
    assert second.run()['status']=='stopped'
    assert store.get(key)['stop_requested'] is True
    store.resume(key,additional_seconds=10)
    third=OptimizationLoop(store,key,agent,backend,knowledge=lambda _:pytest.fail('cached retrieval'),canonical=no_replay)
    assert third.run()['status']=='running'
    assert (backend.prepares,backend.evaluations,len(model.contexts))==calls
    assert third.state['best']==first.state['best']
