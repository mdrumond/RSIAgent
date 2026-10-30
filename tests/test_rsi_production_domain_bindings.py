"""Default public constructors, actual canonical stages, CPU-only host runtimes."""
import json
from types import SimpleNamespace

import pytest

from rsi.contracts import Stage
import rsi.domain_bindings as bindings


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


def test_default_a3_trial_preserves_typed_host_verification_and_memory(tmp_path, monkeypatch):
    from test_a3_trial_loop import (_cell, FakeCandidate, FakeKnowledge, FakeProfiler, Resolver, SOURCE)
    from benchmarks.a3_experiments import KnowledgeMode
    from benchmarks.a3_model_profiles import load_a3_model_profile
    from benchmarks.a3kernels.phase1_evidence import EvidenceLedger
    from benchmarks.a3kernels.phase1_memory import Phase1LearningJournal
    from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
    from benchmarks.a3kernels.trial import A3TrialLoop, trial_protocol_sha256
    session = CanonicalSession(initial=SOURCE, repaired=SOURCE)
    class Adapter:
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
    actor = SimpleNamespace(session=session, close=lambda: None)
    trial = A3TrialLoop(cell=cell, proposal=DEFAULT_PROPOSALS[0],
        profile=load_a3_model_profile(cell.backend_model), actor=actor,
        candidate=FakeCandidate(resolver), knowledge=FakeKnowledge(True), profiler=FakeProfiler(resolver),
        evidence=EvidenceLedger(tmp_path / 'evidence.jsonl'), memory=memory, workdir=tmp_path / 'work')
    trial._run = lambda: pytest.fail('old A3 trial loop ran')
    result = trial.run()
    assert result.status == 'passed' and result.verified.passed
    assert result.memory_entry_sha256
    assert result.knowledge_queries == 2
    from benchmarks.a3kernels.candidate import CANDIDATE_SOURCE_CONTRACT
    presented = [json.loads(context['task']['domain']['instruction']) for context in session.contexts]
    assert all(row['source_contract'] == CANDIDATE_SOURCE_CONTRACT.as_dict() for row in presented)
    assert {context['stage'] for context in session.contexts} == {s.value for s in Stage if s != Stage.PREFLIGHT}
