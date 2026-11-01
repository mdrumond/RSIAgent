"""Canonical CPU kernel fixtures: scripted decisions, real numeric host evidence."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_digest
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.a5kernels.phase1_adaptive_actor import _ProjectActions, AdaptiveActionResult
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor, CompileObservation, RuntimeObservation
from benchmarks.a5kernels.phase1_provider import DirectActorDriver, DirectCompletionProvider, direct_profile
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.trial import parse_trial_action
from rsi.domain_execution import DomainRunner
from rsi.strategy import knowledge_db_query
from test_rsi_evidence_io import FixtureEmbeddings
from test_rsi_production_domain_bindings import CanonicalSession


@pytest.fixture
def canonical_kernel(tmp_path, monkeypatch):
    """Keep the real domain host alive across stop/reconstruction in one test."""
    from test_a5_phase1_adaptive_actor import _request
    import rsi.domain_bindings as bindings
    native_adapter = bindings.PiSessionAdapter
    state = SimpleNamespace(now=1000., calls=[], contexts=[], before_select=None,
                            after_select=None, after_host=None, invalid=False, select=None, wires=[], wire=None)
    def tick():
        state.now += .01
    class Backend:
        def compile(self, request):
            state.calls.append('compile')
            tick()
            compile(request.candidate_source, 'fixture-candidate.py', 'exec')
            result = CompileObservation(True, canonical_digest(dict(source=request.candidate_source)),
                                        'CPU Python compiled', 'CPU')
            if state.after_host:
                state.after_host('compile')
            return result
        def run(self, request, compilation):
            state.calls.append('run')
            tick()
            namespace = {}
            exec(compile(request.candidate_source, 'fixture-candidate.py', 'exec'), namespace)
            observed = tuple(namespace['solve'](list(request.inputs[0].storage)))
            result = RuntimeObservation(True, canonical_digest(dict(observed=observed)),
                                        'CPU numeric output', observed)
            if state.after_host:
                state.after_host('run')
            return result
    executor = ProjectExecutor(Backend(), EvidenceLedger(tmp_path / 'host.jsonl'))
    request = _request(tmp_path, executor)
    artifacts = ContentAddressedArtifacts(request.branch_root / 'artifacts')
    database = KnowledgeDB(tmp_path / 'knowledge.sqlite3', FixtureEmbeddings())
    seed = tmp_path / 'baseline.py'
    seed.write_text('def relu(x): return max(0, x)\n')
    manifest = database.index(tmp_path, [seed], collection='cpu-baseline', language='python')
    query = knowledge_db_query(database, ('cpu-baseline',), negative_query=lambda _: [
        row for row in state.runner.workflow.state['history'] if row.get('passed') is False])
    def knowledge(payload, context):
        return AdaptiveActionResult(query(payload))
    knowledge.canonical_identity = dict(manifest=json.loads(manifest.to_json()))
    query.canonical_identity = knowledge.canonical_identity
    actions = _ProjectActions(executor, request, executor.prepare(request.spec), artifacts,
        max_actions=1, max_seconds=120, max_source_bytes=128000, query_knowledge=knowledge,
        request_profile=lambda *_: None, request_timing=lambda *_: None)
    from benchmarks.ascend.agent_presentation import MemoryArtifactReader
    actions.read_artifacts = MemoryArtifactReader(request, artifacts, admitted=actions.admitted)
    width = request.inputs[0].physical_shape[-1]
    logical = request.inputs[0].shape[-1]
    repaired = (f'def solve(xs):\n    return [max(0,x) if i % {width} < {logical} else 0 '
                'for i,x in enumerate(xs)]\n')
    class Session(CanonicalSession):
        def next_action(self, context, allowed, **kwargs):
            state.contexts.append(deepcopy(context))
            tick()
            if state.before_select:
                state.before_select(context, allowed)
            if state.invalid:
                from coding.pi.session import ActionError
                state.invalid = False
                raise ActionError('malformed fixture decision')
            selected = (state.select(context, allowed) if state.select else
                        super().next_action(context, allowed, **kwargs))
            if state.after_select:
                state.after_select(context, selected)
            return selected
    state.session = Session(initial='def solve(xs):\n    return list(xs)\n', repaired=repaired)
    monkeypatch.setattr('rsi.domain_bindings.PiSessionAdapter', lambda **_: state.session)
    monkeypatch.setattr('coding.pi.actor.runtime_preflight', lambda: None)
    original_create = DomainRunner.create.__func__
    original_init = DomainRunner.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **dict(kwargs, test_backend=True, clock=lambda: state.now))
        state.runner = self
    def create(cls, *args, **kwargs):
        return original_create(cls, *args, **dict(kwargs, test_backend=True, clock=lambda: state.now))
    monkeypatch.setattr(DomainRunner, '__init__', init)
    monkeypatch.setattr(DomainRunner, 'create', classmethod(create))
    state.provider = DirectCompletionProvider({'OPENAI_API_KEY': 'TEST', 'DEEPSEEK_API_KEY': 'TEST'})
    state.workspace = tmp_path / 'actor'
    state.actions, state.manifest, state.database = actions, manifest, database
    state.executor, state.request, state.knowledge = executor, request, knowledge
    state.baseline_query = query
    state.profile = direct_profile(MODEL_IDENTITIES[0])
    def driver(kind='direct', **kwargs):
        if kind == 'pi':
            from coding.pi.actor import PiActorDriver
            cls = PiActorDriver
        else:
            cls = DirectActorDriver
        return cls(state.provider, clock=lambda: state.now, max_seconds=120, **kwargs)
    def run(actor=None):
        state.outcome = (actor or state.driver)('CPU ReLU with a zero padded tail',
            profile=state.profile, workspace=state.workspace, context=None,
            turn_parser=parse_trial_action, action_executor=actions)
        return state.outcome
    def enable_native():
        """Drive the installed Pi SDK; replace upstream HTTP decisions only."""
        from test_rsi_pi_controller import response
        state.profile = direct_profile(next(model for model in MODEL_IDENTITIES
            if model.backend_model.value == 'deepseek-flash'))
        class Adapter(native_adapter):
            def __init__(self, **kwargs):
                super().__init__(context_window=16384, request_bytes=49152, **kwargs)
            def prepare(self, timeout, *, schemas):
                state.session.prepare(timeout, schemas=schemas)
                return super().prepare(timeout, schemas=schemas)
        def provider(*args, **kwargs):
            wire = json.loads(kwargs['data'])
            state.wires.append(wire)
            if state.wire:
                result = state.wire(wire)
                if result is not None:
                    return result
            if not wire.get('tools'):
                return response([], len(state.wires), tokens=len(kwargs['data']) // 4)
            runner = state.runner
            operation = runner.workflow.pending['id']
            reads = [row for row in runner.events if row['type'] == 'skill-usage'
                     and row.get('kind') == 'read' and row['operation_id'] == operation]
            names = runner.bundle.skills_for(runner.workflow.stage)
            read_names = {row['skill'] for row in reads}
            source = runner.snapshot()
            evidence = [dict(name=row['name'], result=dict(reference=row['reference'],
                            evidence=runner.store.read_json(row['reference'])))
                        for row in runner.events if row['type'] == 'project-tool'
                        and row['operation_id'] == operation]
            host = runner.backend.host
            context = dict(stage=runner.workflow.stage.value, evidence=evidence,
                unread_skills=[name for name in names if name not in read_names],
                read_skills=[dict(receipt_id=runner.service.read_receipt(row['receipt_id'])) for row in reads],
                task=dict(domain=dict(instruction=host.instruction, feedback=host.feedback)),
                source=dict(path=source.path, text=source.text, sha256=source.sha256,
                            project_hash=source.project_hash))
            action = state.session.next_action(context, [],
                seconds=120, record=lambda *_: None, source=None)
            return response([action], len(state.wires), tokens=len(kwargs['data']) // 4)
        monkeypatch.setattr(bindings, 'PiSessionAdapter', Adapter)
        monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    state.enable_native = enable_native
    state.make_driver, state.run = driver, run
    state.driver = driver()
    try:
        yield state
    finally:
        database.close()
