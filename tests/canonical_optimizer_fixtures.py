"""Explicit CPU fixture admission; only model decisions are scripted.

The production OptimizerHost freezes the baseline and owns every host effect.
DomainRunner retains the actual workflow, receipts, source and recovery state.
"""
import json
from rsi.event_journal import read_event_refs
from core.artifacts import canonical_bytes

import pytest

from rsi.contracts import Stage
from rsi.domain_bindings import optimizer_adoption
from rsi.domain_execution import CanonicalAdoption
from test_rsi_production_domain_bindings import CanonicalSession


class OptimizerSession(CanonicalSession):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.proposed = False
        self.selected = False
        self.optimization_edit = None

    def next_action(self, context, allowed, **kwargs):
        # Preserve Pi's argument-validation boundary for scripted decisions.
        from coding.pi.schema import validate
        from coding.pi.session import ActionError
        self.owner.agent.canonical_remaining_seconds = kwargs['seconds']
        name, args = self.decide(context, allowed, **kwargs)
        if name not in allowed:
            raise ActionError(f'action unavailable in canonical stage: {name}')
        try:
            validate(args, self.schemas[name])
        except ValueError as error:
            raise ActionError(str(error)) from error
        return name, args

    def decide(self, context, allowed, **kwargs):
        if context['unread_skills']:
            return super().next_action(context, allowed, **kwargs)
        stage = Stage(context['stage'])
        # Required strategy decisions and their receipt references are handled
        # by the same canonical session used by production binding tests.
        if stage in (Stage.DESIGN, Stage.OPTIMISATION):
            name, args = super().next_action(context, allowed, **kwargs)
            if name == 'optimisation_decision' and getattr(self.owner.agent, 'optimize', False):
                action = self.owner.agent.complete('optimize', context, self.owner._record)
                if action['action'] == 'candidate':
                    self.optimization_edit = action
                    return name, dict(action='edit', reason=action['summary'])
                if action['action'] != 'finish':
                    raise AssertionError('optimisation fixture must choose an edit or finish')
            if name == 'perform' and self.optimization_edit is not None:
                action, self.optimization_edit = self.optimization_edit, None
                return name, dict(proposal=action['summary'], edit=dict(
                    path=context['source']['path'], sha256=context['source']['sha256'],
                    source=json.dumps(dict(source=action['source'], block_count=action['block_count']))))
            if name == 'optimisation_decision' and getattr(self.owner.agent, 'profile', False):
                return name, dict(action='profile', reason='Inspect the real fixture host profiler')
            return name, args
        if stage == Stage.TASK_CONSTRUCTION and not self.proposed and 'baseline' not in self.owner.state:
            action = self.owner.agent.complete('cases', context, self.owner._record)
            self.proposed = True
            assert action['action'] == 'cases'
            return 'propose_cases', {key: action[key] for key in ('signatures', 'summary')}
        failed = next((row for row in reversed(self.owner.state['history'])
                       if 'speedup' in row), None)
        needs_candidate = (stage == Stage.IMPLEMENTATION and not self.selected
                           or stage == Stage.DEBUG and failed is not None and failed['speedup'] is None)
        if needs_candidate:
            action = self.owner.agent.complete('optimize', context, self.owner._record)
            name = action['action'].replace('-', '_')
            if name == 'query_knowledge':
                name = 'query_kdb'
            if name in allowed and name != 'perform':
                return name, {key: value for key, value in action.items() if key != 'action'}
            if name != 'candidate':
                return name, {key: value for key, value in action.items() if key != 'action'}
            self.selected = True
            return 'perform', dict(proposal=action.get('summary', 'CPU fixture candidate'),
                edit=dict(path=context['source']['path'], sha256=context['source']['sha256'],
                          source=json.dumps(dict(source=action['source'], block_count=action['block_count']))))
        return 'perform', dict(proposal='Inspect the actual frozen CPU host result')

    def close(self):
        pass


def fixture_adoption():
    def build(owner, **inputs):
        session_factory = getattr(owner.agent, 'canonical_session_factory', OptimizerSession)
        adoption = optimizer_adoption(session_factory=lambda: session_factory(owner),
                                      test_backend=True, clock=owner.clock)
        runner = adoption.build(owner, **inputs)
        owner.fixture_runner = runner
        return runner
    return CanonicalAdoption(build, optimizer_adoption().finish)


@pytest.fixture(autouse=True)
def canonical_optimizer_backend(monkeypatch):
    """Opt in marked scripted agents, leaving registered production models alone."""
    from optimizer.loop import OptimizationLoop
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    from rsi.strategy import knowledge_db_query
    from test_rsi_evidence_io import FixtureEmbeddings
    original = OptimizationLoop.__init__
    databases = []

    def initialize(self, *args, **kwargs):
        agent = args[2] if len(args) > 2 else kwargs['agent']
        if getattr(agent, 'canonical_fixture', False) and kwargs.get('canonical') is None:
            kwargs['canonical'] = fixture_adoption()
            if hasattr(agent, 'canonical_clock'):
                kwargs.setdefault('clock', agent.canonical_clock)
            original(self, *args, **kwargs)
            if self.knowledge is None:
                database = KnowledgeDB(self.directory / 'fixture-knowledge.sqlite3', FixtureEmbeddings())
                # Index only the job's already supplied frozen reference.
                # No learned memory or synthetic experience enters baseline KDB.
                reference = self.directory / 'fixture-reference.py'
                reference.write_text(self.store.submission(self.key)['source'])
                database.index(self.directory, [reference], collection='cpu-baseline', language='python')
                query = knowledge_db_query(database, ('cpu-baseline',), negative_query=lambda _: [])
                self.knowledge = lambda text: dict(query(dict(query=text)), available=True)
                databases.append(database)
        else:
            original(self, *args, **kwargs)
    monkeypatch.setattr(OptimizationLoop, '__init__', initialize)
    yield
    for database in databases:
        database.close()


def optimizer_inputs(context):
    """The actual frozen host instruction delivered to the canonical session."""
    return json.loads(context['task']['domain']['instruction'])


def evidence_results(context, name):
    return [row['result'] for row in context['evidence'] if row['name'] == name]


def retained_native(execution):
    """Read committed native evidence without constructing or running a host."""
    from types import SimpleNamespace
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    state = execution._state() if hasattr(execution, '_state') else execution.state
    directory = (execution.job_directory / 'execution' if hasattr(execution, 'job_directory')
                 else execution.directory)
    root = directory / ('canonical-' + str(state['canonical_active']['index']))
    store = ContentAddressedArtifacts(directory / 'canonical-artifacts')
    return SimpleNamespace(root=root, store=store,
        workflow=json.loads((root / 'workflow.json').read_bytes()),
        events=[store.read_json(reference) for reference in read_event_refs(root)])


def scripted_optimizer_http(monkeypatch, agent, *, compile_if_offered=False):
    """Run the real Pi controller; script only provider HTTP tool decisions."""
    from types import SimpleNamespace
    from rsi.actor import PiSessionAdapter
    from test_rsi_pi_controller import response
    state = SimpleNamespace(session=None, requests=[])

    class Session(PiSessionAdapter):
        def __init__(self, owner):
            super().__init__(model='deepseek-flash', environ={'DEEPSEEK_API_KEY': 'fake-provider-secret'})
            self.owner, self.decisions = owner, OptimizerSession(owner)
            state.session = self
        def run_stage(self, dispatcher, turn, actor):
            self.payload = turn[0]
            return super().run_stage(dispatcher, turn, actor)

    def provider(*_, **kwargs):
        wire = json.loads(kwargs['data'])
        state.requests.append(wire)
        if not wire.get('tools'):
            return response([], len(state.requests))
        if compile_if_offered and any(tool["function"]["name"] == "compile" for tool in wire["tools"]):
            return response([("compile", dict(source="pass", block_count=1,
                                             summary="Probe advertised compiler"))], len(state.requests))
        session = state.session
        runner = session.owner.fixture_runner
        operation = runner.workflow.pending['id']
        events = [row for row in runner.events if row.get('operation_id') == operation]
        reads = [row for row in events if row['type'] == 'skill-usage' and row.get('kind') == 'read']
        names = runner.bundle.skills_for(runner.workflow.stage)
        source = runner.snapshot()
        payload = dict(session.payload,
            unread_skills=[name for name in names if name not in {row['skill'] for row in reads}],
            read_skills=[dict(receipt_id=runner.service.read_receipt(row['receipt_id'])) for row in reads],
            evidence=[dict(name=row['name'], result=row['result']) for row in events
                      if row['type'] == 'evidence-tool-result'],
            source=dict(path=source.path, text=source.text, sha256=source.sha256,
                        project_hash=source.project_hash))
        name, args = session.decisions.decide(payload, list(session.session.allowed))
        return response([(name, args)], len(state.requests))

    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    agent.canonical_session_factory = Session
    return state


def unresolved_authority(native):
    """Only original authority may close unknown work; retries cannot add effects."""
    return dict(pending=native.workflow['pending'], source=native.workflow['source'],
        reservations=native.workflow['execution_reservations'],
        dispatched=[row for row in native.events if row['type'] in
                    ('domain-dispatch', 'actor-request', 'query-dispatch')])
