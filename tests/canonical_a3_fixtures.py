"""Explicit CPU canonical compositions; scripted decisions never certify a result."""
from pathlib import Path

from rsi.contracts import ProjectSpec, Stage
from rsi.domain_bindings import KernelActionsHost, baseline_contract, fingerprint, kernel_adoption
from rsi.domain_execution import CanonicalAdoption, DomainRunner
from test_rsi_production_domain_bindings import CanonicalSession
from test_expanded_kdb import Embeddings


NATIVE_SOURCE = ('extern "C" __global__ __aicore__ void project_kernel('
                 'GM_ADDR gm_input0, GM_ADDR gm_output) {}')


def native_knowledge_config(root):
    """Index real fixture source; the host retrieves with the ordinary SQLite KDB."""
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    root = Path(root)
    source = root / 'native-reference.cpp'
    source.write_text(NATIVE_SOURCE)
    database = root / 'baseline.sqlite3'
    with KnowledgeDB(database, Embeddings()) as store:
        store.index(root, [source], collection='native-baseline', language='ascend-c')
    cache = root / 'embedding-cache'
    cache.mkdir(exist_ok=True)
    return dict(kdb=str(database), collection='native-baseline', bge_cache=str(cache))


def embedding_factory(**_):
    return Embeddings()


def native_baseline(root, workspace):
    from dataclasses import asdict
    import json
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, KnowledgeQuery, ProgressiveMemoryJournal
    database = Path(root) / 'baseline.sqlite3'
    def identity():
        with KnowledgeDB.open_read_only(database, Embeddings()) as view:
            return dict(manifests={'native-baseline': json.loads(view.manifest('native-baseline').to_json())})
    def query(payload):
        agent = KnowledgeAgent(enabled=True,
            database=KnowledgeDB.open_read_only(database, Embeddings()), collection='native-baseline',
            journal=ProgressiveMemoryJournal(Path(workspace) / 'baseline-queries.jsonl'))
        try:
            rows = agent.query(KnowledgeQuery(payload['query'], payload.get('limit', 5)))
            return dict(results=[asdict(row) for row in rows], negative_records=[], provenance=identity())
        finally:
            agent.close()
    query.canonical_identity = identity
    return query


def install_native_kernel(monkeypatch, *, baseline_root, revise=False, observe=None):
    """Bind scripted decisions explicitly, preserving the production kernel host.

    The CLI still builds its ordinary candidate backend, frozen numerical oracle,
    Knowledge Agent and memory. Only model decisions are replaced. No accepted
    result, compilation, verifier result or receipt is supplied by this fixture.
    """
    original = kernel_adoption()
    sessions = []

    class Session(CanonicalSession):
        def __init__(self):
            super().__init__(initial=NATIVE_SOURCE, repaired=NATIVE_SOURCE + ' // repaired')
            self.revised = False

        def next_action(self, context, allowed, **kwargs):
            if not self.contexts and observe:
                observe(context)
            if revise and context['stage'] == Stage.OPTIMISATION.value:
                evidence = {row['name'] for row in context['evidence']}
                if 'select_strategy' in evidence and 'optimisation_decision' not in evidence and not self.revised:
                    self.contexts.append(context)
                    return 'optimisation_decision', dict(action='edit', reason='Measure a source revision')
                if 'optimisation_decision' in evidence and not self.revised:
                    self.revised = True
                    self.contexts.append(context)
                    return 'perform', dict(proposal='Measure the revised candidate', edit=dict(
                        path=context['source']['path'], sha256=context['source']['sha256'],
                        source=NATIVE_SOURCE + ' // variant'))
            name, args = super().next_action(context, allowed, **kwargs)
            if context['stage'] == Stage.DEBUG.value and name == 'perform':
                args.pop('edit', None)
            return name, args

    def build(owner, **inputs):
        baseline = native_baseline(baseline_root, inputs["workspace"])
        host = KernelActionsHost(owner, inputs, baseline_query=baseline)
        bundle, limits = baseline_contract(owner, owner.max_seconds)
        spec = ProjectSpec('kernel', inputs['instruction'], ('host-kernel-executor',), budgets=limits)
        root = Path(inputs['workspace']) / 'canonical'
        session = Session()
        sessions.append(session)
        injections = dict(query=host.query, backend_identity=fingerprint(host.identity),
                          session_factory=lambda: session, test_backend=True)
        if (root / 'domain.json').exists():
            runner = DomainRunner(root, host, **injections)
        else:
            runner = DomainRunner.create(root, spec, bundle, host, **injections)
        session.runner = runner
        return runner

    adoption = CanonicalAdoption(build, original.finish, original.distill)
    monkeypatch.setattr('rsi.domain_bindings.kernel_adoption', lambda: adoption)
    return sessions


class A3Embeddings(Embeddings):
    dimension = 3
    embed_documents = Embeddings.embed
    embed_queries = Embeddings.embed


class A3Session(CanonicalSession):
    """Repair failed host evidence; successful debug preserves the current source."""
    repairs = 0
    def next_action(self, context, allowed, **kwargs):
        name, args = super().next_action(context, allowed, **kwargs)
        if context['stage'] == Stage.DEBUG.value and name == 'perform':
            stages = [row for row in context['task']['domain']['observations'] if 'stage' in row]
            if stages and stages[-1]['passed']:
                args.pop('edit', None)
            elif 'edit' in args:
                self.repairs += 1
                args['edit']['source'] = self.repaired + ('\n// revision ' + str(self.repairs))
        return name, args


class RevisionSession(A3Session):
    revised = False
    def next_action(self, context, allowed, **kwargs):
        if context['stage'] == Stage.OPTIMISATION.value:
            evidence = {row['name'] for row in context['evidence']}
            if 'select_strategy' in evidence and 'optimisation_decision' not in evidence and not self.revised:
                self.contexts.append(context)
                return 'optimisation_decision', dict(action='edit', reason='Measure a source revision')
            if 'optimisation_decision' in evidence and not self.revised:
                self.revised = True
                self.contexts.append(context)
                return 'perform', dict(proposal='Retest current revision', edit=dict(
                    path=context['source']['path'], sha256=context['source']['sha256'],
                    source=context['source']['text'] + '\n// optimisation revision'))
        return super().next_action(context, allowed, **kwargs)


def make_trial(root, *, cell=None, candidate=None, profiler=None, session=None,
               budgets=None, clock=None, execution_profile='gz-a3', proposal=None):
    """Real A3 host and SQLite baseline, separate optional experience collection."""
    from types import SimpleNamespace
    from benchmarks.a3_experiments import KnowledgeMode
    from benchmarks.a3_model_profiles import load_a3_model_profile
    from benchmarks.a3kernels.phase1_evidence import EvidenceLedger
    from benchmarks.a3kernels.phase1_memory import Phase1LearningJournal
    from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
    from benchmarks.a3kernels.trial import A3TrialLoop, trial_protocol_sha256
    from test_a3_trial_loop import _cell, FakeCandidate, FakeProfiler, Resolver, SOURCE

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cell, proposal = cell or _cell(), proposal or DEFAULT_PROPOSALS[0]
    baseline = a3_knowledge(root / 'baseline', enabled=True, collection='baseline')
    knowledge = a3_knowledge(root / 'experience',
        enabled=cell.knowledge is KnowledgeMode.WITH_KDB, collection='experience')
    resolver = Resolver()
    candidate = candidate or FakeCandidate(resolver)
    profiler = profiler or FakeProfiler(resolver)
    if hasattr(candidate, 'resolver'):
        candidate.resolver = resolver
    if hasattr(profiler, 'resolver'):
        profiler.resolver = resolver
    session = session or A3Session(initial=SOURCE, repaired=SOURCE + '\n// repaired')
    memory = Phase1LearningJournal(root / 'memory.jsonl', (proposal,), cell_id=cell.cell_id,
        lineage_id='canonical-a3-cpu', evidence_resolver=resolver,
        trial_protocol_sha256=trial_protocol_sha256())
    options = dict(budgets=budgets) if budgets is not None else {}
    owner = A3TrialLoop(cell=cell, proposal=proposal, profile=load_a3_model_profile(cell.backend_model),
        actor=SimpleNamespace(session=session, close=session.close), candidate=candidate,
        knowledge=knowledge, profiler=profiler, evidence=EvidenceLedger(root / 'evidence.jsonl'),
        memory=memory, workdir=root / 'work', execution_profile=execution_profile, clock=clock,
        canonical=a3_test_adoption(session, baseline), **options)
    owner.baseline_knowledge = baseline
    return owner, session


def a3_test_adoption(session, baseline):
    from dataclasses import asdict
    import json
    from benchmarks.a3kernels.knowledge_agent import KnowledgeQuery
    from rsi.domain_bindings import A3Host, a3_adoption
    def query(payload):
        rows = baseline.query(KnowledgeQuery(payload['query'], payload.get('limit', 5)))
        return dict(results=[asdict(row) for row in rows], negative_records=[],
                    provenance=dict(manifest=json.loads(baseline.manifest.to_json())))
    query.canonical_identity = lambda: dict(manifest=json.loads(baseline.manifest.to_json()))
    adoption = a3_adoption(baseline_query=query)
    def build(owner, **inputs):
        host = A3Host(owner, baseline_query=query)
        bundle, limits = baseline_contract(owner, owner.budgets.max_seconds)
        spec = ProjectSpec(owner.proposal.project_id, host.instruction, ('A3-host-candidate',), budgets=limits)
        directory = owner.workdir / 'canonical'
        injections = dict(query=host.query, backend_identity=fingerprint(host.identity),
            session_factory=lambda: session, test_backend=True, clock=owner.clock)
        if (directory / 'domain.json').exists():
            return DomainRunner(directory, host, **injections)
        return DomainRunner.create(directory, spec, bundle, host, **injections)
    return CanonicalAdoption(build, adoption.finish)


def a3_knowledge(root, *, enabled, collection):
    from hashlib import sha256
    from benchmarks.a3kernels.corpus import CorpusDocument
    from benchmarks.a3kernels.knowledge import KnowledgeDB
    from benchmarks.a3kernels.knowledge_agent import KnowledgeAgent, QueryJournal
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    database = None
    if enabled:
        path = root / 'knowledge.sqlite3'
        embeddings = A3Embeddings()
        if not path.exists():
            text = ('Host reference: vector add and bounded tails.' if collection == 'baseline'
                    else 'Prior experience: use DataCopyPad for bounded tails.')
            digest = sha256(text.encode()).hexdigest()
            with KnowledgeDB.create(path, embeddings) as writer:
                writer.index([CorpusDocument(collection, 'a3', 'cpu-fixture', digest,
                    collection + '.md', digest, text)])
        database = KnowledgeDB.open_read_only(path, embeddings)
    return KnowledgeAgent(enabled=enabled, database=database,
        collection=collection if enabled else None, journal=QueryJournal(root / 'queries.jsonl'))


def install_live_trials(monkeypatch, root):
    """Use the public canonical injection seam of each actual LiveComposition trial."""
    from benchmarks.a3kernels.trial import A3TrialLoop
    from test_a3_live_composition import SOURCE
    sessions = []

    def trial(**kwargs):
        actor = getattr(kwargs['actor'], 'canonical_actor', kwargs['actor'])
        session = getattr(actor, 'canonical_session', None) or A3Session(
            initial=SOURCE, repaired=SOURCE + '\n// repaired')
        for attribute, option in [('canonical_clock', 'clock'), ('canonical_budgets', 'budgets')]:
            if hasattr(actor, attribute):
                kwargs[option] = getattr(actor, attribute)
        baseline = a3_knowledge(Path(root) / 'baseline', enabled=True, collection='baseline')
        kwargs['canonical'] = a3_test_adoption(session, baseline)
        result = A3TrialLoop(**kwargs)
        session.owner = result
        sessions.append(session)
        return result

    monkeypatch.setattr('benchmarks.a3kernels.live_composition.A3TrialLoop', trial)
    return sessions


def recording_candidate(*, compile_failures=0, compile_failure_at=(), run_failures=0, run_exit=0,
                        exception=None, exception_stage='compile', clock=None, seconds=0,
                        compile_detail="CPU fixture compile rejected current source"):
    """Trusted CPU host: inject real failed attestations or numerically wrong output."""
    from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt, FailedEvidence, VerifiedResult
    from test_a3_trial_loop import FakeCandidate

    class Candidate(FakeCandidate):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.compiles = self.runs = 0
            self.compiled_results = []

        def record(self, name, source, options):
            self.calls.append((name, source, dict(options)))
            if clock:
                clock.advance(seconds)
            if exception is not None and name == exception_stage:
                raise exception

        def compile(self, source, workdir, **options):
            self.record('compile', source, options)
            self.compiles += 1
            if self.compiles <= compile_failures or self.compiles in compile_failure_at:
                plan = self._plan(source, options['request_id'], options['attempt_id'],
                    options['length'], options['project_id'], options.get('padded_length'),
                    options.get('block_count', 1), options.get('execution_profile', 'gz-a3'))
                return FailedEvidence.create(plan, stage='compile', error_type='CompileError',
                    detail=compile_detail)
            result = super().compile(source, workdir, **options)
            self.compiled_results.append(result)
            return result

        def run(self, source, workdir, **options):
            self.record('run', source, options)
            self.runs += 1
            if self.runs <= run_failures:
                plan = self._plan(source, options['request_id'], options['attempt_id'],
                    options['length'], options['project_id'], options.get('padded_length'),
                    options.get('block_count', 1), options.get('execution_profile', 'gz-a3'))
                output = [a + b for a, b in zip(plan.input_a, plan.input_b)]
                output[0] += 1.0
                return VerifiedResult.from_receipt(plan, ExecutionReceipt(run_exit, tuple(output)),
                                                   max_abs_error=1.0 if run_exit == 0 else None)
            return super().run(source, workdir, **options)

    return Candidate()


class InvalidDecisionsSession(A3Session):
    def __init__(self, *, invalid=1, clock=None, seconds=0, tokens=1, **kwargs):
        super().__init__(**kwargs)
        self.invalid, self.clock, self.seconds, self.tokens = invalid, clock, seconds, tokens
        self.completion_tokens = 0
        self.requests = []

    def next_action(self, context, allowed, **kwargs):
        from coding.pi.session import ActionError
        self.requests.append(context)
        self.completion_tokens += self.tokens
        if self.clock:
            self.clock.advance(self.seconds)
        if self.invalid:
            self.invalid -= 1
            raise ActionError('scripted malformed provider decision')
        return super().next_action(context, allowed, **kwargs)
