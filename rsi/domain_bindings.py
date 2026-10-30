"""Production bindings to existing kernel host actions, not legacy actor loops."""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import time

from rsi.actor import PiSessionAdapter
from rsi.contracts import BudgetLimits, ProjectSpec, Stage
from rsi.domain_execution import (CanonicalAdoption, CanonicalConfigurationError,
                                  DomainObservation, DomainRunner)
from rsi.skills import builtin_bundle, SkillBundle
from rsi.runtime import WorkflowError
from rsi.strategy import QueryUnavailable


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def baseline_contract(owner, parent_seconds):
    bundle = getattr(owner, 'canonical_skills', None) or builtin_bundle()
    if isinstance(bundle, dict):
        bundle = SkillBundle.from_dict(bundle)
    limits = getattr(owner, 'canonical_budgets', None)
    if isinstance(limits, dict):
        limits = BudgetLimits.from_dict(limits)
    limits = limits or BudgetLimits(actor_seconds=parent_seconds)
    allocation = parent_seconds if limits.actor_seconds is None else min(parent_seconds, limits.actor_seconds)
    return bundle, replace(limits, actor_seconds=allocation)


def unpack_query(raw, store, provenance, negatives):
    """Read the existing host's retained result index; never fabricate misses."""
    if not isinstance(raw, dict) or raw.get('available') is False or raw.get('error'):
        raise QueryUnavailable('configured domain Knowledge Agent is unavailable')
    rows = raw.get('results')
    if raw.get('results_index_sha256'):
        rows = store.read_json(raw['results_index_sha256'])['results']
    if not isinstance(rows, (list, tuple)):
        raise QueryUnavailable('domain Knowledge Agent omitted its result list')
    return dict(results=list(rows), negative_records=list(negatives), provenance=provenance)


class SourceHost:
    editable_files = ('kernel',)

    def project_hash(self):
        return sha256(self.source.encode()).hexdigest()

    def read(self, path):
        if path not in self.editable_files:
            raise ValueError('source path is outside the frozen editable set')
        return dict(content=self.source, project_hash=self.project_hash())

    def edit(self, path, text, expected_file_hash, expected_project_hash):
        self.read(path)
        if self.project_hash() != expected_file_hash or self.project_hash() != expected_project_hash:
            raise ValueError('stale domain source edit')
        self.write(text)
        return self.project_hash()

    def context(self):
        from benchmarks.ascend.actor_control import ActorControl, ActorStopped
        feedback = self.feedback
        try:
            ActorControl(self.control_root).boundary(dict(feedback=feedback), feedback)
        except ActorStopped:
            self.workflow.stop()
            raise WorkflowError('workflow stopped by user') from None
        return dict(instruction=self.instruction, feedback=feedback, observations=self.observations,
                    source_contract=self.identity)

    def preflight(self, context):
        self.read(self.editable_files[0])

    def call(self, activity, name, category, callback):
        return activity.run(name, category, callback, execution=True)


class KernelActionsHost(SourceHost):
    """A5 and shared A3 adaptive actions retain all their host gates/artifacts."""
    def __init__(self, owner, inputs):
        self.owner, self.inputs = owner, inputs
        self.actions = inputs['action_executor']
        self.adaptive = hasattr(self.actions, 'prepared')
        self.store = self.actions.artifacts
        if self.store is None:
            raise CanonicalConfigurationError('kernel binding requires its existing evidence store')
        self.runtime = self.actions.executor if self.adaptive else self.actions.backend
        self.instruction = inputs['instruction']
        self.control_root, self.feedback, self.observations = inputs['workspace'], [], []
        self.initial_source = self.source
        request = getattr(self.actions, 'request', None)
        if request is None and inputs.get('context') is None:
            raise CanonicalConfigurationError('kernel canonical binding requires a retained baseline memory context')
        memory = request.pre_wave_memory if request is not None else Path(inputs['context']).read_bytes()
        self.identity = dict(target=fingerprint(dict(runtime=getattr(self.runtime, 'runtime_identity',
            type(self.runtime).__qualname__), prepared=str(getattr(self.actions, 'prepared', '')))),
            provider=inputs['profile'].identity.as_dict(), kdb=fingerprint(dict(
                query=getattr(getattr(self.actions, 'query_knowledge', None), '__qualname__', None) if self.adaptive else
                    getattr(self.actions.knowledge, 'collection', None))),
            memory=sha256(memory).hexdigest(), verifier=fingerprint(str(
                getattr(self.actions, 'prepared', inputs.get('context')))))

    @property
    def source(self):
        if self.adaptive:
            return self.actions.source or ''
        from benchmarks.a5kernels.trial import _SOURCE_PATHS
        path = self.actions.workspace / _SOURCE_PATHS[self.actions.language]
        return path.read_text() if path.exists() else ''

    def write(self, text):
        self.action('write', dict(slot='kernel', content=text))
        if self.source != text:
            raise ValueError('host rejected source update')

    def action(self, name, payload=None):
        from benchmarks.a5kernels.trial import TrialAction
        outcome = self.actions(TrialAction(name, payload or {}))
        raw = json.loads(outcome.observation)
        self.observations.append(dict(action=name, observation=raw))
        return outcome, raw.get('host', raw)

    def preflight(self, context):
        if self.adaptive:
            if getattr(self.actions.query_knowledge, '__name__', '') == 'disabled_knowledge_action':
                raise CanonicalConfigurationError('canonical kernel workflow requires enabled baseline KDB')
        elif not self.actions.knowledge.enabled:
            raise CanonicalConfigurationError('canonical kernel workflow requires enabled baseline KDB')
        super().preflight(context)

    def context(self):
        from coding.pi.actor import task_contract
        result = super().context()
        result.update(kernel_contract=task_contract(self.actions), target_instructions=self.owner.system_prompt)
        return result

    def on_actor_event(self, kind, data):
        self.owner._trace(kind, data)

    def evidence_tools(self):
        from rsi.actor import EvidenceTool
        from coding.pi.schema import STRING, INTEGER, obj
        def read(stage, args, operation, seconds):
            outcome, raw = self.action('read-artifact', args)
            return raw
        return dict(read_artifact=EvidenceTool(obj(dict(sha256=STRING, offset=INTEGER), ['sha256']), read)) if self.adaptive else {}

    def query(self, payload):
        _, raw = self.action('query-knowledge', payload)
        raw = raw.get('knowledge_query', raw)
        negatives = [row for row in self.observations if row['action'] in ('compile', 'run')
                     and ('fail' in json.dumps(row).lower() or 'host-mismatch' in json.dumps(row) or
                          row['observation'].get('host', {}).get('passed') is False)]
        return unpack_query(raw, self.store, dict(binding=self.identity['kdb'],
            host_query=self.store.put_json(raw)), negatives)

    def verified(self):
        if self.adaptive:
            latest = self.actions.latest
            return bool(latest and self.actions.latest_source_sha256 == self.actions.source_sha256
                        and latest.evidence.verified)
        latest = self.actions.latest_run
        return bool(latest and latest.verified.passed)

    def execute(self, stage, args, operation, activity):
        passed = True
        commands = ['compile', 'run'] if stage == Stage.TEST else ['run'] if stage == Stage.VALIDATION else []
        if stage == Stage.OPTIMISATION and args['optimisation_decision']['action'] == 'profile':
            commands.append('request-profile')
        if stage == Stage.VALIDATION and self.adaptive:
            spec = self.actions.request.spec
            if spec.timing_lengths:
                commands.append('request-timing')
            if spec.profile_metric is not None:
                commands.append('request-profile')
        if stage == Stage.ACCEPTANCE:
            commands.append('submit')
        for name in commands:
            category = 'compilation' if name == 'compile' else 'profiling' if name.startswith('request-') else 'execution'
            outcome, raw = self.call(activity, name, category, lambda seconds, name=name: self.action(name))
            if raw.get('error') or raw.get('passed') is False:
                passed = False
            if name == 'compile' and self.adaptive:
                passed = raw.get('passed') is True
            if name == 'run':
                passed = self.verified()
            if name == 'submit':
                passed = passed and outcome.terminal and self.verified()
                if not self.adaptive:
                    passed = passed and bool(self.actions.final_run and self.actions.final_run.verified.passed)
            if not passed:
                break
        refs = tuple(getattr(self.actions, 'admitted', getattr(self.actions, 'admitted_artifacts', {})))
        return DomainObservation(passed, refs, self.verified() if stage == Stage.VALIDATION else None,
                                 performance='not_requested')


@dataclass(frozen=True)
class KernelContinuation:
    runner: DomainRunner
    profile: object
    instruction: str


def kernel_adoption():
    def build(owner, **inputs):
        host = KernelActionsHost(owner, inputs)
        bundle, limits = baseline_contract(owner, owner.max_seconds)
        spec = ProjectSpec('kernel', inputs['instruction'], ('host-kernel-executor',), budgets=limits)
        root = Path(inputs['workspace']) / 'canonical'
        injections = dict(query=host.query, backend_identity=fingerprint(host.identity),
            session_factory=lambda: PiSessionAdapter(model=inputs['profile'].identity.backend_model.value,
                                                     environ=owner.provider.coding_environment()))
        if (root / 'domain.json').exists():
            return DomainRunner(root, host, **injections)
        return DomainRunner.create(root, spec, bundle, host, **injections)

    def finish(owner, runner, result, inputs):
        from benchmarks.a5kernels.trial import ActorOutcome
        host = runner.backend.host
        admitted = dict(getattr(host.actions, 'admitted', getattr(host.actions, 'admitted_artifacts', {})))
        return ActorOutcome('host-verified' if result['status'] == 'accepted' else
            'user-stopped' if runner.workflow.state['stopped'] else result['status'],
            len(result['history']), None, runner.workflow.accounting.seconds(),
            admitted_artifacts=admitted,
            continuation=KernelContinuation(runner, inputs['profile'], inputs['instruction']))

    return CanonicalAdoption(build, finish, distill_kernel)


def distill_kernel(owner, outcome, *, admitted_artifacts=None, host_summaries=()):
    """Reuse registered direct completion and the existing terminal parser."""
    from benchmarks.a5kernels.phase1_provider import DISTILLATION_PROMPT
    from benchmarks.a5kernels.phase1_memory import parse_agent_distillation
    continuation = outcome.continuation
    if not isinstance(continuation, KernelContinuation):
        raise ValueError('missing retained canonical kernel continuation')
    runner = continuation.runner
    admitted = dict(outcome.admitted_artifacts)
    admitted.update(admitted_artifacts or {})
    payloads, remaining_bytes = {}, 64000
    for ref in admitted:
        data = runner.store.read_bytes(ref)
        if len(data) <= min(8192, remaining_bytes):
            payloads[ref] = data.decode(errors='replace')
            remaining_bytes -= len(data)
    messages = [dict(role='user', content=continuation.instruction), dict(role='user', content=
        DISTILLATION_PROMPT + '\n' + json.dumps(dict(admitted_artifacts=admitted,
            artifact_payloads=payloads, host_summaries=list(host_summaries)[:16],
            workflow_history=runner.workflow.state['history'][-20:])))]
    tokens = 0
    while True:
        seconds = runner.actor._available()
        token = runner.workflow.accounting.start('thinking')
        try:
            completion = owner.provider.complete(continuation.profile, messages, max_seconds=seconds)
        finally:
            runner.workflow.accounting.end(token)
        tokens += completion.completion_tokens
        owner._trace('message', dict(role='assistant', content=completion.text))
        try:
            distilled = parse_agent_distillation(completion.text, admitted)
        except ValueError as exc:
            messages.extend((dict(role='assistant', content=completion.text),
                             dict(role='user', content='Correct terminal JSON: ' + str(exc))))
            continue
        return replace(outcome, tokens=tokens, terminal_response=completion.text,
                       distillation=distilled, admitted_artifacts=admitted, continuation=None)


class A3Host(SourceHost):
    def __init__(self, owner):
        from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
        from benchmarks.a3kernels.candidate import CANDIDATE_SOURCE_CONTRACT
        self.owner, self.runtime = owner, owner.candidate
        self.control_root = owner.workdir
        self.store = ContentAddressedArtifacts(owner.workdir / 'canonical-artifacts')
        self.source_path = owner.workdir / 'canonical-source'
        self.source = self.source_path.read_text() if self.source_path.exists() else ''
        self.compilation = self.verified = self.profile = self.timing = None
        recovery = getattr(owner.actor, 'canonical_recovery_source', None)
        if recovery and not self.source:
            self.source_path.parent.mkdir(parents=True, exist_ok=True)
            self.source_path.write_text(recovery)
            self.source = recovery
        self.feedback, self.observations, self.queries, self.actions = [], [], 0, []
        self.instruction = json.dumps(dict(proposal=asdict(owner.proposal),
            profile=owner.profile.as_dict(), execution_profile=owner.execution_profile,
            source_contract=CANDIDATE_SOURCE_CONTRACT.as_dict(),
            memory=json.loads(owner.memory.project_context())), default=str)
        self.identity = dict(target=owner.execution_profile, provider=owner.profile.as_dict(),
            kdb=fingerprint(dict(collection=getattr(owner.knowledge, 'collection', None),
                                 enabled=owner.knowledge.enabled)),
            memory=fingerprint(dict(cell=owner.memory.cell_id, lineage=owner.memory.lineage_id)),
            verifier=fingerprint(dict(proposal=asdict(owner.proposal), parameters=dict(owner.proposal.parameters))))

    def write(self, text):
        if not text.strip():
            raise ValueError('candidate source must not be empty')
        self.source_path.parent.mkdir(parents=True, exist_ok=True)
        self.source_path.write_text(text)
        self.source = text
        self.compilation = self.verified = self.profile = self.timing = None
        self.actions.append('write_source')

    def query(self, payload):
        from benchmarks.a3kernels.knowledge_agent import KnowledgeQuery
        self.queries += 1
        rows = self.owner.knowledge.query(KnowledgeQuery(payload['query'], payload.get('limit', 5)))
        if not self.owner.knowledge.enabled:
            raise QueryUnavailable('A3 canonical workflow requires an enabled baseline Knowledge Agent')
        return dict(results=[asdict(row) for row in rows], negative_records=[row for row in
            self.observations if row.get('passed') is False], provenance=dict(binding=self.identity['kdb']))

    def execute(self, stage, args, operation, activity):
        from benchmarks.a3kernels.candidate import CandidateCompilation
        from benchmarks.a3kernels.phase1_protocol import FailedEvidence, VerifiedResult
        from benchmarks.a3kernels.profiling import (CandidateBinding, ProfileMetric, ProfileRequest,
                                                   ProfilingTreatment, StudyDimensions)
        from benchmarks.a3_experiments import ProfilingGuidance
        owner = self.owner
        passed = True
        if stage == Stage.TEST:
            self.actions.append('compile')
            self.compilation = self.call(activity, 'compile', 'compilation', lambda seconds:
                owner.candidate.compile(self.source, owner.workdir, request_id=owner.cell.cell_id,
                    attempt_id=operation, project_id=owner.proposal.project_id, length=owner._length(),
                    padded_length=owner._padded_length(), block_count=owner._block_count(), seed=0,
                    execution_profile=owner.execution_profile))
            passed = isinstance(self.compilation, CandidateCompilation)
        if stage in (Stage.TEST, Stage.VALIDATION) and passed:
            if self.compilation is None:
                raise ValueError('A3 validation requires its actual retained compilation')
            self.actions.append('run')
            self.verified = self.call(activity, 'run', 'execution', lambda seconds:
                owner.candidate.run(self.source, owner.workdir, request_id=owner.cell.cell_id,
                    attempt_id=operation, project_id=owner.proposal.project_id, length=owner._length(),
                    padded_length=owner._padded_length(), block_count=owner._block_count(), seed=0,
                    execution_profile=owner.execution_profile))
            passed = isinstance(self.verified, VerifiedResult) and self.verified.passed
            if passed and self.verified.source_fingerprint != self.compilation.plan.source_fingerprint:
                raise ValueError('A3 compiler/verification source identity differs')
        if stage == Stage.VALIDATION and passed:
            binding = CandidateBinding(self.verified.execution_id, self.verified.source_fingerprint)
            if owner.cell.profiling is ProfilingGuidance.WITH_GUIDANCE:
                request = ProfileRequest(binding, StudyDimensions(owner._length(), owner._block_count(), 5, 20),
                    ProfileMetric.PIPE_UTILIZATION, treatment=ProfilingTreatment.ON, execution_profile=owner.execution_profile)
                self.profile = self.call(activity, 'profile', 'profiling', lambda seconds: owner.profiler.profile(request))
                passed = self.profile is not None
            if owner.timing_dimensions is not None:
                self.timing = self.call(activity, 'timing', 'profiling', lambda seconds:
                    owner.profiler.time(binding, owner.timing_dimensions))
        if stage == Stage.ACCEPTANCE:
            self.interpretation = args['proposal']
            passed = owner._submit_gate(self.source, self.compilation, self.verified, self.profile) is None
            self.actions.append('submit')
        observation = dict(stage=stage.value, passed=passed, source=self.project_hash(),
                           compilation=str(self.compilation), verification=str(self.verified))
        self.observations.append(observation)
        reference = self.store.put_json(observation)
        from benchmarks.a3kernels.phase1_evidence import EvidenceKind
        owner.evidence.append(EvidenceKind.ARTIFACT, owner._identity(dict(canonical_stage=stage.value,
            operation_id=operation, artifact_sha256=reference)))
        return DomainObservation(passed, (reference,),
            bool(isinstance(self.verified, VerifiedResult) and self.verified.passed) if stage == Stage.VALIDATION else None,
            performance='not_requested')


def a3_adoption():
    def build(owner, **inputs):
        host = A3Host(owner)
        actor = getattr(owner.actor, 'canonical_actor', owner.actor)
        # Production A3Actor owns an existing registered Pi session. Preserve its
        # exact model/credential route instead of creating a new provider stack.
        session = getattr(actor, 'session', None)
        if session is None:
            raise CanonicalConfigurationError('A3 canonical binding needs the registered Pi actor session')
        def session_factory():
            from rsi.actor import SYSTEM
            session.system_prompt = SYSTEM
            session.first_action_only = True
            adapter = object.__new__(PiSessionAdapter)
            adapter.session = session
            return adapter
        root = owner.workdir / 'canonical'
        injections = dict(query=host.query, backend_identity=fingerprint(host.identity), session_factory=session_factory)
        if (root / 'domain.json').exists():
            return DomainRunner(root, host, **injections)
        bundle, limits = baseline_contract(owner, owner.budgets.max_seconds)
        spec = ProjectSpec(owner.proposal.project_id, host.instruction, ('A3-host-candidate',), budgets=limits)
        return DomainRunner.create(root, spec, bundle, host, **injections)

    def finish(owner, runner, result, inputs):
        from benchmarks.a3kernels.trial import TrialResult
        from benchmarks.a3kernels.phase1_memory import AgentInterpretation, HostFact, ProjectMemory
        from benchmarks.a3kernels.phase1_evidence import EvidenceKind
        host = runner.backend.host
        entry_ref = None
        if result['status'] == 'accepted':
            facts = [HostFact('compile', 'host compiled current candidate', host.compilation.attestation_sha256),
                     HostFact('host-verification', 'host verified current candidate', host.verified.evidence_sha256)]
            if host.profile is not None:
                facts.append(HostFact('profiling', owner._profile_statement(host.profile), host.profile.evidence_sha256))
            if host.timing is not None:
                facts.append(HostFact('profiling', 'host timing samples captured', host.timing.evidence_sha256))
            memory = ProjectMemory(owner.proposal, owner.proposal.project_id, owner.cell.cell_id,
                owner.memory.lineage_id, host.verified.source_fingerprint, tuple(host.actions), tuple(facts),
                (AgentInterpretation(host.interpretation, ()),))
            entry_ref = owner.memory.append(memory)['entry_sha256']
            owner.evidence.append(EvidenceKind.RESULT, owner._identity(dict(verified=host.verified,
                                                                          memory_entry_sha256=entry_ref)))
        return TrialResult('passed' if result['status'] == 'accepted' else result['status'],
            len(result['history']), sum(e.get('completion_tokens', 0) for e in runner.events), host.verified, host.profile, host.queries,
            tuple(str(row) for row in host.observations if row.get('passed') is False), entry_ref)
    return CanonicalAdoption(build, finish)


class OptimizerHost(SourceHost):
    editable_files = ('candidate.json',)

    def __init__(self, owner, inputs):
        from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
        from optimizer.memory import verify_snapshot
        self.owner, self.runtime, self.inputs = owner, owner.backend, inputs
        self.boundary_stage = Stage.TASK_CONSTRUCTION if inputs.get('prepare_only') else None
        self.submission = owner.store.submission(owner.key)
        snapshot = inputs.get('memory_snapshot') or self.submission['memory_snapshot']
        verify_snapshot(snapshot)
        self.submission = dict(self.submission, memory_snapshot=snapshot)
        owner.memory_snapshot = snapshot
        owner.context = inputs.get('context') or {}
        self.store = ContentAddressedArtifacts(owner.directory / 'canonical-artifacts')
        self.control_root, self.feedback, self.observations = owner.directory.parent, [], []
        self.source_path = owner.directory / 'canonical-candidate.json'
        self.source = self.source_path.read_text() if self.source_path.exists() else json.dumps(
            dict(source='', block_count=1))
        self.spec = owner._read(owner.state['spec']) if 'spec' in owner.state else owner._spec(self.submission)
        self.baseline = owner._read(owner.state['baseline']) if 'baseline' in owner.state else None
        self.evaluation, self.value = None, None
        self.instruction = json.dumps(dict(submission=self.spec, memory=snapshot,
            guide=getattr(getattr(owner.agent, 'agent', owner.agent), 'guide', ''),
            optimizer_context=owner.context, source_format='candidate.json requires source and block_count; '
            'use native optimizer_kernel ABI from the frozen baseline contracts'), default=str)
        self.identity = dict(target=fingerprint(getattr(owner.backend, 'identity', type(owner.backend).__qualname__)),
            provider=owner.agent.identity, kdb=fingerprint(dict(config=self.submission.get('config'),
                available=owner.knowledge is not None)), memory=fingerprint(snapshot), verifier=fingerprint(owner._spec(self.submission)))

    def write(self, text):
        value = json.loads(text)
        if (set(value) != {'source', 'block_count'} or not isinstance(value['source'], str)
                or not value['source'].strip() or type(value['block_count']) is not int
                or not 1 <= value['block_count'] <= 65535):
            raise ValueError('candidate.json needs nonempty source and block_count in [1,65535]')
        self.source_path.write_text(text)
        self.source, self.evaluation, self.value = text, None, None

    def query(self, payload):
        if self.owner.knowledge is None:
            raise QueryUnavailable('optimizer requires configured baseline KDB')
        raw = self.owner.knowledge(payload['query'])
        if isinstance(raw, (list, tuple)):
            raw = dict(results=list(raw), available=True)
        negatives = [row for row in self.owner.state['history'] if row.get('speedup') is None
                     and row.get('content') is not None]
        return unpack_query(raw, self.store, dict(binding=self.identity['kdb'],
            host_query=self.store.put_json(raw)), negatives)

    def evidence_tools(self):
        from rsi.actor import EvidenceTool
        from coding.pi.adapters import OPTIMIZER_SCHEMAS
        from optimizer.cases import validate_submission
        def propose(stage, args, operation, seconds):
            if self.baseline is not None:
                raise ValueError('optimizer cases already frozen')
            accepted, rejected = [], []
            for signature in args['signatures']:
                try:
                    if len(self.spec['signatures']) + len(accepted) >= self.owner.max_cases:
                        raise ValueError('case proposal exceeds configured case capacity')
                    validate_submission(dict(self.spec, signatures=[signature]))
                    accepted.append(signature)
                except (ValueError, TypeError, KeyError) as exc:
                    rejected.append(dict(signature=signature, error=str(exc)))
            self.spec['signatures'] += accepted
            self.owner.store.add_event(self.owner.key, 'case-proposals',
                dict(accepted=accepted, rejected=rejected, summary=args['summary']))
            return dict(accepted=accepted, rejected=rejected)
        def read_memory(stage, args, operation, seconds):
            snapshot = self.submission['memory_snapshot']
            index = args['index']
            if type(index) is not int or not 0 <= index < len(snapshot['memory_entries']):
                raise ValueError('memory index is outside the admitted snapshot')
            row = snapshot['memory_entries'][index]
            if self.owner.memory_observer is not None:
                row = self.owner.memory_observer('read-memory', snapshot, dict(result=row), operation)
            return row
        def read_artifact(stage, args, operation, seconds):
            import base64
            reference, offset = args['sha256'], args.get('offset', 0)
            if offset < 0:
                raise ValueError('artifact offset must be nonnegative')
            try:
                data = self.owner.store.read_artifact(self.owner.key, reference)
            except (ValueError, FileNotFoundError):
                row = self.submission['memory_snapshot']['files'].get(reference)
                if row is None:
                    raise ValueError('artifact is absent from admitted job/memory')
                data = base64.b64decode(row['base64'], validate=True)
            text = data.decode('utf-8')
            return dict(text=text[offset:offset+8192], next_offset=offset+8192 if len(text)>offset+8192 else None)
        return dict(propose_cases=EvidenceTool(OPTIMIZER_SCHEMAS['cases'], propose, (Stage.TASK_CONSTRUCTION,)),
                    read_memory=EvidenceTool(OPTIMIZER_SCHEMAS['read-memory'], read_memory),
                    read_artifact=EvidenceTool(OPTIMIZER_SCHEMAS['read-artifact'], read_artifact))

    def context(self):
        if self.owner.store.get(self.owner.key)['stop_requested']:
            self.workflow.stop()
            raise WorkflowError('workflow stopped by user')
        value = super().context()
        value.update(baseline=self.baseline, history=self.owner.state['history'],
                     best=self.owner.state.get('best'))
        return value

    def execute(self, stage, args, operation, activity):
        from optimizer.loop import score
        owner = self.owner
        if stage == Stage.TASK_CONSTRUCTION and self.baseline is None:
            owner.state['spec'] = owner._artifact('frozen-reference-submission', self.spec)
            owner._save()
            self.baseline = self.call(activity, 'prepare', 'execution', lambda seconds:
                owner.backend.prepare(self.spec, owner.directory / 'baseline'))
            if not self.baseline.get('suite_sha256') or not self.baseline.get('cases'):
                raise ValueError('optimizer baseline must retain a nonempty frozen suite')
            owner.state['baseline'] = owner._artifact('baseline', self.baseline)
            owner._save()
        passed = True
        if stage == Stage.TEST:
            candidate = json.loads(self.source)
            directory = owner.directory / fingerprint(candidate) / operation
            if stage == Stage.TEST:
                owner.store.record_attempt(owner.key, candidate_sha256=sha256(candidate['source'].encode()).hexdigest(),
                                           result='evaluating')
            self.evaluation = self.call(activity, 'evaluate', 'execution', lambda seconds:
                owner.backend.evaluate(candidate['source'], directory, block_count=candidate['block_count']))
            self.value = score(self.baseline, self.evaluation)
            passed = self.value is not None
            ref = owner._artifact('evaluation', self.evaluation)
            row = dict(candidate_sha256=sha256(candidate['source'].encode()).hexdigest(),
                configuration_sha256=fingerprint(candidate), block_count=candidate['block_count'],
                result=ref, source=owner._artifact('candidate.cpp', candidate['source']),
                content=self.evaluation, speedup=self.value, summary=args['proposal'])
            owner.state['seen'].append(row['configuration_sha256'])
            owner.state['candidate'] = row['source']
            owner.state['history'].append(row)
            if passed and self.value > owner.state.get('best', {}).get('speedup', 0):
                owner.state['best'] = dict(row, faster_than_torch=self.value > 1)
            owner._save()
        if stage == Stage.VALIDATION:
            self.value = activity.run('verify-evaluation', 'execution', lambda seconds:
                score(self.baseline, self.evaluation) if self.evaluation is not None else None, execution=True)
            passed = self.value is not None
        if stage == Stage.OPTIMISATION and args['optimisation_decision']['action'] == 'profile':
            candidate = json.loads(self.source)
            report = self.call(activity, 'profile', 'profiling', lambda seconds:
                owner.backend.profile(candidate['source'], owner.directory / fingerprint(candidate) / 'profile',
                                      block_count=candidate['block_count']))
            owner.state['history'].append(dict(tool={'action': 'profile'}, content=report,
                                              result=owner._artifact('profiling', report)))
            owner._save()
        ref = self.store.put_json(dict(stage=stage.value, passed=passed, evaluation=self.evaluation,
                                       speedup=self.value, source=self.project_hash()))
        return DomainObservation(passed, (ref,), self.value is not None if stage == Stage.VALIDATION else None,
                                 performance='not_requested')


def optimizer_adoption():
    def build(owner, **inputs):
        row = owner.store.get(owner.key)
        if row['status'] == 'queued':
            owner.store.transition(owner.key, 'running')
        host = OptimizerHost(owner, inputs)
        agent = getattr(owner.agent, 'agent', owner.agent)
        if not hasattr(agent, 'model'):
            raise CanonicalConfigurationError('optimizer canonical binding requires a registered Pi model or explicit CanonicalAdoption')
        active = owner.state.get('canonical_active')
        if (active is None or (inputs.get('single_evaluation') and active.get('status') == 'accepted')
                or (not inputs.get('prepare_only') and active.get('status') == 'prepared')):
            active = dict(index=owner.state.get('canonical_count', 0) + 1, status='new')
            owner.state['canonical_count'] = active['index']
            owner.state['canonical_active'] = active
            owner._save()
        root = owner.directory / ('canonical-' + str(active['index']))
        injections = dict(query=host.query, backend_identity=fingerprint(host.identity),
                          session_factory=lambda: PiSessionAdapter(model=agent.model))
        if (root / 'domain.json').exists():
            runner = DomainRunner(root, host, **injections)
            if runner.workflow.state.pop('domain_boundary', None):
                runner.workflow.resume()
            return runner
        bundle, limits = baseline_contract(owner, max(0.001, row['max_seconds'] - row['elapsed_seconds']))
        return DomainRunner.create(root, ProjectSpec(owner.key, host.instruction, ('optimizer-host',),
            budgets=limits), bundle, host, **injections)

    def finish(owner, runner, result, inputs):
        elapsed = runner.workflow.accounting.seconds()
        owner.store.account_canonical_elapsed(owner.key, runner.contract_ref, elapsed)
        owner.state['canonical_active']['status'] = ('prepared' if runner.workflow.state.get('domain_boundary')
                                                      else result['status'])
        owner._save()
        row = owner.store.get(owner.key)
        if row['status'] == 'running' and not runner.workflow.state.get('domain_boundary') and not (
                inputs.get('single_evaluation') and result['status'] == 'accepted'):
            status = 'completed' if result['status'] == 'accepted' else (
                'stopped' if runner.workflow.state['stopped'] else
                'budget-exhausted' if result['status'] == 'budget_exhausted' else 'failed')
            owner.store.transition(owner.key, status, stage=result['stage'])
        return owner.store.get(owner.key)
    return CanonicalAdoption(build, finish)
