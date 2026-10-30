"""Production bindings to existing kernel host actions, not legacy actor loops."""
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
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
from rsi.strategy import QueryUnavailable, StrategyRejected


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def knowledge_identity(knowledge):
    """Freeze selected collection content and admitted negative provenance."""
    explicit = getattr(knowledge, 'canonical_identity', None)
    if explicit is not None:
        return explicit() if callable(explicit) else explicit
    manifests = getattr(knowledge, 'manifests', None)
    manifest = getattr(knowledge, 'manifest', None)
    if manifests is None and manifest is not None:
        manifests = {getattr(knowledge, 'collection', None): manifest}
    if manifests is None:
        if getattr(knowledge, 'enabled', None) is False:
            return dict(enabled=False)
        raise CanonicalConfigurationError('native KDB requires collection manifests or explicit canonical_identity')
    return dict(enabled=getattr(knowledge, 'enabled', True),
        manifests={name: json.loads(value.to_json()) for name, value in manifests.items()},
        negative_snapshots={name: snapshot.manifest for name, snapshot in
                            getattr(knowledge, 'snapshots', {}).items()})


def combine_queries(baseline, experience):
    """Preserve both real retrieval namespaces; never synthesize availability."""
    if any(not isinstance(row, dict) or not isinstance(row.get('results'), (list, tuple))
           for row in (baseline, experience)):
        raise QueryUnavailable('configured baseline/experience query omitted results')
    return dict(results=[*baseline['results'], *experience['results']],
        negative_records=[*baseline.get('negative_records', []), *experience.get('negative_records', [])],
        provenance=dict(baseline=baseline.get('provenance'), experience=experience.get('provenance')))


def frozen_context_bytes(path):
    """Freeze all named context bytes, including a genuinely empty directory."""
    path = Path(path)
    if path.is_symlink():
        raise CanonicalConfigurationError('baseline context must not be a symlink')
    if path.is_file():
        return path.read_bytes()
    if not path.is_dir():
        raise CanonicalConfigurationError('baseline context is absent')
    rows = []
    for child in sorted(path.rglob('*')):
        if child.is_symlink():
            raise CanonicalConfigurationError('baseline context contains a symlink')
        if child.is_file():
            rows.append(dict(path=child.relative_to(path).as_posix(),
                             sha256=sha256(child.read_bytes()).hexdigest()))
    return json.dumps(dict(kind='context-directory-v1', files=rows), sort_keys=True).encode()


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

    checkpoint_fields = ('source', 'feedback', 'observations')

    def checkpoint(self):
        return dict(adapter=type(self).__name__, fields={name:getattr(self,name,None) for name in self.checkpoint_fields})

    def restore_checkpoint(self, snapshot, sources):
        if snapshot is None or snapshot['adapter'] != type(self).__name__:
            raise WorkflowError('native host checkpoint adapter differs')
        expected = sources[self.editable_files[0]]
        if snapshot['fields']['source'] != expected:
            raise WorkflowError('native retained source bytes differ from typed checkpoint')
        if self.source != expected and (not hasattr(self,'source_path') or self.source_path.exists()):
            raise WorkflowError('native source changed outside retained checkpoint')
        if hasattr(self,'source_path') and not self.source_path.exists():
            self.source_path.parent.mkdir(parents=True,exist_ok=True)
            self.source_path.write_text(expected)
        if set(snapshot['fields']) != set(self.checkpoint_fields):
            raise WorkflowError('native checkpoint field contract differs')
        for name,value in snapshot['fields'].items():
            setattr(self,name,value)

    def activity_exception_terminal(self, name, error):
        # Source edits are synchronous local writes, never compiler/job dispatch.
        return name == 'edit' and isinstance(error, (ValueError, OSError))

    def activity_observed_terminal(self, name, value):
        # These adapters own synchronous observed-result host interfaces.
        # Exceptions stay unresolved in DomainActivity; A3 overrides ambiguous
        # FailedEvidence returns below. No model output establishes closure.
        return True

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
        if text != self.source:
            self.write(text)
        return self.project_hash()

    def context(self):
        from benchmarks.ascend.actor_control import ActorControl, ActorStopped
        feedback = self.feedback
        control = ActorControl(self.control_root)
        checkpoint = control.status().get('checkpoint') or {}
        retained = checkpoint.get('feedback', [])
        for row in retained:
            if row not in feedback:
                feedback.append(row)
        try:
            control.boundary(dict(feedback=feedback), feedback)
        except ActorStopped:
            self.workflow.stop()
            raise WorkflowError('workflow stopped by user') from None
        return deepcopy(dict(instruction=self.instruction, feedback=feedback, observations=self.observations,
                             source_contract=self.identity))

    def context_refresh(self, delivered):
        # Consume real inbox receipts only at the shared actor action boundary.
        # The delivered context is an immutable snapshot, never an alias of the
        # host's evolving feedback list.
        before = [row['feedback_id'] for row in delivered.get('feedback', [])]
        fresh = self.context()
        after = [row['feedback_id'] for row in fresh['feedback']]
        return fresh if before != after else None

    def preflight(self, context):
        self.read(self.editable_files[0])

    def call(self, activity, name, category, callback):
        return activity.run(name, category, callback, execution=True)


class KernelActionsHost(SourceHost):
    """A5 and shared A3 adaptive actions retain all their host gates/artifacts."""
    def __init__(self, owner, inputs, *, baseline_query=None):
        self.owner, self.inputs = owner, inputs
        self.baseline_query = baseline_query
        self.actions = inputs.get('action_executor')
        if not callable(self.actions):
            raise CanonicalConfigurationError(
                'kernel execution requires a metadata-backed callable host or explicit CanonicalAdoption')
        self.adaptive = hasattr(self.actions, 'prepared')
        self.store = getattr(self.actions, 'artifacts', None)
        if self.store is None:
            raise CanonicalConfigurationError('kernel host requires its actual artifacts, frozen verifier, memory and KDB; use a configured host or explicit CanonicalAdoption')
        self.runtime = getattr(self.actions, 'executor' if self.adaptive else 'backend', None)
        if self.runtime is None:
            raise CanonicalConfigurationError('kernel host requires its existing execution backend')
        self.instruction = inputs['instruction']
        self.control_root, self.feedback, self.observations = inputs['workspace'], [], []
        self.initial_source = self.source
        request = getattr(self.actions, 'request', None)
        if request is None and inputs.get('context') is None:
            raise CanonicalConfigurationError('kernel canonical binding requires a retained baseline memory context')
        memory = request.pre_wave_memory if request is not None else frozen_context_bytes(inputs['context'])
        experience = self.actions.query_knowledge if self.adaptive else self.actions.knowledge
        self.experience_enabled = getattr(experience, 'enabled', True)
        self.identity = dict(target=fingerprint(dict(runtime=getattr(self.runtime, 'runtime_identity',
            type(self.runtime).__qualname__), prepared=str(getattr(self.actions, 'prepared', '')))),
            provider=inputs['profile'].identity.as_dict(), kdb=fingerprint(knowledge_identity(
                baseline_query if baseline_query is not None else
                self.actions.query_knowledge if self.adaptive else self.actions.knowledge)),
            experience=fingerprint(knowledge_identity(experience)),
            memory=sha256(memory).hexdigest(), verifier=fingerprint(str(
                getattr(self.actions, 'prepared', inputs.get('context')))))

    def checkpoint(self):
        target = self.actions
        while 'observe' in vars(target) and 'actions' in vars(target):
            target = target.actions
        names = ('source','source_sha256','latest','latest_source_sha256','admitted','citations',
                 'host_facts','observed_failures','latest_run_at','last_compile_source_sha256','actions','metered_actions') if self.adaptive else (
                 'latest_run','final_run','final_profile','total_actions','actions','compile_probes','admitted_artifacts')
        return dict(adapter='kernel', adaptive=self.adaptive, source=self.source,
            fields={name:getattr(target,name,None) for name in names},
            feedback=self.feedback, observations=self.observations)

    def restore_checkpoint(self, snapshot, sources):
        if snapshot is None or snapshot['adapter'] != 'kernel' or snapshot['adaptive'] != self.adaptive:
            raise WorkflowError('kernel checkpoint adapter differs')
        expected = sources[self.editable_files[0]]
        if snapshot['source'] != expected:
            raise WorkflowError('kernel checkpoint source bytes differ')
        target = self.actions
        while 'observe' in vars(target) and 'actions' in vars(target):
            target = target.actions
        if self.source != expected:
            if self.adaptive and target.source is not None:
                raise WorkflowError('kernel source changed outside checkpoint')
            if not self.adaptive:
                from benchmarks.a5kernels.trial import _SOURCE_PATHS
                path = target.workspace / _SOURCE_PATHS[target.language]
                if path.exists():
                    raise WorkflowError('kernel source changed outside checkpoint')
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(expected)
        latest = snapshot['fields'].get('latest')
        if latest is not None:
            from rsi.domain_checkpoint import encode
            if encode(latest.prepared) != encode(target.prepared):
                raise WorkflowError('kernel prepared oracle/cases changed on resume')
        if set(snapshot['fields']) != set(self.checkpoint()['fields']):
            raise WorkflowError('kernel checkpoint field contract differs')
        for name,value in snapshot['fields'].items():
            setattr(target,name,value)
        self.feedback, self.observations = snapshot['feedback'], snapshot['observations']

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

    def accepted_checkpoint_valid(self):
        if not self.verified():
            return False
        return self.adaptive or bool(self.actions.final_run and self.actions.final_run.verified.passed
            and self.actions.final_run.candidate_source_sha256 == sha256(self.source.encode()).hexdigest())

    def preflight(self, context):
        if self.baseline_query is not None:
            super().preflight(context)
            return
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
        baseline = self.baseline_query(payload) if self.baseline_query is not None else None
        if baseline is not None and not self.experience_enabled:
            return baseline
        _, raw = self.action('query-knowledge', payload)
        raw = raw.get('knowledge_query', raw)
        negatives = [row for row in self.observations if row['action'] in ('compile', 'run')
                     and ('fail' in json.dumps(row).lower() or 'host-mismatch' in json.dumps(row) or
                          row['observation'].get('host', {}).get('passed') is False)]
        experience = unpack_query(raw, self.store, dict(binding=self.identity['kdb'],
            host_query=self.store.put_json(raw)), negatives)
        return combine_queries(baseline, experience) if baseline is not None else experience

    def verified(self):
        if self.adaptive:
            latest = self.actions.latest
            return bool(latest and self.actions.latest_source_sha256 == self.actions.source_sha256
                        and latest.evidence.verified)
        latest = self.actions.latest_run
        return bool(latest and latest.verified.passed)

    def before_edit(self, stage, edit, operation, activity):
        if not self.adaptive or stage != Stage.OPTIMISATION or edit['source'] == self.source:
            return
        spec = self.actions.request.spec
        if 'sync-performance' not in spec.learning_focus:
            return
        if not self.verified():
            raise WorkflowError('sync-performance comparison requires current host-verified source')
        # Measure the already verified pre-edit source. _ProjectActions._write
        # retains its actual receipt as comparison-* when the source changes;
        # final VALIDATION independently measures the new verified revision.
        if spec.profile_metric is not None:
            name, role = 'request-profile', 'profiling'
        elif spec.timing_lengths:
            name, role = 'request-timing', 'performance-study'
        else:
            raise CanonicalConfigurationError('sync-performance requires a configured profile or timing study')
        current = self.actions.source_sha256
        if any(admitted_role == role and self.store.read_json(reference).get('candidate_sha256') == current
               for reference, admitted_role in self.actions.admitted.items() if admitted_role == role):
            return
        _, raw = self.call(activity, 'pre-edit-' + name, 'profiling', lambda seconds: self.action(name))
        if raw.get('error') or raw.get('passed') is not True:
            raise WorkflowError('pre-edit comparison measurement did not pass; source remains unchanged')
        if not any(admitted_role == role and self.store.read_json(reference).get('candidate_sha256') == current
                   for reference, admitted_role in self.actions.admitted.items() if admitted_role == role):
            raise WorkflowError('pre-edit comparison measurement omitted its actual source-bound receipt')

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


def kernel_adoption(*, baseline_query=None):
    def build(owner, **inputs):
        host = KernelActionsHost(owner, inputs, baseline_query=baseline_query)
        bundle, limits = baseline_contract(owner, owner.max_seconds)
        spec = ProjectSpec('kernel', inputs['instruction'], ('host-kernel-executor',), budgets=limits)
        root = Path(inputs['workspace']) / 'canonical'
        injections = dict(query=host.query, backend_identity=fingerprint(host.identity),
            session_factory=lambda: PiSessionAdapter(model=inputs['profile'].identity.backend_model.value,
                                                     environ=owner.provider.coding_environment()))
        if (root / 'domain.json').exists():
            runner = DomainRunner(root, host, **injections)
            from benchmarks.ascend.actor_control import ActorControl
            if runner.workflow.state['stopped'] and not ActorControl(host.control_root).status()['stop_requested']:
                runner.workflow.resume()  # Keep every pending dispatch and elapsed interval.
            return runner
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
    from benchmarks.ascend.agent_context import WorkingContext
    from benchmarks.a5kernels.phase1_provider import _ProviderEvidenceRecorder
    recorder = _ProviderEvidenceRecorder(runner.backend.host.control_root)
    host = runner.backend.host
    # Reuse the existing direct-provider context policy for this noncoding task.
    # Full host material remains in CAS; summaries never acquire host authority.
    working = WorkingContext([dict(role='system', content=owner.system_prompt),
        dict(role='user', content=continuation.instruction)])
    for row in runner.workflow.state['history']:
        working.append(dict(role='user', content=json.dumps(dict(workflow_step=row))))
    for summary in host_summaries:
        working.append(dict(role='user', content=json.dumps(dict(host_summary=summary))))
    working.pinned = [dict(role=row['role'], content=row['content']) for row in host.feedback]
    terminal = dict(role='user', content=DISTILLATION_PROMPT + '\n' + json.dumps(dict(
        admitted_artifacts=admitted, artifact_payloads=payloads,
        workflow_history=runner.workflow.state['history'][-1:])))
    working.append(terminal)
    tokens, turn = 0, 0
    while True:
        seconds = runner.actor._available()
        driver = owner.with_time_budget(seconds)
        token = runner.workflow.accounting.start('thinking')
        try:
            try:
                messages, summary_tokens = driver._prepare_context(
                    working, continuation.profile, recorder, turn=turn)
                tokens += summary_tokens
                driver._request(recorder, continuation.profile, messages, stage='distillation', turn=turn)
                completion = driver._complete(recorder, continuation.profile, messages,
                                               stage='distillation', turn=turn)
            except Exception as failure:
                recorder.failure(failure, stage='distillation', turn=turn)
                ref = runner.store.put_json(dict(kind='domain-distillation-failure-v1',
                    exception_type=type(failure).__name__, source_hash=host.project_hash(),
                    contract_ref=runner.contract_ref, host_outcome=outcome.status))
                runner.append(dict(type='domain-distillation-failure', reference=ref))
                # The already verified host outcome survives a separate summary
                # failure. No candidate or execution is replayed.
                return replace(outcome, tokens=tokens, distillation_failure=ref,
                               wall_time_s=runner.workflow.accounting.seconds())
        finally:
            runner.workflow.accounting.end(token)
        tokens += completion.completion_tokens
        recorder.response(completion.text, stage='distillation', turn=turn)
        owner._trace('message', dict(role='assistant', content=completion.text))
        try:
            distilled = parse_agent_distillation(completion.text, admitted)
        except ValueError as exc:
            working.append(dict(role='assistant', content=completion.text))
            working.append(dict(role='user', content='Correct terminal JSON: ' + str(exc)))
            turn += 1
            continue
        return replace(outcome, tokens=tokens, terminal_response=completion.text,
                       wall_time_s=runner.workflow.accounting.seconds(),
                       distillation=distilled, admitted_artifacts=admitted, continuation=None)


class A3Host(SourceHost):
    def __init__(self, owner, *, baseline_query=None):
        self.baseline_query = baseline_query
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
        self.failures, self.failure_classification = [], None
        memory_context = owner.memory.project_context()
        # An accepted run appends exactly its own learning entry. Reopening may
        # reuse that admitted baseline only for the retained, exact post-commit
        # bytes; subsequent journal changes must still invalidate the baseline.
        event_path = owner.workdir / 'canonical' / 'events.json'
        if event_path.exists():
            events = [self.store.read_json(ref) for ref in json.loads(event_path.read_text())]
            commit = next((row for row in reversed(events) if row.get('type') == 'domain-memory-commit'), None)
            if commit is not None:
                if memory_context.encode() != self.store.read_bytes(commit['after']):
                    raise CanonicalConfigurationError('native memory changed after retained learning commit')
                memory_context = self.store.read_bytes(commit['before']).decode()
        self.admitted_memory = memory_context
        self.instruction = json.dumps(dict(proposal=asdict(owner.proposal),
            profile=owner.profile.as_dict(), execution_profile=owner.execution_profile,
            source_contract=CANDIDATE_SOURCE_CONTRACT.as_dict(),
            memory=json.loads(memory_context)), default=str)
        self.identity = dict(target=owner.execution_profile, provider=owner.profile.as_dict(),
            kdb=fingerprint(knowledge_identity(baseline_query if baseline_query is not None else owner.knowledge)),
            experience=fingerprint(knowledge_identity(owner.knowledge)),
            memory=sha256(memory_context.encode()).hexdigest(),
            verifier=fingerprint(dict(proposal=asdict(owner.proposal), parameters=dict(owner.proposal.parameters))))

    checkpoint_fields = SourceHost.checkpoint_fields + ('compilation','verified','profile','timing',
        'queries','actions','failures','failure_classification','interpretation')

    def checkpoint(self):
        result = super().checkpoint()
        callback = getattr(self.owner.candidate,'checkpoint_state',None)
        result['candidate_state'] = callback() if callback else None
        return result

    def restore_checkpoint(self, snapshot, sources):
        super().restore_checkpoint(snapshot,sources)
        if snapshot.get('candidate_state') is not None:
            callback = getattr(self.owner.candidate,'restore_checkpoint_state',None)
            if callback is None:
                raise WorkflowError('native candidate checkpoint adapter is unavailable')
            callback(snapshot['candidate_state'])

    def accepted_checkpoint_valid(self):
        from benchmarks.a3kernels.phase1_protocol import VerifiedResult
        return bool(isinstance(self.verified, VerifiedResult) and self.verified.passed and
            self.owner._submit_gate(self.source,self.compilation,self.verified,self.profile) is None)

    def activity_observed_terminal(self, name, value):
        from benchmarks.a3kernels.phase1_protocol import FailedEvidence
        from benchmarks.a3kernels.trial import infrastructure_unavailable
        unavailable = isinstance(value, FailedEvidence) and (
            infrastructure_unavailable(value) or self.owner._retryable_failure(name, value))
        if unavailable:
            # DomainActivity must suspend before returning an unobserved value;
            # retain the native classification/diagnostic at that boundary.
            self.failure_classification = 'infrastructure-unverified'
            self.owner._retain_candidate_failure(len(self.actions), name, value,
                self.failures, self.observations, next_action='terminate-project')
        return not unavailable

    def write(self, text):
        if not text.strip():
            raise ValueError('candidate source must not be empty')
        self.source_path.parent.mkdir(parents=True, exist_ok=True)
        self.source_path.write_text(text)
        self.source = text
        self.compilation = self.verified = self.profile = self.timing = None
        self.actions.append('write_source')

    def query(self, payload):
        baseline = self.baseline_query(payload) if self.baseline_query is not None else None
        if baseline is not None and not self.owner.knowledge.enabled:
            return baseline
        from benchmarks.a3kernels.knowledge_agent import KnowledgeQuery
        self.queries += 1
        rows = self.owner.knowledge.query(KnowledgeQuery(payload['query'], payload.get('limit', 5)))
        if not self.owner.knowledge.enabled:
            raise QueryUnavailable('A3 canonical workflow requires an enabled baseline Knowledge Agent')
        experience = dict(results=[asdict(row) for row in rows], negative_records=[row for row in
            self.observations if row.get('passed') is False], provenance=dict(binding=self.identity['kdb']))
        return combine_queries(baseline, experience) if baseline is not None else experience

    def retain_failure(self, operation, result):
        from benchmarks.a3kernels.phase1_protocol import FailedEvidence, VerifiedResult
        from benchmarks.a3kernels.trial import infrastructure_unavailable
        turn = len(self.actions)
        if isinstance(result, FailedEvidence):
            unavailable = infrastructure_unavailable(result) or self.owner._retryable_failure(operation, result)
            self.owner._retain_candidate_failure(turn, operation, result, self.failures,
                self.observations, next_action='terminate-project' if unavailable else 'write_source')
            if unavailable:
                self.failure_classification = 'infrastructure-unverified'
                raise WorkflowError('A3 retained host operation requires infrastructure observation/recovery')
        elif isinstance(result, VerifiedResult) and not result.passed:
            self.owner._retain_verification_failure(turn, result, self.failures, self.observations)

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
            self.retain_failure('compile', self.compilation)
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
            self.retain_failure('run', self.verified)
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


def a3_adoption(*, baseline_query=None):
    def build(owner, **inputs):
        host = A3Host(owner, baseline_query=baseline_query)
        actor = getattr(owner.actor, 'canonical_actor', owner.actor)
        # Production A3Actor owns an existing registered Pi session. Preserve its
        # exact model/credential route instead of creating a new provider stack.
        session = getattr(actor, 'session', None)
        if session is None:
            raise CanonicalConfigurationError('A3 canonical binding needs the registered Pi actor session')
        def session_factory():
            return PiSessionAdapter.from_existing_session(session)
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
            existing = next((entry for entry in owner.memory.read() if entry['memory'] == memory.as_dict()), None)
            entry_ref = (existing or owner.memory.append(memory))['entry_sha256']
            if runner.latest('domain-memory-commit') is None:
                runner.append(dict(type='domain-memory-commit', entry_sha256=entry_ref,
                    before=host.store.put_bytes(host.admitted_memory.encode()),
                    after=host.store.put_bytes(owner.memory.project_context().encode())))
            if not any(e.kind == EvidenceKind.RESULT.value and e.payload.get('memory_entry_sha256') == entry_ref
                       for e in owner.evidence.entries):
                owner.evidence.append(EvidenceKind.RESULT, owner._identity(dict(verified=host.verified,
                                                                              memory_entry_sha256=entry_ref)))
        from benchmarks.a3kernels.phase1_protocol import VerifiedResult
        return TrialResult('passed' if result['status'] == 'accepted' else
            host.failure_classification or result['status'], len(result['history']),
            getattr(getattr(getattr(owner.actor, 'canonical_actor', owner.actor), 'session', None),
                'completion_tokens', sum((e.get('usage') or {}).get('completion_tokens', 0) or 0 for e in runner.events)),
            host.verified if isinstance(host.verified, VerifiedResult) else None,
            host.profile, host.queries, tuple(host.failures), entry_ref)
    return CanonicalAdoption(build, finish)


class OptimizerHost(SourceHost):
    editable_files = ('candidate.json',)

    def boundary_reached(self, stage):
        return stage != Stage.TEST or self.value is None

    def admit_tool(self, name, stage, args, operation):
        if (self.inputs.get('single_evaluation') and self.evaluation is not None
                and name == 'optimisation_decision' and args['action'] == 'edit'):
            raise StrategyRejected('This public attempt already evaluated its candidate. '
                'Choose profile or noop for this source; a new candidate needs a new public attempt.')

    def admit_proposal(self, stage, args, operation):
        edit = args.get('edit')
        if (self.inputs.get('single_evaluation') and self.evaluation is not None
                and edit is not None and edit['source'] != self.source):
            raise StrategyRejected('The single-evaluation candidate is fixed after TEST. '
                'Retain this source; use noop/profile, or start another public attempt for a new candidate.')

    def activity_exception_terminal(self, name, error):
        from optimizer.control import ObservedBudgetStop
        if name == 'profile' and isinstance(error, ObservedBudgetStop):
            error.validate(self.owner.directory)
            return True
        return super().activity_exception_terminal(name, error)

    def refresh_public_result(self, retained):
        row = self.owner.store.get(self.owner.key)
        # CanonicalAdoption invokes this only for an authenticated accepted
        # checkpoint. A normal run settles the current job, whereas explicit
        # single-evaluation calls leave it available for another public attempt.
        if (row['status'] == 'running' and not row['stop_requested']
                and not self.inputs.get('single_evaluation') and not self.inputs.get('prepare_only')):
            self.owner.store.transition(self.owner.key, 'completed', stage=Stage.ACCEPTANCE.value)
            row = self.owner.store.get(self.owner.key)
        return row

    def on_actor_event(self, kind, data):
        observer = self.owner.memory_observer
        if kind != 'pi-provider-request' or observer is None:
            return
        # Only exact summary rows observed in the actual gateway request count;
        # building/filtering a context or admitting a CAS reference is not exposure.
        shown = []
        def visit(value):
            if isinstance(value, str):
                try:
                    decoded = json.loads(value)
                except (ValueError, TypeError):
                    return
                if not isinstance(decoded, str):
                    visit(decoded)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for row in value.get('memory_summary', []):
                    if row in self.presentable_memory and row not in shown:
                        shown.append(row)
                for item in value.values():
                    visit(item)
        visit(data['request'])
        if shown:
            delivered = observer('presentation', self.submission['memory_snapshot'], dict(memory_summary=shown),
                                 data['reservation_id'])
            if delivered.get('memory_summary') != shown:
                raise WorkflowError('memory availability changed before provider dispatch; refresh context')


    def __init__(self, owner, inputs):
        from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
        from optimizer.memory import verify_snapshot
        self.owner, self.runtime, self.inputs = owner, owner.backend, inputs
        self.boundary_stage = (Stage.TASK_CONSTRUCTION if inputs.get('prepare_only') else
                               Stage.TEST if inputs.get('single_evaluation') else None)
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
        if len(self.spec['signatures']) > owner.max_cases:
            raise ValueError('supplied representatives exceed configured case budget')
        self.baseline = owner._read(owner.state['baseline']) if 'baseline' in owner.state else None
        self.evaluation, self.value = None, None
        self.presentable_memory = []
        self.instruction = json.dumps(dict(submission=self.spec,
            memory=dict(snapshot_id=snapshot['snapshot_id'], entries=len(snapshot['memory_entries']),
                        files={name:row['sha256'] for name,row in snapshot['files'].items()}),
            guide=getattr(getattr(owner.agent, 'agent', owner.agent), 'guide', ''),
            optimizer_context=owner.context, source_format='candidate.json requires source and block_count; '
            'use native optimizer_kernel ABI from the frozen baseline contracts'), default=str)
        self.identity = dict(target=fingerprint(getattr(owner.backend, 'identity', type(owner.backend).__qualname__)),
            provider=owner.agent.identity, kdb=fingerprint(dict(config=self.submission.get('config'),
                available=owner.knowledge is not None)), memory=fingerprint(snapshot), verifier=fingerprint(owner._spec(self.submission)))

    checkpoint_fields = SourceHost.checkpoint_fields + ('spec','baseline','evaluation','value')

    def accepted_checkpoint_valid(self):
        from optimizer.loop import score
        actual = score(self.baseline, self.evaluation) if self.evaluation is not None else None
        return actual is not None and actual == self.value

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
        def compile_candidate(stage, args, operation, seconds):
            from rsi.domain_execution import DomainActivity
            callback = getattr(self.owner.backend, 'compile', None)
            if callback is None:
                raise CanonicalConfigurationError('configured optimizer backend has no compile diagnostics capability')
            candidate = dict(source=args['source'], block_count=args['block_count'])
            directory = self.owner.directory / 'compile' / fingerprint(candidate) / operation
            request = self.runner.latest('actor-request', operation)
            if not request or not request.get('request_id'):
                raise WorkflowError('compile diagnostic requires an actual current actor request')
            import uuid
            invocation = self.store.put_json(dict(kind='domain-tool-invocation-v1', call_id=uuid.uuid4().hex,
                name='compile-diagnostic', operation_id=operation, request_id=request['request_id'],
                contract_ref=self.runner.contract_ref, source_hash=self.project_hash(), arguments=args))
            self.runner.append(dict(type='domain-tool-invocation', operation_id=operation,
                                    reference=invocation))
            report = DomainActivity(self.runner, operation).run('compile-diagnostic', 'compilation',
                lambda remaining: callback(candidate['source'], directory, block_count=candidate['block_count']),
                execution=True, invocation=invocation)
            ref = self.owner._artifact('compile', report)
            self.owner.state['history'].append(dict(tool=dict(action='compile', **args), result=ref, content=report))
            self.owner._save()
            return report
        def read_memory(stage, args, operation, seconds):
            snapshot = self.submission['memory_snapshot']
            index = args['index']
            if type(index) is not int or not 0 <= index < len(snapshot['memory_entries']):
                raise StrategyRejected(
                    f'memory index is outside admitted entries [0, {len(snapshot["memory_entries"])}); '
                    'use read_memory with an admitted index or read_context for a paged tool result')
            row = snapshot['memory_entries'][index]
            if self.owner.memory_observer is not None:
                row = self.owner.memory_observer('read-memory', snapshot, dict(action=dict(index=index), result=row), operation)
            return row
        def read_artifact(stage, args, operation, seconds):
            import base64
            reference, offset = args['sha256'], args.get('offset', 0)
            if offset < 0:
                raise StrategyRejected('artifact offset must be nonnegative')
            job, memory = self.artifact_scope()
            if reference not in job and reference not in memory:
                raise StrategyRejected(json.dumps(dict(
                    reason='reference is outside admitted job and memory artifacts',
                    guidance='Use read_context for paged query results or read_memory for snapshot entries. '
                             'Query receipts are not artifacts; use exact admitted IDs from artifact_scope.',
                    job_ids=sorted(job), memory_ids=sorted(memory))))
            if reference in job:
                data = self.owner.store.read_artifact(self.owner.key, reference)
            else:
                data = base64.b64decode(memory[reference]['base64'], validate=True)
                if sha256(data).hexdigest() != reference:
                    raise ValueError('admitted memory artifact digest mismatch')
            text = data.decode('utf-8')
            result = dict(offset=offset, text=text[offset:offset+8192],
                          next_offset=offset+8192 if len(text)>offset+8192 else None)
            if reference not in job and reference in memory and self.owner.memory_observer is not None:
                result = self.owner.memory_observer('read-artifact', self.submission['memory_snapshot'],
                    dict(action=dict(sha256=reference, offset=offset), result=result), operation)
            return result
        return dict(compile=EvidenceTool(OPTIMIZER_SCHEMAS['compile'], compile_candidate),
                    propose_cases=EvidenceTool(OPTIMIZER_SCHEMAS['cases'], propose, (Stage.TASK_CONSTRUCTION,)),
                    read_memory=EvidenceTool(OPTIMIZER_SCHEMAS['read-memory'], read_memory),
                    read_artifact=EvidenceTool(OPTIMIZER_SCHEMAS['read-artifact'], read_artifact))

    def artifact_scope(self):
        job = {row['sha256'] for row in self.owner.store.artifacts(self.owner.key)}
        memory = {row['sha256']: row for row in self.submission['memory_snapshot']['files'].values()}
        return job, memory

    def context(self):
        if self.owner.store.get(self.owner.key)['stop_requested']:
            self.workflow.stop()
            raise WorkflowError('workflow stopped by user')
        value = super().context()
        job, memory = self.artifact_scope()
        value['artifact_scope'] = dict(job_ids=sorted(job), memory_ids=sorted(memory),
            memory_entries=len(self.submission['memory_snapshot']['memory_entries']),
            query_results='Use returned content/read_context pages; query receipt IDs are not artifact IDs.')
        summaries = dict(memory_summary=[dict(index=index, summary=str(entry)[:1200])
            for index,entry in enumerate(self.submission['memory_snapshot']['memory_entries'])])
        observer = self.owner.memory_observer
        if observer is not None:
            filter_presentation = getattr(observer, 'filter_presentation', None)
            if filter_presentation is not None:
                summaries = filter_presentation(self.submission['memory_snapshot'], summaries)
        self.presentable_memory = summaries['memory_summary']
        value.update(summaries)
        value['workload'] = dict(single_evaluation=bool(self.inputs.get('single_evaluation')),
            evaluated=self.evaluation is not None,
            candidate_locked=bool(self.inputs.get('single_evaluation') and self.evaluation is not None))
        value.update(baseline=self.baseline, history=self.owner.state['history'],
                     best=self.owner.state.get('best'))
        return value

    def retain_evaluation(self, evaluation, args, operation):
        from optimizer.loop import score
        owner = self.owner
        self.evaluation, self.value = evaluation, score(self.baseline, evaluation)
        existing = next((row for row in owner.state['history'] if row.get('domain_operation') == operation), None)
        if existing is not None:
            if self.owner._read(existing['result']) != evaluation:
                raise WorkflowError('retained evaluation history differs from observed terminal result')
            return existing
        candidate = json.loads(self.source)
        ref = owner._artifact('evaluation', evaluation)
        row = dict(candidate_sha256=sha256(candidate['source'].encode()).hexdigest(),
            configuration_sha256=fingerprint(candidate), block_count=candidate['block_count'],
            result=ref, source=owner._artifact('candidate.cpp', candidate['source']),
            content=evaluation, speedup=self.value, summary=args['proposal'], domain_operation=operation)
        owner.state['seen'].append(row['configuration_sha256'])
        owner.state['candidate'] = row['source']
        owner.state['history'].append(row)
        if self.value is not None and self.value > owner.state.get('best', {}).get('speedup', 0):
            owner.state['best'] = dict(row, faster_than_torch=self.value > 1)
        owner._save()
        return row

    def recover_terminal_boundary(self, stage, operation, args, receipts):
        if stage != Stage.TEST or not self.inputs.get('single_evaluation'):
            return None
        if len(receipts) != 1 or receipts[0]['dispatch']['name'] != 'evaluate':
            raise WorkflowError('single evaluation recovery requires exactly its original terminal evaluation')
        self.retain_evaluation(receipts[0]['result'], args, operation)
        reference = self.store.put_json(dict(stage=stage.value, passed=self.value is not None,
            evaluation=self.evaluation, speedup=self.value, source=self.project_hash()))
        return DomainObservation(self.value is not None, (reference,))

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
            if self.inputs.get('single_evaluation') and self.evaluation is not None:
                raise WorkflowError('single-evaluation workload already consumed its candidate evaluation')
            candidate = json.loads(self.source)
            directory = owner.directory / fingerprint(candidate) / operation
            if stage == Stage.TEST:
                owner.store.record_attempt(owner.key, candidate_sha256=sha256(candidate['source'].encode()).hexdigest(),
                                           result='evaluating')
            self.evaluation = self.call(activity, 'evaluate', 'execution', lambda seconds:
                owner.backend.evaluate(candidate['source'], directory, block_count=candidate['block_count']))
            self.value = score(self.baseline, self.evaluation)
            passed = self.value is not None
            self.retain_evaluation(self.evaluation, args, operation)
        if stage == Stage.VALIDATION:
            self.value = activity.run('verify-evaluation', 'execution', lambda seconds:
                score(self.baseline, self.evaluation) if self.evaluation is not None else None, execution=True)
            passed = self.value is not None
        if stage == Stage.OPTIMISATION and args['optimisation_decision']['action'] == 'profile':
            candidate = json.loads(self.source)
            from optimizer.control import ObservedBudgetStop
            try:
                report = self.call(activity, 'profile', 'profiling', lambda seconds:
                    owner.backend.profile(candidate['source'], owner.directory / fingerprint(candidate) / 'profile',
                                          block_count=candidate['block_count']))
            except ObservedBudgetStop as error:
                if owner.store.get(owner.key)['stop_requested'] or self.value is None:
                    raise
                proof = error.validate(owner.directory)
                report = dict(status='unavailable', reason='observed elapsed stop', terminal_receipt=proof)
                owner.store.add_event(owner.key, 'profiling-unavailable', report)
            owner.state['history'].append(dict(tool={'action': 'profile'}, content=report,
                                              result=owner._artifact('profiling', report)))
            owner._save()
        ref = self.store.put_json(dict(stage=stage.value, passed=passed, evaluation=self.evaluation,
                                       speedup=self.value, source=self.project_hash()))
        return DomainObservation(passed, (ref,), self.value is not None if stage == Stage.VALIDATION else None,
                                 performance='not_requested')


def optimizer_adoption(*, session_factory=None, test_backend=False, clock=None):
    def build(owner, **inputs):
        if owner.state.get('pending') and not owner.state.get('canonical_active'):
            raise CanonicalConfigurationError('retained legacy native operation needs explicit canonical checkpoint migration; observe original handles first')
        identity = owner.agent.identity
        if 'model' in owner.state and owner.state['model'] != identity:
            raise CanonicalConfigurationError('optimizer model identity changed')
        if 'model' not in owner.state:
            owner.state['model'] = json.loads(json.dumps(identity))
            owner._save()
        row = owner.store.get(owner.key)
        if row['status'] == 'queued':
            owner.store.transition(owner.key, 'running')
        host = OptimizerHost(owner, inputs)
        agent = getattr(owner.agent, 'agent', owner.agent)
        if session_factory is None and not hasattr(agent, 'model'):
            raise CanonicalConfigurationError('optimizer canonical binding requires a registered Pi model or explicit CanonicalAdoption')
        active = owner.state.get('canonical_active')
        if (active is None or (inputs.get('single_evaluation') and active.get('status') in ('accepted', 'evaluated'))
                or (not inputs.get('prepare_only') and active.get('status') == 'prepared')):
            active = dict(index=owner.state.get('canonical_count', 0) + 1, status='new')
            owner.state['canonical_count'] = active['index']
            owner.state['canonical_active'] = active
            owner._save()
        root = owner.directory / ('canonical-' + str(active['index']))
        injections = dict(query=host.query, backend_identity=fingerprint(host.identity),
                          session_factory=session_factory or (lambda: PiSessionAdapter(model=agent.model)),
                          test_backend=test_backend, **(dict(clock=clock) if clock is not None else {}))
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
        boundary = runner.workflow.state.get('domain_boundary')
        owner.state['canonical_active']['status'] = ('prepared' if boundary == Stage.TASK_CONSTRUCTION.value else
                                                      'evaluated' if boundary == Stage.TEST.value else result['status'])
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
