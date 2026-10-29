"""Compose the unified subject campaign using configured production bindings."""
import argparse
from copy import copy
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import time

from optimizer.jobs import atomic, canonical
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.phase1_adaptive import AdaptiveProjectRequest
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent, CurriculumContext, WaveDecision
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_project import materialize_inputs, evaluate_reference
from benchmarks.a5kernels.phase1_provider import direct_profile
from .project_verifier import ProjectVerifierAgent, bounded_review_messages
from .subject_campaign import SubjectCampaign
from .subject_execution import SubjectExecutor, optimizer_snapshot
from .subjects import load_subjects
from .subject_memory import digest as memory_digest
from .subject_knowledge import SubjectKnowledge, memory_artifacts


def generation_memory(snapshot, subject_id, store, *, max_bytes=8192):
    """Small advisory context; complete immutable records remain in the snapshot."""
    envelope = json.loads(snapshot)
    seed = envelope['initial_memory'].encode('utf-8')
    header = dict(snapshot_sha256=store.put_bytes(snapshot), memory_count=len(envelope['memories']),
        initial_memory=dict(sha256=store.put_bytes(seed), bytes=len(seed), preview=seed[:1024].decode('utf-8', errors='ignore')))
    rows = [header]
    records = list(reversed(envelope['memories']))
    records.sort(key=lambda record: record['provenance'].get('subject_id') != subject_id)
    for record in records:
        proposal = record['proposal']
        row = dict(memory_id=record['memory_id'], version=memory_digest(record),
            subject_id=record['provenance'].get('subject_id'),
            **{key: proposal[key][:384] for key in ('summary', 'preconditions', 'expectations')})
        if len(canonical([*rows, row])) <= max_bytes:
            rows.append(row)
    return tuple(rows)


class _SelectionResults(tuple):
    def __new__(cls, rows, details):
        value = super().__new__(cls, rows)
        value.search_details = details
        return value


def referenced_settings(settings, *, contents=False):
    """Freeze ignored runtime JSON bytes without persisting their contents."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--config'); parser.add_argument('--worker-config')
    options, _ = parser.parse_known_args(settings.get('phase1_arguments', []))
    paths = dict(config=options.config, worker=options.worker_config)
    if isinstance(settings.get('optimizer_worker_config'), str):
        paths['optimizer_worker'] = settings['optimizer_worker_config']
    raw = {role: Path(path).read_bytes() for role, path in paths.items() if path is not None}
    return raw if contents else {role: hashlib.sha256(value).hexdigest() for role, value in raw.items()}


def bounded_completion(provider, profile, messages, remaining_seconds):
    """Use a private provider copy and bound both retry and HTTP timeouts."""
    from benchmarks.a5kernels.phase1_provider import openai_compatible_transport, DirectCompletion
    deadline = provider.clock() + remaining_seconds
    selected = copy(provider)
    selected.retry_max_seconds = remaining_seconds
    transport = provider._transport
    def bounded_transport(**options):
        available = deadline - provider.clock()
        if available <= 0:
            raise TimeoutError('subject role provider elapsed allowance exhausted')
        if transport is openai_compatible_transport:
            from openai import OpenAI
            response = OpenAI(api_key=options['api_key'], base_url=options['base_url'],
                timeout=available, max_retries=0).chat.completions.create(**options['request'])
            result = DirectCompletion(response.choices[0].message.content, response.usage.completion_tokens)
        else:
            # Injected transports own their I/O bounds, just as in the Actor API.
            result = transport(**options)
        if provider.clock() >= deadline:
            raise TimeoutError('subject role provider returned after its elapsed allowance')
        return result
    selected._transport = bounded_transport
    from benchmarks.a5kernels.phase1_live import InfrastructureFailure
    try:
        return selected.complete(profile, messages).text
    except InfrastructureFailure:
        if provider.clock() >= deadline:
            raise TimeoutError('subject role provider elapsed allowance exhausted') from None
        raise


def compose_production(settings, *, target, root, initial_memory, max_seconds, native_options=None):
    """Capture existing CLI composition without invoking its learning loop."""
    from .phase1_cli import main as native_main
    from .production import build_native_adaptive_phase1
    from benchmarks.a5kernels.phase1_adaptive_cli import main as a5_main
    from benchmarks.a5kernels.phase1_adaptive_production import build_adaptive_phase1
    arguments = settings['phase1_arguments']
    if not isinstance(arguments, list) or any(not isinstance(item, str) for item in arguments):
        raise ValueError('phase1_arguments must be a string list')
    forbidden = ('--root', '--objective', '--initial-memory', '--target', '--max-seconds', '--project-target')
    if any(item.split('=', 1)[0] in forbidden for item in arguments):
        raise ValueError('phase1_arguments cannot override subject campaign identity or budget')
    captured = []
    root.mkdir(parents=True, exist_ok=True)
    seed = root / 'initial-memory.bin'
    if seed.exists() and seed.read_bytes() != initial_memory:
        raise ValueError('initial memory changed')
    seed.write_bytes(initial_memory)

    def capture(builder, options):
        built = builder(**options)
        coordinator = getattr(built, 'coordinator', built)
        actor = coordinator.execute_project
        captured.append(dict(coordinator=coordinator, actor=actor,
            provider=actor.actor.provider, profile=actor.profile,
            paths=options.get('paths', options.get('config')), embeddings=options.get('embeddings')))
        return SimpleNamespace(run=lambda: {})

    argv = ['--target', target, 'run', '--root', str(root / 'binding'),
            '--objective', 'Subject curriculum production binding', '--initial-memory', str(seed),
            '--max-seconds', str(max_seconds), *arguments]
    with redirect_stdout(io.StringIO()):
        native_main(argv, native_parallel=False, **(native_options or {}),
            native_builder=lambda **options: capture(build_native_adaptive_phase1, options),
            a5_main=lambda args: a5_main(args, builder=lambda **options: capture(build_adaptive_phase1, options)))
    return captured[0]


def build_subject_campaign(settings, *, target, root, subjects, initial_memory,
                           max_seconds=7200, compose=compose_production, previous_runtime_configs=None):
    subjects = tuple(subjects)
    obsolete = {'memory_admission_mode', 'memory_verification', 'memory_review_seconds',
                'memory_trial_seconds', 'memory_arm_seconds', 'memory_verifier_model',
                'verify_memory_usefulness', 'usefulness_pair_seconds'} & settings.keys()
    if obsolete:
        raise ValueError('Removed memory A/B settings: ' + ', '.join(sorted(obsolete)))
    memory_delivery = settings.get('actor_memory_delivery', 'indexed')
    if memory_delivery not in {'indexed', 'kdb-only'}:
        raise ValueError('actor_memory_delivery must be indexed or kdb-only')
    if memory_delivery == 'kdb-only' and initial_memory:
        raise ValueError('kdb-only Actor memory delivery does not support nonempty initial memory')
    if memory_delivery == 'kdb-only' and any((subject.task or {}).get('kind') == 'optimizer' for subject in subjects):
        raise ValueError('kdb-only Actor memory delivery currently requires graph subjects')
    if target == 'a5' and any((getattr(subject, 'task', None) or {}).get('kind') == 'optimizer'
                              for subject in subjects):
        raise ValueError('A5 Catlass optimizer subjects are unsupported; use qualified graph/performance targets')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    resumed_runtime = (root / 'subject-runtime.json').exists()
    from .runtime_allocation import admit_identity
    identity = admit_identity(root, dict(target=target, max_seconds=max_seconds,
        settings_sha256=hashlib.sha256(canonical({key: value for key, value in settings.items()
            if key != 'native_operation_root'})).hexdigest()),
        referenced_settings(settings, contents=True), previous=previous_runtime_configs)
    bound = compose(settings, target=target, root=root, initial_memory=initial_memory,
                    max_seconds=max_seconds)
    actor, provider, profile = bound['actor'], bound['provider'], bound['profile']
    policy = bound['coordinator'].curriculum.target_policy
    store = ContentAddressedArtifacts(root / 'artifacts')
    plans_path = root / 'project-plans.json'
    plans = json.loads(plans_path.read_bytes()) if plans_path.exists() else {}
    campaign = None

    def role_complete(role):
        name = settings.get(role + '_model')
        selected = profile if name is None else direct_profile(next(
            item for item in MODEL_IDENTITIES if item.backend_model.value == name))
        def complete(messages, *, remaining_seconds):
            return bounded_completion(provider, selected, messages, remaining_seconds)
        complete.model = selected.identity.backend_model.value
        return complete

    curriculum_complete, verifier_complete = role_complete('curriculum'), role_complete('verifier')
    from .pi_project_verifier import session_factory
    verifier_identity_path = root / 'project-verifier-runtime.json'
    verifier_settings = dict(settings)
    saved_verifier = (json.loads(verifier_identity_path.read_bytes())
                      if verifier_identity_path.exists() else {})
    legacy_default = ('verifier_engine' not in settings and
        (saved_verifier.get('engine_origin') == 'legacy-resume' or
         resumed_runtime and not verifier_identity_path.exists()))
    if legacy_default:
        # Historical checkpoints retain their completion engine on resume.
        verifier_settings['verifier_engine'] = 'completion'
    verifier_sessions = session_factory(verifier_settings, provider, verifier_complete)
    from coding.pi import ENGINE_IDENTITY
    verifier_identity = dict(engine='pi' if verifier_sessions is not None else 'completion',
        engine_origin='legacy-resume' if legacy_default else 'configured' if 'verifier_engine' in settings else 'provider-default',
        model=verifier_complete.model, role='project-verifier',
        runtime=dict(ENGINE_IDENTITY, context_policy='native-review-whole-request-v1',
                     verifier_loop='native-reads-standalone-terminal-v1') if verifier_sessions is not None else None,
        context_policy='native-review-whole-request-v1' if verifier_sessions is not None else 'bounded-messages-v1',
        selection_policy='native-read-loop-standalone-terminal-v1' if verifier_sessions is not None else 'json-action-v1')
    if saved_verifier and saved_verifier != verifier_identity:
        raise ValueError('project verifier engine or model changed on resume')
    atomic(verifier_identity_path, canonical(verifier_identity))
    paths, embeddings = bound.get('paths'), bound.get('embeddings')
    contextual_reference = getattr(bound['coordinator'], 'contextual_reference_query', None)
    knowledge = SubjectKnowledge(root, store, lambda: campaign.memory, paths=paths, embeddings=embeddings,
        reference_query=(None if contextual_reference is not None else
                         getattr(bound['coordinator'].curriculum, 'query_knowledge', None)),
        contextual_reference_query=contextual_reference)
    if memory_delivery == 'kdb-only' and knowledge.database_path is None:
        raise ValueError('kdb-only Actor memory delivery requires enabled KDB retrieval')
    reference_query = knowledge.reference_query

    def check(request, payload, *, remaining_seconds):
        from .execution import process_deadline
        item = plans[request.project_id]
        if 'spec' not in item:
            raise ValueError('extra graph checks require a graph subject project')
        change = payload['check']
        if set(change) != {'spec'}:
            raise ValueError('additional check requires a complete variant spec')
        variant = ProjectSpec.from_mapping(change['spec'])
        policy.admit(variant)
        source = store.read_bytes(request.source_sha256).decode('utf-8')
        with process_deadline(time.monotonic() + remaining_seconds):
            result = actor.executor.execute(variant, source)
        refs = {name: store.put_json(value) for name, value in result.artifacts.items()}
        return dict(host_verified=result.evidence.verified, artifacts=refs,
                    classification=result.evidence.classification.value, original_spec=item['spec'],
                    variant_spec=variant.as_dict())

    class RuntimeVerifier(ProjectVerifierAgent):
        def review(self, request, **options):
            identity = hashlib.sha256(canonical(asdict(request))).hexdigest()
            caller = dict(root=root / 'verifier-librarian' / identity, role='verifier',
                task='Audit retained evidence for ' + request.project_id,
                context=dict(project_id=request.project_id, source_sha256=request.source_sha256))
            def query(payload, *, remaining_seconds):
                return knowledge.query(payload, remaining_seconds=remaining_seconds, reference_context=caller)
            def request_check(payload, *, remaining_seconds):
                return check(request, payload, remaining_seconds=remaining_seconds)
            # Each invocation owns its callbacks; concurrent reviews cannot replace
            # another caller's task or additional-check request context.
            reviewer = ProjectVerifierAgent(self.complete, self.read_artifact,
                query_knowledge=query, request_check=request_check if self.request_check is not None else None,
                clock=self.clock, trace=self.trace, session_factory=self.session_factory)
            try:
                return reviewer.review(request, **options)
            finally:
                self.last_trace = reviewer.last_trace

    verifier = RuntimeVerifier(verifier_complete, knowledge.read_artifact,
        query_knowledge=knowledge.query, request_check=check, session_factory=verifier_sessions)
    optimizer_config = settings.get('optimizer_worker_config')
    if isinstance(optimizer_config, str):
        optimizer_config = json.loads(Path(optimizer_config).read_bytes())
    if optimizer_config is not None:
        from .phase2_optimizer_execution import OptimizerTaskExecutor
        def optimizer_factory(directory, task, snapshot):
            runtime = getattr(actor.executor, 'runtime_identity', None)
            if (runtime is None or optimizer_config.get('arch') != runtime.get('architecture', getattr(paths, 'arch', None))
                    or str(optimizer_config.get('physical_device')) != runtime.get('device')):
                raise ValueError('optimizer worker must match the composed target architecture and device')
            return OptimizerTaskExecutor(directory, task, snapshot, optimizer_config)
    else:
        optimizer_factory = None
    executor = SubjectExecutor(actor, optimizer_factory)

    if knowledge.database_path is not None:
        def actor_knowledge(payload, context):
            from .execution import _deadline
            allowance = _deadline.get() - time.monotonic()
            if context.remaining_seconds is not None:
                allowance = min(allowance, context.remaining_seconds)
            if hasattr(campaign, 'budget'):
                allowance = min(allowance, campaign._remaining())
            return knowledge.actor_query(payload, context, remaining_seconds=allowance)
        actor.query_knowledge = actor_knowledge
        actor.observe_knowledge_read = knowledge.actor_read

    def execute(item, snapshot, directory, *, remaining_seconds):
        if memory_delivery == 'kdb-only' and json.loads(snapshot).get('initial_memory'):
            raise ValueError('kdb-only Actor memory delivery does not support nonempty initial memory')
        previous = plans.get(item['project_id'])
        if previous is not None and {key: value for key, value in previous.items() if key != '_ordinal'} != item:
            raise ValueError('retained project plan changed')
        checkpoint = Path(directory) / 'subject-execution.json'
        ordinal = (previous or {}).get('_ordinal')
        if ordinal is None:
            ordinal = (json.loads(checkpoint.read_bytes())['identity']['ordinal'] if checkpoint.exists()
                       else max((row.get('_ordinal', 0) for row in plans.values()), default=0) + 1)
        plans[item['project_id']] = dict(item, _ordinal=ordinal)
        atomic(plans_path, canonical(plans))
        if item.get('task', {}).get('kind') == 'optimizer':
            envelope = json.loads(snapshot)
            seed = envelope['initial_memory'].encode('utf-8')
            files = {'initial-memory.txt': seed}
            for record in envelope['memories']:
                files['memories/' + memory_digest(record) + '.json'] = canonical(record)
                for digest in memory_artifacts(record):
                    files['artifacts/' + digest] = knowledge.read_artifact(digest, remaining_seconds=remaining_seconds)
            entries = [dict(kind='initial-memory', text=envelope['initial_memory'],
                            artifact_sha256=hashlib.sha256(seed).hexdigest()), *envelope['memories']]
            frozen = optimizer_snapshot(entries, files=files, source_identity={'target': target})
            manual = Path(directory) / 'manual-execution.json'
            attempt_id = json.loads(manual.read_bytes())['attempt_id'] if manual.exists() else item['project_id']
            return executor.optimizer(directory, item['task']['task'], frozen,
                attempt_id=attempt_id, remaining_seconds=remaining_seconds, artifacts=store)
        spec = ProjectSpec.from_mapping(item['spec'])
        envelope = json.loads(snapshot)
        branch_artifacts = ContentAddressedArtifacts(Path(directory) / 'artifacts')
        if memory_delivery == 'indexed':
            for memory in envelope['memories']:
                for digest in memory_artifacts(memory):
                    branch_artifacts.put_bytes(knowledge.read_artifact(digest, remaining_seconds=remaining_seconds))
        inputs = materialize_inputs(spec)
        request = AdaptiveProjectRequest(spec, ordinal, snapshot, directory, inputs,
            evaluate_reference(spec, inputs), memory_root=root, remaining_seconds=remaining_seconds,
            publish_learning=False, actor_memory_delivery=memory_delivery)
        contract = item.get('task', {}).get('contract')
        if contract is not None:
            from .phase2_targets import TargetContract
            qualification = ContentAddressedArtifacts(Path(item['task']['artifact_root']))
            target_contract = TargetContract.from_mapping(contract)
            reference = qualification.read_json(target_contract.reference_sha256)
            for digest in [target_contract.reference_sha256, reference['source_sha256'],
                           *reference['artifacts'].values(), *(row['evidence_sha256'] for row in reference['timings'])]:
                store.put_bytes(qualification.read_bytes(digest))
        else:
            target_contract = None
        recovery = item.get('recovery')
        if recovery is None:
            return executor.graph(request, target=target_contract, artifacts=store, policy=policy, allow_device_migration=True,
                binding_root=root / 'binding')
        if (not isinstance(recovery, dict) or set(recovery) != {'reason', 'artifacts'}
                or not isinstance(recovery['reason'], str) or not isinstance(recovery['artifacts'], dict)):
            raise ValueError('recovery requires a reason and retained artifact references')
        if 'before-source' in recovery['artifacts']:
            request = replace(request, initial_source=store.read_bytes(recovery['artifacts']['before-source']).decode('utf-8'))
        for digest in set(recovery['artifacts'].values()):
            branch_artifacts.put_bytes(knowledge.read_artifact(digest, remaining_seconds=remaining_seconds))
        registry = branch_artifacts.put_bytes(canonical(recovery))
        from .project_verifier import artifact_preview
        preview = dict(reason=recovery['reason'][:1024], artifacts=artifact_preview(recovery['artifacts']),
                       registry_sha256=registry, artifact_count=len(recovery['artifacts']))
        previous = actor.immutable_context
        actor.immutable_context = previous + b'\nOriginal failure context (not learned memory; read full registry by SHA): ' + canonical(preview)
        try:
            return executor.graph(request, target=target_contract, artifacts=store, policy=policy, allow_device_migration=True,
                binding_root=root / 'binding')
        finally:
            actor.immutable_context = previous

    def select_wave(context, snapshot, history, *, remaining_seconds, project_limit=64):
        deadline = time.monotonic() + remaining_seconds
        def available():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError('subject wave authoring elapsed allowance exhausted')
            return seconds
        subject = context['current_subject']
        repairs = [row for row in context['repairs'] if row['subject_id'] == subject['subject_id']]
        if subject['task'] is not None:
            task = subject['task']
            item = dict(project_id=f"{subject['subject_id']}-{len(history) + 1}", task=task,
                        repairs=[row['repair_id'] for row in repairs])
            if task.get('kind') == 'target':
                item['spec'] = task['contract']['spec']
            elif task.get('kind') != 'optimizer':
                raise ValueError('subject task kind must be target or optimizer')
            return [item]
        def complete_wave(messages):
            wire = list(messages)
            while True:
                working = bounded_review_messages(wire, {'history_sha256':context['selection_artifacts']['history']}) if context.get('selection_artifacts') else wire
                response = curriculum_complete(working, remaining_seconds=available())
                if context.get('selection_artifacts'):
                    try:
                        action = json.loads(response)
                        if set(action) == {'read_artifact'}:
                            request = action['read_artifact']
                            sha = request['sha256']
                            if sha not in context['selection_artifacts'].values():
                                raise ValueError('artifact is not admitted to transfer selection')
                            data = store.read_bytes(sha)
                            offset = request.get('offset', 0)
                            if type(offset) is not int or not 0 <= offset <= len(data):
                                raise ValueError('invalid artifact offset')
                            observation = dict(sha256=sha, text=data[offset:offset+8192].decode(errors='replace'),
                                next_offset=offset+8192 if offset+8192<len(data) else None)
                            wire.extend((dict(role='assistant', content=response),
                                         dict(role='user', content=json.dumps(observation))))
                            continue
                    except (ValueError, TypeError, KeyError) as error:
                        if response.lstrip().startswith('{'):
                            wire.extend((dict(role='assistant', content=response),
                                         dict(role='user', content=str(error))))
                            continue
                try:
                    header, body = response.strip().split('\n', 1)
                    ids = [row['spec']['project_id'] for row in json.loads(body)['projects']] if header == 'DECISION: WAVE' else []
                    reused = set(ids) & history.keys()
                except (ValueError, KeyError, TypeError):
                    return response  # The ordinary selector supplies structural feedback.
                if not reused:
                    return response
                wire.extend(({'role': 'assistant', 'content': response}, {'role': 'user', 'content':
                    'These project IDs were already executed; author fresh IDs for the same frozen wave context: '
                    + json.dumps(sorted(reused))}))
        def query_wave(action, *, check_remaining=None):
            from benchmarks.a5kernels.knowledge_agent import KnowledgeResult, Citation
            if check_remaining is not None:
                check_remaining()
            result = knowledge.query({'query': action.query, 'limit': action.limit}, remaining_seconds=available(),
                snapshot=snapshot, branch_root=root / 'curriculum-selection' / hashlib.sha256(snapshot).hexdigest(),
                reference_context=dict(root=root / 'curriculum-librarian' / hashlib.sha256(canonical(dict(
                    objective=value.objective, snapshot_sha256=value.pre_wave_memory_sha256,
                    wave_index=value.wave_index))).hexdigest(), role='curriculum', task=value.objective,
                    context=dict(subject_id=subject['subject_id'], wave_index=value.wave_index)))
            hits = result['hits'][:action.limit]
            allowance = max(64, 4096 // max(1, len(hits)))
            rows, pointers = [], []
            for index, hit in enumerate(hits):
                data = hit['text'].encode('utf-8')
                pointers.append(dict(result_index=index, sha256=store.put_bytes(data), bytes=len(data),
                                     truncated=len(data) > allowance))
                rows.append(KnowledgeResult(Citation.from_mapping(hit['citation']),
                    data[:allowance].decode('utf-8', errors='ignore'), hit['score']))
            available()
            return _SelectionResults(rows, dict(snapshot_sha256=hashlib.sha256(snapshot).hexdigest(), artifacts=pointers))
        selector = CurriculumAgent(complete_wave, target_policy=policy,
            unavailable_capabilities=bound['coordinator'].curriculum.unavailable_capabilities,
            query_knowledge=query_wave if knowledge.database_path is not None or reference_query is not None else None,
            require_fresh_project_ids=True)
        objective = subject['objective'] + '\nOpen repair obligations: ' + json.dumps(repairs)
        objective += '\nAuthor experiments that satisfy these obligations; project IDs must be new: ' + json.dumps(list(history))
        value = CurriculumContext(objective=objective, wave_index=len(campaign.state['waves']) + 1,
            pre_wave_memory_sha256=hashlib.sha256(snapshot).hexdigest(), pre_wave_coverage=(),
            prior_outcomes=(), curriculum_memory=generation_memory(snapshot, subject['subject_id'], store),
            remaining_seconds=remaining_seconds, remaining_projects=project_limit)
        decision, _ = selector.select(value, remaining_projects=project_limit)
        if not isinstance(decision, WaveDecision):
            raise ValueError('subject curriculum must author a wave before progression review')
        items = [dict(project_id=project.spec.project_id, spec=project.spec.as_dict(), repairs=[])
                 for project in decision.wave.projects]
        if repairs:
            messages = [{'role': 'system', 'content': (
                'Link only experiments that address these repair obligations. Return JSON '
                '{"links":[{"project_id":"...","repair_ids":["..."]}]}. Include controls '
                'with their probe when both are necessary. Unrelated projects have empty repair_ids.')},
                {'role': 'user', 'content': json.dumps(dict(projects=items, repairs=repairs))}]
            while True:
                response = curriculum_complete(messages, remaining_seconds=available())
                try:
                    links = json.loads(response)['links']
                    by_id = {row['project_id']: row['repair_ids'] for row in links}
                    known_repairs = {row['repair_id'] for row in repairs}
                    if (set(by_id) != {item['project_id'] for item in items}
                            or any(not isinstance(ids, list) or not set(ids) <= known_repairs for ids in by_id.values())):
                        raise ValueError('links must name authored projects and existing repair IDs')
                    for item in items:
                        item['repairs'] = by_id[item['project_id']]
                    break
                except (KeyError, ValueError, TypeError) as error:
                    messages.extend(({'role': 'assistant', 'content': response},
                                     {'role': 'user', 'content': str(error)}))
        available()
        return items

    campaign = SubjectCampaign(root, subjects, initial_memory, select_wave=select_wave,
        execute=execute, verifier=verifier, complete=curriculum_complete,
        max_seconds=max_seconds, index_memory=knowledge.index, memory_ledger=knowledge.ledger)
    verifier.read_artifact = campaign.memory_maintenance.read_artifact
    def review_retained(item, request, **options):
        if item.get('project_id') != request.project_id or 'spec' not in item:
            raise ValueError('retained review requires its original graph project')
        policy.admit(ProjectSpec.from_mapping(item['spec']))
        previous = plans.get(request.project_id)
        if previous is not None and {k: v for k, v in previous.items() if k != '_ordinal'} != item:
            raise ValueError('retained project plan changed')
        plans[request.project_id] = dict(item, _ordinal=(previous or {}).get('_ordinal',
            max((row.get('_ordinal', 0) for row in plans.values()), default=0) + 1))
        atomic(plans_path, canonical(plans))
        return verifier.review(campaign.memory_maintenance.prepare(request), **options)

    campaign.review_retained = review_retained
    if 'check_environment' in bound:
        campaign.check_environment = bound['check_environment']
    from .subject_editing import SubjectEdits, validate_project
    def admit_edit(item):
        observed_runtime = getattr(actor.executor, 'runtime_identity', None)
        if item.get('task', {}).get('kind') == 'target' and observed_runtime is None:
            raise ValueError('target approval requires an observed composed runtime')
        validate_project(item, runtime=observed_runtime, policy=policy)
        if item.get('task', {}).get('kind') == 'optimizer':
            if target == 'a5' or optimizer_factory is None:
                raise ValueError('optimizer projects are unavailable for this campaign runtime')
        else:
            policy.admit(ProjectSpec.from_mapping(item['spec']))
        return item
    campaign.curriculum_edits = SubjectEdits(root, campaign.scheduler, curriculum_complete,
                                             validate_item=admit_edit, target_policy=policy)
    campaign.role_complete = role_complete
    campaign.model_identity = profile.identity.as_dict()
    campaign.runtime_identity = getattr(actor.executor, 'runtime_identity', None)


    return campaign


def main(argv=None, *, builder=build_subject_campaign):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'resume', 'report'))
    parser.add_argument('--subjects', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--target', choices=('a3', 'a5'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--initial-memory', type=Path)
    parser.add_argument('--max-seconds', type=float, default=7200)
    args = parser.parse_args(argv)
    if args.action == 'report':
        from .memory_status import review_report
        state = json.loads((args.root / 'subject-campaign.json').read_bytes())
        state = review_report(state)
        state['subjects'] = json.loads((args.root / 'subjects.json').read_bytes())
        print(json.dumps(state, sort_keys=True))
        return state
    if any(value is None for value in (args.config, args.subjects, args.target, args.initial_memory)):
        parser.error('run/resume requires --config, --subjects, --target and --initial-memory')
    campaign = builder(json.loads(args.config.read_bytes()), target=args.target, root=args.root,
        subjects=load_subjects(args.subjects), initial_memory=args.initial_memory.read_bytes(),
        max_seconds=args.max_seconds)
    result = getattr(campaign, args.action)()
    print(json.dumps(result, sort_keys=True))
    return result
