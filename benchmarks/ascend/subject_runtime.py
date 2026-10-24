"""Compose the unified subject campaign using configured production bindings."""
import argparse
from copy import copy
from contextlib import redirect_stdout
from dataclasses import asdict
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
from .project_verifier import ProjectVerifierAgent
from .subject_campaign import SubjectCampaign
from .subject_execution import SubjectExecutor, optimizer_snapshot
from .subjects import load_subjects


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


def compose_production(settings, *, target, root, initial_memory, max_seconds):
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
        native_main(argv, native_parallel=False,
            native_builder=lambda **options: capture(build_native_adaptive_phase1, options),
            a5_main=lambda args: a5_main(args, builder=lambda **options: capture(build_adaptive_phase1, options)))
    return captured[0]


def build_subject_campaign(settings, *, target, root, subjects, initial_memory,
                           max_seconds=7200, compose=compose_production):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(target=target, max_seconds=max_seconds, settings_sha256=hashlib.sha256(canonical(settings)).hexdigest())
    identity_path = root / 'subject-runtime.json'
    if identity_path.exists() and json.loads(identity_path.read_bytes()) != identity:
        raise ValueError('subject runtime settings or target changed on resume')
    atomic(identity_path, canonical(identity))
    bound = compose(settings, target=target, root=root, initial_memory=initial_memory,
                    max_seconds=max_seconds)
    actor, provider, profile = bound['actor'], bound['provider'], bound['profile']
    policy = bound['coordinator'].curriculum.target_policy
    store = ContentAddressedArtifacts(root / 'artifacts')
    plans_path = root / 'project-plans.json'
    plans = json.loads(plans_path.read_bytes()) if plans_path.exists() else {}
    current = {}
    campaign = None

    def role_complete(role):
        name = settings.get(role + '_model')
        selected = profile if name is None else direct_profile(next(
            item for item in MODEL_IDENTITIES if item.backend_model.value == name))
        def complete(messages, *, remaining_seconds):
            return bounded_completion(provider, selected, messages, remaining_seconds)
        return complete

    curriculum_complete, verifier_complete = role_complete('curriculum'), role_complete('verifier')
    paths, embeddings = bound.get('paths'), bound.get('embeddings')
    database_path = getattr(paths, 'kdb', None)
    reference_query = getattr(bound['coordinator'].curriculum, 'query_knowledge', None)
    if database_path is not None and reference_query is None:
        from benchmarks.a5kernels.phase1_adaptive_production import curriculum_knowledge
        reference_query, _ = curriculum_knowledge(paths, embeddings)

    def index(version, record):
        if database_path is None:
            return
        from benchmarks.a5kernels.knowledge import KnowledgeDB
        directory = root / 'memory-index' / version
        directory.mkdir(parents=True, exist_ok=True)
        document = directory / 'memory.json'
        atomic(document, canonical(record))
        with KnowledgeDB(database_path, embeddings) as database:
            database.index(directory, [document], collection='subject-memory-' + version, language='memory')

    def knowledge(payload, *, remaining_seconds):
        from benchmarks.a5kernels.knowledge_agent import KnowledgeQuery, validate_knowledge_hit
        started = time.monotonic()
        def check_remaining():
            if time.monotonic() - started >= remaining_seconds:
                raise TimeoutError('subject knowledge query elapsed allowance exhausted')
        memories = campaign.memory.search(payload['query'])
        for memory in memories:
            memory.setdefault('artifacts', {}).update({str(i): digest for i, digest in enumerate(
                memory['proposal']['evidence'] + [memory['proposal']['after']['sha256']])})
            if memory['proposal']['before'] is not None:
                memory['artifacts']['before'] = memory['proposal']['before']['sha256']
        hits = []
        if reference_query is not None:
            hits.extend(reference_query(KnowledgeQuery(payload['query']), check_remaining=check_remaining))
        if database_path is not None and memories:
            from benchmarks.a5kernels.knowledge import KnowledgeDB
            with KnowledgeDB.open_read_only(database_path, embeddings) as database:
                for memory in memories:
                    check_remaining()
                    collection = 'subject-memory-' + memory['version']
                    hits.extend(validate_knowledge_hit(database, collection, hit)
                                for hit in database.query(collection, payload['query'], limit=5))
        return dict(memories=memories, citations=[asdict(hit.citation) for hit in hits],
                    hits=[dict(citation=asdict(hit.citation), text=hit.text, score=hit.score) for hit in hits])

    def read(digest, *, remaining_seconds):
        return store.read_bytes(digest)

    def check(payload, *, remaining_seconds):
        from .execution import process_deadline
        request = current['request']
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
            current['request'] = request
            return super().review(request, **options)

    verifier = RuntimeVerifier(verifier_complete, read, query_knowledge=knowledge, request_check=check)
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

    if database_path is not None:
        def actor_knowledge(payload, context):
            from core.knowledge_presentation import knowledge_presentation
            from benchmarks.a5kernels.knowledge_agent import KnowledgeQuery, Citation
            from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActionResult
            action = KnowledgeQuery(**payload)
            result = knowledge({'query': action.query}, remaining_seconds=campaign._remaining())
            admitted = {}
            def retain(content):
                digest = context.artifacts.put_bytes(content)
                admitted[digest] = 'knowledge'
                return digest
            observation = knowledge_presentation(action.query, result['hits'][:action.limit], retain)
            return AdaptiveActionResult(observation,
                citations=tuple(Citation.from_mapping(hit['citation']) for hit in result['hits'][:action.limit]),
                admitted_artifacts=admitted)
        actor.query_knowledge = actor_knowledge

    def execute(item, snapshot, directory, *, remaining_seconds):
        plans[item['project_id']] = item
        atomic(plans_path, canonical(plans))
        if item.get('task', {}).get('kind') == 'optimizer':
            envelope = json.loads(snapshot)
            frozen = optimizer_snapshot(envelope['memories'], source_identity={'target': target})
            return executor.optimizer(directory, item['task']['task'], frozen,
                attempt_id=item['project_id'], remaining_seconds=remaining_seconds, artifacts=store)
        spec = ProjectSpec.from_mapping(item['spec'])
        envelope = json.loads(snapshot)
        branch_artifacts = ContentAddressedArtifacts(Path(directory) / 'artifacts')
        for memory in envelope['memories']:
            proposal = memory['proposal']
            reachable = set(proposal['evidence']) | {proposal['after']['sha256']}
            if proposal['before'] is not None:
                reachable.add(proposal['before']['sha256'])
            reachable.update(memory['provenance'].get('request', {}).get('artifacts', {}).values())
            for digest in reachable:
                branch_artifacts.put_bytes(store.read_bytes(digest))
        inputs = materialize_inputs(spec)
        request = AdaptiveProjectRequest(spec, len(plans), snapshot, directory, inputs,
            evaluate_reference(spec, inputs), memory_root=root, remaining_seconds=remaining_seconds,
            publish_learning=False)
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
        return executor.graph(request, target=target_contract, artifacts=store, policy=policy)

    def select_wave(context, snapshot, history, *, remaining_seconds):
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
        selector = CurriculumAgent(lambda messages: curriculum_complete(messages,
            remaining_seconds=available()), target_policy=policy,
            unavailable_capabilities=bound['coordinator'].curriculum.unavailable_capabilities,
            query_knowledge=reference_query, require_fresh_project_ids=True)
        objective = subject['objective'] + '\nOpen repair obligations: ' + json.dumps(repairs)
        objective += '\nAuthor experiments that satisfy these obligations; project IDs must be new: ' + json.dumps(list(history))
        value = CurriculumContext(objective=objective, wave_index=len(campaign.state['waves']) + 1,
            pre_wave_memory_sha256=hashlib.sha256(snapshot).hexdigest(), pre_wave_coverage=(),
            prior_outcomes=(), curriculum_memory=tuple(json.loads(snapshot)['memories']),
            remaining_seconds=remaining_seconds)
        decision, _ = selector.select(value)
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
        max_seconds=max_seconds, index_memory=index)
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
        state = json.loads((args.root / 'subject-campaign.json').read_bytes())
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
