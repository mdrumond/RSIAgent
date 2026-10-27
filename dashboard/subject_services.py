"""Role-only production bindings; never acquire the campaign admission lease."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace


def configured_subject(config_path, campaign_id):
    config = json.loads(Path(config_path).read_bytes())
    entry = next((row for row in config.get('subject_campaigns', []) if row['id'] == campaign_id), None)
    if entry is None:
        raise ValueError('unknown subject campaign')
    if entry.get('target') not in {'a3', 'a5'}:
        raise ValueError('subject campaign requires a configured target')
    root = Path(entry['root'])
    if not (root / 'subject-campaign.json').is_file():
        raise ValueError('subject campaign has no retained state')
    return entry, root


def subject_editor(root, complete=None, *, validate_item=None, policy=None):
    from benchmarks.ascend.subject_editing import SubjectEdits
    from benchmarks.ascend.subjects import Subject, SubjectScheduler
    state = json.loads((Path(root) / 'subjects.json').read_bytes())
    scheduler = SubjectScheduler(root, [Subject.from_mapping(row) for row in state['subjects']])
    options = {} if validate_item is None else dict(validate_item=validate_item)
    return SubjectEdits(root, scheduler, complete, target_policy=policy, **options)


@contextmanager
def bind_subject(config_path, campaign_id, *, max_seconds, compose=None):
    from benchmarks.ascend.subject_runtime import compose_production, bounded_completion, referenced_settings
    from benchmarks.ascend.subject_memory import SubjectMemory
    from benchmarks.ascend.subject_knowledge import SubjectKnowledge
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import direct_profile
    from optimizer.jobs import canonical
    entry, root = configured_subject(config_path, campaign_id)
    settings = json.loads(Path(entry['config']).read_bytes())
    identity = json.loads((root / 'subject-runtime.json').read_bytes())
    if (identity['target'] != entry['target']
            or identity['settings_sha256'] != hashlib.sha256(canonical(settings)).hexdigest()
            or identity['referenced_json_sha256'] != referenced_settings(settings)):
        raise ValueError('subject runtime settings changed; reconcile configuration first')
    initial = Path(entry['initial_memory']).read_bytes()
    memory = SubjectMemory(root / 'memory', initial)
    store = ContentAddressedArtifacts(root / 'artifacts')
    directory = root / 'dashboard-bindings'
    directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory, prefix='role-') as temporary:
        bound = (compose or compose_production)(settings, target=entry['target'], root=Path(temporary),
                                               initial_memory=initial, max_seconds=identity['max_seconds'])
        provider, profile = bound['provider'], bound['profile']
        policy = bound['coordinator'].curriculum.target_policy
        knowledge = SubjectKnowledge(root, store, lambda: memory, paths=bound.get('paths'),
            embeddings=bound.get('embeddings'),
            reference_query=getattr(bound['coordinator'].curriculum, 'query_knowledge', None))
        def complete(role, messages, *, remaining_seconds):
            name = settings.get(role + '_model', settings.get('curriculum_model') if role == 'librarian' else None)
            selected = profile if name is None else direct_profile(next(
                row for row in MODEL_IDENTITIES if row.backend_model.value == name))
            return bounded_completion(provider, selected, messages, min(max_seconds, remaining_seconds))
        yield SimpleNamespace(root=root, entry=entry, settings=settings, bound=bound,
            provider=provider, profile=profile, policy=policy, knowledge=knowledge, complete=complete)
