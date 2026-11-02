"""Optional production A3 librarian over pinned references and an immutable memory export."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sqlite3
import time
import uuid

from core.knowledge_presentation import knowledge_presentation
from benchmarks.a5kernels.adaptive_knowledge import AdaptiveKnowledgeAgent
from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.knowledge import CollectionManifest, KnowledgeDB, collection_names
from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, KnowledgeQuery, ProgressiveMemoryJournal
from benchmarks.a5kernels.memory_snapshot import MemorySnapshot
from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActionResult
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_provider import direct_profile
from benchmarks.ascend.time_budget import AttemptTimeBudget


DEEPSEEK = direct_profile(next(item for item in MODEL_IDENTITIES if item.backend_model.value == 'deepseek-flash'))


def _manifest(path, names):
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as connection:
        manifests = {}
        for name in names:
            row = connection.execute('SELECT manifest_json FROM collections WHERE name=?', (name,)).fetchone()
            if row is None:
                raise ValueError('selected knowledge collection is missing')
            manifests[name] = CollectionManifest.from_json(row[0])
        return manifests


def adaptive_knowledge_identity(config):
    manifests = _manifest(config.kdb, collection_names(config.collection))
    memory = None
    if config.memory_kdb is not None:
        if config.memory_collection in manifests:
            raise ValueError('memory collection collides with reference collection')
        extra = _manifest(config.memory_kdb, (config.memory_collection,))
        snapshot = MemorySnapshot(Path(config.memory_snapshot))
        snapshot.bind(extra[config.memory_collection])
        memory = snapshot.fingerprint
        manifests.update(extra)
    if len({(m.embedding_model, m.embedding_revision) for m in manifests.values()}) != 1:
        raise ValueError('reference and memory embedding identities differ')
    return dict(enabled=True, model=DEEPSEEK.identity.as_dict(), seconds=config.librarian_seconds,
                manifests={name: manifest.fingerprint for name, manifest in manifests.items()}, snapshot=memory)


def deepseek_librarian(environment, *, root):
    """A typed native role; host librarian evidence admission remains unchanged."""
    import base64
    from benchmarks.ascend.native_curriculum import NativeSubjectRole
    role = NativeSubjectRole(Path(root), DEEPSEEK, environment)
    def evidence():
        if role.last_store is None:
            return {}
        evidence = {}
        for path in role.last_store.objects.glob('*/*'):
            reference = path.parent.name + path.name
            evidence[reference] = base64.b64encode(role.last_store.read_bytes(reference)).decode('ascii')
        return evidence
    def complete(messages, seconds, *, validate=None):
        role.validator = validate
        response = role(messages, remaining_seconds=seconds)
        usage = role.last_receipt['usage']
        tokens = sum(row.get('totalTokens', row.get('input', 0)+row.get('output', 0)) for row in usage)
        return dict(text=response, tokens=tokens, native_evidence=evidence())
    complete.native_validation = True
    complete.native_evidence = evidence
    return complete


class _Results(tuple):
    def __new__(cls, results, details):
        value = super().__new__(cls, results)
        value.search_details = details
        return value


class NativeAdaptiveKnowledge:
    def __init__(self, config, embeddings, complete):
        self.config, self.embeddings, self.complete = config, embeddings, complete
        self.identity = adaptive_knowledge_identity(config)
        manifest = next(iter(_manifest(config.kdb, collection_names(config.collection)).values()))
        if (manifest.embedding_model, manifest.embedding_revision) != (embeddings.model, embeddings.revision):
            raise ValueError('adaptive knowledge embedding backend differs from manifest')
        self.selection = None

    def _gate(self, root):
        config = self.config
        if adaptive_knowledge_identity(config) != self.identity:
            raise ValueError('adaptive knowledge collection identity changed')
        with ExitStack() as stack:
            reference = stack.enter_context(KnowledgeDB.open_read_only(config.kdb, self.embeddings))
            stores = {name: reference for name in collection_names(config.collection)}
            snapshots = {}
            if config.memory_kdb is not None:
                stores[config.memory_collection] = stack.enter_context(
                    KnowledgeDB.open_read_only(config.memory_kdb, self.embeddings))
                snapshots[config.memory_collection] = MemorySnapshot(Path(config.memory_snapshot))
            return KnowledgeAgent(enabled=True, databases=stores, snapshots=snapshots,
                                  journal=ProgressiveMemoryJournal(root / 'knowledge.jsonl'))

    def _search(self, action, root, role, task, context, remaining):
        root.mkdir(parents=True, exist_ok=True)
        budget = AttemptTimeBudget(root / 'librarian-budget.json', self.config.librarian_seconds)
        try:
            seconds = min(budget.deadline-time.monotonic(), remaining())
            if seconds <= 0:
                return (), dict(status='insufficient', gaps=['Shared librarian elapsed allowance exhausted.'])
            gate = self._gate(root)
            audit_root = root / 'librarian-audit'
            audit_root.mkdir(exist_ok=True)
            def retain(audit):
                retained = getattr(self.complete, 'native_evidence', None)
                if retained and audit.get('requests') and 'response' not in audit['requests'][-1]:
                    audit['native_failure_evidence'] = retained()
                (audit_root / (uuid.uuid4().hex + '.json')).write_bytes(canonical_bytes(audit))
            seconds = min(budget.deadline-time.monotonic(), remaining())
            if seconds <= 0:
                gate.close()
                return (), dict(status='insufficient', gaps=['Shared librarian elapsed allowance exhausted.'])
            agent = AdaptiveKnowledgeAgent(gate, self.complete, seconds=seconds, audit=retain)
            agent.configure_context(task=task, role=role, context=context)
            try:
                outcome = agent.search(action, seconds=min(seconds, remaining()))
            finally:
                agent.close()
            remaining()
            return outcome.results, dict(status=outcome.status, gaps=list(outcome.gaps),
                librarian_audit=outcome.audit,
                selection_reasons=[dict(citation=outcome.audit['selected_citations'][key], reason=reason)
                    for key, reason in outcome.audit['selected'].items()])
        finally:
            budget.close()

    def context(self):
        return dict(target=self.config.target, arch=self.config.arch,
                    environment=self.config.environment_identity or {})

    def reference_query(self, action, *, root, role, task, context, check_remaining):
        """Search for one explicit caller without borrowing curriculum state."""
        results, details = self._search(action, Path(root), role, task,
            dict(context, **self.context()), check_remaining)
        return _Results(results, details)

    def actor_query(self, payload, context):
        if set(payload) - {'query', 'limit'}:
            raise ValueError('knowledge action fields must be query and optional limit')
        action = KnowledgeQuery(payload.get('query'), payload.get('limit', 5))
        deadline = time.monotonic() + (context.remaining_seconds if context.remaining_seconds is not None else self.config.librarian_seconds)
        def remaining():
            value = deadline-time.monotonic()
            if value <= 0:
                raise TimeoutError('actor knowledge caller elapsed allowance exhausted')
            return value
        results, details = self._search(action, Path(context.request.branch_root), 'writer',
            context.prepared.instruction, dict(project=context.request.spec.as_dict(), **self.context()), remaining)
        admitted = {}
        def retain(data):
            digest = context.artifacts.put_bytes(data)
            admitted[digest] = 'knowledge'
            return digest
        observation = knowledge_presentation(action.query, [dict(citation=result.citation.as_dict(),
            text=result.text, score=result.score) for result in results], retain)
        observation.update(status=details['status'], search_details_sha256=retain(canonical_bytes(details)))
        return AdaptiveActionResult(observation, citations=tuple(result.citation for result in results),
                                    admitted_artifacts=admitted)

    def begin_selection(self, context):
        # A normal wave and each lesson assessment can share a wave/prior pair.
        # Bind the authored objective too; retries of the same selection reuse it.
        selection_id = hashlib.sha256(canonical_bytes(dict(
            objective=context.objective, wave_index=context.wave_index,
            pre_wave_memory_sha256=context.pre_wave_memory_sha256,
            coverage=[item.value for item in context.pre_wave_coverage],
            prior_projects=[item.project_id for item in context.prior_outcomes]))).hexdigest()
        parent = Path(self.config.root) / 'curriculum-librarian'
        legacy = parent / f'{context.wave_index:03}-{context.pre_wave_memory_sha256}'
        root = parent / f'{legacy.name}-{selection_id}'
        old_budget = legacy / 'librarian-budget.json'
        if old_budget.is_file():
            # Old directories did not bind the objective. Conservatively inherit
            # their spent allowance, including an interrupted active interval,
            # once per selection. Subsequent selections own independent files.
            root.mkdir(parents=True, exist_ok=True)
            try:
                with (root / 'librarian-budget.json').open('xb') as stream:
                    stream.write(old_budget.read_bytes())
            except FileExistsError:
                pass
        self.selection = (root, context.objective,
            dict(wave_index=context.wave_index, coverage=[item.value for item in context.pre_wave_coverage],
                 prior_projects=[item.project_id for item in context.prior_outcomes], **self.context()))

    def curriculum_query(self, action, *, check_remaining=None):
        if self.selection is None:
            raise ValueError('curriculum librarian requires a selection context')
        root, objective, context = self.selection
        results, details = self._search(action, root, 'curriculum', objective, context,
            check_remaining or (lambda: self.config.librarian_seconds))
        return _Results(results, details)
