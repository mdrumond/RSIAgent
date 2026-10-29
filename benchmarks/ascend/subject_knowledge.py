"""Accepted and frozen subject-memory KDB views with exact artifact provenance."""
from dataclasses import asdict
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
import re
import time

from optimizer.jobs import atomic, canonical
from .subject_memory import digest as memory_digest
from .memory_ledger import MemoryLedger


def memory_artifacts(record):
    """Artifacts reachable from one accepted or explicitly supplied draft record."""
    proposal, provenance = record['proposal'], record['provenance']
    result = {proposal['after']['sha256'], *proposal['evidence'],
              *provenance.get('request', {}).get('artifacts', {}).values()}
    if proposal['before'] is not None:
        result.add(proposal['before']['sha256'])
    if provenance.get('review_sha256'):
        result.add(provenance['review_sha256'])
    return result


class SubjectKnowledge:
    """Use the live accepted catalog for review and frozen records for Actors.

    ``memory_getter`` returns the campaign's SubjectMemory after composition.
    Frozen snapshots are indexed solely inside their branch namespace,
    with withdrawn versions excluded from actor retrieval.
    """
    def __init__(self, root, store, memory_getter, *, paths=None, embeddings=None,
                 reference_query=None, contextual_reference_query=None, clock=time.monotonic):
        self.root, self.store, self.memory_getter = Path(root), store, memory_getter
        self.database_path, self.embeddings = getattr(paths, 'kdb', None), embeddings
        self.ledger = (MemoryLedger.for_kdb(self.database_path) if self.database_path is not None
                       else MemoryLedger(self.root / 'memory-ledger.sqlite3'))
        from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
        self.shared_artifacts = ContentAddressedArtifacts(Path(str(self.ledger.path) + '.artifacts'))
        self.reference_query, self.clock = reference_query, clock
        self.contextual_reference_query = contextual_reference_query
        if self.database_path is not None and self.reference_query is None and contextual_reference_query is None:
            from benchmarks.a5kernels.phase1_adaptive_production import curriculum_knowledge
            self.reference_query, _ = curriculum_knowledge(paths, embeddings)

    def index(self, version, record):
        self.shared_artifacts.put_bytes(canonical(record))
        for sha in memory_artifacts(record):
            self.shared_artifacts.put_bytes(self.read_artifact(sha, remaining_seconds=float('inf')))
        if self.database_path is None:
            return
        from benchmarks.a5kernels.knowledge import KnowledgeDB
        directory = self.root / 'memory-index' / version
        directory.mkdir(parents=True, exist_ok=True)
        document = directory / 'memory.json'
        atomic(document, canonical(record))
        with KnowledgeDB(self.database_path, self.embeddings) as database:
            database.index(directory, [document], collection='subject-memory-' + version, language='memory')

    def read_artifact(self, digest, *, remaining_seconds):
        if remaining_seconds <= 0:
            raise TimeoutError('memory artifact read allocation exhausted')
        try:
            return self.store.read_bytes(digest)
        except FileNotFoundError:
            content = self.shared_artifacts.read_bytes(digest)
            self.store.put_bytes(content)
            return content

    def query(self, payload, *, remaining_seconds, snapshot=None, branch_root=None, reference_context=None):
        from benchmarks.a5kernels.knowledge_agent import KnowledgeQuery
        from benchmarks.a5kernels.knowledge_candidates import build_candidate_pool
        action = KnowledgeQuery(payload['query'], payload.get('limit', 5))
        started = self.clock()
        def check_remaining():
            available = remaining_seconds - (self.clock() - started)
            if available <= 0:
                raise TimeoutError('subject knowledge query elapsed allowance exhausted')
            return available
        frozen_records = None if snapshot is None else json.loads(snapshot)['memories']
        records = self.memory_getter().records() if frozen_records is None else frozen_records
        words = set(re.findall(r'\w+', action.query.casefold()))
        memories, documents = [], {}
        for record in records:
            if self.database_path is None and not words & set(re.findall(r'\w+', json.dumps(record).casefold())):
                continue
            # Freeze the original record before adding query-only version and
            # artifact fields. Its full document is exactly the versioned bytes.
            version = memory_digest(record)
            if not self.ledger.available(version):
                continue
            documents[version] = canonical(record)
            memory = dict(record, version=version, artifacts=dict(record.get('artifacts', {})))
            memory['artifacts'].update({str(i): digest
                for i, digest in enumerate(sorted(memory_artifacts(record)))})
            memories.append(memory)
        reference_hits, memory_hits = [], []
        if self.contextual_reference_query is not None:
            caller = reference_context or dict(root=self.root / 'reference-librarian',
                role='verifier', task=action.query, context={})
            reference_hits.extend(self.contextual_reference_query(action,
                check_remaining=check_remaining, **caller))
        elif self.reference_query is not None:
            reference_hits.extend(self.reference_query(action, check_remaining=check_remaining))
        if self.database_path is not None and memories:
            from benchmarks.a5kernels.knowledge import KnowledgeDB
            selected_database = self.database_path
            if frozen_records is not None:
                snapshot_id = hashlib.sha256(snapshot).hexdigest()
                directory = (Path(branch_root) if branch_root is not None else self.root) / 'frozen-memory' / snapshot_id
                directory.mkdir(parents=True, exist_ok=True)
                selected_database = directory / 'knowledge.sqlite'
                with KnowledgeDB(selected_database, self.embeddings) as database:
                    for record in frozen_records:
                        check_remaining()
                        version = memory_digest(record)
                        if version not in documents:
                            continue
                        source = directory / version / 'memory.json'
                        source.parent.mkdir(exist_ok=True)
                        atomic(source, canonical(record))
                        database.index(source.parent, [source], collection='subject-memory-' + version,
                                       language='memory')
            with KnowledgeDB.open_read_only(selected_database, self.embeddings) as database:
                collections = {'subject-memory-' + memory['version']: database for memory in memories}
                for collection in collections:
                    check_remaining()
                    manifest = database.manifest(collection)
                    if (manifest.embedding_model, manifest.embedding_revision) != (self.embeddings.model, self.embeddings.revision):
                        raise ValueError('embedding backend does not match collection manifest')
                memory_hits = list(build_candidate_pool(collections, action, size=100,
                    check_remaining=check_remaining).page(limit=action.limit))
            selected = {hit.citation.collection for hit in memory_hits}
            memories = [memory for memory in memories if 'subject-memory-' + memory['version'] in selected]
        for memory in memories:
            memory['artifacts']['memory-document'] = self.store.put_bytes(documents[memory['version']])
        # Reference retrieval may be an adaptive opaque callback with its own
        # pinned identities. Preserve it, but never let its prefix starve memory.
        hits = [hit for pair in zip_longest(memory_hits, reference_hits) for hit in pair if hit is not None]
        return dict(memories=memories, citations=[asdict(hit.citation) for hit in hits],
                    hits=[dict(citation=asdict(hit.citation), text=hit.text, score=hit.score) for hit in hits])

    def actor_query(self, payload, context, *, remaining_seconds):
        """Present and retain exact results from this Actor's pinned memory view."""
        from core.knowledge_presentation import knowledge_presentation
        from benchmarks.a5kernels.knowledge_agent import KnowledgeQuery, Citation
        from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActionResult
        action = KnowledgeQuery(**payload)
        started = self.clock()
        reference_context = None
        if self.contextual_reference_query is not None:
            reference_context = dict(root=Path(context.request.branch_root) / 'reference-librarian',
                role=context.actor_role, task=context.prepared.instruction,
                context=dict(project=context.request.spec.as_dict()))
        result = self.query({'query': action.query, 'limit': action.limit}, remaining_seconds=remaining_seconds,
            snapshot=context.request.pre_wave_memory, branch_root=context.request.branch_root,
            reference_context=reference_context)
        admitted = {}
        def retain(content):
            digest = context.artifacts.put_bytes(content)
            admitted[digest] = 'knowledge'
            return digest
        selected = result['hits'][:action.limit]
        collections = {hit['citation']['collection'] for hit in selected}
        # A query may run outside the production campaign's eager snapshot copy.
        # Admit only the records actually presented from this frozen view, never
        # unrelated accepted or pending lessons from the shared artifact store.
        for memory in result['memories']:
            if 'subject-memory-' + memory['version'] in collections:
                for digest in set(memory['artifacts'].values()):
                    if self.clock() - started >= remaining_seconds:
                        raise TimeoutError('Actor knowledge artifact copy elapsed allowance exhausted')
                    content = self.read_artifact(digest, remaining_seconds=remaining_seconds - (self.clock() - started))
                    if hashlib.sha256(content).hexdigest() != digest:
                        raise ValueError('memory artifact bytes differ from their identity')
                    retain(content)
        observation = knowledge_presentation(action.query, selected, retain)
        delivered = {row['citation']['collection'] for row in observation['results']}
        for memory in result['memories']:
            if 'subject-memory-' + memory['version'] not in delivered:
                continue
            event_id = context.access_event_id
            if not event_id:
                raise ValueError('Actor memory delivery requires a durable host access event')
            self.ledger.access(event_id + ':' + memory['version'], memory_id=memory['memory_id'],
                version=memory['version'], campaign_id=str(self.root.resolve()),
                project_id=context.request.spec.project_id, actor_id=context.actor_id,
                actor_role=context.actor_role)
        return AdaptiveActionResult(observation,
            citations=tuple(Citation.from_mapping(hit['citation']) for hit in selected),
            admitted_artifacts=admitted)

    def actor_read(self, payload, context, page):
        """Count returned linked content only for versions this actor retrieved."""
        sha = payload['sha256']
        versions = {row['version'] for row in self.ledger.artifact_versions(sha)}
        delivered = [row for row in self.ledger.exposures(str(self.root.resolve()), context.request.spec.project_id)
                     if row['actor_id'] == context.actor_id and row['actor_role'] == context.actor_role
                     and row['version'] in versions]
        if not delivered:
            return  # Reference/project bytes do not inherit unrelated memory authority.
        available = [row for row in delivered if self.ledger.available(row['version'])]
        if not available:
            raise ValueError('memory artifact withdrawn pending evidence-backed resolution')
        if not page.get('text_available') or not page.get('page_bytes'):
            return
        for row in available:
            self.ledger.require_actor_artifact(sha, row['version'])
            self.ledger.access(context.access_event_id + ':' + row['version'],
                **{key: row[key] for key in ('memory_id','version','campaign_id','project_id','actor_id','actor_role')})
