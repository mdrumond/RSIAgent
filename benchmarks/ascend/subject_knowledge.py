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
    Frozen snapshots, including pending treatment proposals, are indexed solely
    inside their branch namespace, never into the accepted/reference database.
    """
    def __init__(self, root, store, memory_getter, *, paths=None, embeddings=None,
                 reference_query=None, clock=time.monotonic):
        self.root, self.store, self.memory_getter = Path(root), store, memory_getter
        self.database_path, self.embeddings = getattr(paths, 'kdb', None), embeddings
        self.reference_query, self.clock = reference_query, clock
        if self.database_path is not None and self.reference_query is None:
            from benchmarks.a5kernels.phase1_adaptive_production import curriculum_knowledge
            self.reference_query, _ = curriculum_knowledge(paths, embeddings)

    def index(self, version, record):
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
        return self.store.read_bytes(digest)

    def query(self, payload, *, remaining_seconds, snapshot=None, branch_root=None):
        from benchmarks.a5kernels.knowledge_agent import KnowledgeQuery
        from benchmarks.a5kernels.knowledge_candidates import build_candidate_pool
        action = KnowledgeQuery(payload['query'], payload.get('limit', 5))
        started = self.clock()
        def check_remaining():
            if self.clock() - started >= remaining_seconds:
                raise TimeoutError('subject knowledge query elapsed allowance exhausted')
        if snapshot is None:
            frozen_records = None
            memories = (self.memory_getter().search(action.query) if self.database_path is None else
                        [dict(record, version=memory_digest(record)) for record in self.memory_getter().records()])
        else:
            words = set(re.findall(r'\w+', action.query.casefold()))
            frozen_records = json.loads(snapshot)['memories']
            memories = [dict(record, version=memory_digest(record)) for record in frozen_records
                        if self.database_path is not None or words & set(re.findall(r'\w+', json.dumps(record).casefold()))]
        for memory in memories:
            memory.setdefault('artifacts', {}).update({str(i): digest
                for i, digest in enumerate(sorted(memory_artifacts(memory)))})
        reference_hits, memory_hits = [], []
        if self.reference_query is not None:
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
        result = self.query({'query': action.query, 'limit': action.limit}, remaining_seconds=remaining_seconds,
                            snapshot=context.request.pre_wave_memory, branch_root=context.request.branch_root)
        admitted = {}
        def retain(content):
            digest = context.artifacts.put_bytes(content)
            admitted[digest] = 'knowledge'
            return digest
        observation = knowledge_presentation(action.query, result['hits'][:action.limit], retain)
        return AdaptiveActionResult(observation,
            citations=tuple(Citation.from_mapping(hit['citation']) for hit in result['hits'][:action.limit]),
            admitted_artifacts=admitted)
