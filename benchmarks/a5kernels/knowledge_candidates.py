"""Global vector candidates and collection-balanced lexical evidence."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from itertools import zip_longest
from typing import Callable

from .knowledge import KnowledgeDB
from .knowledge_agent import KnowledgeQuery, KnowledgeResult, validate_knowledge_hit


@dataclass(frozen=True)
class CandidatePool:
    """One frozen search result; paging never embeds, scans, or reranks again."""
    results: tuple[KnowledgeResult, ...]

    def page(self, offset: int = 0, limit: int = 10) -> tuple[KnowledgeResult, ...]:
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError("page requires a nonnegative offset and positive limit")
        return self.results[offset:offset + limit]


def build_candidate_pool(databases: dict[str, KnowledgeDB], action: KnowledgeQuery, *,
                         size: int = 100, check_remaining: Callable[[], object] | None = None
                         ) -> CandidatePool:
    """Reserve 60% global cosine and 40% local-keyword candidates, then interleave."""
    if type(size) is not int or size < 1:
        raise ValueError("candidate pool size must be positive")
    check = check_remaining or (lambda: None)
    check()
    backend = next(iter(databases.values())).embeddings
    embed = getattr(backend, "embed_query", backend.embed)
    vectors = embed([action.query])
    if len(vectors) != 1:
        raise ValueError("embedding backend must return one query vector")
    all_vector, lexical = [], []
    for name, database in sorted(databases.items()):
        check()
        vector_lane, lexical_lane = database.ranked_candidates(
            name, action.query, limit=size, query_vector=vectors[0], check_remaining=check)
        all_vector.extend((name, hit) for hit in vector_lane)
        lexical.append([(name, hit) for hit in lexical_lane])
    all_vector.sort(key=lambda row: (-row[1].score, row[0], row[1].chunk_id))
    global_ranks = {(name, hit.chunk_id): rank
                    for rank, (name, hit) in enumerate(all_vector, 1)}
    vector_queue = deque(all_vector)
    lexical_queue = deque(row for group in zip_longest(*lexical) for row in group if row)
    seen_text, seen_ranges = set(), {}

    def take(queue, count):
        accepted = []
        while queue and len(accepted) < count:
            check()
            name, hit = queue.popleft()
            manifest = databases[name].manifest(name)
            source = (manifest.repository or name, manifest.revision, hit.path)
            ranges = seen_ranges.setdefault(source, [])
            if hit.text in seen_text or any(
                    hit.start_line <= end and hit.end_line >= start for start, end in ranges):
                continue
            seen_text.add(hit.text)
            ranges.append((hit.start_line, hit.end_line))
            accepted.append((name, replace(hit, vector_rank=global_ranks.get((name, hit.chunk_id)))))
        return accepted

    selected = []
    # Allocate at acceptance time: preselecting vectors would swallow the keyword
    # lane entirely when a small corpus fits inside the vector reservation.
    while len(selected) < size and (vector_queue or lexical_queue):
        for preferred in (vector_queue, vector_queue, lexical_queue,
                          vector_queue, lexical_queue):
            other = lexical_queue if preferred is vector_queue else vector_queue
            accepted = take(preferred, 1) or take(other, 1)
            if accepted:
                name, hit = accepted[0]
                check()
                selected.append(validate_knowledge_hit(databases[name], name, hit))
            if len(selected) == size:
                break
    return CandidatePool(tuple(selected))
