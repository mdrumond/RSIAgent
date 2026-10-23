# Candidate selection across collections

`KnowledgeAgent.query` now selects from a global candidate pool instead of
comparing collection-local reciprocal-rank scores. `KnowledgeDB.query` retains
its historical single-collection behavior.

The default pool reserves 60 candidates for global cosine ranking and 40 for
keyword matches. Keyword candidates come from collection-local FTS5/BM25 lists,
visited round-robin; local BM25 values are never compared across collections.
Unused reservations are filled from the other lane. Identical excerpts and
overlapping excerpts from the same source are suppressed. The selected lanes
use a repeating vector/vector/keyword/vector/keyword pattern: six vector and
four keyword candidates per ten-result batch, and three vector plus two keyword
candidates in the ordinary five-result response when both lanes are available.
Scores in this path are cosine similarities. `vector_rank` is global among
vector candidates; `lexical_rank` remains a collection-local diagnostic.

```python
agent = KnowledgeAgent(
    enabled=True,
    databases={"reference_collection": references, "memory_collection": memory},
    journal=journal,
    snapshots={"memory_collection": snapshot},
)
pool = agent.candidate_pool(KnowledgeQuery("matmul Mmad Fixpipe"),
                            check_remaining=check_remaining)
first = pool.page(offset=0, limit=10)
second = pool.page(offset=10, limit=10)
```

`pool.results` is immutable. Paging neither embeds nor searches again. The
optional `check_remaining` callback is invoked before embedding, between
collections, during scan batches, and during selection/validation; raising a
timeout aborts generation. Callers must also bound database/model operations
in their transport when strict elapsed deadlines are required.

The mapping selects permitted collections in caller configuration, not an
agent action. All selected manifests must share an embedding model and revision.
The gate owns read-only views and closes each once; caller handles stay open.
Existing `database=..., collections=...` construction remains supported. A
candidate pool is not itself an assertion of relevance or correctness; the
adaptive librarian can inspect its excerpts and preserve citation provenance.
