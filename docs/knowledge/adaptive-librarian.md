# Adaptive evidence librarian

`AdaptiveKnowledgeAgent` wraps an existing, selected-collection `KnowledgeAgent`.
Deterministic retrieval remains the default; opt in by returning this wrapper
from an injected knowledge factory. No provider SDK or credentials belong in the
wrapper. Inject `complete(messages, seconds) -> {text, tokens}` with transport
and retry behavior bounded by the supplied elapsed allowance.

```python
from benchmarks.a5kernels.adaptive_knowledge import AdaptiveKnowledgeAgent

librarian = AdaptiveKnowledgeAgent(gate, complete, seconds=90, audit=save_json)
librarian.configure_context(task=task, role="writer", context={"target": "a3"})
results = librarian.query(request, seconds=remaining_arm_seconds)
# Or obtain status, gaps, and raw decisions alongside the original results:
outcome = librarian.last_search_details
```

The librarian initially sees ten original, citation-validated excerpts. It can
inspect the next batch, reformulate the query, read an admitted snapshot artifact,
or finish. Only already-inspected IDs can be selected. Selection/rejection
reasons preserve target/runtime applicability and failed-versus-corrected example
context. The returned excerpt bytes and citations always come from the gate,
not model-generated text. The `execute` adapter exposes selection reasons, status and evidence gaps
to ordinary Actor consumers; tuple-returning callers can inspect the detailed
outcome through `search` or `last_search_details`.

At most100 distinct excerpts are exposed per request as a context/workload bound.
Elapsed time controls execution:90 seconds is shared over the entire wrapper
lifetime, including all candidate searches, provider calls and artifact reads.
Each request is also bounded by its caller's remaining allowance. Keep one
wrapper per arm and pass remaining time through every transport. Provider/tool
exceptions and invalid decisions retain earlier valid selections as `partial`,
or return `insufficient` if none were selected. Empty or fewer-than-requested
results are legitimate abstention. A provider must enforce the elapsed allowance;
post-return checks alone cannot interrupt a blocking transport.

The audit callback receives original model requests/responses, inspected
excerpts, decisions and reasons, artifact reads, token telemetry, errors and
elapsed usage. Treat this as run evidence in ignored storage. The ordinary gate
journal retains citations actually returned to the consumer.

Live acceptance must exercise this librarian with the configured DeepSeek
profile, not merely mocked providers. Freeze diagnostic relevance labels,
collection manifests and snapshots before those calls. The comparison harness
can then use fresh wrappers in adaptive reference and indexed-memory conditions;
never pass hidden rubrics or target repair answers to this wrapper.
