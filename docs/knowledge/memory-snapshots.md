# Publishing retained memory

Publication is explicit. It neither updates producer journals nor modifies old
collections. Load retained memory through `Phase1ProjectMemory.from_mapping`,
then construct one `MemorySource` per record. Its `root` is the directory against
which `memory.artifact_store` resolves (usually the memory journal's parent).

```python
from benchmarks.a5kernels.memory_snapshot import MemorySource, export_snapshot

snapshot = export_snapshot([
    MemorySource(memory, journal.parent, provenance={
        "target": "A3", "runtime": exact_runtime_version,
        "source_revision": exact_revision,
    }),
], new_snapshot_directory)
manifest = snapshot.index(new_database, collection="memory-snapshot-001")
```

Use a new database file to preserve an entire historical database. Indexing into
an existing database adds a separate immutable collection; it never replaces a
collection. Check `snapshot.manifest["excluded"]` before accepting a publication:
a missing or mismatched artifact excludes its whole memory record. Source roots
are not exported. Caller provenance must use portable identities and must not
contain credentials or machine-specific paths.

The original structured memory preserves host facts separately from agent
interpretations and lessons. Artifact hashes join these claims to exact retained
code, reports, and other evidence. UTF-8 artifacts become searchable documents;
binary artifacts remain available by hash. Identical memory and artifact content
is stored once with all producer provenance. Supply exact target, runtime,
toolchain and dependency identities in provenance when the original memory does
not contain them; the exporter does not infer missing facts.

```python
from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent

gate = KnowledgeAgent(enabled=True, database=database, journal=query_journal,
                      collections=(manifest.collection,),
                      snapshots={manifest.collection: snapshot})
page = gate.read_artifact(manifest.collection, digest, limit=16384)
```

The gate validates that each supplied snapshot exactly matches its selected
collection. Read pages contain bytes, total size, and the next offset; callers
can request subsequent pages. Each read revalidates the complete artifact hash.
Reads allow at most 65536 bytes per page and cannot access another collection or
an artifact absent from the selected snapshot. The snapshot and database can be
transferred to fresh consumers without producer directories.

For paired evaluation, `snapshot.direct_documents()` returns the complete
searchable document bundle for a direct-memory treatment. Use exactly that
snapshot for retrieval, and keep held-out answers out of both treatments.
