"""Small retrieval metadata; complete accepted records remain immutable CAS data."""


def project_memory(record, artifacts):
    # raw_refs identify the original task's archived instrumentation. Preserve
    # them in the immutable record, without importing that task into every
    # descendant runner. Packet/evidence dependencies keep their existing closure.
    proposal, provenance = record['proposal'], record.get('provenance', {})
    projected = {key: value for key in ('kind', 'project_id', 'summary')
                 if isinstance((value := proposal.get(key)), str)}
    for key in projected:
        projected[key] = projected[key].encode()[:1024].decode(errors='ignore')
    for key in ('base_versions', 'evidence'):
        values = proposal.get(key, [])
        projected[key] = values[:8]
        projected[key + '_count'] = len(values)
    roots = [provenance.get('packet'), provenance.get('baseline'),
             *provenance.get('queries', []), *proposal.get('evidence', [])]
    closure = artifacts.put_json(dict(schema='rsi-memory-retrieval-closure-v1',
        artifact_refs=[ref for ref in roots if isinstance(ref, str) and len(ref) == 64
                       and all(c in '0123456789abcdef' for c in ref)]))
    return dict(retained=dict(reference=closure), memory_id=record['memory_id'], proposal=projected,
        artifact_ref=artifacts.put_json(record),
        provenance=dict(reference=artifacts.put_json(provenance),
            raw_refs_scope='origin-task-archive',
            counts={key: len(value) for key, value in provenance.items()
                    if isinstance(value, (list, dict))},
            packet=provenance.get('packet'), baseline=provenance.get('baseline')))
