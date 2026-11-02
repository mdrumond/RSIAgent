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
    projected['evidence_scope'] = 'origin-task-archive'
    # Accepted service-result IDs retain their original archival meaning. Their
    # result.events must not import the originating actor's SFT into descendants.
    roots = [provenance.get('packet'), provenance.get('baseline'),
             *provenance.get('queries', [])]
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


def retrieval_facts(row):
    """Useful per-record index facts; full original rows remain retained separately."""
    def mapping(value):
        return value if isinstance(value, dict) else {}
    def text(value):
        return value.encode()[:512].decode(errors='ignore') if isinstance(value, str) else None
    value = mapping(row)
    metadata = mapping(value.get('memory', value))
    proposal = mapping(metadata.get('proposal'))
    observed = mapping(value.get('observations'))
    scope = mapping(observed.get('applicability'))
    candidates = dict(memory_id=metadata.get('memory_id'), kind=proposal.get('kind'),
        project_id=proposal.get('project_id'), summary=proposal.get('summary', value.get('reason', value.get('text', row))),
        goal=scope.get('goal'), outcome=observed.get('outcome'))
    facts = {key: found for key, item in candidates.items() if (found := text(item)) is not None}
    commands = mapping(observed.get('commands')).get('items', [])
    commands = commands if isinstance(commands, list) else []
    facts['commands'] = [dict(command=text(item.get('command')), classification=text(item.get('classification')),
        stderr=text(mapping(item.get('stderr')).get('text')), stdout=text(mapping(item.get('stdout')).get('text')))
        for item in commands[-3:] if isinstance(item, dict)]
    return facts
