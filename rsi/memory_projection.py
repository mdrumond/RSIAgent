"""Small retrieval metadata; complete accepted records remain immutable CAS data."""


def project_memory(record, artifacts):
    proposal, provenance = record['proposal'], record.get('provenance', {})
    projected = {key: value for key in ('kind', 'project_id', 'summary')
                 if isinstance((value := proposal.get(key)), str)}
    for key in projected:
        projected[key] = projected[key].encode()[:1024].decode(errors='ignore')
    for key in ('base_versions', 'evidence'):
        values = proposal.get(key, [])
        projected[key] = values[:8]
        projected[key + '_count'] = len(values)
    return dict(memory_id=record['memory_id'], proposal=projected,
        artifact_ref=artifacts.put_json(record),
        provenance=dict(reference=artifacts.put_json(provenance),
            counts={key: len(value) for key, value in provenance.items()
                    if isinstance(value, (list, dict))},
            packet=provenance.get('packet'), baseline=provenance.get('baseline')))
