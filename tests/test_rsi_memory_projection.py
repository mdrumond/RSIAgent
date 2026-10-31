"""Dense retained memory stays useful without placing raw provenance in context."""
import json
from types import SimpleNamespace

from core.artifacts import ContentAddressedArtifacts, canonical_bytes
from coding.pi.context import ContextRecords, encoded
from rsi.evidence_io import _references
from rsi.memory_projection import project_memory
from rsi.project_composition import ProjectComposition


def read_all(context, pointer):
    pieces, offset = [], 0
    while True:
        page = context.read(pointer['record'], offset, 8192)
        pieces.append(page['text'])
        offset = page['next_offset']
        if offset is None:
            raw = ''.join(pieces)
            try:
                return json.loads(raw)
            except ValueError:
                return raw


def dense_records(store):
    raw = store.put_bytes(b'complete provider instrumentation\n')
    evidence = store.put_json({'stdout': raw})
    records = []
    for i in range(8):
        records.append(dict(memory_id=f'failure-{i}', proposal=dict(kind='failure',
            project_id=f'project-{i}', summary=f'Accumulation failed for shape {i}',
            evidence=[evidence] * 12, base_versions=[], observations=dict(
                outcome='numeric_failure', applicability=dict(goal=f'Compute shape {i}'),
                final_sources=[{'text': 'source ' * 3000}], commands=dict(items=[dict(
                    command='test', classification='numeric_failure', final_source_scope=True,
                    stderr={'text': f'expected {i}, got -1'})]))),
            provenance=dict(raw_refs=[raw] * (65625 if i == 0 else 7795), queries=[evidence])))
    return records, raw, evidence


def composition(tmp_path):
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    records, raw, evidence = dense_records(store)
    snapshot = dict(schema='subject-memory-snapshot-v1', memories=records)
    reference = store.put_json(snapshot)
    services = SimpleNamespace(store=store, test=SimpleNamespace(query=lambda _: dict(
        results=[], negative_records=[], provenance={'backend': 'fixture'})))
    composed = ProjectComposition(services, tmp_path, {'config': {'payload': {}}}, None)
    return composed, reference, records, raw, evidence


def test_dense_query_projects_metadata_without_changing_original_or_evidence_closure(tmp_path):
    composed, reference, records, raw, evidence = composition(tmp_path)
    before = composed.store.read_bytes(reference)
    query = composed.query(reference)
    response = query({'query': 'numeric failures'})
    assert len(response['negative_records']) == 8
    for original, row in zip(records, response['negative_records']):
        assert row['observations'] == original['proposal']['observations']
        assert row['provenance']['counts']['raw_refs'] == len(original['provenance']['raw_refs'])
        assert composed.store.read_json(row['artifact_ref']) == original
        assert composed.store.read_json(row['provenance']['reference']) == original['provenance']
        assert len(encoded({k: v for k, v in row.items() if k != 'observations'})) < 2048
    assert composed.store.read_bytes(reference) == before
    # Follow exactly the same reference vocabulary as runner copy and SFT export.
    pending, visited = list(query.artifact_refs), set()
    while pending:
        ref = pending.pop()
        if ref in visited:
            continue
        visited.add(ref)
        content = composed.store.read_bytes(ref)
        try:
            pending.extend(_references(json.loads(content)))
        except (ValueError, UnicodeDecodeError):
            pass
    assert {raw, evidence} <= visited
    assert all(row['artifact_ref'] in visited for row in response['negative_records'])


def test_all_eight_negative_facts_and_complete_rows_addressable_in_bounded_context(tmp_path):
    composed, reference, records, _, _ = composition(tmp_path)
    response = composed.query(reference)({})
    context = ContextRecords()
    observed = context.resource(dict(reference=reference, evidence=dict(payload=dict(response=response)),
        summary={'negative_records': 8}))
    assert len(encoded(observed)) <= 6144
    groups = observed.get('retrieval', observed.get('evidence', {}).get('payload', {}).get('response'))
    negatives = groups['negative_records']
    assert negatives['count'] == 8
    index = read_all(context, negatives['index'])
    assert [entry['index'] for entry in index] == list(range(8))
    for i, entry in enumerate(index):
        assert entry['facts']['memory_id'] == f'failure-{i}'
        assert entry['facts']['goal'] == f'Compute shape {i}'
        assert entry['facts']['commands'][0]['stderr'] == f'expected {i}, got -1'
        full = read_all(context, entry['full'])
        assert full['observations'] == records[i]['proposal']['observations']
    assert read_all(context, negatives['all_records']) == response['negative_records']


def test_projection_retains_evidence_beyond_inline_preview(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    refs = [store.put_bytes(str(i).encode()) for i in range(12)]
    original = dict(memory_id='one', proposal=dict(kind='failure', evidence=refs),
                    provenance={'raw_refs': refs[::-1]})
    row = project_memory(original, store)
    assert row['proposal']['evidence'] == refs[:8]
    assert row['proposal']['evidence_count'] == 12
    assert store.read_json(row['retained']['reference'])['artifact_refs'] == refs
    assert store.read_json(row['artifact_ref']) == original


def test_mixed_legacy_negatives_remain_recoverable():
    rows = ['old failure', {'reason': 'overflow', 'observations': None}, {'proposal': None}]
    context = ContextRecords()
    result = context.retrieval_rows(rows, budget=1)
    assert result['omitted'] == 3
    index = read_all(context, result['index'])
    assert [read_all(context, item['full']) for item in index] == rows


def test_export_preserves_projected_original_and_complete_raw_instrumentation(tmp_path):
    import base64
    from test_rsi_evidence_io import task_fixture
    from rsi.evidence_io import export_task
    task, store, _, _, _ = task_fixture(tmp_path)
    records, raw, _ = dense_records(store)
    projected = [project_memory(record, store) for record in records]
    query = store.put_json(dict(type='retrieval', response=dict(results=[], negative_records=projected)))
    events = json.loads((task / 'events.json').read_text())
    (task / 'events.json').write_bytes(canonical_bytes([*events, query]))
    export = tmp_path / 'export'
    export_task(task, export)
    exported = {row['id']: base64.b64decode(row['data'])
                for line in (export / 'evidence.jsonl').read_text().splitlines()
                if (row := json.loads(line))['kind'] == 'artifact'}
    assert exported[raw] == store.read_bytes(raw)
    assert json.loads(exported[query])['response']['negative_records'] == projected
    for original, row in zip(records, projected):
        assert json.loads(exported[row['artifact_ref']]) == original
        assert json.loads(exported[row['provenance']['reference']]) == original['provenance']


from test_rsi_strategy import setup as strategy_setup


def test_all_eight_decisions_still_required_and_used_negative_needs_redirection(strategy_setup):
    import pytest
    from rsi.strategy import StrategyRejected
    from test_rsi_strategy import prepare, decisions, select
    _, make, _, _, store, skill, _ = strategy_setup
    service = make(query=lambda _: dict(results=[], negative_records=[
        {'reason': f'numeric failure {i}'} for i in range(8)], provenance={'source': 'accepted'}))
    query, retro = prepare(service, skill)
    choices = decisions(service, query, reuse=True)
    with pytest.raises(StrategyRejected):
        select(service, skill, query, retro, choices[:-1])
    del choices[-1]['redirection']
    with pytest.raises(StrategyRejected):
        select(service, skill, query, retro, choices)
    choices[-1]['redirection'] = 'Change reduction order for this shape'
    ref = select(service, skill, query, retro, choices)
    assert len(store.read_json(ref)['payload']['decisions']) == 8


def test_archival_raw_identities_do_not_import_origin_task_into_descendants(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    historical = 'd' * 64  # Original task archive, deliberately outside this runner's CAS.
    evidence = store.put_bytes(b'host failure facts')
    original = dict(memory_id='old-task', proposal=dict(kind='failure', evidence=[evidence]),
                    provenance=dict(raw_refs=[historical] * 1000))
    row = project_memory(original, store)
    assert row['provenance']['raw_refs_scope'] == 'origin-task-archive'
    assert row['provenance']['counts']['raw_refs'] == 1000
    assert store.read_json(row['artifact_ref']) == original
    assert historical not in set(_references(original))
    assert evidence in store.read_json(row['retained']['reference'])['artifact_refs']
