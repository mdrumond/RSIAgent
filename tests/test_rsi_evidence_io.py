"""Behavioral fixture backends only; production never selects these embeddings."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from core.artifacts import ContentAddressedArtifacts, canonical_bytes
from rsi.cli_evidence import register
from rsi.evidence_io import export_task, ingest_repository


class FixtureEmbeddings:
    model, revision = 'TEST', 'fixture-v1'

    def embed(self, texts):
        return [[1., float(len(text) % 3 + 1)] for text in texts]


def repository(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args])
    git('init', '-q')
    (source / 'solution.py').write_text('def add(a, b):\n    return a + b\n')
    (source / 'README.md').write_text('Ordinary source documentation\n')
    (source / '.env').write_text('DO_NOT_INDEX=fixture\n')
    (source / 'credentials.txt').write_text('DO_NOT_INDEX\n')
    (source / 'link.py').symlink_to('/etc/hosts')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'snapshot')
    revision = git('rev-parse', 'HEAD').decode().strip()
    return source, revision


def test_ingestion_uses_committed_snapshot_existing_db_and_preserves_negative(tmp_path):
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    source, revision = repository(tmp_path)
    (source / 'solution.py').write_text('uncommitted change\n')
    negatives = [dict(id='negative-1', outcome='numeric_failure',
                      evidence_ref='prior-host-result', reason='unstable accumulation')]
    with KnowledgeDB(tmp_path / 'existing.sqlite', FixtureEmbeddings()) as database:
        receipt = ingest_repository(database, source, collection='repo-v1',
            repository='fixture/project', negative_records=negatives, test_backend=True)
        assert receipt['revision'] == revision
        assert receipt['negative_records'] == negatives
        assert {row['path'] for row in receipt['sources']} == {'solution.py', 'README.md'}
        hits = database.query('repo-v1', 'add', limit=5)
        assert any('return a + b' in hit.text for hit in hits)
        assert all('uncommitted' not in hit.text for hit in hits)
        assert database.manifest('repo-v1').revision == revision
        again = ingest_repository(database, source, collection='repo-v1',
            repository='fixture/project', negative_records=negatives, test_backend=True)
        assert again == receipt


def test_fixture_backend_requires_explicit_opt_in(tmp_path):
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    source, _ = repository(tmp_path)
    with KnowledgeDB(tmp_path / 'existing.sqlite', FixtureEmbeddings()) as database:
        with pytest.raises(ValueError, match='pinned BGE'):
            ingest_repository(database, source, collection='repo', repository='fixture', negative_records=[])


def task_fixture(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'receipts').mkdir()
    store = ContentAddressedArtifacts(task / 'artifacts')
    contract = store.put_json(dict(source_hash='a' * 64, skills={'bundle_id': 'b' * 64},
                                   backend_identity='fixture', model='TEST-model'))
    output = store.put_bytes(b'numeric failure: expected 1, observed 0\n')
    receipt = store.put_json(dict(operation='op1', stdout=output, classification='numeric_failure'))
    events = [dict(type='model-request', operation='op1', messages=[{'role': 'user', 'content': 'implement'}]),
              dict(type='model-response', operation='op1', tool_calls=[{'name': 'read_skill', 'id': 'tool1'}]),
              dict(type='skill-usage', operation='op1', action='read', skill='debug'),
              dict(type='retrieval', negative_records=[{'outcome': 'failed', 'id': 'n1'}]),
              dict(type='test', reference=receipt)]
    refs = [store.put_json(event) for event in events]
    state = dict(stage='debug', outcome=None, pending={'id': 'op2'},
                 intervals={'thinking1': ['thinking', 1., 3.], 'setup1': ['setup', 0., 1.]},
                 limits={'actor_seconds': 20}, history=[{'passed': False, 'artifact_refs': [receipt]}])
    for name, value in [('runner.json', {'contract': contract}), ('manifest.json', {'contract': contract}),
                        ('events.json', refs), ('workflow.json', state)]:
        (task / name).write_bytes(canonical_bytes(value))
    (task / 'receipts/command.json').write_bytes(canonical_bytes({'artifact': receipt}))
    return task, store, events, state, output


def test_export_preserves_order_raw_evidence_and_incomplete_budget_state(tmp_path):
    task, store, events, state, output = task_fixture(tmp_path)
    before = {p.relative_to(task): p.read_bytes() for p in task.rglob('*') if p.is_file()}
    first, second = tmp_path / 'export1', tmp_path / 'export2'
    manifest = export_task(task, first)
    assert export_task(task, second) == manifest
    payload = (first / 'evidence.jsonl').read_bytes()
    assert payload == (second / 'evidence.jsonl').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest['sha256']
    rows = [json.loads(line) for line in payload.splitlines()]
    assert [r['sequence'] for r in rows] == list(range(len(rows)))
    assert [r['data'] for r in rows if r['kind'] == 'event'] == events
    assert next(r['data'] for r in rows if r['id'] == 'workflow.json') == state
    artifact = next(r for r in rows if r['kind'] == 'artifact' and r['id'] == output)
    assert base64.b64decode(artifact['data']) == store.read_bytes(output)
    assert before == {p.relative_to(task): p.read_bytes() for p in task.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        export_task(task, first)


@pytest.mark.parametrize('corrupt', [False, True])
def test_export_missing_or_corrupt_artifact_leaves_no_dataset(tmp_path, corrupt):
    task, store, _, _, output = task_fixture(tmp_path)
    path = store.objects / output[:2] / output[2:]
    if corrupt:
        path.write_bytes(b'changed')
    else:
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        export_task(task, tmp_path / 'export')
    assert not (tmp_path / 'export').exists()


def test_export_rejects_credential_structures(tmp_path):
    task, store, _, _, _ = task_fixture(tmp_path)
    ref = store.put_json({'type': 'request', 'api_key': 'fixture-secret'})
    (task / 'events.json').write_bytes(canonical_bytes([ref]))
    with pytest.raises(ValueError, match='credential'):
        export_task(task, tmp_path / 'export')
    assert not (tmp_path / 'export').exists()


def test_extension_dispatches_without_cli_module_changes(tmp_path):
    task, _, _, _, _ = task_fixture(tmp_path)
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    args = parser.parse_args(['evidence-export', str(task), str(tmp_path / 'export')])
    result = args.evidence_handler(args)
    assert result['task_contract'] == json.loads((task / 'runner.json').read_text())['contract']


def test_existing_a3_database_and_durable_negative_receipt(tmp_path):
    from benchmarks.a3kernels.knowledge import KnowledgeDB
    from rsi.evidence_io import ingest_to_store
    class A3Fixture(FixtureEmbeddings):
        dimension = 2
        embed_documents = FixtureEmbeddings.embed
        embed_queries = FixtureEmbeddings.embed
    source, revision = repository(tmp_path)
    negatives = [{'id': 'failed-microbench', 'outcome': 'rejected', 'source_revision': revision}]
    store = ContentAddressedArtifacts(tmp_path / 'evidence')
    with KnowledgeDB.create(tmp_path / 'a3.sqlite', A3Fixture()) as database:
        result = ingest_to_store(database, source, store, collection='repo',
            repository='fixture/project', negative_records=negatives, test_backend=True)
        assert database.query('repo', 'add')[0].source_revision == revision
    retained = store.read_json(result['reference'])
    assert retained['negative_records'] == negatives
    assert retained['kind'] == 'indexed-source-material'


def test_ingestion_rejects_credential_bearing_repository_identity(tmp_path):
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    source, _ = repository(tmp_path)
    with KnowledgeDB(tmp_path / 'db.sqlite', FixtureEmbeddings()) as database:
        with pytest.raises(ValueError, match='repository identity'):
            ingest_repository(database, source, collection='repo', repository='https://user:pass@example.invalid/repo',
                              negative_records=[], test_backend=True)
