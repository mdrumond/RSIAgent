"""Pure SQLite/configuration/CAS IO; service admission tested by producer."""
import argparse
import hashlib
import json
import subprocess
import sqlite3

import pytest

from core.artifacts import ContentAddressedArtifacts
from rsi.cli_evidence import register
from rsi.configuration import set_configuration, show_configuration
from rsi.instance_ingestion import ingest_instance, load_ingestion_records
from test_rsi_configuration import instance as configuration_instance


class Embedding:
    model, revision = 'TEST', 'fixture-v1'

    def embed(self, texts):
        return [[1., 0.] for _ in texts]


@pytest.fixture
def instance(configuration_instance):
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    root = configuration_instance
    (root / 'memory').mkdir()
    with KnowledgeDB(root / 'memory/base.db', Embedding()):
        pass
    checksum = hashlib.sha256((root / 'memory/base.db').read_bytes()).hexdigest()
    (root / 'instance.json').write_text(json.dumps({'files': {'memory/base.db': checksum}}))
    return root


def source_repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'safe.py').write_text('answer = 42\n')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    git('init', '-q')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'source')
    return root, git('rev-parse', 'HEAD').decode().strip()


@pytest.mark.parametrize('explicit_baseline', [False, True])
def test_new_revision_preserves_baseline_and_frozen_sqlite_snapshot(instance, tmp_path, explicit_baseline):
    root = instance
    baseline = root / 'memory/base.db'
    original = baseline.read_bytes()
    inventory = (root / 'instance.json').read_bytes()
    # A real detached SQLite snapshot models immutable IO, not a fake service.
    frozen = root / 'frozen.sqlite3'
    with sqlite3.connect(baseline.as_uri() + '?mode=ro', uri=True) as original_db:
        with sqlite3.connect(frozen) as target:
            original_db.backup(target)
    old_db = frozen.read_bytes()
    if explicit_baseline:
        payload = show_configuration(root)['payload']
        payload['baseline_kdb'] = payload.pop('kdb')
        set_configuration(root, payload)
    before = show_configuration(root)
    source, head = source_repo(tmp_path)
    negative = {'id': 'prior-failure', 'outcome': 'numeric_failure', 'source_revision': head}
    result = ingest_instance(root, source, collection='added-repo', repository='fixture/repo',
        negative_records=[negative], expect_revision=before['revision'], test_embeddings=Embedding())
    assert result['revision'] == head
    assert baseline.read_bytes() == original
    assert (root / 'instance.json').read_bytes() == inventory
    assert frozen.read_bytes() == old_db
    assert before['payload']['baseline_kdb' if explicit_baseline else 'kdb'] == 'memory/base.db'
    config = show_configuration(root)
    assert config['payload']['kdb_collections'] == ['added-repo']
    assert config['payload']['kdb_ingestion_refs'] == [result['reference']]
    new_db = root / result['baseline_kdb']
    assert hashlib.sha256(new_db.read_bytes()).hexdigest() == result['kdb_sha256']
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    with KnowledgeDB.open_read_only(new_db, Embedding()) as database:
        assert any('answer = 42' in hit.text for hit in database.query('added-repo', 'answer'))
    with KnowledgeDB.open_read_only(frozen, Embedding()) as database:
        with pytest.raises(KeyError):
            database.manifest('added-repo')
    evidence = load_ingestion_records(ContentAddressedArtifacts(root / 'artifacts'), [result['reference']])
    assert evidence['negative_records'] == [negative]
    assert evidence['provenance'][0]['reference'] == result['reference']
    receipt = ContentAddressedArtifacts(instance / 'artifacts').read_json(result['reference'])
    assert receipt['previous_kdb_sha256'] == hashlib.sha256(original).hexdigest()
    assert receipt['kdb_sha256'] == result['kdb_sha256']
    assert receipt['previous_configuration'] == before['revision']


def test_racing_configuration_edit_is_not_overwritten(instance, tmp_path, monkeypatch):
    import rsi.instance_ingestion as ingestion
    source, _ = source_repo(tmp_path)
    before = show_configuration(instance)
    original = (instance / 'memory/base.db').read_bytes()
    publish = ingestion.set_configuration
    retained = {}
    def race(root, payload, **kwargs):
        steering = dict(before['payload'], prompts={'actor': 'concurrent steering'})
        retained.update(publish(root, steering, expect_revision=before['revision']))
        return publish(root, payload, **kwargs)
    monkeypatch.setattr(ingestion, 'set_configuration', race)
    with pytest.raises(ValueError, match='revision conflict'):
        ingest_instance(instance, source, collection='racing-repo', repository='fixture/repo',
                        test_embeddings=Embedding())
    assert show_configuration(instance) == retained
    assert 'baseline_kdb' not in retained['payload']
    assert (instance / 'memory/base.db').read_bytes() == original


def test_ingestion_receipt_chain_survives_multiple_revisions(instance, tmp_path):
    source, _ = source_repo(tmp_path)
    first = ingest_instance(instance, source, collection='first', repository='fixture/repo',
        negative_records=[{'outcome': 'failed', 'id': 'first-negative'}], test_embeddings=Embedding())
    second = ingest_instance(instance, source, collection='second', repository='fixture/repo',
        negative_records=[{'outcome': 'rejected', 'id': 'second-negative'}], test_embeddings=Embedding())
    config = show_configuration(instance)['payload']
    assert config['kdb_collections'] == ['first', 'second']
    assert config['kdb_ingestion_refs'] == [first['reference'], second['reference']]
    receipt = ContentAddressedArtifacts(instance / 'artifacts').read_json(second['reference'])
    assert receipt['previous_ingestion_refs'] == [first['reference']]
    assert receipt['previous_kdb_sha256'] == first['kdb_sha256']
    evidence = load_ingestion_records(ContentAddressedArtifacts(instance / 'artifacts'), config['kdb_ingestion_refs'])
    assert [r['id'] for r in evidence['negative_records']] == ['first-negative', 'second-negative']
    with pytest.raises(FileNotFoundError):
        load_ingestion_records(ContentAddressedArtifacts(instance / 'artifacts'), ['f' * 64])


def test_instance_cli_routes_configured_copy_on_write_without_standalone_flags(instance, monkeypatch):
    import rsi.instance_ingestion as ingestion
    calls = []
    monkeypatch.setattr(ingestion, 'ingest_instance', lambda *args, **kwargs: calls.append((args, kwargs)) or {'ok': True})
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance')
    register(parser.add_subparsers(required=True))
    args = parser.parse_args(['--instance', str(instance), 'kdb-add-repo', '/node/repo',
                              '--collection', 'node', '--repository', 'fixture/node'])
    assert args.evidence_handler(args) == {'ok': True}
    assert calls[0][0][0] == str(instance)
    assert calls[0][1]['negative_records'] == []
    assert calls[0][1]['collection'] == 'node'
