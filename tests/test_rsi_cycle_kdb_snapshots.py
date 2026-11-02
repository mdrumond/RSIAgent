"""Cycle KDB snapshots share immutable storage, never mutable SQLite state."""
import hashlib
import sqlite3

import pytest

from benchmarks.a5kernels.knowledge import KnowledgeDB
from rsi.configuration import set_configuration, show_configuration
from rsi.instance_ingestion import ingest_instance
from rsi.project_composition import file_identity, verify_kdb
from rsi.python_execution import atomic_json
from test_rsi_instance_ingestion import source_repo
from test_rsi_service_wave import wave_instance
from test_rsi_services import Embedding, instance


def pin(instance, path, **settings):
    config = show_configuration(instance.root)['payload']
    config.update(kdb_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **settings)
    set_configuration(instance.root, config)


def snapshot(instance, name):
    cycle = instance.create_cycle(name, 'phase1')
    try:
        return cycle.root/'knowledge.sqlite3', cycle.binding['kdb']
    finally:
        cycle.close()


def index(instance):
    path = instance.root/'memory/knowledge.sqlite3'
    source = instance.root/'source/solution.py'
    with KnowledgeDB(path, Embedding()) as database:
        database.index(source.parent, [source], collection='initial', language='python')
    pin(instance, path, kdb_collections=['initial'])
    return path


def test_cycles_share_verified_cas_and_read_only_queries_do_not_write(instance, monkeypatch):
    baseline = index(instance)
    first, reference = snapshot(instance, 'first')
    second, other = snapshot(instance, 'second')
    cas = instance.store._retention_path(reference)
    assert reference == other
    assert first.samefile(second) and first.samefile(cas)
    assert not first.samefile(baseline)
    before = first.read_bytes()
    def no_migrations(*args):
        pytest.fail('read-only cycle queries must not create or migrate schema')
    monkeypatch.setattr(KnowledgeDB, '_create_schema', no_migrations)
    for path in (first, second):
        with KnowledgeDB.open_read_only(path, Embedding()) as database:
            assert database.query('initial', 'answer')[0].text == 'answer = 42\n'
            with pytest.raises(sqlite3.OperationalError):
                database.connection.execute('CREATE TABLE forbidden (value)')
    assert first.read_bytes() == second.read_bytes() == cas.read_bytes() == before


def test_instance_ingestion_publishes_new_snapshot_without_changing_old_cycles(instance, tmp_path):
    baseline = index(instance)
    first, reference = snapshot(instance, 'first')
    second, _ = snapshot(instance, 'second')
    old = first.read_bytes()
    source, _ = source_repo(tmp_path)
    result = ingest_instance(instance.root, source, collection='added', repository='fixture/repo',
                             test_embeddings=Embedding())
    new_baseline = instance.root/result['baseline_kdb']
    third, new_reference = snapshot(instance, 'third')
    assert reference != new_reference == result['kdb_sha256']
    assert third.samefile(instance.store._retention_path(new_reference))
    assert not third.samefile(new_baseline) and not third.samefile(first)
    assert first.samefile(second) and first.read_bytes() == second.read_bytes() == old
    assert baseline.read_bytes() == old
    with KnowledgeDB.open_read_only(first, Embedding()) as database:
        assert database.query('initial', 'answer')
        with pytest.raises(KeyError):
            database.query('added', 'answer')
    with KnowledgeDB.open_read_only(third, Embedding()) as database:
        assert database.query('added', 'answer')


def test_cached_reader_reauthenticates_after_another_cycle_links_snapshot(instance, monkeypatch):
    import benchmarks.a5kernels.embeddings as embeddings
    index(instance)
    cycle = instance.create_cycle('first', 'phase1')
    path = cycle.root/'knowledge.sqlite3'
    cycle.kdb_stat = verify_kdb(path, cycle.binding['kdb'])
    snapshot(instance, 'second')
    assert file_identity(path) != cycle.kdb_stat
    monkeypatch.setattr(Embedding, '_load', lambda self: None, raising=False)
    monkeypatch.setattr(embeddings, 'PinnedBGEEmbeddings', lambda **kwargs: Embedding())
    instance.test = None
    try:
        query = cycle.query(cycle.spec.baseline_kdb)
        assert query({'query': 'answer'})['results']
        assert cycle.kdb_stat == file_identity(path)
    finally:
        cycle.close()


@pytest.mark.parametrize('corrupt', [False, True])
def test_wave_admission_rechecks_content_when_shared_inode_metadata_changes(wave_instance, corrupt):
    cycle = wave_instance.create_cycle('first', 'phase1')
    path = cycle.root/'knowledge.sqlite3'
    atomic_json(cycle.root/'kdb-admission.json',
                dict(sha256=cycle.binding['kdb'], stat=verify_kdb(path, cycle.binding['kdb'])))
    snapshot(wave_instance, 'second')
    if corrupt:
        path.write_bytes(b'changed snapshot bytes')
    class Prepared(Exception):
        pass
    def prepare_only(*args):
        assert cycle.waves._prepare(2)['admission_ref']
        raise Prepared()
    cycle.scheduler.runner = prepare_only
    try:
        if corrupt:
            with pytest.raises(ValueError, match='frozen KDB changed'):
                cycle.scheduler.step()
        else:
            with pytest.raises(Prepared):
                cycle.scheduler.step()
    finally:
        cycle.close()


def test_active_wal_fails_closed_without_omitting_committed_rows(instance):
    baseline = instance.root/'memory/knowledge.sqlite3'
    with sqlite3.connect(baseline) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('CREATE TABLE snapshot_probe (value)')
        writer.commit()
        writer.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        pin(instance, baseline)
        writer.execute('INSERT INTO snapshot_probe VALUES (1)')
        writer.commit()
        with pytest.raises(ValueError, match='active WAL'):
            snapshot(instance, 'wal-first')
        assert not (instance.root/'runs/wal-first').exists()
        assert writer.execute('SELECT value FROM snapshot_probe').fetchall() == [(1,)]
