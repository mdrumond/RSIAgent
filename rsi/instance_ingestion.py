"""Copy-on-write instance KDB revisions; admitted cycles keep their own DBs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid

from core.artifacts import ContentAddressedArtifacts
from rsi.configuration import set_configuration, show_configuration
from rsi.evidence_io import _safe, ingest_repository


def load_ingestion_records(artifacts, references):
    """Query-service adapter: verify receipts and retain negatives plus provenance.

    Merge these negative_records with accepted memory failures, and attach this
    provenance to the KDB query receipt. Never trust a config list without CAS
    verification. No ingestion record is promoted to a positive result.
    """
    if (not isinstance(references, list) or not all(isinstance(r, str) for r in references)
            or len(set(references)) != len(references)):
        raise ValueError('ingestion references must be a unique list')
    negatives, provenance = [], []
    for reference in references:
        receipt = artifacts.read_json(reference)
        if (receipt.get('schema') != 'rsi-repository-ingestion-v1'
                or not isinstance(receipt.get('negative_records'), list)
                or not isinstance(receipt.get('manifest'), dict)):
            raise ValueError('invalid KDB ingestion receipt')
        _safe(receipt)
        negatives.extend(copy.deepcopy(receipt['negative_records']))
        provenance.append(dict(reference=reference, repository=receipt['repository'],
            revision=receipt['revision'], manifest=receipt['manifest']['fingerprint']))
    return dict(negative_records=negatives, provenance=provenance)


def ingest_instance(instance, source, *, collection, repository, negative_records=(),
                    expect_revision=None, test_embeddings=None):
    """Backup configured generic KDB, index copy, CAS receipt, optimistic publish.

    Production uses the same generic A5 KnowledgeDB/PinnedBGE as CPU services.
    test_embeddings is an explicit fixture injection, never a CLI fallback.
    Original baseline inventory, original DB and admitted cycle DBs are read-only.
    A racing config edit leaves the old pointer intact and the candidate unused.
    """
    from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    root = Path(instance).resolve(strict=True)
    snapshot = show_configuration(root)
    if expect_revision is not None and snapshot['revision'] != expect_revision:
        raise ValueError('configuration revision conflict')
    config = snapshot['payload']
    def path(value):
        selected = Path(value).expanduser()
        return (selected if selected.is_absolute() else root / selected).resolve(strict=True)
    baseline = path(config.get('baseline_kdb', config['kdb']))
    if baseline.is_dir():
        baseline = baseline / config.get('kdb_filename', 'knowledge.sqlite3')
    inventory = json.loads((root / 'instance.json').read_text())
    relative = str(baseline.relative_to(root)) if baseline.is_relative_to(root) else None
    expected = config.get('kdb_sha256') or inventory.get('files', {}).get(relative)
    if not expected or hashlib.sha256(baseline.read_bytes()).hexdigest() != expected:
        raise ValueError('verified baseline KDB checksum required')
    collections = config.get('kdb_collections', [])
    if not isinstance(collections, list) or not all(isinstance(c, str) for c in collections):
        raise ValueError('kdb_collections must be a list')
    if not isinstance(collection, str) or not collection.strip() or collection in collections:
        raise ValueError('new distinct collection name required')
    store = ContentAddressedArtifacts(root / 'artifacts')
    prior = config.get('kdb_ingestion_refs', [])
    load_ingestion_records(store, prior)  # Invalid/missing previous receipts fail closed.
    settings = dict(config.get('embeddings', {}))
    if settings.get('cache_dir'):
        settings['cache_dir'] = path(settings['cache_dir'])
    embeddings = test_embeddings if test_embeddings is not None else PinnedBGEEmbeddings(**settings)
    directory = root / 'memory' / 'kdb-revisions'
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ingest-', dir=directory) as temporary:
        candidate = Path(temporary) / 'knowledge.sqlite3'
        with sqlite3.connect(baseline.as_uri() + '?mode=ro', uri=True) as original:
            with sqlite3.connect(candidate) as target:
                original.backup(target)
        # Authenticate the source again; a checksum change cannot slip into a backup.
        if hashlib.sha256(baseline.read_bytes()).hexdigest() != expected:
            raise ValueError('baseline changed during ingestion')
        with KnowledgeDB(candidate, embeddings) as database:
            # Validate the copied schema/collections through existing API.
            for name in collections:
                database.manifest(name)
            receipt = ingest_repository(database, source, collection=collection,
                repository=repository, negative_records=negative_records,
                test_backend=test_embeddings is not None)
        data = candidate.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        receipt.update(previous_kdb_sha256=expected, kdb_sha256=checksum,
            previous_configuration=snapshot['revision'], previous_ingestion_refs=prior)
        reference = store.put_json(receipt)
        final = directory / (uuid.uuid4().hex + '.sqlite3')
        # New name owned by this invocation; no baseline or cycle path is replaced.
        with final.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        payload = copy.deepcopy(config)
        payload.update(baseline_kdb=str(final.relative_to(root)), kdb_sha256=checksum,
                       kdb_collections=[*collections, collection],
                       kdb_ingestion_refs=[*prior, reference])
        # On conflict retain the unused immutable file/CAS receipt for audit.
        # Never delete a file after an ambiguous atomic-pointer publication.
        published = set_configuration(root, payload, expect_revision=snapshot['revision'])
    return dict(reference=reference, revision=receipt['revision'], collection=collection,
                baseline_kdb=published['payload']['baseline_kdb'], kdb_sha256=checksum,
                configuration_revision=published['revision'], negative_records=len(receipt['negative_records']))
