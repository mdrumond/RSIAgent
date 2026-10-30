"""Existing KDB ingestion and lossless canonical evidence export; no agent loop."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from core.artifacts import ContentAddressedArtifacts, canonical_bytes

SCHEMA = 'rsi-evidence-v1'
SHA = re.compile(r'[0-9a-f]{64}')
FORBIDDEN = {'env', 'environ', 'environment', 'api_key', 'apikey', 'authorization',
             'password', 'secret', 'access_token', 'refresh_token', 'credentials'}
TEXT_SUFFIXES = {'.py', '.md', '.rst', '.txt', '.c', '.cc', '.cpp', '.h', '.hpp',
                 '.cu', '.cuh', '.mlir', '.toml'}
REF_KEYS = {'reference', 'contract', 'contract_ref', 'execution_contract_ref',
            'host_failure_ref', 'query_ref', 'retrospective_ref', 'stdout', 'stderr',
            'profile', 'artifact', 'artifact_ref', 'selection_ref'}
REF_LISTS = {'artifact_refs', 'skill_refs', 'retrieval_refs', 'events'}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _safe(value):
    if isinstance(value, dict):
        if any(str(key).lower() in FORBIDDEN for key in value):
            raise ValueError('credential or environment structure is not exportable')
        for item in value.values():
            _safe(item)
    elif isinstance(value, list):
        for item in value:
            _safe(item)


def _allowed(path):
    return (not any(part.startswith('.') or part in {'node_modules', '__pycache__',
                'venv', 'artifacts', 'logs'} for part in path.parts)
            and path.suffix.lower() in TEXT_SUFFIXES
            and not any(word in path.name.lower() for word in ('credential', 'secret', 'password')))


def ingest_repository(database, source, *, collection, repository, negative_records,
                      test_backend=False):
    """Index committed code/docs with the existing generic collection API.

    database is the configured A3 or generic A5 KnowledgeDB collection store.
    negative_records comes from the existing knowledge-agent outcome store and
    is retained unchanged in the returned ingestion receipt. No validity claim.
    """
    from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
    from benchmarks.a3kernels.embeddings import PinnedBGEEmbeddings as A3Embeddings
    from benchmarks.a3kernels.knowledge import KnowledgeDB as A3KnowledgeDB
    if not test_backend and not isinstance(database.embeddings, (PinnedBGEEmbeddings, A3Embeddings)):
        raise ValueError('production ingestion requires pinned BGE embeddings')
    from urllib.parse import urlsplit
    parsed_repository = urlsplit(repository)
    if parsed_repository.username or parsed_repository.password or parsed_repository.query or parsed_repository.fragment:
        raise ValueError('repository identity must not contain credentials or URL parameters')
    source = Path(source).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], stderr=subprocess.DEVNULL)
    revision = git('rev-parse', '--verify', 'HEAD').decode().strip()
    snapshots = []
    for entry in git('ls-tree', '-rz', '--full-tree', revision).split(b'\0'):
        if not entry:
            continue
        metadata, raw_path = entry.split(b'\t', 1)
        mode, kind, blob = metadata.decode().split()
        path = Path(raw_path.decode('utf-8'))
        if kind != 'blob' or mode not in {'100644', '100755'} or not _allowed(path):
            continue
        data = git('cat-file', 'blob', blob)
        data.decode('utf-8')
        snapshots.append((path, data))
    if not snapshots:
        raise ValueError('repository contains no admitted code or documents')
    negatives = list(negative_records)
    _safe(negatives)
    with tempfile.TemporaryDirectory(prefix='rsi-repo-snapshot-') as temporary:
        root = Path(temporary)
        for path, data in snapshots:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if isinstance(database, A3KnowledgeDB):
            from benchmarks.a3kernels.corpus import CorpusDocument
            manifest = database.index([CorpusDocument(collection, 'a3', repository, revision,
                path.as_posix(), _digest(data), data.decode('utf-8')) for path, data in snapshots])
        else:
            manifest = database.index(root, [root / path for path, _ in snapshots],
                collection=collection, language='repository', repository=repository,
                revision=revision)
    result = dict(schema='rsi-repository-ingestion-v1', kind='indexed-source-material',
                  repository=repository, revision=revision, manifest=json.loads(manifest.to_json()),
                  sources=[dict(path=path.as_posix(), sha256=_digest(data)) for path, data in snapshots],
                  negative_records=negatives, test_backend=test_backend)
    _safe(result)
    return result


def ingest_to_store(database, source, artifacts, **options):
    """Durably retain the complete provenance/negative receipt in existing CAS."""
    receipt = ingest_repository(database, source, **options)
    return dict(reference=artifacts.put_json(receipt), collection=options['collection'],
                revision=receipt['revision'], negative_records=len(receipt['negative_records']))


def _references(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in REF_KEYS and isinstance(item, str) and SHA.fullmatch(item):
                yield item
            elif key in REF_LISTS and isinstance(item, list):
                yield from (ref for ref in item if isinstance(ref, str) and SHA.fullmatch(ref))
            yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def export_task(task, destination):
    """Export a quiescent canonical task. Missing/corrupt evidence fails closed.

    Retains event order, raw artifacts, failures and incomplete state. Does not
    invent completions, elapsed measurements or outcomes. No environment read.
    """
    task, destination = Path(task).resolve(strict=True), Path(destination).absolute()
    if destination.is_relative_to(task):
        raise ValueError('export destination must be outside the task')
    store = ContentAddressedArtifacts(task / 'artifacts')
    files = [task / name for name in ('runner.json', 'manifest.json', 'workflow.json', 'events.json')]
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError('canonical task contract/state/event index required')
    files += sorted((task / 'receipts').glob('*.json'))
    if any(path.is_symlink() for path in files):
        raise ValueError('symlink receipt is not exportable')
    before = {path: path.read_bytes() for path in files}
    states = {path.name: json.loads(before[path]) for path in files[:4]}
    for name in ('runner.json', 'manifest.json'):
        store.read_json(states[name]['contract'])
    if not isinstance(states['events.json'], list):
        raise ValueError('invalid canonical event index')
    objects = {}
    if store.objects.is_symlink():
        raise ValueError('symlink artifact store is not exportable')
    for path in sorted(store.objects.glob('*/*')):
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('symlink artifact is not exportable')
        reference = path.parent.name + path.name
        objects[reference] = store.read_bytes(reference)
    rows, parsed = [], []
    def add(kind, identity, value):
        _safe(value)
        parsed.append(value)
        rows.append(dict(schema=SCHEMA, sequence=len(rows), kind=kind, id=identity, data=value))
    for name in ('runner.json', 'manifest.json', 'workflow.json'):
        add('state', name, states[name])
    for index, reference in enumerate(states['events.json']):
        add('event', dict(index=index, reference=reference), store.read_json(reference))
    for path in files[4:]:
        add('receipt', path.name, json.loads(before[path]))
    for reference, data in objects.items():
        try:
            value = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            value = None
        if value is not None:
            _safe(value)
            parsed.append(value)
        rows.append(dict(schema=SCHEMA, sequence=len(rows), kind='artifact', id=reference,
                         encoding='base64', data=base64.b64encode(data).decode('ascii')))
    for value in parsed:
        for reference in _references(value):
            if reference not in objects:
                raise ValueError('referenced artifact is missing: ' + reference)
    if (sorted((task / 'receipts').glob('*.json')) != files[4:]
            or {p.parent.name + p.name for p in store.objects.glob('*/*')} != set(objects)
            or any(path.read_bytes() != data for path, data in before.items())):
        raise ValueError('task changed during export; stop its writer first')
    payload = b''.join(canonical_bytes(row) + b'\n' for row in rows)
    manifest = dict(schema=SCHEMA, format='canonical-evidence-jsonl', records=len(rows),
                    sha256=_digest(payload), bytes=len(payload),
                    task_contract=states['runner.json']['contract'],
                    artifacts=len(objects), completion_policy='observed-only')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rsi-export-', dir=destination.parent) as temporary:
        root = Path(temporary)
        for name, content in [('evidence.jsonl', payload), ('manifest.json', canonical_bytes(manifest))]:
            with (root / name).open('wb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        # Reserve once: never overwrite an earlier evidence dataset.
        destination.mkdir()
        for name in ('evidence.jsonl', 'manifest.json'):
            os.replace(root / name, destination / name)
        for directory in (destination, destination.parent):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return manifest
