"""Immutable retention shares disk storage without weakening CAS integrity."""
from contextlib import nullcontext
import errno
import os
from types import SimpleNamespace

import pytest

from core.artifacts import ContentAddressedArtifacts


def object_path(store, ref):
    return store.objects / ref[:2] / ref[2:]


@pytest.fixture
def stores(tmp_path):
    return (ContentAddressedArtifacts(tmp_path / 'source'),
            ContentAddressedArtifacts(tmp_path / 'target'))


def test_retention_links_and_survives_source_unlink(stores):
    source, target = stores
    ref = source.put_bytes(b'immutable evidence')
    before = object_path(source, ref).stat()
    assert target.retain_from(source, ref) == ref
    assert os.path.samefile(object_path(source, ref), object_path(target, ref))
    assert object_path(source, ref).stat().st_mtime_ns == before.st_mtime_ns
    object_path(source, ref).unlink()
    assert target.read_bytes(ref) == b'immutable evidence'


@pytest.mark.parametrize('error', [errno.EXDEV, errno.EOPNOTSUPP, errno.EPERM])
def test_retention_copy_fallback(stores, monkeypatch, error):
    source, target = stores
    data = b'large binary\x00\xff' * 100000
    ref = source.put_bytes(data)
    def unsupported(*args):
        raise OSError(error, 'link unavailable')
    monkeypatch.setattr('core.artifacts.os.link', unsupported)
    assert target.retain_from(source, ref) == ref
    assert not os.path.samefile(object_path(source, ref), object_path(target, ref))
    object_path(source, ref).unlink()
    assert target.read_bytes(ref) == data


@pytest.mark.parametrize('fallback', [False, True])
def test_existing_target_is_verified_without_replacement(stores, monkeypatch, fallback):
    source, target = stores
    ref = source.put_json({'evidence': 1})
    target.put_bytes(source.read_bytes(ref))
    if fallback:
        def unsupported(*args):
            raise OSError(errno.EXDEV, 'cross device')
        monkeypatch.setattr('core.artifacts.os.link', unsupported)
    inode = object_path(target, ref).stat().st_ino
    target.retain_from(source, ref)
    assert object_path(target, ref).stat().st_ino == inode
    object_path(target, ref).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        target.retain_from(source, ref)
    assert object_path(target, ref).read_bytes() == b'corrupt'


@pytest.mark.parametrize('existing_target', [False, True])
def test_corrupt_source_is_rejected(stores, existing_target):
    source, target = stores
    ref = source.put_bytes(b'original')
    if existing_target:
        target.put_bytes(b'original')
    object_path(source, ref).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        target.retain_from(source, ref)
    if existing_target:
        assert target.read_bytes(ref) == b'original'
    else:
        assert not object_path(target, ref).exists()


def test_missing_source_and_invalid_reference_are_rejected(stores):
    source, target = stores
    with pytest.raises(FileNotFoundError):
        target.retain_from(source, '0' * 64)
    with pytest.raises(ValueError):
        target.retain_from(source, '../invalid')
    assert not target.objects.exists()


def test_failed_fallback_removes_partial_target(stores, monkeypatch):
    source, target = stores
    ref = source.put_bytes(b'original')
    def unsupported(*args):
        raise OSError(errno.EXDEV, 'cross device')
    def failed_copy(input_stream, output, **kwargs):
        output.write(b'partial')
        raise OSError(errno.ENOSPC, 'full')
    monkeypatch.setattr('core.artifacts.os.link', unsupported)
    monkeypatch.setattr('core.artifacts.shutil.copyfileobj', failed_copy)
    with pytest.raises(OSError):
        target.retain_from(source, ref)
    assert not object_path(target, ref).exists()
    assert source.read_bytes(ref) == b'original'


@pytest.mark.parametrize('reopen', [False, True])
@pytest.mark.parametrize('layered', [False, True])
def test_composition_retains_transitive_query_closure(tmp_path, reopen, layered):
    from rsi.project_composition import ProjectComposition
    from rsi.services import BoundCycle

    store = ContentAddressedArtifacts(tmp_path / 'instance')
    binary = store.put_bytes(b'\xff\x00host evidence')
    nested = store.put_json({'artifact_ref': binary})
    observations = store.put_json({'artifact_refs': [nested]})
    packet = store.put_json({'observations': {'artifact_refs': [nested]},
                             'observations_ref': observations})
    memory = store.put_json({'memories': [
        {'proposal': {'observations_ref': observations}, 'provenance': {'packet': packet}}]})
    configuration = store.put_json({'payload': {'provider': {}, 'prompts': {'actor': 'guidance'}}})
    measures = []
    def measure(label, scope):
        measures.append((label, scope))
        return nullcontext()
    parent_store = store
    if layered:
        from rsi.service_wave import WorkerArtifacts
        store = WorkerArtifacts(tmp_path / 'worker', parent_store.root)
    services = SimpleNamespace(store=store, test=SimpleNamespace(),
                               setup=SimpleNamespace(measure=measure), executor_factory=None)
    composition = ProjectComposition(services, tmp_path, {'config': {'payload': {}}},
                                     SimpleNamespace(configuration_ref=configuration))
    composition.query = lambda ref: (lambda payload: {})
    class Runner:
        def __init__(self, root, **injections):
            self.store = ContentAddressedArtifacts(root / 'artifacts')
            self.context = lambda: {'stage': 'inference'}
        @classmethod
        def create(cls, root, spec, bundle, config, **injections):
            return cls(root, **injections)
    composition.runner_class = Runner
    runner = composition.runner(tmp_path / 'runner', memory_ref=memory,
                                backend_identity='test', reopen=reopen)
    for ref in (binary, nested, observations, memory, configuration):
        assert os.path.samefile(object_path(parent_store, ref), object_path(runner.store, ref))
    assert runner.context()['user_guidance'] == {'actor': 'guidance'}
    assert measures == [('project-open' if reopen else 'project-snapshot-copy-and-bind',
                         str(tmp_path / 'runner'))]
    result = runner.store.put_json({'artifact_ref': binary, 'result': 'retained'})
    BoundCycle.retain(services, runner)
    assert os.path.samefile(object_path(store, result), object_path(runner.store, result))
    for source_store in {store, parent_store}:
        for path in source_store.objects.glob('*/*'):
            path.unlink()
    assert runner.store.read_bytes(binary) == b'\xff\x00host evidence'
    assert runner.store.read_json(result)['result'] == 'retained'


@pytest.mark.parametrize('local', [False, True])
def test_layered_store_retention_uses_verified_backing_object(tmp_path, local):
    from rsi.service_wave import WorkerArtifacts
    parent = ContentAddressedArtifacts(tmp_path / 'parent')
    worker = WorkerArtifacts(tmp_path / 'worker', parent.root)
    target = ContentAddressedArtifacts(tmp_path / 'target')
    ref = parent.put_bytes(b'parent evidence')
    if local:
        worker.put_bytes(b'parent evidence')
    backing = worker if local else parent
    target.retain_from(worker, ref)
    assert os.path.samefile(object_path(backing, ref), object_path(target, ref))
    object_path(backing, ref).unlink()
    assert target.read_bytes(ref) == b'parent evidence'


def test_layered_retention_rejects_corrupt_local_shadow(tmp_path):
    from rsi.service_wave import WorkerArtifacts
    parent = ContentAddressedArtifacts(tmp_path / 'parent')
    worker = WorkerArtifacts(tmp_path / 'worker', parent.root)
    target = ContentAddressedArtifacts(tmp_path / 'target')
    ref = parent.put_bytes(b'parent evidence')
    worker.put_bytes(b'parent evidence')
    object_path(worker, ref).write_bytes(b'corrupt child shadow')
    with pytest.raises(ValueError):
        target.retain_from(worker, ref)
    assert not object_path(target, ref).exists()
    assert parent.read_bytes(ref) == b'parent evidence'
