import base64
from dataclasses import replace
import hashlib
import json
import shutil

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_bytes
from benchmarks.a5kernels.phase1_curriculum import WaveSpec, WaveProject
from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory, HostFact
from benchmarks.ascend.history_import import import_history
from benchmarks.ascend.project_verifier import ProjectReview
from test_a5_phase1_wave_runtime import project


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def producer(root, *, spec=None, terminal='verified', published=True):
    spec = spec or project('copy-project')
    write(root / 'campaign-identity.json', dict(target='a3', arch='dav-2201',
        language='ascend-c', environment_identity=dict(cann_version='9.1.0')))
    wave = WaveSpec('wave-001', hashlib.sha256(b'{}').hexdigest(), (), (WaveProject(spec),))
    wr = root / 'waves/wave-001'
    branch = wr / 'branches' / spec.project_id
    branch.mkdir(parents=True)
    write(wr / 'plan.json', dict(schema='a5-phase1-live-wave-plan-v1', wave=wave.as_dict(),
                                  wave_fingerprint=wave.fingerprint))
    store = ContentAddressedArtifacts(branch / 'artifacts')
    source = store.put_bytes(b'kernel source\n')
    observed = store.put_json(dict(output=[1, 2, 3]))
    host = store.put_json(dict(host=dict(event='candidate-written', source_sha256=source)))
    EvidenceLedger(branch / 'evidence.jsonl').append('artifact', dict(role='host-observation', sha256=host))
    memory = Phase1ProjectMemory(spec, source, ('write', 'run'),
        (HostFact('host-verification', 'exact oracle output', observed),),
        admitted_artifact_sha256=(source, observed), artifact_store='artifacts')
    raw = canonical_bytes(memory.as_dict())
    body = dict(project_id=spec.project_id, wave_fingerprint=wave.fingerprint)
    if terminal == 'interrupted':
        write(branch / 'interruption.json', dict(body, classification='infrastructure-failure',
                                                summary='remote command interrupted'))
    else:
        payload = dict(classification='host-verified' if terminal == 'verified' else 'actor-unverified',
                       source_sha256=source, artifacts={'observed': observed})
        data = canonical_bytes(payload)
        delta = canonical_bytes(dict(schema='a5-phase1-adaptive-memory-delta-v1', project_id=spec.project_id,
                memory_base64=base64.b64encode(raw).decode(), summary='retained repair'))
        (branch / 'execution-evidence.bin').write_bytes(data)
        (branch / 'memory-delta.bin').write_bytes(delta)
        write(branch / 'outcome.json', dict(body, authored_ordinal=1,
            pre_wave_memory_sha256=wave.pre_wave_memory_sha256, verified=terminal == 'verified',
            classification=payload['classification'], memory_delta_sha256=hashlib.sha256(delta).hexdigest(),
            evidence_sha256=hashlib.sha256(data).hexdigest()))
    write(root / 'state.json', dict(memory_bytes_base64=[base64.b64encode(raw).decode()]
                                    if terminal == 'verified' and published else []))
    return branch, source


def test_published_record_portable_after_original_deleted(tmp_path):
    root = tmp_path / 'producer';branch, source = producer(root)
    store = ContentAddressedArtifacts(tmp_path / 'consumer')
    inventory = import_history([root], store)
    row, = inventory['memories']
    request = ProjectReview(**row['review_request'])
    assert request.host_verified and request.source_sha256 == source
    shutil.rmtree(root)
    assert store.read_bytes(request.source_sha256) == b'kernel source\n'
    assert json.loads(store.read_bytes(request.artifacts['historical-memory'])) == row['memory']
    assert 'actor-log' in request.artifacts and 'execution-result' in request.artifacts
    assert inventory['failures'] == [] and inventory['exclusions'] == []


@pytest.mark.parametrize('target', ['source', 'outcome', 'plan', 'journal'])
def test_tampering_never_imports_a_verified_candidate(tmp_path, target):
    root = tmp_path / 'producer';branch, source = producer(root)
    if target == 'source':
        ContentAddressedArtifacts(branch / 'artifacts')._path(source).write_bytes(b'changed source')
    elif target == 'outcome':
        p = branch / 'outcome.json';d = json.loads(p.read_bytes());d['authored_ordinal'] = 2;write(p, d)
    elif target == 'plan':
        p = branch.parent.parent / 'plan.json';d = json.loads(p.read_bytes());d['wave_fingerprint'] = '0' * 64;write(p, d)
    else:
        p = branch / 'evidence.jsonl';p.write_text(p.read_text().replace('candidate', 'tampered'))
        # Alter the admitted role in the ledger without updating its chain digest.
        d = json.loads(p.read_text());d['payload']['role'] = 'changed';p.write_text(json.dumps(d)+'\n')
    inventory = import_history([root], ContentAddressedArtifacts(tmp_path / 'consumer'))
    assert inventory['memories'] == [] and inventory['exclusions']


def test_unpublished_verified_and_interrupted_results_do_not_become_candidates(tmp_path):
    one, two = tmp_path / 'unpublished', tmp_path / 'failed'
    producer(one, published=False)
    producer(two, spec=replace(project('failed'), hypothesis='Different broken API'), terminal='interrupted')
    result = import_history([one, two], ContentAddressedArtifacts(tmp_path / 'consumer'))
    assert result['memories'] == []
    failure, = result['failures']
    assert not failure['review_request']['host_verified']
    assert failure['reason'] == 'remote command interrupted'
    assert failure['review_request']['source_sha256'] is not None


def test_exact_attempt_dedup_and_same_id_distinct_contract(tmp_path):
    one, duplicate, different = tmp_path / 'one', tmp_path / 'duplicate', tmp_path / 'different'
    producer(one, terminal='interrupted');shutil.copytree(one, duplicate)
    producer(different, spec=replace(project('copy-project'), hypothesis='Different logical tail failure'), terminal='interrupted')
    result = import_history([one, duplicate, different, one], ContentAddressedArtifacts(tmp_path / 'consumer'))
    assert result['counts']['attempts'] == 2
    assert len(result['failures']) == 2
    assert len({x['id'] for x in result['failures']}) == 2


def test_successful_retry_with_new_id_resolves_old_failure(tmp_path):
    one, two = tmp_path / 'old', tmp_path / 'recovered'
    producer(one, terminal='interrupted')
    producer(two, spec=replace(project('copy-project'), project_id='copy-project-retry'))
    result = import_history([one, two], ContentAddressedArtifacts(tmp_path / 'consumer'))
    assert result['failures'] == [] and len(result['memories']) == 1
    assert result['counts']['resolved_contracts'] == 1


def test_unverified_semantic_payload_remains_recoverable(tmp_path):
    root = tmp_path / 'failed';branch, _ = producer(root, terminal='unverified')
    delta = canonical_bytes(dict(classification='actor-unverified', summary='no final submit'))
    (branch / 'memory-delta.bin').write_bytes(delta)
    outcome = json.loads((branch / 'outcome.json').read_bytes())
    outcome['memory_delta_sha256'] = hashlib.sha256(delta).hexdigest();write(branch / 'outcome.json', outcome)
    result = import_history([root], ContentAddressedArtifacts(tmp_path / 'consumer'))
    assert len(result['failures']) == 1 and result['exclusions'] == []
    assert not result['failures'][0]['review_request']['host_verified']


def test_duplicate_published_roots_and_relocation_keep_identity(tmp_path):
    one, copy = tmp_path / 'one', tmp_path / 'copy'
    producer(one);shutil.copytree(one, copy)
    store = ContentAddressedArtifacts(tmp_path / 'consumer')
    first = import_history([one], store);second = import_history([one, copy], store)
    assert len(second['memories']) == 1 and second['exclusions'] == []
    assert first['memories'][0]['id'] == second['memories'][0]['id']
    moved = import_history([copy], store)
    assert moved['memories'][0]['id'] == first['memories'][0]['id']


def test_terminal_hash_and_verified_source_binding_are_checked(tmp_path):
    root = tmp_path / 'producer';branch, _ = producer(root)
    payload = json.loads((branch / 'execution-evidence.bin').read_bytes())
    payload['source_sha256'] = 'f' * 64
    data = canonical_bytes(payload);(branch / 'execution-evidence.bin').write_bytes(data)
    outcome = json.loads((branch / 'outcome.json').read_bytes())
    outcome['evidence_sha256'] = hashlib.sha256(data).hexdigest();write(branch / 'outcome.json', outcome)
    result = import_history([root], ContentAddressedArtifacts(tmp_path / 'consumer'))
    assert result['memories'] == [] and result['exclusions']


def test_corrupted_failure_journal_retains_contract_without_claiming_verification(tmp_path):
    root = tmp_path / 'failed';branch, _ = producer(root, terminal='interrupted')
    (branch / 'evidence.jsonl').write_text('{unfinished')
    result = import_history([root], ContentAddressedArtifacts(tmp_path / 'consumer'))
    failure, = result['failures']
    assert failure['warnings'] and not failure['review_request']['host_verified']
    assert failure['spec'] == project('copy-project').as_dict()


@pytest.mark.parametrize('journal', ['missing', 'empty'])
def test_published_candidate_needs_retained_actor_log(tmp_path, journal):
    root = tmp_path / 'producer';branch, _ = producer(root)
    path = branch / 'evidence.jsonl'
    if journal == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'')
    result = import_history([root], ContentAddressedArtifacts(tmp_path / 'consumer'))
    assert result['memories'] == [] and result['exclusions']


@pytest.mark.parametrize('published_first', [False, True])
def test_unpublished_copy_cannot_hide_published_attempt(tmp_path, published_first):
    one, copy = tmp_path / 'published', tmp_path / 'unpublished'
    producer(one);shutil.copytree(one, copy)
    write(copy / 'state.json', dict(memory_bytes_base64=[]))
    roots = [one, copy] if published_first else [copy, one]
    result = import_history(roots, ContentAddressedArtifacts(tmp_path / 'consumer'))
    assert len(result['memories']) == 1 and result['exclusions'] == []
    assert result['counts']['attempts'] == 1
    assert result['memories'][0]['producer']['root'] == 'published'


@pytest.mark.parametrize('different', ['target', 'toolchain'])
def test_verification_never_resolves_another_binding(tmp_path, different):
    good, failed = tmp_path / 'verified', tmp_path / 'failed'
    producer(good);producer(failed, terminal='interrupted')
    path = failed / 'campaign-identity.json';identity = json.loads(path.read_bytes())
    if different == 'target':
        identity.update(target='a5', arch='dav-3510')
    else:
        identity['environment_identity']['cann_version'] = '9.3.0'
    identity['model'] = {'credential_env': 'PRIVATE', 'secret': 'do-not-import'}
    identity['workspace'] = '/private/machine/path'
    write(path, identity)
    store = ContentAddressedArtifacts(tmp_path / 'consumer')
    result = import_history([good, failed], store)
    assert len(result['memories']) == 1 and len(result['failures']) == 1
    assert result['counts']['resolved_contracts'] == 0
    binding = store.read_bytes(result['failures'][0]['review_request']['artifacts']['campaign-binding'])
    assert b'do-not-import' not in binding and b'/private/' not in binding
    assert json.loads(binding)['target'] == identity['target']


@pytest.mark.parametrize('identity', ['missing', 'target', 'arch'])
def test_unknown_same_basename_producers_cannot_resolve_each_other(tmp_path, identity):
    good, failed = tmp_path / 'a/run', tmp_path / 'b/run'
    producer(good);producer(failed, terminal='interrupted')
    for root in (good, failed):
        path = root / 'campaign-identity.json'
        if identity == 'missing':
            path.unlink()
        else:
            value = json.loads(path.read_bytes());value.pop(identity);write(path, value)
    store = ContentAddressedArtifacts(tmp_path / 'consumer')
    result = import_history([good, failed], store)
    assert len(result['memories']) == 1 and len(result['failures']) == 1
    assert result['counts']['resolved_contracts'] == 0
    binding = store.read_bytes(result['failures'][0]['review_request']['artifacts']['campaign-binding'])
    assert str(tmp_path).encode() not in binding
    assert import_history([failed], store)['failures'][0]['id'] == result['failures'][0]['id']


@pytest.mark.parametrize('bad_state', [None, [], {'memory_bytes_base64': None},
    {'memory_bytes_base64': [None]}, {'memory_bytes_base64': [123]},
    {'memory_bytes_base64': base64.b64encode(b'{}').decode()},
    {'memory_bytes_base64': [base64.b64encode(b'null').decode()]}])
def test_malformed_state_excludes_only_its_producer(tmp_path, bad_state):
    bad, good = tmp_path / 'bad', tmp_path / 'good'
    producer(bad);producer(good)
    write(bad / 'state.json', bad_state)
    result = import_history([bad, good], ContentAddressedArtifacts(tmp_path / 'consumer'))
    assert len(result['memories']) == 1 and result['failures'] == []
    assert result['exclusions'] == [{'producer': 'bad', 'reason': 'ValueError'}]
