import json
from dataclasses import replace

import pytest

from optimizer.memory import discover_sources, freeze_source, verify_snapshot


def completed(tmp_path, *, target_driven=False):
    from tests.test_a3_reviewed_phase2 import native_source
    from tests.test_a5_phase1_adaptive import _project, _result, _wave
    from benchmarks.ascend.phase2 import admit_native_snapshot
    from benchmarks.ascend.phase2_curriculum import AdaptivePhase2Coordinator
    from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    source, review, _ = native_source(tmp_path)
    root = tmp_path / 'phase2'
    seed = admit_native_snapshot(source, b'', review, root)
    def execute(request):
        result = _result(request.spec, True, 'verified')
        objects = ContentAddressedArtifacts(request.branch_root / 'artifacts')
        src = objects.put_bytes(b'source')
        evidence = objects.put_bytes(result.evidence_payload)
        return replace(result, memory=replace(result.memory, source_revision=src,
            admitted_artifact_sha256=(evidence,), lessons=(), no_new_knowledge=True,
            host_facts=tuple(replace(f, evidence_sha256=evidence) for f in result.memory.host_facts)))
    if target_driven:
        from benchmarks.ascend.phase2_target_loop import TargetDrivenPhase2Coordinator
        from benchmarks.ascend.phase2_target_selector import TargetCurriculumAgent
        from tests.test_ascend_phase2_targets import qualified, RUNTIME
        target, artifacts, _ = qualified(tmp_path / 'reference')
        runner = TargetDrivenPhase2Coordinator(root, 'learn', seed,
            TargetCurriculumAgent(lambda _: 'DECISION: READY\n{"reason":"target verified"}'),
            execute, target=target, artifacts=artifacts, runtime=RUNTIME)
    else:
        runner = AdaptivePhase2Coordinator(root, 'learn', seed,
            CurriculumAgent(lambda msgs: _wave(msgs[-1]['content'], [_project('phase2', 'simd-vector')])),
            execute, project_target=1)
    runner.run()
    return root


def test_snapshot_is_portable_authenticated_and_contains_both_phases(tmp_path):
    root = completed(tmp_path)
    result = discover_sources([root])
    assert len(result) == 1 and result[0]['available']
    snapshot = freeze_source(root)
    assert len(snapshot['memory_entries']) == 2
    assert snapshot['files'] and snapshot['source_identity']['language'] == 'ascend-c'
    root.rename(tmp_path / 'moved')
    verify_snapshot(snapshot)
    snapshot['memory_entries'][0]['source_revision'] = '0' * 64
    with pytest.raises(ValueError, match='snapshot'):
        verify_snapshot(snapshot)


@pytest.mark.parametrize('corruption', ['publication', 'object', 'state', 'pending'])
def test_bad_provenance_never_available(tmp_path, corruption):
    root = completed(tmp_path)
    if corruption == 'publication':
        next(root.glob('waves/*/published.json')).write_text('{}')
    elif corruption == 'object':
        next(root.glob('waves/*/branches/*/artifacts/objects/*/*')).write_bytes(b'bad')
    elif corruption == 'pending':
        (root / 'pending.json').write_text('{}')
    else:
        path = root / 'state.json'
        state = json.loads(path.read_bytes())
        state['memory_bytes_base64'] = []
        path.write_text(json.dumps(state))
    assert not discover_sources([root])[0]['available']
    with pytest.raises((ValueError, KeyError)):
        freeze_source(root)


def test_legacy_and_terminal_only_roots_explicitly_unavailable(tmp_path):
    (tmp_path / 'state.json').write_text('{"terminal":{"status":"READY"}}')
    assert 'unsupported' in discover_sources([tmp_path])[0]['reason']


def test_explicit_external_reviewed_snapshot_root(tmp_path):
    root = completed(tmp_path)
    external = tmp_path / 'seeds'
    external.mkdir()
    (root / 'cells').rename(external / 'cells')
    assert not discover_sources([root])[0]['available']
    assert discover_sources([root], snapshot_roots=[external])[0]['available']
    verify_snapshot(freeze_source(root, snapshot_roots=[external]))


def test_target_ready_requires_published_verified_target(tmp_path):
    root = completed(tmp_path, target_driven=True)
    snapshot = freeze_source(root)
    assert snapshot['terminal']['status'] == 'READY'
    path = root / 'state.json'
    state = json.loads(path.read_bytes())
    state['attempts'][-1]['verified'] = False
    path.write_text(json.dumps(state))
    assert not discover_sources([root])[0]['available']


def test_cli_discover_create_inspect_runtime(tmp_path, capsys):
    from optimizer.__main__ import main
    root = completed(tmp_path, target_driven=True)
    source = tmp_path / 'reference.py'
    source.write_text('def forward(x): return x + 1\n')
    signature = tmp_path / 'signatures.json'
    signature.write_text(json.dumps({'signatures': [{'args': [{'shape': [257], 'dtype': 'float32'}]}]}))
    main(['discover', '--memory-root', str(root)])
    assert json.loads(capsys.readouterr().out)[0]['available']
    store = tmp_path / 'jobs'
    main(['create', '--store', str(store), '--source', str(source), '--entrypoint', 'forward',
          '--signature', str(signature), '--memory-root', str(root)])
    created = json.loads(capsys.readouterr().out)
    main(['inspect', '--store', str(store), created['job_id']])
    assert json.loads(capsys.readouterr().out) == created
