import json
import base64

import pytest

from benchmarks.ascend import campaign_cli as cli
from benchmarks.ascend.manual_retry import ManualPhase1Retry
from benchmarks.ascend.feedback import FeedbackStore
from benchmarks.ascend.reviewed_memory import reviewed_phase1_coordinator
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from test_a5_phase1_adaptive import _project, _result
from reviewed_fixtures import snapshot
from test_generic_failure_feedback import phase1




@pytest.mark.parametrize('verified', [True, False])
def test_manual_retry_executes_original_spec_with_reviewed_memory_and_links_lifecycle(tmp_path, verified):
    original_spec = _project('original', 'gm-ub-movement')
    phase1(tmp_path/'failed', spec=original_spec)
    store = FeedbackStore(tmp_path/'feedback')
    store.import_phase1(tmp_path/'failed', target='a3', language='ascend-c')
    original = store.records()[0]
    requests = []
    def execute(request):
        requests.append(request)
        return _result(request.spec, verified, 'retained host acceptance')
    runner = reviewed_phase1_coordinator(tmp_path/'retry', 'Retry frozen project', snapshot(),
        CurriculumAgent(lambda _: pytest.fail('original retry must not ask curriculum for a new project')),
        execute, project_target=1)
    job = dict(root=str(runner.root), target='a3', feedback_root=str(store.root), retry_max_seconds=30)
    retry = ManualPhase1Retry(runner, job, dict(selected=[original]))
    assert retry.resume()['status'] == ('FIXED' if verified else 'FAILED')
    assert requests[0].spec.as_dict() == original_spec
    envelope = json.loads(requests[0].pre_wave_memory)
    reviewed = json.loads(base64.b64decode(envelope['initial_memory_base64']))
    assert reviewed == dict(schema='ascend-reviewed-memory-seed-v1',
                            snapshot_id=runner.snapshot.snapshot_id,
                            phase1_memory=list(runner.snapshot.memory_entries()))
    assert envelope['project_memory_base64'] == []
    first = requests[0].pre_wave_memory
    assert retry.resume()['status'] == ('FIXED' if verified else 'FAILED')
    assert len(requests) == 1
    assert (runner.root/'retry-memory.bin').read_bytes() == first
    original = next(row for row in store.records() if row['feedback_id'] == original['feedback_id'])
    assert original['lifecycle'] == ('fixed' if verified else 'failed')
    assert len(original['investigation']['retries']) == 1


@pytest.mark.parametrize('phase', [1, 2])
def test_phase1_feedback_start_pins_spec_and_skips_optimizer_goal(tmp_path, phase):
    from test_ascend_campaign_cli import start
    setup = tmp_path/'setup'
    setup.mkdir()
    prepared = start(setup)
    original_spec = _project('original', 'gm-ub-movement')
    phase1(tmp_path/'failed', spec=original_spec)
    store = FeedbackStore(tmp_path/'feedback')
    store.import_phase1(tmp_path/'failed', target='a3', language='ascend-c')
    original = store.records()[0]
    arguments = ['start', '--control', str(tmp_path/'manual-control'), '--root', str(tmp_path/'manual'),
        '--target', 'a3', '--phase', str(phase), '--objective', 'Retry unchanged original',
        '--config', prepared['config'], '--initial-memory', str(setup/'seed'),
        '--project-target', '1', '--phase1-root', str(tmp_path/'reviewed'), '--review', str(tmp_path/'review.json'),
        '--feedback-root', str(store.root), '--retry-feedback-id', original['feedback_id']]
    screens = []
    if phase == 2:
        with pytest.raises(SystemExit) as error:
            cli.main(arguments, screen=lambda *args: screens.append(args))
        assert error.value.code == 2
        assert not screens
        assert not (tmp_path/'manual').exists() and not (tmp_path/'manual-control').exists()
        return
    cli.main(arguments, screen=lambda *args: 'manual')
    job = cli._read(tmp_path/'manual-control/job.json')
    assert job['phase'] == 1
    assert job['phase1_original_retry'] is True
    assert job['retry_spec']['original_spec'] == original_spec
    assert not job.get('optimizer_goal') and not job.get('autonomous_goal')


@pytest.mark.parametrize('damage', ['seed-bytes', 'missing-seed', 'state-seed'])
def test_original_retry_persists_reviewed_seed_for_evidence_exports_and_resume(tmp_path, damage):
    from dataclasses import replace
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.evidence import canonical_bytes
    from benchmarks.ascend.training_annex import _verify_adaptive_state, inventory_campaign, package_campaign
    from optimizer.memory import freeze_source, verify_snapshot
    from test_reviewed_memory import native_source

    source, review, _ = native_source(tmp_path/'reviewed')
    original_spec = _project('original', 'gm-ub-movement')
    phase1(tmp_path/'failed', spec=original_spec)
    store = FeedbackStore(tmp_path/'feedback')
    store.import_phase1(tmp_path/'failed', target='a3', language='ascend-c')
    original, = store.records()
    root, control = tmp_path/'retry', tmp_path/'control'
    root.mkdir()
    control.mkdir()
    seed_path = root/'initial-memory.bin'
    seed_path.write_bytes(b'')  # The launcher copies the source campaign's initial seed.
    job = dict(root=str(root), control=str(control), target='a3',
        phase1_root=str(source), review=str(review), feedback_root=str(store.root), retry_max_seconds=30)
    admitted = cli._snapshot(job, {})
    requests = []

    def execute(request):
        requests.append(request)
        result = _result(request.spec, True, 'host verified original repair')
        objects = ContentAddressedArtifacts(request.branch_root/'artifacts')
        source_sha = objects.put_bytes(b'fixed original source')
        fact = objects.put_json(dict(passed=True, source_sha256=source_sha))
        payload = canonical_bytes(dict(classification='host-verified', source_sha256=source_sha,
                                       artifacts=dict(classification=fact)))
        memory = replace(result.memory, source_revision=source_sha, admitted_artifact_sha256=(fact,),
            lessons=(), no_new_knowledge=True,
            host_facts=tuple(replace(item, evidence_sha256=fact) for item in result.memory.host_facts))
        return replace(result, memory=memory, evidence_payload=payload)

    def compose():
        return reviewed_phase1_coordinator(root, 'Retry frozen original', cli._snapshot(job, {}),
            CurriculumAgent(lambda _: pytest.fail('original retry must not select another project')),
            execute, project_target=1)

    runner = compose()
    expected = runner.initial_memory
    assert seed_path.read_bytes() == expected != b''
    (root/'campaign-identity.json').write_bytes(canonical_bytes(admitted.destination.source_identity))
    retry = ManualPhase1Retry(runner, job, dict(selected=[original]))
    assert retry.resume()['status'] == 'FIXED'
    state = json.loads((root/'state.json').read_bytes())
    assert state['project_target'] == 1
    _verify_adaptive_state(root, state)
    assert inventory_campaign(root)['complete']
    assert package_campaign(root, tmp_path/'retry.tar.gz')['complete']
    exported = freeze_source(root)
    verify_snapshot(exported)
    assert len(exported['memory_entries']) == 2
    assert base64.b64decode(exported['files']['initial-memory.bin']['base64']) == expected
    assert len(requests) == 1
    stamp = seed_path.stat().st_mtime_ns
    assert ManualPhase1Retry(compose(), job, dict(selected=[original])).resume()['status'] == 'FIXED'
    assert len(requests) == 1 and seed_path.stat().st_mtime_ns == stamp

    if damage == 'seed-bytes':
        seed_path.write_bytes(b'changed seed')
    elif damage == 'missing-seed':
        seed_path.unlink()
    else:
        state['initial_memory_sha256'] = '0'*64
        (root/'state.json').write_bytes(canonical_bytes(state))
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='retained reviewed seed'):
        compose()
    assert {path: path.read_bytes() for path in root.rglob('*') if path.is_file()} == before
