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
