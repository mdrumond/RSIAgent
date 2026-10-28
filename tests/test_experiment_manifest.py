"""Inspect completed trials without changing execution or copying evidence."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.experiment_manifest import export_trial


def trial_fixture(root, *, decision='accept', task=True):
    root.mkdir()
    (root / '.evaluation.lock').touch()
    store = ContentAddressedArtifacts(root / 'artifacts')
    baseline = store.put_bytes(b'{"memories":[]}')
    treatment = store.put_json({'memories': [{'summary': 'use aligned copies'}]})
    proposal = store.put_json({'summary': 'use aligned copies'})
    log = store.put_bytes(b'actor performed aligned copy\n')
    outcome = dict(host_verified=True, source_sha256=None,
                   artifacts={'log:actor': log}, observations={})
    selected = {'project_id': 'copy', 'inputs': [3, 5]} if task else None
    identity = dict(proposal_sha256=proposal, baseline_sha256=baseline,
        treatment_sha256=treatment, sequence=1, max_seconds=3600, arm_seconds=1200,
        model={'model': 'provider/model', 'api_key': 'private-token', 'base_url': 'https://private'},
        runtime={'target': 'a3', 'product': 'Ascend910', 'cann_version': '9.1',
                 'workspace': '/private/workspace', 'compiler_version': '/private/compiler'})
    judgment = dict(evaluation_id='retained-evaluation-id', trial_task=selected,
        decision=decision, reason='observed transfer', instructions=None,
        validity={'treatment': True, 'control': True}, evidence=[log],
        artifacts={'proposal': proposal, 'control:log:actor': log, 'treatment:log:actor': log})
    state = dict(identity=identity, allocation=3500, arm_allocation=1100,
        order=['treatment', 'control'], task=selected,
        arms={name: {'outcome': outcome, 'valid': True} for name in ('control', 'treatment')},
        decision=judgment)
    (root / 'evaluation.json').write_text(json.dumps(state))
    return state


def save(root, state):
    (root / 'evaluation.json').write_text(json.dumps(state))


def test_export_keeps_source_unchanged_and_points_to_real_evidence(tmp_path):
    trial = tmp_path / 'trial'
    state = trial_fixture(trial)
    before = {str(p.relative_to(trial)): p.read_bytes() for p in trial.rglob('*') if p.is_file()}
    output = tmp_path / 'inspection'
    manifest = json.loads(export_trial(trial, output).read_text())
    assert manifest['evaluation_id'] == state['decision']['evaluation_id']
    assert manifest['order'] == state['order']
    assert manifest['allocation_seconds'] == {
        'trial': 3500, 'configured_trial': 3600, 'configured_arm': 1200, 'arm': 1100}
    assert json.loads((output / manifest['task']).read_text()) == state['task']
    assert json.loads((output / manifest['memory']['control']).read_text()) == {'memories': []}
    assert set(manifest['memory']) == {'control', 'treatment'}
    assert manifest['cited_evidence'] == ['control:log:actor', 'treatment:log:actor']
    for reference in manifest['evidence'].values():
        path = output / reference
        assert path.is_file()
        assert trial in path.resolve().parents
    assert not (output / 'artifacts').exists()
    assert manifest['model'] == {'model': 'provider/model'}
    assert manifest['runtime'] == {'target': 'a3', 'product': 'Ascend910', 'cann_version': '9.1'}
    assert before == {str(p.relative_to(trial)): p.read_bytes() for p in trial.rglob('*') if p.is_file()}
    other = tmp_path / 'second'
    second = json.loads(export_trial(trial, other).read_text())
    assert second == manifest


def test_inconclusive_missing_peer_and_no_selected_task_are_inspectable(tmp_path):
    trial = tmp_path / 'trial'
    state = trial_fixture(trial, decision='inconclusive', task=False)
    state.pop('arm_allocation')
    state.pop('order')
    state['arms'] = {}
    state['decision'].update(validity={}, evidence=[], artifacts={'proposal': state['identity']['proposal_sha256']})
    save(trial, state)
    manifest = json.loads(export_trial(trial, tmp_path / 'out').read_text())
    assert manifest['task'] is None
    assert manifest['order'] == []
    assert manifest['allocation_seconds']['arm'] is None
    assert manifest['result']['decision'] == 'inconclusive'
    assert manifest['result']['validity'] == {}


@pytest.mark.parametrize('change', ['pending', 'missing-log', 'missing-memory', 'task-mismatch', 'unlisted-citation'])
def test_invalid_or_unavailable_source_does_not_publish_output(tmp_path, change):
    trial = tmp_path / 'trial'
    state = trial_fixture(trial)
    if change == 'pending':
        state['decision'] = None
    elif change in ('missing-log', 'missing-memory'):
        digest = (state['decision']['evidence'][0] if change == 'missing-log'
                  else state['identity']['baseline_sha256'])
        (trial / 'artifacts' / 'objects' / digest[:2] / digest[2:]).unlink()
    elif change == 'task-mismatch':
        state['decision']['trial_task'] = {'different': 'task'}
    else:
        state['decision']['evidence'].append('a' * 64)
    save(trial, state)
    with pytest.raises(ValueError):
        export_trial(trial, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['trial']


def test_active_lock_existing_destination_and_source_output_are_rejected(tmp_path):
    trial = tmp_path / 'trial'
    trial_fixture(trial)
    with (trial / '.evaluation.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match='active'):
            export_trial(trial, tmp_path / 'out')
    output = tmp_path / 'out'
    output.mkdir()
    marker = output / 'keep'
    marker.write_text('preserve')
    with pytest.raises(ValueError, match='already exists'):
        export_trial(trial, output)
    assert marker.read_text() == 'preserve'
    with pytest.raises(ValueError, match='outside'):
        export_trial(trial, trial / 'out')


def test_plain_text_memory_and_cli(tmp_path):
    trial = tmp_path / 'trial'
    state = trial_fixture(trial)
    state['identity']['baseline_sha256'] = ContentAddressedArtifacts(trial / 'artifacts').put_bytes(b'prior lesson\n')
    save(trial, state)
    output = tmp_path / 'out'
    command = [sys.executable, '-m', 'benchmarks.ascend.experiment_manifest', 'export',
               '--trial', str(trial), '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == output / 'manifest.json'
    assert (output / 'control-memory.txt').read_bytes() == b'prior lesson\n'
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode == 2
