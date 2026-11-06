import copy
import io
import json
from pathlib import Path
import tarfile

import pytest
from ascend_tools.baselines import file_hash
from ascend_tools.phase2_focus import (FOCUSED_LAUNCHER, evaluate_focused,
                                      prepare_focused_project, select_focus, validate_focus)
from test_baseline_projects import qualification


def timed_qualification(workload='fa', target='a3'):
    receipt, identity = qualification(workload, target)
    receipt['timing'].update(method='synchronized-call-wall-latency', warmup=5, repeats=20)
    for row in receipt['timing']['samples']:
        row['seconds'] = [row['id'] / 100] * 20
    return receipt, identity


def contract_for(tmp_path, workload='fa', target='a3'):
    receipt, identity = timed_qualification(workload, target)
    project = prepare_focused_project(workload, target, tmp_path / 'project', receipt, identity)
    with tarfile.open(project['verifier']['files'][1]) as tar:
        contract = json.load(tar.extractfile('contract.json'))
        focus = json.load(tar.extractfile('focus.json'))
    return project, contract, focus


def test_top_five_uses_medians_ties_and_target_evidence():
    receipt, _ = timed_qualification()
    receipt['timing']['samples'][0]['seconds'] = [.001] * 19 + [100000.]
    receipt['timing']['samples'][-2]['seconds'] = [.2] * 20
    before = copy.deepcopy(receipt)
    focus = select_focus(receipt)
    assert focus['caseIds'] == [19, 20, 18, 17, 16]
    assert focus['cases'][0]['medianSeconds'] == .2
    assert focus['fullCaseCount'] == 20 and receipt == before
    other = copy.deepcopy(receipt)
    other['target'] = 'a5'; other['runtime']['fingerprint'] = 'other-target'
    other['timing']['samples'][0]['seconds'] = [1.] * 20
    assert select_focus(other)['caseIds'] == [1, 19, 20, 18, 17]
    assert select_focus(other)['scopeHash'] != focus['scopeHash']


@pytest.mark.parametrize('mutation', ['missing_case', 'nonfinite', 'protocol', 'unqualified'])
def test_invalid_qualification_cannot_define_focus(mutation):
    receipt, _ = timed_qualification()
    if mutation == 'missing_case': receipt['timing']['samples'].pop()
    elif mutation == 'nonfinite': receipt['timing']['samples'][0]['seconds'][0] = float('nan')
    elif mutation == 'protocol': receipt['timing']['samples'][0]['seconds'].pop()
    else: receipt['qualified'] = False
    with pytest.raises(ValueError): select_focus(receipt)


def test_five_real_case_decisions_and_paired_scoring(tmp_path):
    from test_baseline_candidate import tiny_inputs, correct
    _, contract, focus = contract_for(tmp_path)
    built = []
    def build(workload, case):
        built.append(case['case_id']); return tiny_inputs(workload, case)
    result = evaluate_focused(contract, focus, correct, build=build)
    assert result['passed'] and result['complete'] and result['targetEvidence'] is False
    assert built == sorted(focus['caseIds']) and len(result['cases']) == 5
    assert result['evaluationScope'] == focus
    for candidate_seconds, passed in [(1., True), (4., False)]:
        calls = []
        def measure(label, operation):
            calls.append(label); return operation(), 2. if label == 'baseline' else candidate_seconds
        result = evaluate_focused(contract, focus, correct, build=tiny_inputs, performance=True,
                                  measure=measure, warmup=0, repeats=2)
        assert result['passed'] is passed and result['complete']
        assert result['geometricMeanSpeedup'] == pytest.approx(2. / candidate_seconds)
        assert len(calls) == 20 and calls[:4] == ['baseline', 'candidate', 'candidate', 'baseline']
    with pytest.raises(ValueError, match='Numerical mismatch'):
        evaluate_focused(contract, focus, lambda q, k, v, **kw: q.clone(), build=tiny_inputs)
    changed = copy.deepcopy(focus); changed['caseIds'][0] = 1
    with pytest.raises(ValueError, match='selection'):
        evaluate_focused(contract, changed, correct, build=tiny_inputs)


def test_missing_or_failed_selected_case_cannot_pass(tmp_path, monkeypatch):
    from ascend_tools.baselines import project_verifier
    _, contract, focus = contract_for(tmp_path)
    for rows in [[{'id': i, 'passed': True} for i in focus['caseIds'][:-1]],
                 [{'id': i, 'passed': i != focus['caseIds'][0]} for i in focus['caseIds']]]:
        monkeypatch.setattr(project_verifier, 'evaluate', lambda *a, **kw: {
            'cases': rows, 'passed': True, 'complete': True, 'geometricMeanSpeedup': 10.})
        result = evaluate_focused(contract, focus, lambda: None, performance=True)
        assert not result['passed'] and not result['complete']


@pytest.mark.parametrize('target', ['a3', 'a5'])
def test_focused_bundle_launch_binding_and_legacy_compatibility(tmp_path, target):
    from ascend_tools.campaign_launch import inspect_launch
    from test_campaign_launch import launch_fixture
    request = launch_fixture(tmp_path / 'full', target)
    assert inspect_launch(request)['verified'] == 7
    receipt = next(r for r in request['qualification']['receipts'] if r['workload'] == 'fa')
    receipt['timing'].update(method='synchronized-call-wall-latency', warmup=5, repeats=20)
    for row in receipt['timing']['samples']: row['seconds'] = [row['id'] / 100] * 20
    project = prepare_focused_project('fa', target, tmp_path / 'focus', receipt, receipt['runtime'])
    request['phase2Projects']['fa'] = project
    assert inspect_launch(request)['verified'] == 7
    assert len(project['verifier']['files']) == 2
    archive = Path(project['verifier']['files'][1])
    with tarfile.open(archive) as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    contract = json.loads(files['contract.json']); focus = json.loads(files['focus.json'])
    assert contract['qualification'] == receipt and focus == project['benchmark_focus']
    validate_focus(contract, focus)
    changed = copy.deepcopy(request)
    changed['phase2Projects']['fa']['benchmark_focus']['caseIds'].reverse()
    with pytest.raises(ValueError, match='focus'): inspect_launch(changed)
    changed = copy.deepcopy(request)
    changed['phase2Projects']['fa']['verifier']['command'] = 'true'
    with pytest.raises(ValueError, match='command'): inspect_launch(changed)
    # A matching archive hash cannot authorize changed focus evidence or scoring code.
    for name in ['focus.json', 'ascend_tools/phase2_focus.py']:
        damaged = dict(files); damaged[name] += b' '
        with tarfile.open(archive, 'w:gz') as tar:
            for filename, data in damaged.items():
                member = tarfile.TarInfo(filename); member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        Path(project['verifier']['files'][0]).write_text(
            FOCUSED_LAUNCHER.replace('ARCHIVE_HASH', file_hash(archive)))
        with pytest.raises(ValueError, match='differs'): inspect_launch(request)
