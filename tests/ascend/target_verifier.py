"""CPU behavior only; these checks cannot establish NPU execution."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from ascend_tools.targets.runner import compare, verify
from ascend_tools.targets.health import assess, AdmissionFailure


def request(tmp_path, source='def run(a,b):\n return [x+y for x,y in zip(a,b)]\n'):
    editable = tmp_path/'editable'
    editable.mkdir()
    candidate = editable/'candidate.py'
    candidate.write_text(source)
    cases = tmp_path/'cases.json'
    cases.write_text(json.dumps(dict(operation='add', cases=[dict(left=[1., -3., 0.25], right=[-2., 7., 0.5])])))
    runtime = tmp_path/'runtime.json'
    runtime.write_text(json.dumps(dict(target='cpu', python=sys.version.split()[0])))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return SimpleNamespace(target='cpu', mode='test', candidate=str(candidate), cases=str(cases),
        cases_sha256=digest(cases), runtime=str(runtime), runtime_sha256=digest(runtime),
        source_sha256=digest(candidate), artifacts=str(tmp_path/'outputs'))


def test_independent_numeric_control_and_no_device_claim(tmp_path):
    args = request(tmp_path)
    report = verify(args)
    assert report['status'] == 'passed'
    assert report['correctness'] == 'passed'
    assert report['launch'] == 'not-run' and report['device_evidence'] is False
    assert report['cases'][0]['actual'] == [-1., 4., .75]


def test_bad_numeric_candidate_retains_outputs(tmp_path):
    args = request(tmp_path, 'def run(a,b):\n print("candidate diagnostic")\n return a\n')
    report = verify(args)
    assert report['status'] == 'failed'
    assert report['cases'][0]['actual'] == [1., -3., .25]
    assert 'candidate diagnostic' in (Path(args.artifacts)/'stdout.log').read_text()
    assert report['device_evidence'] is False


@pytest.mark.parametrize('field', ['runtime_sha256', 'cases_sha256', 'source_sha256'])
def test_frozen_identity_failure_does_not_execute_candidate(tmp_path, field):
    args = request(tmp_path, 'raise RuntimeError("candidate executed")\n')
    setattr(args, field, '0'*64)
    report = verify(args)
    assert report['status'] == 'failed'
    assert report['error_type'] == 'ValueError'
    assert report['launch'] == 'not-run'


def test_oracle_rejects_extent_nonfinite_and_wrong_values():
    for actual in ([1.], [float('nan'), 2.], [1., 3.]):
        with pytest.raises(ValueError):
            compare(actual, [1., 2.])


def test_health_never_infers_admission_from_unavailable_queries():
    with pytest.raises(AdmissionFailure) as error:
        assess(dict(physical_device=2, queries={}), physical_device=2,
               configuration=dict(product='a3', chip=0))
    assert error.value.kind == 'metric_unavailable'


def test_runtime_consumer_mismatch_prevents_candidate_execution(tmp_path):
    args = request(tmp_path, 'raise RuntimeError("candidate executed")\n')
    consumer = tmp_path/'compiler'
    consumer.write_text('changed consumer')
    profile = json.loads(Path(args.runtime).read_text())
    profile['compiler_consumer_files'] = [dict(path=str(consumer), sha256='0'*64)]
    Path(args.runtime).write_text(json.dumps(profile))
    args.runtime_sha256 = hashlib.sha256(Path(args.runtime).read_bytes()).hexdigest()
    report = verify(args)
    assert report['status'] == 'failed'
    assert report['error_type'] == 'ValueError'
    assert report['launch'] == 'not-run'


def test_nonfinite_candidate_retains_a_valid_failure_report(tmp_path):
    args = request(tmp_path, 'def run(a,b):\n return [float("nan")]*len(a)\n')
    report = verify(args)
    retained = json.loads((Path(args.artifacts)/'result.json').read_text())
    assert report['status'] == 'failed'
    assert retained['correctness'] == 'failed'
    assert retained['cases'][0]['actual'] == ['nan', 'nan', 'nan']


@pytest.mark.parametrize('physical', [0, 1])
def test_same_frozen_toolchain_accepts_explicit_leased_device(tmp_path, monkeypatch, physical):
    from ascend_tools.targets.runner import bind_device
    manifest = tmp_path/'runtime.json'
    manifest.write_text(json.dumps(dict(target='a3-triton', device_health=dict(product='a3', chip=0))))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name in ('ASCEND_PHYSICAL_DEVICE', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
        monkeypatch.delenv(name, raising=False)
    assert bind_device(SimpleNamespace(physical_device=physical), json.loads(manifest.read_text()), 'a3') == physical
    assert __import__('os').environ['ASCEND_PHYSICAL_DEVICE'] == str(physical)
    assert __import__('os').environ['ASCEND_RT_VISIBLE_DEVICES'] == str(physical)
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('name', ['ASCEND_PHYSICAL_DEVICE', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'])
def test_execution_binding_rejects_mask_or_assignment_mismatch(monkeypatch, name):
    from ascend_tools.targets.runner import bind_device
    monkeypatch.setenv(name, '0')
    with pytest.raises(ValueError, match='mismatch'):
        bind_device(SimpleNamespace(physical_device=1), dict(device_health=dict(product='a3')), 'a3')


def test_frozen_device_affinity_is_not_silently_reinterpreted():
    from ascend_tools.targets.runner import runtime_check
    with pytest.raises(ValueError, match='must not bind'):
        runtime_check(dict(physical_device=0), 'a3-triton')
