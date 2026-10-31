import hashlib
import json
from pathlib import Path

import pytest

from rsi.triton_project_config import budgets, project_config, project_spec, trusted_resources
from rsi.triton_verifier import audit_candidate, exercise_evidence, launch_capture


def write_json(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_freezes_commands_inputs_and_elapsed_budget(tmp_path):
    source = tmp_path/'project'
    source.mkdir()
    (source/'candidate.py').write_text('candidate source')
    cases, runtime = tmp_path/'cases.json', tmp_path/'runtime.json'
    write_json(cases, {'cases': [{'case_id': 'tail'}]})
    write_json(runtime, {'physical_device': 3, 'device_health': {'product': 'a3', 'chip': 0}})
    guard = tmp_path/'guard.py'
    config = project_config(source, cases, runtime, executor_identity='admitted-triton',
                            python='/runtime/python', physical_device=3, health_guard=guard)
    assert config.editable_files == ('candidate.py',)
    assert config.generated_paths == ('outputs', 'caches')
    assert config.retained_paths == ('outputs',)
    spec = project_spec('conversion', 'Translate a frozen Ascend C kernel to Triton', config,
                        correctness_ref=str(cases))
    assert spec.command == tuple(config.commands['test'])
    assert spec.starting_folder == config.source
    resources = trusted_resources(config)
    assert cases in resources and runtime in resources and guard in resources
    assert Path(config.commands['validation'][config.commands['validation'].index('--exercise-producer')+1]) in resources
    for mode in ('compile', 'test', 'validation'):
        argv = config.commands[mode]
        assert argv[:4] == ['/runtime/python', '{verifier}', '--mode', mode]
        assert argv[argv.index('--cases-sha256')+1] == hashlib.sha256(cases.read_bytes()).hexdigest()
        assert argv[argv.index('--physical-device')+1] == '3'
    allocation = budgets()
    assert allocation.actor_seconds == 7200
    assert allocation.design_iterations is None and allocation.execution_tries is None
    runtime.write_text('{"physical_device":4}')
    with pytest.raises(ValueError):
        project_config(source, cases, runtime, executor_identity='admitted-triton',
                       python='/runtime/python', physical_device=3, health_guard=guard)


def test_config_keeps_trusted_cases_outside_actor_snapshot(tmp_path):
    (tmp_path/'candidate.py').write_text('candidate source')
    cases, runtime = tmp_path/'cases.json', tmp_path/'runtime.json'
    write_json(cases, {'cases': [1]})
    write_json(runtime, {'physical_device': 0})
    with pytest.raises(ValueError):
        project_config(tmp_path, cases, runtime, executor_identity='binding',
                       python='python3', physical_device=0, health_guard=tmp_path/'guard.py')


@pytest.mark.parametrize('expression', ['x + y', 'torch.add(x, y)', 'x.sum()'])
def test_wrapper_compute_is_rejected_before_candidate_import(tmp_path, expression):
    path = tmp_path/'candidate.py'
    path.write_text('@triton.jit\ndef add(X, O):\n    pass\n'
                    'class ModelNew:\n    def forward(self, x, y):\n        return '+expression+'\n')
    with pytest.raises(ValueError):
        audit_candidate(path)


def test_allocation_and_real_kernel_dispatch_wrapper_is_admitted(tmp_path):
    path = tmp_path/'candidate.py'
    path.write_text('import torch\nimport triton\n@triton.jit\ndef add(X, O):\n    pass\n'
                    'def get_inputs():\n    return [torch.randn(257)]\n'
                    'class ModelNew:\n    def forward(self, x):\n'
                    '        output = torch.empty_like(x)\n        add[(1,)](x, output)\n'
                    '        return output\n')
    assert audit_candidate(path) == ['add']


def test_launch_capture_excludes_warmup_archives_and_restores_after_failure(tmp_path):
    class Compiled:
        asm = {'ttir': 'module {}', 'kernel': b'binary'}

    class Jit:
        __name__ = 'add'

        def run(self, **kwargs):
            return Compiled()

    original = Jit.run
    with pytest.raises(RuntimeError):
        with launch_capture(Jit, tmp_path) as launches:
            Jit().run(warmup=True)
            assert not launches
            Jit().run(warmup=False)
            assert len(launches) == 1
            for artifact in launches[0]['artifacts'].values():
                assert hashlib.sha256((tmp_path/artifact['file']).read_bytes()).hexdigest() == artifact['sha256']
            raise RuntimeError('later execution failure')
    assert Jit.run is original


def test_validation_produces_new_source_binding_after_candidate_edits(tmp_path):
    from types import SimpleNamespace
    producer = Path(__file__).parents[1]/'rsi/triton_exercises.py'
    candidate = tmp_path/'candidate.py'
    args = SimpleNamespace(exercise_profile=None, exercise_producer=str(producer),
                           exercise_producer_sha256=hashlib.sha256(producer.read_bytes()).hexdigest())
    results = []
    for index, text in enumerate(('first candidate', 'repaired candidate')):
        candidate.write_text(text)
        context = {'source_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
                   'cases_sha256': 'frozen-cases', 'runtime_manifest_sha256': 'frozen-runtime',
                   'compiler_target': {'backend': 'ascend'}, 'observed_api': {'device_print': True}}
        directory = tmp_path/str(index)
        directory.mkdir()
        write_json(directory/'native-result.json', context)
        args.artifacts = str(directory)
        result = exercise_evidence(args, {'required_evidence': ['numeric_verification']}, context)
        results.append(result)
        assert result['facts']['numeric_verification']['source_sha256'] == context['source_sha256']
    assert results[0]['source_sha256'] != results[1]['source_sha256']
    assert results[0]['cases_sha256'] == results[1]['cases_sha256']


def test_missing_study_profile_is_not_a_passing_attestation(tmp_path):
    from types import SimpleNamespace
    from rsi.triton_exercises import Producer, StudyFailure
    context = {'source_sha256': 'current', 'cases_sha256': 'cases',
               'runtime_manifest_sha256': 'runtime', 'compiler_target': {'backend': 'ascend'},
               'observed_api': {'device_print': True}}
    write_json(tmp_path/'native-result.json', context)
    args = SimpleNamespace(artifacts=str(tmp_path), exercise_profile=None)
    producer = Producer(args, {'required_evidence': ['msprof_capability', 'profile_raw']}, context)
    with pytest.raises(StudyFailure) as failure:
        producer.produce()
    assert failure.value.facts['facts']['profile_raw']['status'] == 'unconfigured'
    assert (tmp_path/'exercises/evidence.json').is_file()


@pytest.mark.parametrize('fails', [False, True])
def test_driver_retains_raw_output_and_artifacts_on_both_outcomes(tmp_path, monkeypatch, fails):
    import os
    from types import SimpleNamespace
    from rsi import triton_verifier as driver

    candidate = tmp_path/'candidate.py'
    candidate.write_text('frozen candidate')
    args = SimpleNamespace(mode='test', artifacts=str(tmp_path/'outputs'),
                           candidate=str(candidate), cases=None, runtime_manifest=None,
                           exercise_evidence=None)

    def execute(request):
        print('python output')
        os.write(2, b'native diagnostic\n')
        (Path(request.artifacts)/'kernel.bin').write_bytes(b'raw binary')
        if fails:
            raise ValueError('compile failure')
        return {'status': 'passed'}

    monkeypatch.setattr(driver, 'execute', execute)
    if fails:
        with pytest.raises(ValueError):
            driver.run_driver(args)
    else:
        driver.run_driver(args)
    folder = Path(args.artifacts)
    assert (folder/'stdout.log').read_text() == 'python output\n'
    assert b'native diagnostic\n' in (folder/'stderr.log').read_bytes()
    assert (folder/'kernel.bin').read_bytes() == b'raw binary'
    assert (folder/'inputs/candidate.py').read_bytes() == candidate.read_bytes()
    outcome = json.loads((folder/'result.json').read_text())
    assert outcome['status'] == ('failed' if fails else 'passed')
    manifest = json.loads((folder/'manifest.json').read_text())
    for artifact in manifest['artifacts']:
        assert hashlib.sha256((folder/artifact['path']).read_bytes()).hexdigest() == artifact['sha256']
    if fails:
        assert outcome['error_type'] == 'ValueError'


def test_shared_probe_alarm_permission_is_preserved_but_profile_is_strict(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rsi.device_health_probe import standalone_hash, standalone_source
    from rsi.triton_verifier import fresh_device_health

    source = tmp_path/'probe.py'
    source.write_text(standalone_source())
    args = SimpleNamespace(health_guard=str(source), health_guard_sha256=standalone_hash(),
                           physical_device='2')
    identity = {'device_health': {'product': 'a3', 'chip': 0, 'allow_alarm': True}}
    responses = {
        'board': 'NPU ID : 2\nChip Name : 910B4-1\n',
        'health': 'Health Status : Alarm\nError Code : 0x9999\n',
        'usages': 'HBM Capacity(MB) : 32768\nHBM Usage Rate(%) : 5\n'
                  'Aicore Usage Rate(%) : 0\nAivector Usage Rate(%) : 0\n',
        'proc-mem': 'No process in device.\n',
    }
    calls = []

    def query(argv, **kwargs):
        calls.append(argv)
        assert argv[argv.index('-i')+1] == '2'
        assert argv[argv.index('-c')+1] == '0'
        return SimpleNamespace(returncode=0, stdout=responses[argv[argv.index('-t')+1]].encode(), stderr=b'')

    monkeypatch.setattr('subprocess.run', query)
    ordinary = fresh_device_health(args, identity, tmp_path/'ordinary')
    assert ordinary['admitted'] and ordinary['degraded']
    assert ordinary['measurements']['hardware_error_code'] == '0x9999'
    assert (tmp_path/'ordinary/health.stdout').read_text() == responses['health']
    with pytest.raises(ValueError):
        fresh_device_health(args, identity, tmp_path/'profile', require_idle=True)
    assert json.loads((tmp_path/'profile/guard.json').read_text())['admitted'] is False
    assert len(calls) == 8


@pytest.mark.parametrize('call', ['ceil_div(x.numel(), 256)', 'ceil_div(x.shape[0], block=256)'])
def test_scalar_helper_arguments_admit_shape_arithmetic(tmp_path, call):
    path = tmp_path/'candidate.py'
    path.write_text('import torch\nimport triton\n'
        'def ceil_div(n, block):\n    quotient = (n + block - 1) // block\n    return quotient\n'
        '@triton.jit\ndef kernel(X, O):\n    pass\n'
        'class ModelNew:\n    def forward(self, x):\n'
        '        output = torch.empty_like(x)\n'
        f'        kernel[({call},)](x, output)\n        return output\n')
    assert audit_candidate(path) == ['kernel']


@pytest.mark.parametrize('body', ['return x + 1', 'alias = x\n    return alias * 2'])
def test_helper_tensor_arguments_still_reject_host_compute(tmp_path, body):
    path = tmp_path/'candidate.py'
    path.write_text('import triton\ndef helper(x):\n    '+body+'\n'
        '@triton.jit\ndef kernel(X):\n    pass\n'
        'class ModelNew:\n    def forward(self, x):\n        return helper(x)\n')
    with pytest.raises(ValueError):
        audit_candidate(path)


def test_copy_layout_requires_contiguous_output():
    from types import SimpleNamespace
    from rsi.triton_verifier import check_output_layout
    transposed = SimpleNamespace(is_contiguous=lambda: False)
    with pytest.raises(ValueError):
        check_output_layout(transposed, 'strided_copy')
    check_output_layout(SimpleNamespace(is_contiguous=lambda: True), 'strided_copy')
    check_output_layout(transposed, 'add')


@pytest.mark.parametrize('path_kind', ['absolute', 'parent', 'missing', 'hash_changed'])
def test_fixture_closure_rejects_nonportable_or_changed_inputs(tmp_path, path_kind):
    from rsi.triton_project_config import validate_fixture_closure
    source = tmp_path/'source'
    source.mkdir()
    fixture = source/'fixture.cpp'
    fixture.write_text('frozen original')
    value = {'project_path': 'fixture.cpp', 'sha256': hashlib.sha256(fixture.read_bytes()).hexdigest()}
    if path_kind == 'absolute': value['project_path'] = str(fixture)
    if path_kind == 'parent': value['project_path'] = '../source/fixture.cpp'
    if path_kind == 'missing': value['project_path'] = 'missing.cpp'
    if path_kind == 'hash_changed': fixture.write_text('changed')
    with pytest.raises(ValueError):
        validate_fixture_closure(source, {'ascendc_reference': value})


def test_actual_cpl_staging_preserves_external_case_fixture_closure(tmp_path):
    from dataclasses import replace
    import sys
    from test_cpl_execution import fixture
    from rsi.cpl_execution import CplExecutor
    from rsi.python_execution import PythonExecution
    from rsi.triton_project_config import validate_fixture_closure

    code = ('import hashlib,json,pathlib,sys\n'
            'contract=json.loads(pathlib.Path(sys.argv[1]).read_text())\n'
            'for value in contract.values():\n'
            ' p=pathlib.Path(value["project_path"])\n'
            ' assert not p.is_absolute()\n'
            ' assert hashlib.sha256(p.read_bytes()).hexdigest()==value["sha256"]\n'
            'print("fixture closure verified after transfer")\n')
    executor, request, observe, observations, _ = fixture(tmp_path, code)
    cases = tmp_path/'external-cases.json'
    contract = {}
    for name in ('compiler_failure', 'ascendc_reference', 'feature_probe', 'historical_environment'):
        file = request.project/(name+'.txt')
        file.write_text('immutable '+name)
        contract[name] = {'project_path': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    cases.write_text(json.dumps(contract))
    validate_fixture_closure(request.project, contract)
    backend = PythonExecution(request.project.parent)
    request = replace(request, source_hash=backend.project_hash(), argv=(sys.executable, 'main.py', str(cases)))
    executor = CplExecutor('frozen', replace(executor.config, trusted_resources=(cases,)))
    result = executor.execute(request, observe=observe, remaining=lambda: 60)
    assert result.returncode == 0, result.stderr
    assert result.stdout == b'fixture closure verified after transfer\n'
    remote = next((tmp_path/'remote').glob('rsi-*/project'))
    assert remote != request.project
    assert all((remote/value['project_path']).is_file() for value in contract.values())


def test_printf_child_enables_native_print_and_matches_verified_sample(tmp_path):
    import sys
    from types import SimpleNamespace
    from rsi.triton_exercises import Producer
    source = tmp_path/'candidate.py'
    source.write_text('@triton.jit\ndef kernel(O, value):\n    tl.store(O, value)\n')
    context = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'cases_sha256': 'cases', 'runtime_manifest_sha256': 'runtime',
               'compiler_target': {'backend': 'npu'}, 'observed_api': {'device_print': True}}
    def native_command(directory, candidate):
        # Real stdlib child: the flag must cross the subprocess environment
        # boundary. No candidate/kernel or framework is executed in this test.
        program = ('import hashlib,json,os,pathlib,sys\n'
            'assert os.environ.get("TRITON_DEVICE_PRINT")=="1"\n'
            'directory=pathlib.Path(sys.argv[1]);directory.mkdir(parents=True)\n'
            'result={"status":"passed","source_sha256":hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest(),'
            '"cases_sha256":"cases","runtime_manifest_sha256":"runtime","cases":[{"actual_sample":[1.25]}]}\n'
            '(directory/"result.json").write_text(json.dumps(result))\n'
            'print("RSI_DEVICE_NUMERIC 1.250000")\n')
        return [sys.executable, '-c', program, str(directory/'application'), str(candidate)]
    args = SimpleNamespace(artifacts=str(tmp_path), candidate=str(source), exercise_profile=None,
                           native_command=native_command, cases_sha256='cases', runtime_sha256='runtime')
    producer = Producer(args, {}, context)
    producer.printf()
    assert producer.facts['device_printf']['status'] == 'passed'
    assert producer.facts['numeric_diagnosis']['verified_samples'] == [1.25]
