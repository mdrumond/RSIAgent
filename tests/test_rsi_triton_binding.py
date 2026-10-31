import hashlib
import json
from pathlib import Path

import pytest

from rsi.triton_project_config import budgets, executor_factory, project_config, project_spec
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
    write_json(runtime, {'physical_device': 3})
    config = project_config(source, cases, runtime, executor_identity='admitted-triton',
                            python='/runtime/python', physical_device=3)
    assert config.editable_files == ('candidate.py',)
    assert config.generated_paths == ('artifacts/',)
    spec = project_spec('conversion', 'Translate a frozen Ascend C kernel to Triton', config,
                        correctness_ref=str(cases))
    assert spec.command == tuple(config.commands['test'])
    assert spec.starting_folder == config.source
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
                       python='/runtime/python', physical_device=3)


def test_config_keeps_trusted_cases_outside_actor_snapshot(tmp_path):
    (tmp_path/'candidate.py').write_text('candidate source')
    cases, runtime = tmp_path/'cases.json', tmp_path/'runtime.json'
    write_json(cases, {'cases': [1]})
    write_json(runtime, {'physical_device': 0})
    with pytest.raises(ValueError):
        project_config(tmp_path, cases, runtime, executor_identity='binding',
                       python='python3', physical_device=0)


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


def test_validation_requires_source_bound_host_artifacts(tmp_path):
    from types import SimpleNamespace
    args = SimpleNamespace(exercise_evidence=None, exercise_evidence_sha256=None,
                           cases_sha256='frozen-cases', runtime_sha256='frozen-runtime')
    contract = {'required_evidence': ['numeric_verification', 'device_printf']}
    with pytest.raises(ValueError):
        exercise_evidence(args, contract, 'candidate-head')
    output = tmp_path/'device.txt'
    output.write_bytes(b'device observation')
    manifest = tmp_path/'evidence.json'
    value = {'source_sha256': 'candidate-head', 'cases_sha256': 'frozen-cases',
             'runtime_manifest_sha256': 'frozen-runtime',
             'artifacts': {'device_printf': {'producer_operation_id': 'retained-op',
                            'path': output.name, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}}}
    args.exercise_evidence = str(manifest)
    args.exercise_evidence_sha256 = write_json(manifest, value)
    assert exercise_evidence(args, contract, 'candidate-head') == value['artifacts']
    output.write_bytes(b'changed observation')
    with pytest.raises(ValueError):
        exercise_evidence(args, contract, 'candidate-head')
    with pytest.raises(ValueError):
        exercise_evidence(args, contract, 'new-candidate-head')


def test_factory_binds_common_executor_without_dispatch(tmp_path, monkeypatch):
    from rsi.cpl_execution import CplConfig, CplExecutor

    def reject_dispatch(*args, **kwargs):
        raise AssertionError('Factory construction must not dispatch')

    monkeypatch.setattr('rsi.cpl_execution.subprocess.run', reject_dispatch)
    config = CplConfig(target='registered-target', runtime='named-runtime',
                       workspace='/owned/workspace', physical_device=2,
                       installed_client=tmp_path/'cpl-remote', profile_identity='admitted',
                       command_cwd='admitted-runtime')
    factory = executor_factory({'admitted': config})
    executor = factory('admitted')
    assert isinstance(executor, CplExecutor)
    assert executor.metadata['physical_device'] == 2
    assert executor.metadata['logical_device'] == 0
    with pytest.raises(ValueError):
        factory('unknown')
