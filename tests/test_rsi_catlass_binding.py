"""Host-only binding contracts; no NPU or provider calls. Not yet executed."""
import hashlib
import json
import sys
import runpy
from types import SimpleNamespace
from pathlib import Path

import pytest

from rsi.catlass_project_config import freeze_verifier, project_config, worker_environment, project_spec
from rsi.catlass_verifier import case_inputs, project_hash, reference


def test_conversion_cases_reject_identity_and_include_tie_rounding():
    left, right = case_inputs({'length':6, 'seed':17}, 'convert')
    expected = reference('convert', left, right)
    assert any(actual != want for actual, want in zip(left, expected))
    assert expected[:4] == [1.0009765625, -1.0009765625, 1.0, 1.001953125]


def test_independent_numeric_oracles_preserve_sign_and_operations():
    left, right = [-2., 0., 3.], [4., -1., .5]
    assert reference('add', left, right) == [2., -1., 3.5]
    assert reference('multiply', left, right) == [-8., 0., 1.5]
    assert reference('relu', left, right) == [0., 0., 3.]
    with pytest.raises(ValueError):
        reference('invented', left, right)


def test_source_hash_detects_added_inputs_but_ignores_declared_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'candidate.py').write_text('pass\n')
    contract = {'execution_inputs':['candidate.py'], 'generated_paths':['outputs']}
    initial = project_hash(contract)
    (tmp_path/'outputs').mkdir()
    (tmp_path/'outputs'/'result.json').write_text('{}')
    assert project_hash(contract) == initial
    (tmp_path/'helper.py').write_text('pass\n')
    assert project_hash(contract) != initial


def test_standalone_frozen_oracle_detaches_cases_from_actor_changes(tmp_path):
    runtime = dict(catlass_source='/profile/source', source_revision='src', dependency_gitlink='dep',
                   dependency_revision='dep', consumer_files=[{'path':'/profile/compiler','sha256':'hash'}],
                   cann_version='9.3.0', cann_version_file='/profile/version', cann_version_contents='9.3.0',
                   pipeline_manifest='/profile/manifest', pipeline_manifest_sha256='hash',
                   catlass_module_sha256='hash', compiler_paths={'TlaCompile':'/profile/compiler','hivmc-a5':'/profile/backend'})
    seed = {'operation':'convert','exercise':{'kind':'conversion'},'cases':[{'length':6,'block_count':1,'seed':17}]}
    path = tmp_path/'verify.py'
    digest = freeze_verifier(path, seed=seed, runtime_identity=runtime, physical_device=2, outer_cpl_handle='remote:bz-a5:session:admitted')
    seed['cases'][0]['length'] = 1
    frozen = runpy.run_path(str(path))
    assert frozen['FROZEN_CONTRACT']['cases'][0]['length'] == 6
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        freeze_verifier(path, seed=seed, runtime_identity=runtime, physical_device=2, outer_cpl_handle='remote:bz-a5:session:admitted')



def test_default_local_config_and_worker_env_do_not_rebind_controller(tmp_path):
    inherited = {'PATH':'/registry/bin','ASCEND_RT_VISIBLE_DEVICES':'old'}
    env = worker_environment(inherited=inherited, outer_cpl_handle='remote:bz-a5:session:actual',
                             physical_device=3, cache_directory=tmp_path/'project-cache')
    assert inherited['ASCEND_RT_VISIBLE_DEVICES'] == 'old'
    assert env['ASCEND_RT_VISIBLE_DEVICES'] == '3' and env['PATH'] == '/registry/bin'
    config = project_config(source=tmp_path/'source', verifier=tmp_path/'verify.py',
                            target_python='/profile/python3.11')
    assert config.executor_identity is None
    assert config.commands['validation'][0] == '/profile/python3.11'
    with pytest.raises(ValueError):
        worker_environment(inherited={}, outer_cpl_handle='invented', physical_device=3,
                           cache_directory=tmp_path/'cache')


def test_exercise_gate_does_not_accept_an_unregistered_numeric_task(monkeypatch):
    from rsi.catlass_capabilities import capability_gate
    monkeypatch.setenv('RSI_EXPECTED_SOURCE_HASH', 'source')
    contract = {'exercise':{'kind':'numeric-only'},'case_hash':'cases','runtime_identity':{}}
    with pytest.raises(ValueError, match='unregistered exercise'):
        capability_gate(contract, compile_original=lambda path:None, candidate_hash='candidate')


def test_repair_requires_real_expected_failure_and_binds_both_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from rsi.catlass_capabilities import compiler_repair_gate
    original = 'invalid original kernel'
    contract = {'exercise':{'original_source':original,'diagnostic_pattern':'staged-loop type mismatch'}}
    def compiler(path):
        assert path.read_text() == original
        raise RuntimeError('staged-loop type mismatch')
    result = compiler_repair_gate(contract, compiler, 'corrected-source')
    assert result['original_sha256'] == hashlib.sha256(original.encode()).hexdigest()
    assert result['corrected_sha256'] == 'corrected-source'
    with pytest.raises(ValueError, match='unexpectedly succeeded'):
        compiler_repair_gate(contract, lambda path:None, 'corrected-source')
    with pytest.raises(ValueError, match='outside the frozen'):
        compiler_repair_gate(contract, lambda path:(_ for _ in ()).throw(ConnectionError('transport lost')), 'corrected-source')


def test_tool_absence_is_study_outcome_only_when_frozen_contract_allows(tmp_path):
    from rsi.catlass_capabilities import CapabilityUnavailable, tool_command
    spec = {'path':str(tmp_path/'missing-msprof'), 'sha256':'unused'}
    with pytest.raises(ValueError, match='missing'):
        tool_command(spec, ['--help'])
    spec['allow_unavailable'] = True
    with pytest.raises(CapabilityUnavailable):
        tool_command(spec, ['--help'])


def test_actor_report_cannot_replace_a_tool_launch_receipt(tmp_path, monkeypatch):
    from rsi.catlass_capabilities import owned_command
    tool = tmp_path/'tool'
    tool.write_text('trusted tool executable fixture')
    # A plain JSON success from the Actor is not a trusted child-driver receipt.
    monkeypatch.setattr('rsi.catlass_capabilities.subprocess.run', lambda *a,**k:
        SimpleNamespace(returncode=0,stdout=b'{"success":true}',stderr=b''))
    spec = {'path':str(tool),'sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),'arguments':[]}
    with pytest.raises(ValueError, match='no trusted child launch receipt'):
        owned_command(spec, {}, contract={})


@pytest.fixture
def fake_consumer(tmp_path, monkeypatch):
    """Exercise the real driver decisions with fake target numerical operations."""
    from rsi import catlass_verifier as driver
    root = tmp_path/'project'
    root.mkdir()
    (root/'candidate.py').write_text('pass\n')
    monkeypatch.chdir(root)
    monkeypatch.delenv('RSI_EXPECTED_SOURCE_HASH', raising=False)
    monkeypatch.setenv('RSI_OUTER_CPL_HANDLE', 'remote:bz-a5:session:owned')
    monkeypatch.setenv('RSI_CPL_RUNTIME', 'catlass-dsl')
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '2')
    monkeypatch.setenv('CATLASS_DSL_CACHE_DIR', str(tmp_path/'bc-cache'))
    calls = []
    class Tensor:
        def __init__(self, values): self.values = values
        def __getitem__(self, item): return Tensor(self.values[item])
        def cpu(self): return self
        def tolist(self): return self.values
        def mark_compact_shape_dynamic(self, _): return self
    def compile_kernel(kernel, a, b, out, **options):
        calls.append('compile')
        def launch(*args, block_num):
            calls.append(('launch', block_num))
            out.values = [x+y for x,y in zip(a.values,b.values)]
        return launch
    torch = SimpleNamespace(float32='float32', tensor=lambda values, **kwargs:Tensor(values),
        full_like=lambda tensor,value:Tensor([value]*len(tensor.values)),
        npu=SimpleNamespace(set_device=lambda device:None,synchronize=lambda:None))
    tla = SimpleNamespace(arch=SimpleNamespace(RowMajor='row-major'), compile=compile_kernel)
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'catlass',SimpleNamespace(tla=tla))
    monkeypatch.setitem(sys.modules,'catlass.tla',tla)
    monkeypatch.setitem(sys.modules,'catlass.tla.runtime',SimpleNamespace(from_dlpack=lambda tensor,**kw:tensor))
    monkeypatch.setattr(driver,'check_runtime',lambda *args:{'source_revision':'src'})
    monkeypatch.setattr(driver,'load_kernel',lambda *args:object())
    monkeypatch.setattr(driver.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0))
    contract = {'abi_version':1,'operation':'add','case_hash':'cases',
        'outer_cpl_handle':'remote:bz-a5:session:owned','execution_runtime':'catlass-dsl',
        'physical_device':2,'allowed_helpers':[],
        'runtime_identity':{}, 'cases':[{'length':3,'block_count':2,'seed':17}],
        'execution_inputs':['candidate.py'],'generated_paths':['outputs','cache'],
        'exercise':{'kind':'numeric-only'}}
    return driver, contract, calls


def test_validation_rejects_numeric_success_without_exercise_evidence(fake_consumer):
    driver, contract, calls = fake_consumer
    assert driver.run(contract,'test')['correctness'] is True
    with pytest.raises(ValueError, match='unregistered exercise'):
        driver.run(contract,'validation')
    assert calls.count(('launch',2)) == 2
    receipt = json.loads(Path('outputs/catlass-result.json').read_text())
    assert receipt['exercise_validated'] is False


def test_unavailable_study_exits_separately_and_never_claims_validation(fake_consumer, tmp_path, monkeypatch):
    from rsi import catlass_capabilities
    driver, contract, calls = fake_consumer
    trusted = tmp_path/'trusted'
    trusted.mkdir()
    monkeypatch.setattr(catlass_capabilities,'__file__',str(trusted/'verify.py'))
    contract['exercise'] = {'kind':'msprof-basic','tools':{'msprof':{
        'path':str(tmp_path/'absent-msprof'),'sha256':'unused','allow_unavailable':True,'arguments':[]}}}
    monkeypatch.setattr(sys,'argv',['verify.py','--mode','validation'])
    with pytest.raises(SystemExit) as stopped:
        driver.main(contract)
    assert stopped.value.code == 78
    receipt = json.loads(Path('outputs/catlass-capability-study.json').read_text())
    assert receipt['outcome'] == 'capability-study-unavailable' and receipt['exercise_validated'] is False
    assert ('launch',2) in calls


def test_msprof_requires_exact_kernel_report_not_an_unrelated_csv(tmp_path, monkeypatch):
    from rsi import catlass_capabilities as gates
    reports = tmp_path/'msprof'
    reports.mkdir()
    path = reports/'OpBasicInfo.csv'
    path.write_text('Kernel Name,Duration(us)\nother_kernel,12.5\n')
    contract = {'exercise':{'tools':{'msprof':{'kernel_name':'kernel_42',
        'kernel_column':'Kernel Name','duration_column':'Duration(us)'}}},
        'physical_device':2,'cases':[{'length':3,'block_count':2,'seed':17}]}
    monkeypatch.setattr(gates,'owned_command',lambda *args,**kwargs:(None,{'tool_sha256':'tool'}))
    with pytest.raises(ValueError, match='no report'):
        gates.msprof_gate(contract,tmp_path)
    path.write_text('Kernel Name,Duration(us)\nkernel_42,12.5\n')
    evidence = gates.msprof_gate(contract,tmp_path)
    assert evidence['reports'][0]['duration'] == 12.5
    assert evidence['reports'][0]['report']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_compile_only_never_launches_or_reports_correctness(fake_consumer):
    driver, contract, calls = fake_consumer
    receipt = driver.run(contract,'compile')
    assert calls == ['compile']
    assert receipt['correctness'] is False and receipt['exercise_validated'] is False
    assert not receipt['cases'][0]['launched'] and receipt['cases'][0]['compiled']


def test_validation_cannot_reduce_the_frozen_case_set(fake_consumer):
    driver, contract, calls = fake_consumer
    with pytest.raises(ValueError, match='must cover all'):
        driver.run(contract,'validation',case_index=0)
    assert not any(isinstance(call,tuple) and call[0]=='launch' for call in calls)


def test_final_project_command_uses_config_test_argv(tmp_path):
    config = project_config(source=tmp_path/'source', verifier=tmp_path/'frozen.py',
                            target_python='/profile/python3.11')
    spec = project_spec(project_id='translation', goal='Translate frozen Ascend C', config=config)
    assert spec.command == tuple(config.commands['test'])
    assert spec.correctness_ref == config.verifier
    assert spec.budgets.actor_seconds == 7200


def test_resource_reads_use_existing_snapshot_and_cannot_be_edited(tmp_path):
    from rsi.python_execution import PythonExecution
    source = tmp_path/'source'
    reference = source/'.reference-skills'
    reference.mkdir(parents=True)
    (source/'candidate.py').write_text('pass\n')
    resource = reference/'guide.md'
    resource.write_text('A complete owned programming guide.')
    verifier = tmp_path/'verifier.py'
    verifier.write_text('pass\n')
    backend = PythonExecution.snapshot(tmp_path/'snapshot',source,verifier,
        commands={'test':['python','{verifier}']},editable_files=['candidate.py'])
    read = backend.read('.reference-skills/guide.md')
    assert read['content'] == resource.read_text()
    assert read['file_hash'] == hashlib.sha256(resource.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='not in the frozen editable manifest'):
        backend.edit('.reference-skills/guide.md','replacement',
                     expected_file_hash=read['file_hash'],expected_project_hash=read['project_hash'])


def test_ir_gate_rejects_unpinned_helper_even_with_no_shim_boolean(tmp_path):
    from rsi.catlass_capabilities import ir_gate
    helper = tmp_path/'installed-like.py'
    helper.write_text('raise RuntimeError("must never run")\n')
    contract = {'exercise':{'capture_tool':{'path':str(helper),
        'sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'no_absolute_shims':True}},
        'runtime_identity':{'compiler_paths':{'TlaCompile':'/not-run/frontend','hivmc-a5':'/not-run/backend'}}}
    with pytest.raises(ValueError,match='exact owned no-shim helper identity'):
        ir_gate(contract,tmp_path)


def test_compiler_repair_captures_native_and_child_stderr(tmp_path, monkeypatch):
    import os
    import subprocess
    from rsi.catlass_capabilities import compiler_repair_gate
    monkeypatch.chdir(tmp_path)
    contract = {'exercise':{'original_source':'frozen failing source',
                            'diagnostic_pattern':'child compiler staged type mismatch'}}
    def compile_original(path):
        os.write(2, b'native compiler diagnostic\n')
        subprocess.run([sys.executable,'-c',
                        'import sys; sys.stderr.write("child compiler staged type mismatch\\n")'],check=True)
        raise RuntimeError('generic compilation failure')
    receipt = compiler_repair_gate(contract,compile_original,'corrected-source')
    stderr = Path(receipt['original_output']['stderr_path']).read_bytes()
    assert b'native compiler diagnostic' in stderr
    assert b'child compiler staged type mismatch' in stderr
    assert receipt['corrected_compiled'] is False
    assert Path(receipt['archive']['path']).is_file()


def test_failed_tool_keeps_raw_reports_streams_and_typed_failure(tmp_path, monkeypatch):
    import tarfile
    from rsi import catlass_capabilities as gates
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('RSI_EXPECTED_SOURCE_HASH','source')
    tool = tmp_path/'msprof-fixture'
    tool.write_text('trusted fake process fixture')
    calls = []
    def process(argv, **kwargs):
        calls.append(argv)
        if len(calls)==1:
            return SimpleNamespace(returncode=0,stdout=b'probe stdout',stderr=b'probe stderr')
        report = Path(argv[-1]);report.mkdir(parents=True)
        (report/'failed-report.csv').write_text('retained partial vendor report')
        return SimpleNamespace(returncode=1,stdout=b'command stdout',stderr=b'actual tool error')
    monkeypatch.setattr(gates.subprocess,'run',process)
    contract = {'exercise':{'kind':'msprof-basic','tools':{'msprof':{'path':str(tool),
        'sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),'arguments':['--report','{report}']}}},
        'case_hash':'cases','runtime_identity':{},'outer_cpl_handle':'remote:bz-a5:session:actual','physical_device':2}
    with pytest.raises(ValueError,match='required tool command failed'):
        gates.capability_gate(contract,compile_original=lambda path:None,candidate_hash='candidate')
    receipts = list(Path('outputs').glob('*-receipt.json'))
    assert len(receipts)==1
    failure = json.loads(receipts[0].read_text())
    assert failure['outcome']=='capability-failure' and not failure['exercise_validated']
    assert failure['failure']['type']=='ValueError'
    with tarfile.open(failure['archive']['path']) as archive:
        assert archive.extractfile('./tool-command/command.stdout').read()==b'command stdout'
        assert archive.extractfile('./tool-command/command.stderr').read()==b'actual tool error'
        assert archive.extractfile('./msprof/failed-report.csv').read()==b'retained partial vendor report'


def test_language_conversion_binds_frozen_ascend_source_not_dtype_cast(fake_consumer, monkeypatch):
    driver, contract, calls = fake_consumer
    source = Path('.reference-skills')
    source.mkdir()
    original = source/'original.cpp'
    original.write_text('void vector_add(float *out, const float *a, const float *b, int n) {\n'
                        '  for (int i=0;i<n;++i) out[i]=a[i]+b[i];\n}\n')
    provenance = {'repository':'fixture-repository','revision':'frozen-source-revision',
                  'path':'original.cpp','symbol':'vector_add',
                  'sha256':hashlib.sha256(original.read_bytes()).hexdigest()}
    metadata = source/'provenance.json'
    metadata.write_text(json.dumps(provenance))
    contract['exercise'] = {'kind':'language-conversion','source_path':str(original),
                           'provenance_path':str(metadata),'provenance':provenance}
    receipt = driver.run(contract,'validation')
    evidence = receipt['capability_evidence']['evidence']
    assert receipt['exercise_validated'] and evidence['source_language']=='ascend-c'
    assert evidence['original_source']['sha256']==provenance['sha256']
    original.write_text('different source')
    monkeypatch.delenv('RSI_EXPECTED_SOURCE_HASH')
    with pytest.raises(ValueError,match='translation source changed'):
        driver.run(contract,'validation')
