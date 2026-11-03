"""Host-only binding contracts; no NPU or provider calls."""
import hashlib
import json
import sys
import runpy
from types import SimpleNamespace
from pathlib import Path

import pytest

from rsi.catlass_project_config import freeze_verifier, project_config, worker_environment, project_spec
from rsi.catlass_verifier import case_inputs, project_hash, reference


def _with_outer_admission(runtime,tmp_path,handle='remote:bz-a5:job:admitted',physical=2):
    receipt=tmp_path/'outer-submission.json'
    receipt.write_text(json.dumps({'handle':handle,'target':'bz-a5','runtime':'catlass-dsl',
        'npu':{'devices':[physical],'pool':'fixture-pool','checked_at':'fixture-time'}}))
    runtime['device_admission']={'outer_receipt':str(receipt),'outer_runtime':'catlass-dsl',
                                 'allow_busy':False,'allow_alarm':True}
    return runtime


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
    _with_outer_admission(runtime,tmp_path)
    seed = {'operation':'convert','exercise':{'kind':'conversion'},'cases':[{'length':6,'block_count':1,'seed':17}]}
    path = tmp_path/'verify.py'
    digest = freeze_verifier(path, seed=seed, runtime_identity=runtime, physical_device=2, outer_cpl_handle='remote:bz-a5:job:admitted')
    seed['cases'][0]['length'] = 1
    frozen = runpy.run_path(str(path))
    assert frozen['FROZEN_CONTRACT']['cases'][0]['length'] == 6
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        freeze_verifier(path, seed=seed, runtime_identity=runtime, physical_device=2, outer_cpl_handle='remote:bz-a5:job:admitted')



def test_default_local_config_and_worker_env_do_not_rebind_controller(tmp_path):
    inherited = {'PATH':'/registry/bin','ASCEND_RT_VISIBLE_DEVICES':'old'}
    env = worker_environment(inherited=inherited, outer_cpl_handle='remote:bz-a5:job:actual',
                             physical_device=3, cache_directory=tmp_path/'project-cache')
    assert inherited['ASCEND_RT_VISIBLE_DEVICES'] == 'old'
    assert env['ASCEND_RT_VISIBLE_DEVICES'] == '3' and env['PATH'] == '/registry/bin'
    config = project_config(source=tmp_path/'source', verifier=tmp_path/'verify.py',
                            target_python='/profile/python3.11')
    assert config.executor_identity is None
    assert config.retained_paths == ('outputs',)
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
        owned_command(spec, {}, contract={'cases':[{'length':1,'block_count':1,'seed':17}]})


@pytest.fixture
def fake_consumer(tmp_path, monkeypatch):
    """Exercise the real driver decisions with fake target numerical operations."""
    from rsi import catlass_verifier as driver
    from rsi import catlass_capabilities as gates
    for name in ('capability_gate','compiler_repair_gate','launch_output','profile_validation'):
        monkeypatch.setattr(driver,name,getattr(gates,name),raising=False)
    monkeypatch.setattr(driver,'fresh_device_health',lambda *a,**kw:{},raising=False)
    monkeypatch.setattr(gates,'fresh_device_health',lambda *a,**kw:{})
    root = tmp_path/'project'
    root.mkdir()
    (root/'candidate.py').write_text('pass\n')
    monkeypatch.chdir(root)
    monkeypatch.delenv('RSI_EXPECTED_SOURCE_HASH', raising=False)
    monkeypatch.setenv('RSI_OUTER_CPL_HANDLE', 'remote:bz-a5:job:owned')
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
        'outer_cpl_handle':'remote:bz-a5:job:owned','execution_runtime':'catlass-dsl',
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
    contract['profile_capture']=True
    contract['exercise'] = {'kind':'msprof-basic','tools':{'msprof':{
        'device_mode':'direct-physical','basic_info_arguments':['--device','{physical_device}'],
        'path':str(tmp_path/'absent-msprof'),'sha256':'unused','allow_unavailable':True,'arguments':[]}}}
    def unavailable(*args,**kwargs):
        raise catlass_capabilities.CapabilityUnavailable('msprof unavailable after reaped numeric child')
    monkeypatch.setattr(driver,'profile_validation',unavailable)
    monkeypatch.setattr(sys,'argv',['verify.py','--mode','validation'])
    with pytest.raises(SystemExit) as stopped:
        driver.main(contract)
    assert stopped.value.code == 78
    receipt = json.loads(Path('outputs/catlass-capability-study.json').read_text())
    assert receipt['outcome'] == 'capability-study-unavailable' and receipt['exercise_validated'] is False
    assert calls == []  # parent orchestrator never imports/launches NPU code


def test_msprof_derives_current_export_and_rejects_stale_name_or_wrong_device(tmp_path, monkeypatch):
    from rsi import catlass_capabilities as gates
    reports = tmp_path/'basic-info'
    reports.mkdir()
    path = reports/'OpBasicInfo.csv'
    path.write_text('Kernel Name,Duration(us),Device\nstale_kernel,12.5,2\n')
    contract = {'exercise':{'tools':{'msprof':{'kernel_name':'obsolete_frozen_name',
        'device_mode':'direct-physical','basic_info_arguments':['--device','{physical_device}'],
        'kernel_column':'Kernel Name','duration_column':'Duration(us)','device_column':'Device'}}},
        'profile_capture':True,'physical_device':2,'cases':[{'length':3,'block_count':2,'seed':17}]}
    child={'device_addressing':'direct-physical','application_device':2,'physical_device':2,
           'cases':[{'exported_kernel':'current_kernel'}],'source_hash':'current-source',
           'case_hash':'cases','identity':{'cann_version':'9.3.0'}}
    monkeypatch.setattr(gates,'fresh_device_health',lambda *a,**kw:{})
    def command(*args,**kwargs):
        assert 'ASCEND_RT_VISIBLE_DEVICES' not in kwargs['env']
        assert '--profile-capture' in args[1]['workload']
        return None,{'tool_sha256':'tool','launch_receipt':child}
    monkeypatch.setattr(gates,'owned_command',command)
    with pytest.raises(ValueError,match='one current compiled kernel'):
        gates.msprof_gate(contract,tmp_path)
    path.write_text('Kernel Name,Duration(us),Device\ncurrent_kernel,12.5,2\n')
    evidence = gates.msprof_gate(contract,tmp_path)
    assert evidence['kernel']=='current_kernel' and evidence['source_hash']=='current-source'
    path.write_text('Kernel Name,Duration(us),Device\ncurrent_kernel,12.5,0\n')
    with pytest.raises(ValueError,match='report physical device'):
        gates.msprof_gate(contract,tmp_path)


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
    monkeypatch.setattr(gates,'fresh_device_health',lambda *a,**kw:{})
    contract = {'exercise':{'kind':'msprof-basic','tools':{'msprof':{'device_mode':'direct-physical',
        'basic_info_arguments':['--device','{physical_device}','--report','{report}'],'path':str(tool),
        'sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),'arguments':['--report','{report}']}}},
        'profile_capture':True,'case_hash':'cases','runtime_identity':{},'outer_cpl_handle':'remote:bz-a5:job:actual','physical_device':2}
    with pytest.raises(ValueError,match='required tool command failed'):
        gates.capability_gate(contract,compile_original=lambda path:None,candidate_hash='candidate')
    receipts = list(Path('outputs').glob('*-receipt.json'))
    assert len(receipts)==1
    failure = json.loads(receipts[0].read_text())
    assert failure['outcome']=='capability-failure' and not failure['exercise_validated']
    assert failure['failure']['type']=='ValueError'
    with tarfile.open(failure['archive']['path']) as archive:
        assert archive.extractfile('./basic-command/command.stdout').read()==b'command stdout'
        assert archive.extractfile('./basic-command/command.stderr').read()==b'actual tool error'
        assert archive.extractfile('./basic-info/failed-report.csv').read()==b'retained partial vendor report'


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


def _apply_context_in_real_child(context, parent_pid, sender):
    import os
    from rsi.catlass_project_config import apply_worker_context
    receipt = apply_worker_context(context,parent_pid=parent_pid)
    sender.send((receipt,{name:os.environ.get(name) for name in (
        'ASCEND_RT_VISIBLE_DEVICES','CATLASS_DSL_CACHE_DIR','RSI_OUTER_CPL_HANDLE',
        'RSI_CPL_RUNTIME','RSI_EXPECTED_SOURCE_HASH')}))
    sender.close()


def test_worker_context_applies_actual_child_environment_without_parent_mutation(tmp_path,monkeypatch):
    import multiprocessing
    import os
    from rsi.catlass_project_config import apply_worker_context
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES','parent-device')
    monkeypatch.setenv('RSI_EXPECTED_SOURCE_HASH','stale-parent-command')
    context = {'outer_cpl_handle':'remote:bz-a5:job:actual',
               'physical_device':3,'cache_directory':str(tmp_path/'owned-cache')}
    before = dict(os.environ)
    with pytest.raises(ValueError,match='spawned child'):
        apply_worker_context(context,parent_pid=os.getpid())
    spawn = multiprocessing.get_context('spawn')
    receiver,sender = spawn.Pipe(duplex=False)
    process = spawn.Process(target=_apply_context_in_real_child,args=(context,os.getpid(),sender))
    process.start()
    sender.close()
    try:
        assert receiver.poll(10), 'spawned worker did not return environment receipt'
        receipt,environment = receiver.recv()
        process.join(10)
        assert process.exitcode==0
        assert receipt['pid']!=os.getpid() and receipt['physical_device']==3
        assert environment['ASCEND_RT_VISIBLE_DEVICES']=='3'
        assert environment['CATLASS_DSL_CACHE_DIR']==context['cache_directory']
        assert environment['RSI_OUTER_CPL_HANDLE']==context['outer_cpl_handle']
        assert environment['RSI_CPL_RUNTIME']=='catlass-dsl'
        assert environment['RSI_EXPECTED_SOURCE_HASH'] is None
        assert dict(os.environ)==before
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        receiver.close()


def test_profile_child_is_unmasked_direct_physical_only_with_frozen_capture(fake_consumer,monkeypatch):
    driver,contract,calls=fake_consumer
    contract['exercise']={'kind':'msprof-basic'}
    contract['profile_capture']=True
    with pytest.raises(ValueError,match='must be unmasked'):
        driver.run(contract,'test',case_index=0,profile_capture=True)
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES')
    receipt=driver.run(contract,'test',case_index=0,profile_capture=True)
    assert receipt['application_device']==2 and receipt['device_addressing']=='direct-physical'
    assert receipt['logical_device'] is None
    with pytest.raises(ValueError,match='frozen trusted msprof'):
        driver.run(contract,'validation',profile_capture=True)


def test_direct_verifier_reuses_shared_probe_and_preserves_alarm_policy(tmp_path,monkeypatch):
    from rsi import catlass_capabilities as gates
    monkeypatch.chdir(tmp_path)
    runtime=_with_outer_admission({},tmp_path,physical=3)
    outer=Path(runtime['device_admission']['outer_receipt'])
    calls=[]
    admitted=[True]
    def shared(**kwargs):
        calls.append(kwargs)
        return {'admitted':admitted[0],'physical_device':3,'observation':'shared decision',
                'raw_measurements':{'queries':{}},'measurements':{'health':'Alarm','degraded':True}}
    monkeypatch.setattr(gates,'_shared_device_local_probe',shared,raising=False)
    contract={'physical_device':3,'outer_cpl_handle':'remote:bz-a5:job:admitted',
        'execution_runtime':'catlass-dsl','runtime_identity':runtime,
        'outer_admission':{'path':str(outer),'sha256':hashlib.sha256(outer.read_bytes()).hexdigest(),
            'handle':'remote:bz-a5:job:admitted','physical_device':3,'runtime':'catlass-dsl'}}
    assert gates.fresh_device_health(contract)['measurements']['degraded'] is True
    assert calls[0]['physical_device']==3 and calls[0]['configuration']['allow_alarm'] is True
    assert calls[0]['configuration']['chip'] is None
    gates.fresh_device_health(contract,require_idle=True)
    assert calls[1]['configuration']['require_idle'] is True
    admitted[0]=False
    with pytest.raises(ValueError,match='shared device admission denied'):
        gates.fresh_device_health(contract)


def test_health_failure_precedes_catlass_runtime_and_npu_import_path(fake_consumer,monkeypatch):
    driver,contract,calls=fake_consumer
    def blocked(*args,**kwargs):
        raise ValueError('fresh device probe blocked')
    def runtime_imports(*args,**kwargs):
        raise AssertionError('Catlass/NPU runtime imported before health gate')
    monkeypatch.setattr(driver,'fresh_device_health',blocked)
    monkeypatch.setattr(driver,'check_runtime',runtime_imports)
    with pytest.raises(ValueError,match='fresh device probe blocked'):
        driver.run(contract,'test')
    assert not calls


def test_builtin_hubble_guard_adapts_placement_and_reuses_shared_guard(tmp_path,monkeypatch):
    from rsi.catlass_project_config import device_guard
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES','3')
    monkeypatch.setenv('RSI_OUTER_CPL_HANDLE','remote:bz-a5:job:owned')
    monkeypatch.setenv('RSI_CPL_RUNTIME','catlass-dsl')
    calls=[]
    def shared(request,**kwargs):
        calls.append((request,kwargs))
        return {'admitted':True,'physical_device':3,'observation':'shared guard'}
    monkeypatch.setitem(sys.modules,'rsi.device_admission',SimpleNamespace(guard=shared))
    metadata={'controller_placement':'local','physical_device':3,
              'outer_handle':'remote:bz-a5:job:owned','runtime':'catlass-dsl'}
    request=SimpleNamespace(project=tmp_path/'project',command='timing',profile_path=None)
    receipt=device_guard(request,phase='command',metadata=metadata,
                         configuration={'product':'a5','chip':None},remaining=lambda:100.)
    assert receipt['admitted'] and receipt['physical_device']==3
    assert calls[0][1]['metadata']['controller_placement']=='on'
    assert calls[0][1]['metadata']['executor_identity'] is None
    assert metadata['controller_placement']=='local'


def test_health_runs_once_before_all_cases_not_after_own_context(fake_consumer,monkeypatch):
    driver,contract,calls=fake_consumer
    contract['cases'].append({'length':4,'block_count':1,'seed':19})
    checks=[]
    monkeypatch.setattr(driver,'fresh_device_health',lambda *a,**kw:checks.append('health'))
    driver.run(contract,'test')
    assert checks==['health']
    assert calls.count('compile')==2


def test_numeric_child_is_reaped_before_profile_without_parent_npu_context(tmp_path,monkeypatch):
    from rsi import catlass_capabilities as gates
    monkeypatch.chdir(tmp_path)
    Path('candidate.py').write_text('pass\n')
    monkeypatch.setenv('RSI_EXPECTED_SOURCE_HASH','source')
    order=[]
    cases=[{'length':3,'block_count':1,'seed':17}]
    identity={'source_revision':'src','dependency_revision':'dep','cann_version':'9.3.0'}
    contract={'cases':cases,'case_hash':'cases','physical_device':2,
              'outer_cpl_handle':'remote:bz-a5:job:owned','runtime_identity':identity}
    child={'source_hash':'source','candidate_sha256':hashlib.sha256(Path('candidate.py').read_bytes()).hexdigest(),
           'case_hash':'cases','identity':identity,'correctness':True,
           'physical_device':2,'outer_cpl_handle':contract['outer_cpl_handle'],
           'device_addressing':'masked-logical-zero',
           'cases':[{'case':cases[0],'launched':True,'numeric_match':True}]}
    def completed_child(argv,**kwargs):
        assert argv[-2:]==['--mode','test']
        kwargs['directory'].mkdir(parents=True)
        order.append('numeric-exited')
        return SimpleNamespace(returncode=0,stdout=('RSI_CATLASS_RESULT='+json.dumps(child)).encode())
    def profile(*args,**kwargs):
        assert order==['numeric-exited']
        order.append('profile')
        return {'outcome':'required-capability-verified'}
    monkeypatch.setattr(gates,'run_retained',completed_child)
    monkeypatch.setattr(gates,'capability_gate',profile)
    result=gates.profile_validation(contract,source_hash='source',started=0.)
    assert order==['numeric-exited','profile']
    assert result['numeric_child_exited_before_profile'] is True and result['exercise_validated'] is True


@pytest.mark.parametrize('actor_seconds,wall_seconds',
    [(7200.,None),(900.,5400.),(731.25,5170.5),(300.,420.),(28800.,None),(9600.,None)],
    ids=['legacy','root','carried-remaining','diagnostic-child','a5-campaign','a5-diagnostic'])
def test_lease_binder_matches_actual_assignment_and_never_rebuilds_on_resume(
        tmp_path,monkeypatch,actor_seconds,wall_seconds):
    from rsi.catlass_project_config import bind_project,case_identity
    from rsi.contracts import BudgetLimits,ProjectSpec
    from rsi.project_runner import ProjectConfig
    import runpy
    handle='remote:bz-a5:job:leased'
    outer=tmp_path/'outer.json'
    outer.write_text(json.dumps({'handle':handle,'target':'bz-a5','runtime':'catlass-dsl',
        'npu':{'devices':[0,1],'pool':'fixture','checked_at':'fixture'}}))
    runtime=dict(catlass_source='/profile/source',source_revision='src',dependency_gitlink='dep',
        dependency_revision='dep',consumer_files=[{'path':'/profile/compiler','sha256':'hash'}],
        cann_version='9.3.0',cann_version_file='/profile/version',cann_version_contents='9.3.0',
        pipeline_manifest='/profile/manifest',pipeline_manifest_sha256='hash',catlass_module_sha256='hash',
        compiler_paths={'TlaCompile':'/profile/front','hivmc-a5':'/profile/back'},
        device_admission={'outer_receipt':str(outer),'outer_runtime':'catlass-dsl','allow_alarm':True})
    seed={'operation':'convert','exercise':{'kind':'conversion'},
          'cases':[{'length':3,'block_count':1,'seed':17}]}
    seed_path=tmp_path/'seed.json';seed_path.write_text(json.dumps(seed))
    configuration={'runtime_identity':runtime,'target_python':'/profile/python3.11','projects':{}}
    bindings=[]
    # Lease0 remains occupied; lease1 finishes first and becomes the third task's
    # next-free lease. There is no ordinal % device-count assignment in builder.
    for ordinal,physical in enumerate((0,1,1)):
        name='logical-'+str(ordinal)
        source=tmp_path/(name+'-source');source.mkdir();(source/'candidate.py').write_text('pass\n')
        branch=tmp_path/name;branch.mkdir()
        request=SimpleNamespace(root=branch/'runner',device=str(physical),reopen=False,
            dispatch_id='dispatch-'+name,project_id='operation-'+name)
        import os
        (branch/'worker.json').write_text(json.dumps({'pid':os.getpid(),'dispatch_id':request.dispatch_id}))
        budget=BudgetLimits(design_iterations=None,implementation_iterations=None,execution_tries=None,
                            actor_seconds=actor_seconds,wall_seconds=wall_seconds)
        spec=ProjectSpec(name,'preserve exact goal',('staging-command',),budgets=budget,
                         repository_ref='retained-input-lineage',performance_goal='preserve numeric contract')
        config=ProjectConfig(source=str(source),verifier='not-yet-bound',
            commands={'compile':['old'],'test':['old'],'validation':['old']},editable_files=('candidate.py',),
            generated_paths=('outputs','cache'),retained_paths=('outputs',))
        configuration['projects'][name]={'seed_path':str(seed_path),'case_hash':case_identity(seed)}
        metadata={'target':'bz-a5','runtime':'catlass-dsl','controller_placement':'local',
                  'physical_device':physical,'outer_handle':handle}
        monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES',str(physical))
        monkeypatch.setenv('RSI_OUTER_CPL_HANDLE',handle)
        monkeypatch.setenv('RSI_CPL_RUNTIME','catlass-dsl')
        from dataclasses import asdict
        from rsi.project_binding import validate_resolution
        admission={'operation_id':request.project_id,'spec':spec.to_dict(),'config':asdict(config),
                   'memory_ref':'memory','skills_ref':'skills','configuration_ref':'configuration'}
        bound=bind_project(request,admission,metadata=metadata,configuration=configuration,remaining=lambda:actor_seconds)
        frozen=runpy.run_path(bound['config']['verifier'])['FROZEN_CONTRACT']
        assert frozen['physical_device']==physical and frozen['case_hash']==case_identity(seed)
        assert all(bound['spec'][key]==value for key,value in spec.to_dict().items()
                   if key not in ('command','starting_folder','correctness_ref'))
        assert admission['spec']==spec.to_dict()
        assert frozen['cases']==seed['cases'] and frozen['operation']==seed['operation']
        assert bound['config']['source']!=config.source
        assert bound['config']['retained_paths']==['outputs']
        assert bound['spec']['command']==bound['config']['commands']['test']
        assert not request.root.exists()
        assert Path(bound['config']['verifier']).parent==branch/'lease-inputs'
        assert (source/'candidate.py').read_text()=='pass\n'
        evidence=bound['evidence']
        assert evidence['logical_case_identity']==evidence['resolved_case_identity']
        assert evidence['logical_source_identity']==evidence['resolved_source_identity']
        assert validate_resolution(admission,bound,device=str(physical),executor_identity=None)==bound
        original=Path(bound['config']['verifier']).read_bytes()
        # Hubble stores this resolution and does not invoke binder on resume.
        retained=json.loads(json.dumps(bound))
        request.reopen=True
        with pytest.raises(ValueError,match='resume uses stored resolution'):
            bind_project(request,admission,metadata=metadata,configuration=configuration,remaining=lambda:actor_seconds)
        assert Path(retained['config']['verifier']).read_bytes()==original
        bindings.append(bound)
    assert [b['evidence']['physical_device'] for b in bindings]==[0,1,1]



@pytest.mark.parametrize('exitcode',[0,7])
def test_catlass_config_retains_raw_outputs_cas_after_success_or_failure(tmp_path,exitcode):
    from rsi.python_execution import PythonExecution
    from rsi.runtime import Workflow
    from rsi.catlass_project_config import elapsed_budget
    source=tmp_path/'input';source.mkdir()
    (source/'candidate.py').write_text('pass\n')
    verifier=tmp_path/'verifier.py'
    verifier.write_text('from pathlib import Path\nimport sys\n'
        'Path("outputs").mkdir()\nPath("cache").mkdir()\n'
        'Path("outputs/raw-report.bin").write_bytes(b"raw\\x00\\xffdiagnostic")\n'
        'Path("cache/ignored").write_bytes(b"not-evidence")\n'
        f'sys.exit({exitcode})\n')
    config=project_config(source=source,verifier=verifier,target_python=sys.executable)
    backend=PythonExecution.snapshot(tmp_path/'task',source,verifier,commands=config.commands,
        editable_files=config.editable_files,generated_paths=config.generated_paths,
        retained_paths=config.retained_paths)
    backend.workflow=Workflow(tmp_path/'workflow.json',lambda *args:True,
        source_hash=backend.project_hash(),limits=elapsed_budget())
    receipt=backend.run('validation-evidence','validation')
    assert receipt['returncode']==exitcode
    manifest=backend.store.read_json(receipt['artifact_refs'][0])
    assert [row['path'] for row in manifest['files']]==['outputs/raw-report.bin']
    assert backend.store.read_bytes(manifest['files'][0]['reference'])==b'raw\x00\xffdiagnostic'


def test_case_identity_excludes_physical_bindings_but_tracks_workload():
    import copy
    from rsi.catlass_project_config import case_identity
    first={'operation':'add','cases':[{'length':3,'block_count':1,'seed':17}],
           'exercise':{'kind':'msprof-basic','tools':{'msprof':{'path':'/first/tool','physical_device':0}}}}
    rebound=copy.deepcopy(first)
    rebound['exercise']['tools']['msprof']={'path':'/another/tool','physical_device':5,'arguments':['different']}
    assert case_identity(first)==case_identity(rebound)
    rebound['cases'][0]['length']=4
    assert case_identity(first)!=case_identity(rebound)


def test_optional_target_paths_resolve_package_without_pythonpath_and_reject_controller(tmp_path,monkeypatch):
    from rsi.catlass_verifier import configure_target_python_paths
    root=tmp_path/'registry-target-libs'
    package=root/'fixture_target_package'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('identity="registry-derived-target"\n')
    monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.delenv('PYTHONPATH',raising=False)
    before=list(sys.path)
    monkeypatch.setattr(sys,'version_info',(3,12,0))
    with pytest.raises(ValueError,match='controller interpreter'):
        configure_target_python_paths([str(root)])
    assert sys.path==before
    configure_target_python_paths([])
    assert sys.path==before
    monkeypatch.setattr(sys,'version_info',(3,11,0))
    configure_target_python_paths([str(root)])
    import importlib
    loaded=importlib.import_module('fixture_target_package')
    assert loaded.identity=='registry-derived-target'
    assert Path(loaded.__file__).is_relative_to(root)
    sys.modules.pop('fixture_target_package',None)


@pytest.mark.parametrize('outer_handle', ['remote:bz-a5:session:owned', 'remote:bz-a5:screen:owned'])
def test_worker_environment_requires_actual_outer_job(tmp_path, outer_handle):
    inherited = {'ASCEND_RT_VISIBLE_DEVICES': 'unchanged'}
    with pytest.raises(ValueError):
        worker_environment(inherited=inherited, outer_cpl_handle=outer_handle,
                           physical_device=2, cache_directory=tmp_path/'cache')
    assert inherited == {'ASCEND_RT_VISIBLE_DEVICES': 'unchanged'}


@pytest.fixture
def catlass_parent(tmp_path):
    from dataclasses import replace
    from rsi.catlass_project_config import case_identity
    from rsi.contracts import BudgetLimits
    source = tmp_path/'catlass-parent'
    (source/'reference-skills').mkdir(parents=True)
    (source/'candidate.py').write_text('def kernel(): pass\n')
    (source/'reference-skills/oracle.txt').write_text('Retain the independent numeric oracle.\n')
    seed = dict(operation='convert', exercise={'kind':'conversion'},
        cases=[dict(length=n,block_count=1,seed=17+n) for n in (3,7,11)],
        input_dtype='float32',output_dtype='float16',layout='row-major',
        atol=.001,rtol=.0001,capacity=512,padding_multiple=64,
        provenance={'source':'retained-parent','oracle':'independent-host'})
    cases = tmp_path/'catlass-seed.json'
    cases.write_text(json.dumps(seed))
    runtime = dict(catlass_source='/profile/source',source_revision='src',dependency_gitlink='dep',
        dependency_revision='dep',consumer_files=[{'path':'/profile/compiler','sha256':'hash'}],
        cann_version='9.3.0',cann_version_file='/profile/version',cann_version_contents='9.3.0',
        pipeline_manifest='/profile/manifest',pipeline_manifest_sha256='hash',catlass_module_sha256='hash',
        compiler_paths={'TlaCompile':'/profile/front','hivmc-a5':'/profile/back'})
    _with_outer_admission(runtime,tmp_path)
    outer = Path(runtime['device_admission']['outer_receipt'])
    receipt = json.loads(outer.read_text())
    receipt['npu']['devices'] = [2,3]
    outer.write_text(json.dumps(receipt))
    verifier = tmp_path/'catlass-verify.py'
    freeze_verifier(verifier,seed=seed,runtime_identity=runtime,physical_device=2,
                    outer_cpl_handle='remote:bz-a5:job:admitted')
    config = project_config(source=source,verifier=verifier,target_python='/profile/python3.11')
    spec = replace(project_spec(project_id='catlass-original',goal='Preserve original full workload',config=config),
                   budgets=BudgetLimits(actor_seconds=900,wall_seconds=5400))
    binding = dict(runtime_identity=runtime,target_python='/profile/python3.11',projects={
        spec.project_id:dict(seed_path=str(cases),case_hash=case_identity(seed))})
    return spec,config,binding,cases,seed


def test_catlass_semantic_subset_uses_scoped_factory_and_actual_lease(catlass_parent,tmp_path,monkeypatch):
    from dataclasses import asdict
    import os
    from rsi.curriculum_materialization import capability_catalog,materialize_child,read_materialized_child
    from rsi.curriculum_research import InvestigationIdea,Researcher
    from rsi.executor_profiles import ExecutorProfiles
    from rsi.catlass_project_config import case_identity
    spec,config,binding,cases,seed = catlass_parent
    parent_bytes = {p:p.read_bytes() for p in (cases,Path(config.verifier),*Path(config.source).rglob('*')) if p.is_file()}
    parent_binding = json.loads(json.dumps(binding))
    wrapper = dict(config=asdict(config),binding=binding)
    catalog = capability_catalog(spec,wrapper)
    assert len(catalog)==1 and catalog[0]['template_id']=='catlass.case-subset'
    assert catalog[0]['parameters']['case_indices']['items']['enum']==[0,1,2]
    assert catalog[0]['parameters']['case_indices']['maxItems']==2
    idea = Researcher._bind_investigation(InvestigationIdea('Which case isolates the issue?',
        'Investigate one frozen case','Inform the original task'),
        [dict(project_id=spec.project_id,templates=catalog)],catalog)
    options = dict(root=tmp_path/'catlass-diagnostic',remaining=300.,evidence_refs=('a'*64,),operation_id='semantic-child')
    child = materialize_child(spec,wrapper,idea,**options)
    assert materialize_child(spec,wrapper,idea,**options)==child
    assert read_materialized_child(child.provenance['receipt']['path'],sha256=child.provenance['receipt']['sha256'])==child
    child_seed = json.loads(Path(child.spec.correctness_ref).read_text())
    assert child_seed==dict(seed,cases=[seed['cases'][0]])
    assert child.spec.budgets.actor_seconds==300 and child.spec.command==tuple(child.config.commands['test'])
    assert Path(child.config.verifier).read_bytes()==parent_bytes[Path(config.verifier)]
    assert (Path(child.config.source)/'reference-skills/oracle.txt').read_bytes()==parent_bytes[Path(config.source)/'reference-skills/oracle.txt']
    scoped = child.provenance['binding']
    assert set(scoped['projects'])=={child.spec.project_id} and binding==parent_binding
    assert scoped['runtime_identity']==binding['runtime_identity']
    assert capability_catalog(child.spec,dict(config=asdict(child.config),binding=scoped))==[]
    metadata = dict(target='bz-a5',runtime='catlass-dsl',physical_device=3,executor_identity=None,
                    controller_placement='local',outer_handle='remote:bz-a5:job:admitted')
    profiles = tmp_path/'profiles.json'
    profiles.write_text(json.dumps(dict(profiles={'target':dict(metadata=metadata,
        guard='rsi.catlass_project_config:device_guard',project_binder='rsi.catlass_project_config:bind_project',
        project_configuration=binding)})))
    from core.artifacts import ContentAddressedArtifacts
    from rsi.phase1_recovery import Phase1Recovery
    from rsi.python_execution import PythonExecution
    import shutil
    archived = tmp_path/'failed-original-runner'
    shutil.copytree(config.source,archived/'project')
    (archived/'frozen').mkdir()
    shutil.copyfile(config.verifier,archived/'frozen/verify.py')
    local = ContentAddressedArtifacts(archived/'artifacts')
    contract = local.put_json(dict(spec=spec.to_dict(),config=asdict(config)))
    (archived/'runner.json').write_text(json.dumps(dict(contract=contract)))
    (archived/'service-binding.json').write_text(json.dumps(dict(worker_profile='target')))
    store = ContentAddressedArtifacts(tmp_path/'instance-artifacts')
    cycle = SimpleNamespace(store=store,services=SimpleNamespace(
        executor_factory=ExecutorProfiles(str(profiles)),curriculum_researcher=None,curriculum_materializer=None),
        scheduler=SimpleNamespace(state={'derived_projects':{child.spec.project_id:{
            'provenance_ref':store.put_json(child.provenance)}}}))
    recovery = Phase1Recovery(cycle)
    from rsi.services import BoundCycle
    cycle.curriculum = recovery
    cycle.project_executor_factory = BoundCycle.project_executor_factory.__get__(cycle)
    failed_spec,failed_config = recovery.parent(dict(artifact_ref=store.put_json(dict(root=str(archived)))))
    assert failed_config['binding']==parent_binding
    assert capability_catalog(failed_spec,failed_config)==catalog
    factory = recovery.executor_factory(child.spec)
    assert factory.configuration==scoped
    backend = PythonExecution.snapshot(tmp_path/'readonly-seed-snapshot',child.config.source,child.config.verifier,
        commands=child.config.commands,editable_files=child.config.editable_files,
        generated_paths=child.config.generated_paths,retained_paths=child.config.retained_paths)
    seed_name = Path(child.spec.correctness_ref).relative_to(child.config.source).as_posix()
    shown = backend.read(seed_name)
    assert json.loads(shown['content'])==child_seed
    with pytest.raises(ValueError,match='not in the frozen editable manifest'):
        backend.edit(seed_name,'{}',expected_file_hash=shown['file_hash'],expected_project_hash=shown['project_hash'])
    branch = tmp_path/'actual-child-lease' 
    branch.mkdir()
    request = SimpleNamespace(root=branch/'runner',reopen=False,device='3',project_id='child-operation',dispatch_id='child-dispatch')
    (branch/'worker.json').write_text(json.dumps(dict(pid=os.getpid(),dispatch_id=request.dispatch_id)))
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES','3')
    monkeypatch.setenv('RSI_OUTER_CPL_HANDLE',metadata['outer_handle'])
    monkeypatch.setenv('RSI_CPL_RUNTIME','catlass-dsl')
    admission = dict(operation_id=request.project_id,spec=child.spec.to_dict(),config=asdict(child.config))
    resolved = factory.bind_project('target',request,admission,remaining=lambda:300.)
    actual = runpy.run_path(resolved['config']['verifier'])
    frozen = actual['FROZEN_CONTRACT']
    assert frozen['physical_device']==3 and frozen['cases']==child_seed['cases']
    assert frozen['case_hash']==case_identity(child_seed)
    assert frozen['runtime_identity']==binding['runtime_identity'] and frozen['exercise']==seed['exercise']
    assert resolved['spec']['budgets']==child.spec.budgets.to_dict()
    assert resolved['spec']['command']==resolved['config']['commands']['test']
    original = runpy.run_path(config.verifier)
    left,right = original['case_inputs'](seed['cases'][0],seed['operation'])
    assert actual['reference'](seed['operation'],left,right)==original['reference'](seed['operation'],left,right)
    assert all(p.read_bytes()==data for p,data in parent_bytes.items()) and binding==parent_binding
    assert json.loads(profiles.read_text())['profiles']['target']['project_configuration']==parent_binding


@pytest.mark.parametrize('indices',[[],[0,1,2],[1,1],[-1],[True],[3]])
def test_catlass_materialization_rejects_invalid_subsets_before_writes(catlass_parent,tmp_path,indices):
    from dataclasses import asdict
    from rsi.curriculum_materialization import MaterializationRejected,materialize_child
    spec,config,binding,_,_ = catlass_parent
    root = tmp_path/'invalid-child'
    with pytest.raises(MaterializationRejected):
        materialize_child(spec,dict(config=asdict(config),binding=binding),
            dict(name='child',goal='Investigate one case',acceptance='Same oracle',
                 template_id='catlass.case-subset',parameters=dict(case_indices=indices)),
            root=root,remaining=300,evidence_refs=('a'*64,),operation_id='invalid')
    assert not root.exists()


def test_catlass_materialization_rejects_changed_parent_seed(catlass_parent,tmp_path):
    from dataclasses import asdict
    from rsi.curriculum_materialization import capability_catalog
    spec,config,binding,cases,seed = catlass_parent
    seed['cases'][0]['length']=4
    cases.write_text(json.dumps(seed))
    assert capability_catalog(spec,dict(config=asdict(config),binding=binding))==[]


def test_catlass_admitted_seed_tampering_is_rejected(catlass_parent,tmp_path):
    from dataclasses import asdict
    from rsi.curriculum_materialization import MaterializationRejected,materialize_child,read_materialized_child
    spec,config,binding,_,_ = catlass_parent
    child = materialize_child(spec,dict(config=asdict(config),binding=binding),
        dict(name='child',goal='Investigate one case',acceptance='Same oracle',
            template_id='catlass.case-subset',parameters=dict(case_indices=[1])),
        root=tmp_path/'tampered-child',remaining=300,evidence_refs=('a'*64,),operation_id='tamper')
    Path(child.spec.correctness_ref).write_text('{}')
    with pytest.raises(MaterializationRejected):
        read_materialized_child(child.provenance['receipt']['path'],sha256=child.provenance['receipt']['sha256'])


def test_frozen_entrypoint_selects_existing_function_without_rewriting_source(tmp_path, monkeypatch):
    from types import ModuleType
    from rsi.catlass_project_config import case_identity
    from rsi.catlass_verifier import load_kernel
    tla = ModuleType('catlass.tla')
    tla.kernel = lambda function: function
    catlass = ModuleType('catlass')
    catlass.tla = tla
    monkeypatch.setitem(sys.modules, 'catlass', catlass)
    monkeypatch.setitem(sys.modules, 'catlass.tla', tla)
    path = tmp_path/'candidate.py'
    source = 'import catlass.tla as tla\n@tla.kernel\ndef vector_add(a, b, c):\n    return a, b, c\n'
    path.write_text(source)
    chosen = load_kernel(path, entrypoint='vector_add')
    assert chosen(1, 2, 3) == (1, 2, 3)
    assert path.read_text() == source
    with pytest.raises(ValueError):
        load_kernel(path)  # A different trusted entrypoint cannot silently alias.
    seed = dict(operation='add', cases=[dict(length=64, block_count=1, seed=17)])
    assert case_identity(seed) == case_identity(dict(seed, entrypoint='kernel'))
    assert case_identity(seed) != case_identity(dict(seed, entrypoint='vector_add'))


def test_project_spec_accepts_explicit_campaign_budget_without_resetting_it(tmp_path):
    from rsi.contracts import BudgetLimits
    config = project_config(source=tmp_path/'source', verifier=tmp_path/'verify.py', target_python='/registered/python')
    limits = BudgetLimits(actor_seconds=28800, wall_seconds=None)
    spec = project_spec(project_id='fresh-a5', goal='Preserve admitted independent allocation', config=config, budgets=limits)
    assert spec.budgets is limits
    assert spec.to_dict()['budgets']['actor_seconds'] == 28800
    assert spec.command == tuple(config.commands['test'])


def test_original_retry_preserves_full_catlass_workload_and_actual_lease(catlass_parent, tmp_path, monkeypatch):
    from dataclasses import asdict, replace
    import os
    import shutil
    from core.artifacts import ContentAddressedArtifacts
    from rsi.catlass_project_config import case_identity
    from rsi.curriculum_materialization import materialize_child
    from rsi.curriculum_research import DiagnosticProposal
    from rsi.executor_profiles import ExecutorProfiles
    from rsi.phase1_recovery import Phase1Recovery
    from rsi.project_runner import ProjectConfig
    from rsi.services import BoundCycle
    from rsi.setup_telemetry import SetupTelemetry
    spec, config, binding, cases, seed = catlass_parent
    spec = replace(spec, budgets=replace(spec.budgets, actor_seconds=28800, wall_seconds=None))
    child = materialize_child(spec, dict(config=asdict(config), binding=binding),
        DiagnosticProposal('Isolate one case', 'Understand the original failure', 'Retain measured evidence',
                           'catlass.case-subset', {'case_indices':[0]}),
        root=tmp_path/'diagnostic', remaining=9600, evidence_refs=('a'*64,), operation_id='diagnostic')
    assert len(json.loads(Path(child.spec.correctness_ref).read_text())['cases']) == 1
    archived = tmp_path/'original-runner'
    shutil.copytree(config.source, archived/'project')
    (archived/'project/candidate.py').write_text('def kernel():\n    return 7  # latest ORIGINAL source\n')
    (archived/'frozen').mkdir()
    shutil.copyfile(config.verifier, archived/'frozen/verify.py')
    local = ContentAddressedArtifacts(archived/'artifacts')
    (archived/'runner.json').write_text(json.dumps(dict(contract=local.put_json(
        dict(spec=spec.to_dict(), config=asdict(config))))))
    (archived/'service-binding.json').write_text(json.dumps(dict(worker_profile='target')))
    metadata = dict(target='bz-a5', runtime='catlass-dsl', physical_device=3, executor_identity=None,
                    controller_placement='local', outer_handle='remote:bz-a5:job:admitted')
    profiles = tmp_path/'retry-profiles.json'
    profiles.write_text(json.dumps(dict(profiles={'target':dict(metadata=metadata,
        guard='rsi.catlass_project_config:device_guard',
        project_binder='rsi.catlass_project_config:bind_project', project_configuration=binding)})))
    original_profiles = profiles.read_bytes()
    original_seed = cases.read_bytes()
    store = ContentAddressedArtifacts(tmp_path/'retry-artifacts')
    cycle = SimpleNamespace(root=tmp_path/'cycle', store=store, config=dict(project_configs={}),
        services=SimpleNamespace(executor_factory=ExecutorProfiles(str(profiles)),
            curriculum_researcher=None, curriculum_materializer=None, setup=SetupTelemetry(store)),
        scheduler=SimpleNamespace(state=dict(memory='accepted-diagnostic-memory', derived_projects={}), _save=lambda:None))
    recovery = Phase1Recovery(cycle)
    cycle.curriculum = recovery
    cycle.project_executor_factory = BoundCycle.project_executor_factory.__get__(cycle)
    row = dict(artifact_ref=store.put_json(dict(root=str(archived))))
    retry = recovery.original_retry(spec, row, dict(operation_id='retry-original'))
    assert retry.project_id == spec.project_id and retry.goal == spec.goal and retry.budgets == spec.budgets
    retry_config = recovery.project_config(retry)
    assert '--cases' not in retry_config.commands['test']
    assert Path(retry.correctness_ref).read_bytes() == original_seed
    assert Path(retry_config.verifier).read_bytes() == Path(config.verifier).read_bytes()
    assert (Path(retry_config.source)/'candidate.py').read_bytes() == (archived/'project/candidate.py').read_bytes()
    assert (Path(retry_config.source)/'reference-skills/oracle.txt').read_bytes() == (Path(config.source)/'reference-skills/oracle.txt').read_bytes()
    assert retry.command == tuple(retry_config.commands['test'])
    branch = tmp_path/'retry-lease'
    branch.mkdir()
    request = SimpleNamespace(root=branch/'runner', reopen=False, device='3', project_id='retry-operation', dispatch_id='retry-dispatch')
    (branch/'worker.json').write_text(json.dumps(dict(pid=os.getpid(), dispatch_id=request.dispatch_id)))
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES','3')
    monkeypatch.setenv('RSI_OUTER_CPL_HANDLE',metadata['outer_handle'])
    monkeypatch.setenv('RSI_CPL_RUNTIME','catlass-dsl')
    factory = recovery.executor_factory(retry)
    resolved = factory.bind_project('target',request,
        dict(operation_id=request.project_id, spec=retry.to_dict(), config=asdict(retry_config)), remaining=lambda:28800.)
    frozen = runpy.run_path(resolved['config']['verifier'])['FROZEN_CONTRACT']
    assert frozen['physical_device'] == 3 and frozen['cases'] == seed['cases']
    assert frozen['case_hash'] == case_identity(seed)
    assert resolved['spec']['budgets'] == spec.budgets.to_dict()
    assert recovery.original_retry(spec, row, dict(operation_id='retry-original')) == retry
    assert cases.read_bytes() == original_seed and profiles.read_bytes() == original_profiles
    assert len(json.loads(Path(child.spec.correctness_ref).read_text())['cases']) == 1
