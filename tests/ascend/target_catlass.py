"""CPU-only A5 contract behavior; synthetic tensors are never device evidence."""
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import pytest
from ascend_tools.targets import catlass
from ascend_tools.targets.health import AdmissionFailure, assess


def test_preserved_oracles_and_conversion_rounding():
    assert catlass.reference('add', [-2., 0., 3.], [4., -1., .5]) == [2., -1., 3.5]
    assert catlass.reference('multiply', [-2., 0., 3.], [4., -1., .5]) == [-8., 0., 1.5]
    assert catlass.reference('relu', [-2., 0., 3.], []) == [0., 0., 3.]
    left, _ = catlass.case_inputs(dict(length=6, seed=17), 'convert')
    expected = catlass.reference('convert', left, [])
    assert expected[:4] == [1.0009765625, -1.0009765625, 1., 1.001953125]
    assert expected == [struct.unpack('e', struct.pack('e', x))[0] for x in left]
    assert expected != left


def health(status='OK'):
    outputs = dict(board='NPU ID: 2\nChip Name: Ascend950',
        health='Health Status: '+status+'\nError Code: 0',
        usages='HBM Capacity(MB): 1000\nHBM Usage Rate(%): 0\nAicore Usage Rate(%): 0\nAivector Usage Rate(%): 0\nNPU Utilization(%): 0\nAicube Usage Rate(%): 0',
        **{'proc-mem': 'No process in device.', 'current-fault-event': 'Fault Event Count: 0'})
    return dict(physical_device=2, queries={name: dict(exit=0, stdout=text) for name, text in outputs.items()})


def test_alarm_requires_explicit_ordinary_work_opt_in():
    config = dict(product='a5', chip=None)
    with pytest.raises(AdmissionFailure):
        assess(health('ALARM'), physical_device=2, configuration=config)
    ordinary = assess(health('ALARM'), physical_device=2, configuration=dict(config, allow_alarm=True))
    assert ordinary['degraded'] is True
    with pytest.raises(AdmissionFailure):
        assess(health('ALARM'), physical_device=2, configuration=dict(config, allow_alarm=True, require_idle=True))
    assert assess(health(), physical_device=2, configuration=dict(config, require_idle=True))['idle']


def test_unresponsive_alarm_device_still_excluded():
    raw = health('ALARM')
    raw['queries']['proc-mem'] = dict(failure='query_timeout', stdout='', stderr='')
    with pytest.raises(AdmissionFailure) as error:
        assess(raw, physical_device=2, configuration=dict(product='a5', chip=None, allow_alarm=True))
    assert error.value.kind == 'query_timeout'


def test_wrong_source_revision_fails_before_importing_catlass(monkeypatch):
    monkeypatch.setattr(catlass.sys, 'version_info', (3, 11))
    monkeypatch.setattr(catlass, 'checked_git', lambda *args: 'different revision')
    with pytest.raises(ValueError, match='source revision'):
        catlass.check_runtime(dict(catlass_source='unused', source_revision='frozen'))


class Tensor:
    def __init__(self, values): self.values = list(values)
    def __getitem__(self, item): return Tensor(self.values[item])
    def cpu(self): return self
    def tolist(self): return self.values
    def clone(self): return Tensor(self.values)
    def mark_compact_shape_dynamic(self, _): return self


@pytest.mark.parametrize('physical', [0, 1])
@pytest.mark.parametrize('mode,bad', [('compile', False), ('test', False), ('test', True)])
def test_compile_launch_and_numeric_results_are_separate(tmp_path, monkeypatch, mode, bad, physical):
    calls = []
    class Artifact:
        def __call__(self, a, b, out, **kwargs):
            calls.append('launch')
            out.values = [x+y+(1 if bad else 0) for x, y in zip(a.values, b.values)]
    tla = SimpleNamespace(arch=SimpleNamespace(RowMajor='row'), compile=lambda *args, **kw: (calls.append('compile') or Artifact()))
    torch = SimpleNamespace(__version__='test', __file__=__file__, float32='f32',
        tensor=lambda values, **kwargs: Tensor(values), equal=lambda a,b: a.values == b.values,
        save=lambda value,path: Path(path).write_text('synthetic CPU fixture'),
        npu=SimpleNamespace(set_device=lambda n: calls.append(('device',n)), synchronize=lambda: calls.append('sync')))
    npu = SimpleNamespace(__version__='test', __file__=__file__)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch_npu', npu)
    monkeypatch.setitem(sys.modules, 'catlass', SimpleNamespace(tla=tla))
    monkeypatch.setitem(sys.modules, 'catlass.tla', tla)
    monkeypatch.setitem(sys.modules, 'catlass.tla.runtime', SimpleNamespace(from_dlpack=lambda x, **kw:x))
    monkeypatch.setattr(catlass, 'check_runtime', lambda identity: dict(synthetic=True))
    monkeypatch.setattr(catlass, 'load_kernel', lambda *args,**kwargs: object())
    monkeypatch.setattr(catlass, 'local_probe', lambda **kw: dict(admitted=True))
    monkeypatch.setattr(catlass.subprocess, 'run', lambda *args, **kw: calls.append('bc_compile'))
    for name in ('ASCEND_PHYSICAL_DEVICE', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CATLASS_DSL_CACHE_DIR', 'overwritten')
    identity = dict(device_health=dict(product='a5'), expected_versions=dict(torch='test', torch_npu='test'),
                    module_sha256=dict(torch=catlass.sha(__file__), torch_npu=catlass.sha(__file__)))
    report = dict(compile='not-run', launch='not-run', correctness='not-run', device_evidence=False)
    contract = dict(abi_version=1, operation='add', cases=[dict(length=3, block_count=1, seed=17)])
    args = SimpleNamespace(mode=mode, physical_device=physical, candidate=str(tmp_path/'candidate.py'),source_sha256='trusted-synthetic-source')
    if bad:
        with pytest.raises(ValueError, match='numeric mismatch'):
            catlass.execute(args, contract, identity, tmp_path, report, lambda: None)
        assert report['compile'] == report['launch'] == 'passed'
        assert report['correctness'] != 'passed'
    else:
        catlass.execute(args, contract, identity, tmp_path, report, lambda: None)
        assert report['compile'] == 'passed'
        assert ('launch' in calls) == (mode == 'test')
        assert report['correctness'] == ('passed' if mode == 'test' else 'not-run')
    assert calls.index('bc_compile') < calls.index('compile')
    assert __import__('os').environ['ASCEND_PHYSICAL_DEVICE'] == str(physical)
    assert 'physical_device' not in identity


@pytest.mark.parametrize('changed', [None, 'compiler_bridge', 'hivmc-a5'])
def test_current_pipeline_uses_catlass_selected_bridge_and_backend(tmp_path, monkeypatch, changed):
    from contextlib import nullcontext
    files = {}
    for name in ('compiler_bridge', 'hivmc-a5', 'catlass', 'manifest', 'cann'):
        path = tmp_path/name
        path.write_text(name)
        files[name] = path
    selected = {name: files[name] for name in ('compiler_bridge', 'hivmc-a5')}
    execution = SimpleNamespace(resolve_bridge_extension_path=lambda:selected['compiler_bridge'],
                                _find_hivmc_a5=lambda:selected['hivmc-a5'])
    monkeypatch.setitem(sys.modules, 'catlass', SimpleNamespace(__file__=str(files['catlass'])))
    monkeypatch.setitem(sys.modules, 'catlass._mlir', SimpleNamespace(ir=SimpleNamespace(Context=nullcontext)))
    monkeypatch.setitem(sys.modules, 'catlass.execution', execution)
    monkeypatch.setattr(catlass.sys, 'version_info', (3, 11))
    monkeypatch.setattr(catlass, 'checked_git', lambda repo,*args:
        'master-sha' if args==('rev-parse','HEAD') else '160000 commit dep-sha\tdependency' if args[0]=='ls-tree' else '')
    monkeypatch.delenv('TLA_DSL_COMPILE', raising=False)
    monkeypatch.setenv('PATH', '')
    profile = dict(catlass_source=str(tmp_path),source_revision='master-sha',dependency_gitlink='dependency',
        dependency_revision='dep-sha',cann_version='9.3.0',cann_version_file=str(files['cann']),cann_version_contents='cann',
        pipeline_manifest=str(files['manifest']),pipeline_manifest_sha256=catlass.sha(files['manifest']),
        compiler_paths={name:str(files[name]) for name in selected},
        compiler_consumer_files=[dict(path=str(files[name]),sha256=catlass.sha(files[name])) for name in selected],
        module_sha256=dict(catlass=catlass.sha(files['catlass'])))
    if changed:
        selected[changed] = files['manifest']
        with pytest.raises(ValueError,match='compiler path mismatch'):
            catlass.check_runtime(profile)
    else:
        assert catlass.check_runtime(profile)['source_revision']=='master-sha'


def test_kernel_reload_stages_current_bytes_despite_same_size_timestamp_and_cached_source(tmp_path, monkeypatch):
    import ast
    import inspect
    import os
    import py_compile
    # A staging decorator reads source, as the actual DSL does; invoke its staged result.
    def stage(function):
        tree = ast.parse(inspect.getsource(function))
        value = next(node.value.value for node in ast.walk(tree) if isinstance(node, ast.Return))
        return lambda *_: value
    monkeypatch.setitem(sys.modules, 'catlass.tla', SimpleNamespace(kernel=stage))
    monkeypatch.setitem(sys.modules, 'catlass', SimpleNamespace(tla=sys.modules['catlass.tla']))
    path = tmp_path/'candidate.py'
    first = 'import catlass.tla as tla\n@tla.kernel\ndef kernel(a,b,c):\n    return 1\n'
    path.write_text(first)
    timestamp = path.stat().st_mtime_ns
    py_compile.compile(str(path),doraise=True)
    assert catlass.load_kernel(path,expected_sha256=catlass.sha(path))(None,None,None)==1
    path.write_text(first.replace('return 1','return 2'))
    os.utime(path,ns=(timestamp,timestamp))
    assert catlass.load_kernel(path,expected_sha256=catlass.sha(path))(None,None,None)==2
    with pytest.raises(ValueError,match="candidate source identity mismatch"):
        catlass.load_kernel(path,expected_sha256="0"*64)


def test_bitcode_child_selects_validated_package_before_inherited_package_and_keeps_dependencies(tmp_path, monkeypatch):
    import json
    preferred, inherited = tmp_path/'preferred', tmp_path/'inherited'
    output = tmp_path/'child-selection.json'
    for root,label in [(preferred,'preferred'),(inherited,'inherited')]:
        package=root/'catlass';package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        (package/'bc_compile.py').write_text(
            'import json,pathlib,registry_dependency\n'+
            f'pathlib.Path({str(output)!r}).write_text(json.dumps([{label!r},registry_dependency.VALUE]))\n')
    (inherited/'registry_dependency.py').write_text("VALUE='registry-preserved'\n")
    monkeypatch.setenv('PYTHONPATH',str(inherited))
    monkeypatch.setattr(sys,'path',sys.path.copy())
    monkeypatch.setattr(sys,'version_info',(3,11))
    monkeypatch.setattr(catlass,'check_runtime',lambda identity:
        catlass.configure_target_python_paths(identity['target_python_paths']))
    monkeypatch.setattr(catlass,'local_probe',lambda **kwargs:dict(admitted=True))
    monkeypatch.setattr(catlass,'load_kernel',lambda *args,**kwargs:object())
    tla=SimpleNamespace(arch=SimpleNamespace(RowMajor='row'),compile=lambda *args,**kwargs:object())
    torch=SimpleNamespace(__version__='fixture',__file__=__file__,float32='f32',tensor=lambda values,**kwargs:Tensor(values),
                          npu=SimpleNamespace(set_device=lambda _:None))
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace(__version__='fixture',__file__=__file__))
    monkeypatch.setitem(sys.modules,'catlass',SimpleNamespace(tla=tla))
    monkeypatch.setitem(sys.modules,'catlass.tla',tla)
    monkeypatch.setitem(sys.modules,'catlass.tla.runtime',SimpleNamespace(from_dlpack=lambda value,**kwargs:value))
    for name in ('ASCEND_PHYSICAL_DEVICE','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
        monkeypatch.delenv(name,raising=False)
    identity=dict(device_health=dict(product='a5'),target_python_paths=[str(preferred)],
                  expected_versions=dict(torch='fixture',torch_npu='fixture'),
                  module_sha256=dict(torch=catlass.sha(__file__),torch_npu=catlass.sha(__file__)))
    args=SimpleNamespace(mode='compile',physical_device=0,candidate=str(tmp_path/'candidate.py'),source_sha256='trusted')
    report=dict(compile='not-run',launch='not-run',correctness='not-run',device_evidence=False)
    catlass.execute(args,dict(abi_version=1,operation='add',cases=[dict(length=3,block_count=1,seed=17)]),
                    identity,tmp_path,report,lambda:None)
    assert json.loads(output.read_text())==['preferred','registry-preserved']
    assert __import__('os').environ['PYTHONPATH']==str(inherited)


@pytest.mark.parametrize('location',['decorator','annotation','helper-default','missing-decorator'])
def test_definition_time_effects_are_rejected_before_module_execution(tmp_path, monkeypatch, location):
    marker=tmp_path/'side-effect'
    effect=f"__import__('pathlib').Path({str(marker)!r}).write_text('unexpected')"
    decorator='@'+effect if location=='decorator' else '' if location=='missing-decorator' else '@tla.kernel'
    annotation=effect if location=='annotation' else 'tla.Tensor'
    helper=f'def helper(x={effect}):\n    return x\n' if location=='helper-default' else ''
    source=f'import catlass.tla as tla\n{helper}{decorator}\ndef kernel(a: {annotation},b: tla.Tensor,c: tla.Tensor):\n    pass\n'
    path=tmp_path/'candidate.py';path.write_text(source)
    tla=SimpleNamespace(Tensor=object,kernel=lambda function:function)
    monkeypatch.setitem(sys.modules,'catlass',SimpleNamespace(tla=tla))
    monkeypatch.setitem(sys.modules,'catlass.tla',tla)
    with pytest.raises(ValueError,match='ABI'):
        catlass.load_kernel(path,allowed_helpers=['helper'],expected_sha256=catlass.sha(path))
    assert not marker.exists()


def test_approved_tensor_annotations_and_constexpr_helper_defaults_stage_and_call(tmp_path, monkeypatch):
    class Constexpr:
        def __class_getitem__(cls,kind): return kind
    # CPU staging fixture; actual native annotated-kernel proof is retained separately.
    tla=SimpleNamespace(Tensor=list,Constexpr=Constexpr,kernel=lambda function:function,
                        compile=lambda function,*args:function)
    monkeypatch.setitem(sys.modules,'catlass',SimpleNamespace(tla=tla))
    monkeypatch.setitem(sys.modules,'catlass.tla',tla)
    path=tmp_path/'candidate.py'
    path.write_text('import catlass.tla as tla\nOFFSET=1\ndef helper(x: int,offset: tla.Constexpr[int]=OFFSET)->int:\n    return x+offset\n'
                    '@tla.kernel\ndef kernel(a: tla.Tensor,b: tla.Tensor,c: tla.Tensor)->None:\n    c[0]=helper(a[0]+b[0])\n')
    kernel=catlass.load_kernel(path,allowed_helpers=['helper'],expected_sha256=catlass.sha(path))
    out=[0];compiled=tla.compile(kernel,[2],[3],out);compiled([2],[3],out)
    assert out==[6]
