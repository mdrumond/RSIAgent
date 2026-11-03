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
    monkeypatch.setattr(catlass, 'load_kernel', lambda *args: object())
    monkeypatch.setattr(catlass, 'local_probe', lambda **kw: dict(admitted=True))
    monkeypatch.setattr(catlass.subprocess, 'run', lambda *args, **kw: calls.append('bc_compile'))
    for name in ('ASCEND_PHYSICAL_DEVICE', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CATLASS_DSL_CACHE_DIR', 'overwritten')
    identity = dict(device_health=dict(product='a5'), expected_versions=dict(torch='test', torch_npu='test'),
                    module_sha256=dict(torch=catlass.sha(__file__), torch_npu=catlass.sha(__file__)))
    report = dict(compile='not-run', launch='not-run', correctness='not-run', device_evidence=False)
    contract = dict(abi_version=1, operation='add', cases=[dict(length=3, block_count=1, seed=17)])
    args = SimpleNamespace(mode=mode, physical_device=physical, candidate=str(tmp_path/'candidate.py'))
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
