import json
from types import SimpleNamespace

import pytest

from benchmarks.ascend import environment_admission as module
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError


def setup(tmp_path, monkeypatch, *, device=1, cache=None, clock=None):
    root=tmp_path;root.mkdir(parents=True, exist_ok=True)
    transport=SimpleNamespace(runtime_identity={'target':'gz-a3-2','runtime':'named','physical_device':device})
    health=[];probes=[]
    def fresh(t, r):health.append(t.runtime_identity['physical_device']);return {'compiler':'pinned','versions':{}}, 'health.json'
    monkeypatch.setattr(module,'fresh_health',fresh)
    class Probe:
        config=SimpleNamespace(target='Ascend910_9362', arch='dav-2201', execution_profile='gz-a3')
        last_report=root/'probe.json'
        def __call__(self, *args):
            probes.append('probe');self.last_report.write_text(json.dumps({'status':'passed'}))
    admission=module.EnvironmentAdmission(Probe(),[transport],root,cache=cache or root/'cache',dependencies={'compiler':'pinned'},clock=clock or (lambda:1000))
    return admission,health,probes,transport


def test_cache_shared_across_arms_but_health_is_fresh(tmp_path,monkeypatch):
    cache=tmp_path/'cache'
    parent,health,probes,_=setup(tmp_path/'parent',monkeypatch,cache=cache)
    parent(lambda:100);parent(lambda:100)
    assert len(health)==2 and len(probes)==1
    arm,health,probes,_=setup(tmp_path/'arm',monkeypatch,cache=cache)
    arm(lambda:100)
    assert health==[1] and probes==[]
    assert json.loads((tmp_path/'arm/environment-admission.json').read_bytes())['status']=='reused'


def test_expiry_force_device_and_dependency_changes_require_probe(tmp_path,monkeypatch):
    now=[1000]
    admission,health,probes,transport=setup(tmp_path,monkeypatch,clock=lambda:now[0])
    admission(lambda:100);now[0]+=900;admission(lambda:100);admission(lambda:100,force=True)
    transport.runtime_identity['physical_device']=2;admission(lambda:100)
    admission.dependencies={'compiler':'changed'};admission(lambda:100)
    assert len(probes)==5 and len(health)==5


def test_health_failure_and_pending_work_invalidate_cache(tmp_path,monkeypatch):
    admission,_,probes,_=setup(tmp_path,monkeypatch)
    admission(lambda:100)
    def failed(*args):raise ProjectInfrastructureError('unhealthy')
    monkeypatch.setattr(module,'fresh_health',failed)
    with pytest.raises(ProjectInfrastructureError):admission(lambda:100)
    assert not list((tmp_path/'cache').glob('*.json'))
    op=tmp_path/'operation.json';op.write_text(json.dumps({'state':'running','handle':'remote:t:job:pending'}))
    with pytest.raises(ProjectInfrastructureError,match='unsettled'):admission(lambda:100)
    assert len(probes)==1


def test_fresh_health_uses_physical_device_and_rejects_busy_or_forbidden(tmp_path):
    q={kind:{'exit':0,'text':text} for kind,text in {'health':'Health Status : OK','ecc':'Errors : 0','memory':'HBM','usages':'HBM Usage Rate(%) : 0','proc-mem':'No process in device','board':'board'}.items()}
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=0,stdout=json.dumps({'queries':q,'compiler':'pinned','versions':{},'toolkit':None}),job_handle='remote:t:job:health')
    t=SimpleNamespace(runtime_identity={'target':'gz-a3','physical_device':2},run_remote=run)
    module.fresh_health(t,tmp_path);assert calls[-1][-1]=='2'
    q['proc-mem']['text']='PID:123'
    with pytest.raises(ProjectInfrastructureError,match='idle'):module.fresh_health(t,tmp_path)
    t.runtime_identity['physical_device']=3
    with pytest.raises(ProjectInfrastructureError,match='permissions'):module.fresh_health(t,tmp_path)
    assert len(calls)==2


def test_missing_probe_evidence_forces_fresh_probe(tmp_path,monkeypatch):
    admission,health,probes,_=setup(tmp_path,monkeypatch)
    admission(lambda:100)
    (tmp_path/'probe.json').unlink()
    admission(lambda:100)
    assert len(probes)==2 and len(health)==2
