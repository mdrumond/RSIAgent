import json
from types import SimpleNamespace

import pytest

from benchmarks.ascend import environment_admission as module
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError


def setup(tmp_path, monkeypatch, *, device=1, cache=None, clock=None):
    root=tmp_path;root.mkdir(parents=True, exist_ok=True)
    transport=SimpleNamespace(recover_owned_operations=lambda:None, runtime_identity={'target':'gz-a3-2','runtime':'named','physical_device':device})
    health=[];probes=[]
    def fresh(t, r, **kwargs):
        health.append(t.runtime_identity['physical_device'])
        return {'compiler':'pinned','versions':{'torch':'torch-v','torch-npu':'npu-v'},'toolkit':'runtime: cann-v',
                'product':'Ascend910_9362','arch':'dav-2201','soc_version':255}, 'health.json'
    monkeypatch.setattr(module,'fresh_health',fresh)
    class Probe:
        config=SimpleNamespace(target='Ascend910_9362', arch='dav-2201', execution_profile='gz-a3')
        last_report=root/'probe.json'
        def __call__(self, *args):
            probes.append('probe');self.last_report.write_text(json.dumps({'status':'passed'}))
    admission=module.EnvironmentAdmission(Probe(),[transport],root,cache=cache or root/'cache',dependencies={'target':'Ascend910_9362','arch':'dav-2201','compiler_version':'pinned',
        'environment_identity':{'torch':'torch-v','torch_npu':'npu-v','cann_version':'runtime: cann-v','soc_version':255}},clock=clock or (lambda:1000))
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
    admission.dependencies['extra_dependency']='changed';admission(lambda:100)
    assert len(probes)==5 and len(health)==5


def test_health_failure_and_pending_work_invalidate_cache(tmp_path,monkeypatch):
    admission,_,probes,_=setup(tmp_path,monkeypatch)
    admission(lambda:100)
    def failed(*args,**kwargs):raise ProjectInfrastructureError('unhealthy')
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
    module.fresh_health(t,tmp_path,allowed_devices=[0,1,2],health_workspace='/owned/incoming');assert calls[-1][-1]=='2'
    q['proc-mem']['text']='PID:123'
    with pytest.raises(ProjectInfrastructureError,match='idle'):module.fresh_health(t,tmp_path,allowed_devices=[0,1,2],health_workspace='/owned/incoming')
    t.runtime_identity['physical_device']=3
    with pytest.raises(ProjectInfrastructureError,match='permissions'):module.fresh_health(t,tmp_path,allowed_devices=[0,1,2],health_workspace='/owned/incoming')
    assert len(calls)==2


def test_missing_probe_evidence_forces_fresh_probe(tmp_path,monkeypatch):
    admission,health,probes,_=setup(tmp_path,monkeypatch)
    admission(lambda:100)
    (tmp_path/'probe.json').unlink()
    admission(lambda:100)
    assert len(probes)==2 and len(health)==2


@pytest.mark.parametrize('field', ['compiler','toolkit','torch','torch_npu','product','arch','soc_version','missing'])
def test_discovered_mismatch_blocks_probe_and_invalidates_cached_admission(tmp_path,monkeypatch,field):
    admission,_,probes,_=setup(tmp_path,monkeypatch)
    admission(lambda:100)
    original=module.fresh_health
    def changed(*args,**kwargs):
        actual,path=original(*args,**kwargs)
        if field=='torch':actual['versions']['torch']='changed'
        elif field=='torch_npu':actual['versions']['torch-npu']='changed'
        elif field=='missing':actual.pop('compiler')
        else:actual[field]='changed'
        return actual,path
    monkeypatch.setattr(module,'fresh_health',changed)
    with pytest.raises(ProjectInfrastructureError,match='differs from pinned'):
        admission(lambda:100)
    assert len(probes)==1 and not list((tmp_path/'cache').glob('*.json'))


def test_unpinned_dependency_blocks_probe(tmp_path,monkeypatch):
    admission,_,probes,_=setup(tmp_path,monkeypatch)
    admission.dependencies['environment_identity'].pop('torch')
    with pytest.raises(ProjectInfrastructureError,match='requires pinned'):
        admission(lambda:100)
    assert not probes


@pytest.mark.parametrize('registry_root', [None, '/owned'])
def test_first_health_uses_configured_existing_subdirectory_before_any_bundle_upload(tmp_path,registry_root):
    import subprocess
    from benchmarks.ascend.cpl_native import CplNativeCandidateBackend
    queries={kind:{'exit':0,'text':text} for kind,text in {'health':'Health Status : OK',
        'ecc':'Errors : 0','memory':'HBM','usages':'HBM Usage Rate(%) : 0',
        'proc-mem':'No process in device','board':'Chip Name : Ascend910\nNPU Name : 9362'}.items()}
    report={'queries':queries,'compiler':'pinned','versions':{},'toolkit':None}
    calls=[]
    def cpl(argv,**kwargs):
        calls.append(argv)
        action=argv[2]
        if action=='run':
            assert argv[argv.index('--cwd')+1]==('.cpl-remote/incoming' if registry_root else '/owned/.cpl-remote/incoming')
            value={'state':'completed','handle':'remote:fixture:job:health','exit':0}
        else:
            assert action=='logs'
            value={'handle':'remote:fixture:job:health','content':json.dumps(report) if argv[argv.index('--stream')+1]=='stdout' else ''}
        return subprocess.CompletedProcess(argv,0,json.dumps(value),'')
    transport=CplNativeCandidateBackend(cpl_remote='/skill/cpl-remote',target='fixture',runtime='named',
        remote_workspace='/owned/not-created-yet',registry_workspace=registry_root,
        physical_device=0,state_directory=tmp_path/'transport',process_runner=cpl)
    actual,_=module.fresh_health(transport,tmp_path,health_workspace='/owned/.cpl-remote/incoming')
    assert actual['product']=='Ascend910_9362'
    assert [argv[2] for argv in calls]==['run','logs','logs']


def test_missing_health_workspace_rejects_before_any_submission(tmp_path):
    t=SimpleNamespace(runtime_identity={'physical_device':0},run_remote=lambda *a,**k:pytest.fail('remote submission'))
    with pytest.raises(ProjectInfrastructureError,match='health_workspace'):
        module.fresh_health(t,tmp_path)


def test_admission_recovers_owned_handle_before_deciding_pending(tmp_path, monkeypatch):
    admission, health, probes, transport = setup(tmp_path, monkeypatch)
    operation = tmp_path/'operation.json'
    operation.write_text(json.dumps({'state':'running','handle':'remote:t:job:owned'}))
    def recover():
        operation.write_text(json.dumps({'state':'completed','handle':'remote:t:job:owned'}))
    transport.recover_owned_operations = recover
    admission(lambda:100)
    assert health == [1] and probes == ['probe']


def test_shared_scope_keeps_owned_recovery_and_foreign_trial_blocking(tmp_path, monkeypatch):
    admission, health, probes, transport = setup(tmp_path/'binding', monkeypatch)
    admission.pending_root = tmp_path
    own = admission.root/'operation.json'
    own.write_text(json.dumps(dict(state='running', handle='remote:t:job:owned')))
    other = tmp_path/'trial/runtime/operation.json'
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps(dict(state='running', handle='remote:t:job:trial')))
    recovered = []
    def recover():
        recovered.append('owned')
        own.write_text(json.dumps(dict(state='completed', handle='remote:t:job:owned')))
    transport.recover_owned_operations = recover
    with pytest.raises(ProjectInfrastructureError, match='unsettled'):
        admission(lambda:100)
    assert recovered == ['owned'] and not health and not probes
    assert json.loads(other.read_text())['state'] == 'running'
    other.write_text(json.dumps(dict(state='cancelled', handle='remote:t:job:trial')))
    admission(lambda:100)
    assert recovered == ['owned','owned'] and health == [1] and probes == ['probe']


@pytest.mark.parametrize('state', ['running','completed'])
def test_foreground_observes_owning_trial_handle_without_resubmitting(tmp_path, monkeypatch, state):
    import subprocess
    from benchmarks.ascend.cpl_native import CplNativeCandidateBackend
    root = tmp_path/'campaign'
    calls = []
    def client(argv, **kwargs):
        calls.append(argv)
        assert argv[2:] == ('observe','remote:fixture:job:trial')
        return subprocess.CompletedProcess(argv,0,json.dumps(dict(state=state,handle=argv[3])),'')
    foreground = CplNativeCandidateBackend(cpl_remote='/skill/cpl-remote',target='fixture',runtime='named',
        remote_workspace='/owned/foreground',physical_device=0,state_directory=root/'foreground',process_runner=client)
    trial = CplNativeCandidateBackend(cpl_remote='/skill/cpl-remote',target='fixture',runtime='named',
        remote_workspace='/owned/trial',physical_device=0,state_directory=root/'trial/runtime/transport')
    journal = trial.root/'measurement/operation.json'
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps(dict(state='queued',handle='remote:fixture:job:trial',
        identity=dict(route=trial.runtime_identity,workspace=str(trial._workspace),arguments=[]))))
    admission, health, probes, _ = setup(root/'foreground',monkeypatch)
    admission.transports = (foreground,)
    admission.pending_root = root
    if state == 'running':
        with pytest.raises(ProjectInfrastructureError,match='unsettled'):
            admission(lambda:100)
        assert not health and not probes
    else:
        admission(lambda:100)
        assert health == [0] and probes == ['probe']
    assert len(calls) == 1 and json.loads(journal.read_text())['state'] == state
    from benchmarks.ascend.runtime_allocation import unsettled_operations
    assert bool(unsettled_operations(root)) == (state == 'running')
