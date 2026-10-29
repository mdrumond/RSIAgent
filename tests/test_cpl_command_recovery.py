"""Exercise the actual RSI remote scripts behind a transport metadata fault."""
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from benchmarks.ascend.cpl_native import CplNativeCandidateBackend
from benchmarks.ascend.runtime_allocation import unsettled_operations
from benchmarks.a3kernels.remote_candidate import _PendingObservation
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError


class LocalRemote:
    def __init__(self):
        self.jobs={};self.calls=[];self.main=[];self.stale=False;self.delay=0;self.shift=None
        self.lost_read=False;self.damage=None
    def __call__(self,argv,**options):
        self.calls.append(argv);action=argv[2]
        if action=='run':
            script=Path(argv[argv.index('--file')+1]);main=script.name=='command.sh'
            if main and self.delay:time.sleep(self.delay)
            result=subprocess.run(('bash',str(script)),capture_output=True,text=True,timeout=17)
            handle='remote:fixture:job:'+str(len(self.jobs)+1)
            if main:self.main.append(handle)
            output=result.stdout
            if 'RSI_RECEIPT' in script.read_text() and self.damage:
                value=json.loads(output)
                if self.damage=='identity':value['receipt']['binding']['operation_nonce']='foreign'
                if self.damage=='output':value['stdout']='cGFydGlhbA=='
                if self.damage=='settlement':value['receipt']['process_group_reaped']=False
                output=json.dumps(value)
            self.jobs[handle]=dict(returncode=result.returncode,stdout=output,stderr=result.stderr,main=main)
            if self.shift and 'RSI_CLOCK' in script.read_text():self.shift()
            if main and self.stale:value=dict(state='running',handle=handle)
            elif self.lost_read and 'RSI_RECEIPT' in script.read_text():
                self.read_handle=handle;value=dict(state='queued',handle=handle)
            else:value=dict(state='completed' if result.returncode==0 else 'failed',exit=result.returncode,handle=handle)
        elif action in ('observe','result','cancel'):
            handle=argv[3];job=self.jobs[handle]
            if self.lost_read and handle==getattr(self,'read_handle',None):
                self.lost_read=False
                raise subprocess.TimeoutExpired(argv,1)
            value=dict(handle=handle,state='cancelled' if action=='cancel' else 'cancelling') if self.stale and job['main'] else dict(
                state='completed' if job['returncode']==0 else 'failed',handle=handle,exit=job['returncode'])
        else:
            assert action=='logs'
            value=dict(handle=argv[3],content=self.jobs[argv[3]][argv[argv.index('--stream')+1]])
        return subprocess.CompletedProcess(argv,0,json.dumps(value),'')


def backend(tmp_path,remote,**options):
    return CplNativeCandidateBackend(cpl_remote='/skill/cpl-remote',target='fixture',runtime='named',
        remote_workspace=str(tmp_path/'remote'),physical_device=0,state_directory=tmp_path/'transport',
        process_runner=remote,submission_reserve_seconds=0,cleanup_reserve_seconds=0,**options)


@pytest.mark.parametrize('shift',[-3600,3600])
def test_real_remote_clock_mapping_ignores_controller_wall_offset(tmp_path,shift):
    remote=LocalRemote();candidate=backend(tmp_path,remote,timeout=3,wall_clock=lambda:time.time()+shift)
    value=candidate.run_remote((sys.executable,'-c',"import sys;print('exact');print('error',file=sys.stderr);sys.exit(7)"),key='compile')
    assert (value.returncode,value.stdout,value.stderr)==(7,'exact\n','error\n')
    assert len(remote.main)==1
    payload=json.loads((candidate.root/'compile/command.json').read_text())
    assert payload['remote_deadline']<=time.monotonic()+3


def test_same_boot_wall_correction_during_probe_does_not_refund_queue_time(tmp_path):
    remote=LocalRemote();offset=[0]
    candidate=backend(tmp_path,remote,timeout=3,wall_clock=lambda:time.time()+offset[0])
    remote.shift=lambda:offset.__setitem__(0,3600)
    result=candidate.run_remote((sys.executable,'-c',"print('still in original elapsed allocation')"),key='compile')
    payload=json.loads((candidate.root/'compile/command.json').read_text())
    assert result.returncode==0
    assert 0<payload['remote_deadline']-time.monotonic()<3


def test_queue_delay_past_remote_deadline_cannot_start_candidate(tmp_path):
    remote=LocalRemote();remote.delay=.3
    candidate=backend(tmp_path,remote,timeout=5)
    candidate._sample_remote_clock=lambda *args,**options: dict(remote_deadline=time.monotonic()+.05,
        remote_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    marker=tmp_path/'must-not-run'
    with pytest.raises(_PendingObservation,match='allocation expired'):
        candidate.run_remote((sys.executable,'-c',f'from pathlib import Path;Path({str(marker)!r}).touch()'),key='compile')
    assert not marker.exists()
    assert len(remote.main)==1


@pytest.mark.parametrize('damage',[None,'identity','output','settlement'])
def test_stale_cpl_status_uses_only_complete_bound_terminal_proof(tmp_path,damage):
    remote=LocalRemote();remote.stale=True;remote.damage=damage
    left=[5.]
    candidate=backend(tmp_path,remote,timeout=5,remaining_seconds=lambda:left[0])
    # Move the parent work allocation to zero only after the real payload ran.
    original=remote.__call__
    def client(argv,**options):
        value=original(argv,**options)
        if argv[2]=='run' and Path(argv[argv.index('--file')+1]).name=='command.sh':left[0]=0
        return value
    candidate._run=client
    command=(sys.executable,'-c',"import sys;print('complete');print('diagnostic',file=sys.stderr);sys.exit(7)")
    if damage:
        with pytest.raises(ProjectInfrastructureError,match='receipt is invalid'):
            candidate.run_remote(command,key='compile')
        assert unsettled_operations(candidate.root)
    else:
        result=candidate.run_remote(command,key='compile')
        assert (result.returncode,result.stdout,result.stderr)==(7,'complete\n','diagnostic\n')
        assert not unsettled_operations(candidate.root)
        assert json.loads((candidate.root/'compile/operation.json').read_text())['command_receipt']['termination']=='exit'
    assert len(remote.main)==1


def test_cancelled_metadata_and_interrupted_receipt_read_recover_without_resubmit(tmp_path):
    remote=LocalRemote();remote.stale=True;remote.lost_read=True
    left=[5.]
    candidate=backend(tmp_path,remote,timeout=5,remaining_seconds=lambda:left[0])
    original=remote.__call__
    def client(argv,**options):
        result=original(argv,**options)
        if argv[2]=='run' and Path(argv[argv.index('--file')+1]).name=='command.sh':left[0]=0
        return result
    candidate._run=client
    command=(sys.executable,'-c',"print('retained result')")
    with pytest.raises(_PendingObservation):candidate.run_remote(command,key='compile')
    path=candidate.root/'compile/operation.json';record=json.loads(path.read_text())
    record['state']='cancelled';path.write_text(json.dumps(record))
    assert unsettled_operations(candidate.root)
    candidate.recover_owned_operations(seconds=5)
    result=candidate.run_remote(command,key='compile')
    assert result.returncode==0 and result.stdout=='retained result\n'
    assert len(remote.main)==1
    assert not unsettled_operations(candidate.root)


def test_legacy_script_without_retained_handle_is_not_dispatched(tmp_path):
    remote=LocalRemote();candidate=backend(tmp_path,remote,timeout=5)
    directory=candidate.root/'legacy';directory.mkdir(parents=True)
    command=[sys.executable,'-c','pass']
    payload=json.dumps(dict(argv=command,expires_at=time.time()+30))
    (directory/'command.json').write_text(payload);(directory/'command.sh').write_text('exit 0\n')
    with pytest.raises(ProjectInfrastructureError,match='unmapped legacy'):
        candidate.run_remote(command,key='legacy')
    assert remote.calls==[]
    assert (directory/'command.json').read_text()==payload


@pytest.mark.parametrize('wrapper_exit', [1, 125])
def test_wrapper_failure_without_receipt_stays_unresolved_until_proof_arrives(tmp_path, wrapper_exit):
    remote=LocalRemote();candidate=backend(tmp_path,remote,timeout=5)
    original=remote.__call__;saved={}
    def client(argv,**options):
        result=original(argv,**options)
        if argv[2]=='run' and Path(argv[argv.index('--file')+1]).name=='command.sh':
            command=json.loads((candidate.root/'compile/command.json').read_text())
            path=Path(command['result_directory'])/'terminal.json'
            saved['path'],saved['bytes']=path,path.read_bytes();path.unlink()
            handle=json.loads(result.stdout)['handle'];remote.jobs[handle]['returncode']=wrapper_exit
            result.stdout=json.dumps(dict(state='failed',handle=handle,exit=wrapper_exit))
        return result
    candidate._run=client
    command=(sys.executable,'-c',"print('complete proof')")
    with pytest.raises(_PendingObservation,match='receipt is pending'):
        candidate.run_remote(command,key='compile')
    assert unsettled_operations(candidate.root)
    saved['path'].write_bytes(saved['bytes'])
    candidate.recover_owned_operations(seconds=5)
    assert candidate.run_remote(command,key='compile').stdout=='complete proof\n'
    assert len(remote.main)==1 and not unsettled_operations(candidate.root)
