import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import pytest
from ascend_tools.baselines import describe
from ascend_tools.baselines.projects import prepare_project


def qualification(workload='fa',target='a3'):
    expected=describe(workload);identity={'fingerprint':'fixture','torch':'fixture','torchNpu':'fixture','deviceName':'fixture'}
    receipt={k:expected[k] for k in ['workload','sourceHash','caseHash','verifierHash']}
    receipt.update(format=1,qualified=True,target=target,runtime=identity,
        independentChecks={'passed':True,'checkerHash':expected['independentCheckHash']},
        cases=[{'id':i,'passed':True,'referenceHash':'a','outputHash':'b','maxAbsError':0.} for i in expected['caseIds']],
        timing={'passed':True,'samples':[{'id':i,'seconds':[.01]} for i in expected['caseIds']]})
    return receipt,identity


def test_templates_bundle_two_frozen_files_and_independent_starter(tmp_path):
    receipt,identity=qualification()
    project=prepare_project('fa','a3',tmp_path/'a project with spaces',receipt,identity,sys.executable,'fa-round1')
    assert project['actor_seconds']==28800 and project['id']=='fa-round1'
    assert len(project['verifier']['files'])==2
    source=Path(project['source']);namespace={};exec((source/'kernel.py').read_text(),namespace)
    with pytest.raises(NotImplementedError):namespace['run']()
    with tarfile.open(project['verifier']['files'][1]) as tar:
        contract=json.load(tar.extractfile('contract.json'))
    assert contract['qualification']==receipt and contract['identity']==identity
    command=shlex.split(project['verifier']['command'])
    assert command[:2]==[sys.executable,project['verifier']['files'][0]]
    second=prepare_project('fa','a3',tmp_path/'another',receipt,identity,sys.executable)
    assert Path(second['source'])!=source
    assert Path(project['verifier']['files'][1]).read_bytes()==Path(second['verifier']['files'][1]).read_bytes()
    bad=copy.deepcopy(receipt);bad['caseHash']='changed'
    with pytest.raises(ValueError,match='caseHash'):
        prepare_project('fa','a3',tmp_path/'bad',bad,identity)


def test_launcher_rejects_changed_archive_before_candidate_or_npu(tmp_path):
    receipt,identity=qualification();project=prepare_project('fa','a3',tmp_path/'project',receipt,identity,sys.executable)
    archive=Path(project['verifier']['files'][1]);archive.write_bytes(archive.read_bytes()+b'changed')
    result=subprocess.run(project['verifier']['command'],shell=True,cwd=project['source'],text=True,capture_output=True)
    assert result.returncode!=0 and 'Frozen verifier archive changed' in result.stderr


def test_contracts_are_generated_for_all_seven_workloads_and_targets(tmp_path):
    from ascend_tools.baselines import WORKLOADS
    for target in ('a3','a5'):
        for workload in WORKLOADS:
            receipt,identity=qualification(workload,target)
            project=prepare_project(workload,target,tmp_path/(target+'-'+workload),receipt,identity,sys.executable)
            with tarfile.open(project['verifier']['files'][1]) as tar:contract=json.load(tar.extractfile('contract.json'))
            assert contract['expected']['caseIds']==describe(workload)['caseIds']
            assert contract['target']==target and project['phase']=='phase2'
