import copy
import sys
import json
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
from ascend_tools.baselines.inputs import load
from ascend_tools.baselines.project_verifier import evaluate,load_candidate,output_check,validate_contract
from test_baseline_projects import qualification
from ascend_tools.baselines.projects import prepare_project


def tiny_inputs(workload,case):
    _,module,_=load('fa');torch.manual_seed(case['case_id'])
    return module,[torch.randn(1,1,3,2)*.2 for _ in range(3)],{'causal':case['causal']}


def correct(q,k,v,causal=False):
    return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)


def test_full_frozen_case_coverage_and_wrong_clone_rejection():
    result=evaluate('fa',correct,build=tiny_inputs,warmup=0,repeats=1)
    assert result['passed'] and len(result['cases'])==20 and result['targetEvidence'] is False
    assert [r['id'] for r in result['cases']]==list(range(1,21))
    subset=evaluate('fa',correct,case_ids=[1],build=tiny_inputs)
    assert not subset['passed'] and not subset['complete']
    with pytest.raises(ValueError,match='Numerical mismatch'):
        evaluate('fa',lambda q,k,v,**attrs:q.clone(),build=tiny_inputs)
    with pytest.raises(ValueError,match='mutated input'):
        def mutation(q,k,v,**attrs):
            q.add_(.01);return correct(q,k,v,**attrs)
        evaluate('fa',mutation,build=tiny_inputs)


def test_fresh_paired_performance_controls_and_slowdown_rejection():
    for candidate_time,expected in [(1.,True),(4.,False)]:
        calls=[]
        def measure(label,operation):
            calls.append(label);return operation(),2. if label=='baseline' else candidate_time
        result=evaluate('fa',correct,build=tiny_inputs,performance=True,measure=measure,warmup=0,repeats=2)
        assert result['passed'] is expected and result['geometricMeanSpeedup']==pytest.approx(2./candidate_time)
        assert len(result['cases'])==20 and calls[:4]==['baseline','candidate','candidate','baseline']
    with pytest.raises(ValueError,match='timing'):
        evaluate('fa',correct,build=tiny_inputs,performance=True,measure=lambda label,op:(op(),float('nan')),warmup=0,repeats=1)


def test_state_dtype_and_target_placement_contracts():
    q=torch.zeros(1,1,1,2);state=torch.zeros(1,1,2,2)
    with pytest.raises(ValueError,match='structure'):
        output_check('kimi_delta_attention',q,(q,state),[q],'cpu')
    with pytest.raises(ValueError,match='dtype'):
        output_check('fa',q.double(),q,[q],'cpu')
    with pytest.raises(ValueError,match='wrong device'):
        output_check('fa',q,q,[q],'npu:0')


def test_candidate_import_ignores_stale_bytecode(tmp_path):
    source=tmp_path/'kernel.py';source.write_text('def run(): return 1\n')
    one,h1=load_candidate(source);source.write_text('def run(): return 2\n');two,h2=load_candidate(source)
    assert one()==1 and two()==2 and h1!=h2


def test_contract_changes_and_production_cpu_fallback_are_rejected(tmp_path,monkeypatch):
    import sys,types,tarfile
    receipt,identity=qualification();project=prepare_project('fa','a3',tmp_path/'p',receipt,identity)
    with tarfile.open(project['verifier']['files'][1]) as tar:contract=json.load(tar.extractfile('contract.json'))
    validate_contract(contract)
    bad=copy.deepcopy(contract);bad['qualification']['runtime']['fingerprint']='other'
    with pytest.raises(ValueError,match='runtime'):validate_contract(bad)
    bad=copy.deepcopy(contract);bad['expected']['caseHash']='other'
    with pytest.raises(ValueError,match='changed'):validate_contract(bad)
    from ascend_tools.baselines.project_verifier import verify
    monkeypatch.setitem(sys.modules,'torch_npu',types.SimpleNamespace(__version__='fixture'))
    monkeypatch.setattr(torch,'npu',types.SimpleNamespace(get_device_name=lambda i:'wrong device'),raising=False)
    with pytest.raises(ValueError,match='runtime differs'):
        verify(contract,Path(project['source'])/'kernel.py',tmp_path/'evidence','correctness')


def test_frozen_launcher_executes_without_source_checkout_dependency(tmp_path):
    import subprocess,shlex,shutil
    receipt,identity=qualification();project=prepare_project('fa','a3',tmp_path/'original space',receipt,identity)
    relocated=tmp_path/'relocated space';shutil.copytree(Path(project['source']).parent,relocated)
    command=[sys.executable,str(relocated/'frozen/verify.py'),'--candidate',str(relocated/'source/kernel.py'),
             '--mode','correctness','--artifacts',str(tmp_path/'evidence')]
    result=subprocess.run(command,cwd=relocated/'source',text=True,capture_output=True)
    assert result.returncode==1
    report=json.loads(result.stdout)
    assert report['passed'] is False and report['targetEvidence'] is False
    assert Path(report['evidence']).is_file()


@pytest.mark.parametrize('change_source',[False,True])
def test_failure_retains_partial_case_evidence_and_source_identity(tmp_path,monkeypatch,change_source):
    import types,tarfile
    import ascend_tools.baselines.project_verifier as verifier
    receipt,identity=qualification();identity.update(torch=torch.__version__)
    project=prepare_project('fa','a3',tmp_path/'p',receipt,identity)
    with tarfile.open(project['verifier']['files'][1]) as tar:contract=json.load(tar.extractfile('contract.json'))
    monkeypatch.setitem(sys.modules,'torch_npu',types.SimpleNamespace(__version__='fixture'))
    monkeypatch.setattr(torch,'npu',types.SimpleNamespace(get_device_name=lambda i:'fixture'),raising=False)
    candidate=Path(project['source'])/'kernel.py'
    def fail(*args,progress,**kwargs):
        progress({'cases':[{'id':1,'passed':True}],'runningCase':2})
        if change_source:
            candidate.write_text('def run(*args,**attrs): return None\n')
            return {'passed':True,'cases':[]}
        raise ValueError('Numerical mismatch in case 2')
    monkeypatch.setattr(verifier,'evaluate',fail)
    result=verifier.verify(contract,candidate,tmp_path/'evidence','correctness')
    assert result['passed'] is False and result['cases']==[{'id':1,'passed':True}]
    assert result['runningCase']==2 and result['languageAttested'] is False
    assert result['verifierHash']==contract['expected']['verifierHash']
    from ascend_tools.baselines import digest
    assert result['qualificationHash']==digest(contract['qualification'])
    assert json.loads(Path(result['evidence']).read_text())['candidateHash']==result['candidateHash']
