import copy
import json
import math
from pathlib import Path

import pytest
torch=pytest.importorskip('torch')
from ascend_tools.baselines import WORKLOADS, describe, digest, pack, validate_receipt
from ascend_tools.baselines.inputs import call, inputs, load, tensors
from ascend_tools.baselines.qualification import check, execute_case




def test_fa_real_execution_and_complete_state_comparison():
    result=execute_case('fa',1,'cpu',warmup=0,repeats=2)
    assert result['passed'] and len(result['seconds'])==2 and result['maxAbsError']<1e-5
    with pytest.raises(ValueError,match='count'):
        check('fa',[torch.zeros(1)],[torch.zeros(1),torch.zeros(1)])
    with pytest.raises(ValueError,match='nonfinite'):
        check('fa',[torch.tensor([float('nan')])],[torch.ones(1)])
















def test_elapsed_expiry_closes_child_and_preserves_spent_allocation(tmp_path,monkeypatch):
    import sys,types,subprocess
    import ascend_tools.baselines.qualification as q
    monkeypatch.setitem(sys.modules,'torch_npu',types.SimpleNamespace(__version__='fixture'))
    monkeypatch.setattr(torch,'npu',types.SimpleNamespace(get_device_name=lambda i:'fixture'),raising=False)
    identity={'fingerprint':'runtime','torch':torch.__version__,'torchNpu':'fixture','deviceName':'fixture'}
    popen=subprocess.Popen
    monkeypatch.setattr(q.subprocess,'Popen',lambda *a,**kw:popen([sys.executable,'-c','import time;time.sleep(30)'],**kw))
    with pytest.raises(subprocess.TimeoutExpired):q.qualify('fa','a3',identity,tmp_path,.05)
    journal=json.loads((tmp_path/'qualification.json').read_text())
    assert journal['elapsedSeconds']>=.05 and not journal['qualified']
    assert 'runningAt' not in journal and 'owner' not in journal
    with pytest.raises(TimeoutError,match='exhausted'):q.qualify('fa','a3',identity,tmp_path,.05)


def test_supplied_generators_construct_one_cpu_case_with_original_seed():
    from torch.nn.functional import normalize
    _,_,cases=pack('gdn');module,args,_=inputs('gdn',cases[2])
    expected=normalize(torch.randn(1,128,1,8,generator=torch.Generator().manual_seed(58)),dim=-1,eps=1e-6)
    torch.testing.assert_close(args[0],expected)
    assert all(a.device.type=='cpu' for a in args if isinstance(a,torch.Tensor))
    _,_,cases=pack('bsa');_,first,_=inputs('bsa',cases[0]);_,second,_=inputs('bsa',cases[0])
    assert first[0].shape==(8,3,64) and torch.equal(first[0],second[0])


@pytest.mark.parametrize('workload',WORKLOADS)
def test_independent_math_checks_execute_for_every_workload(workload):
    from ascend_tools.baselines.checks import run
    assert run(workload)['passed'] is True


@pytest.mark.parametrize('workload',WORKLOADS[3:])
def test_all_public_case_attributes_generate_legal_small_inputs(workload):
    # Exercise every public dtype, gate, layout and attribute combination without
    # pretending reduced construction probes are full-case NPU qualification.
    _,_,cases=pack(workload)
    for original in cases:
        case=copy.deepcopy(original)
        if workload=='kimi_delta_attention':
            case['input_shape']=[[1,3,1,2]]*4+[[1,3,1],[1,1,2,2]]
        elif workload=='lact_ttt_chunk':
            case['input_shape']=[[1,3,1,2]]*3+[[1,1,3,2],[1,1,2,3],[1,1,3,2],[1,3,1]]
        elif workload=='tensor_program_vm':
            case['input_shape']=[[16,4],[4,8]]
        else:
            bsnd=case['attrs']['layout']=='BSND'
            shapes=[[3,2,4],[3,1,4],[3,2,2],[3,1,2],[3,2],[3,1,4]]
            case['input_shape']=[([1]+s if bsnd else s) for s in shapes]+[[1,3,2]]*2+[
                [1,3,2,2] if bsnd else [3,2,2],[1,3,1,2] if bsnd else [3,1,2]]
            case['attrs'].update(actual_seq_lengths_query=[3],actual_seq_lengths_key=[3])
        module,args,kwargs=inputs(workload,case)
        assert all(torch.isfinite(a).all() for a in args if isinstance(a,torch.Tensor) and a.is_floating_point())
        assert all(torch.isfinite(a).all() for a in tensors(call(workload,module,args,kwargs)))




@pytest.mark.parametrize('case_id',[5,7])
def test_extended_fa_cases_execute_low_precision_and_batch_heads(case_id):
    result=execute_case('fa',case_id,'cpu',warmup=0,repeats=1)
    assert result['passed'] and result['maxAbsError']<.002


def test_qualification_refuses_second_live_controller(tmp_path):
    import fcntl
    from ascend_tools.baselines.qualification import qualify
    with (tmp_path/'.qualification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='already owned'):
            qualify('fa','a3',{},tmp_path,1)
