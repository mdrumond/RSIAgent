"""Small independent mathematical checks executed before full target qualification."""
import sys
import torch
from . import file_hash
from .inputs import load
from .qualification import check, execute_case

def test_kimi_recurrence_matches_independent_numpy_final_state():
    _,m,_=load('kimi_delta_attention');torch.manual_seed(4)
    q,k,v=[torch.randn(1,3,1,2)*.15 for _ in range(3)]
    args=[q,k,v,torch.full_like(q,-.05),torch.full((1,3,1),.4),torch.ones(1,1,2,2)*.1]
    got=m.kimi_delta_attention(*args)
    ref=m.kimi_delta_attention_oracle(*[a.double() for a in args])
    assert check('kimi_delta_attention',list(got),list(ref))<1e-6
    assert torch.equal(args[-1],torch.ones_like(args[-1])*.1)


def test_lact_closed_form_update_matches_independent_autograd():
    _,m,_=load('lact_ttt_chunk');torch.manual_seed(5)
    q,k,v=[torch.randn(1,3,1,2,dtype=torch.float64)*.2 for _ in range(3)]
    weights=[torch.randn(*shape,dtype=torch.float64).requires_grad_() for shape in
             [(1,1,3,2),(1,1,2,3),(1,1,3,2)]]
    lr=torch.full((1,3,1),.01,dtype=torch.float64)
    x=(k/(k.norm(dim=-1,keepdim=True)+1e-6)).permute(0,2,1,3)
    w1,w2,w3=weights;u=torch.nn.functional.silu(x@w1.transpose(-1,-2))*(x@w3.transpose(-1,-2))
    out=u@w2.transpose(-1,-2)
    loss=-(out*v.permute(0,2,1,3)*lr.permute(0,2,1).unsqueeze(-1)).sum()
    grads=torch.autograd.grad(loss,weights)
    expected=[(w-g)/(w-g).norm(dim=-1,keepdim=True).add(1e-6) for w,g in zip(weights,grads)]
    actual=m.lact_ttt_chunk_oracle(q,k,v,*weights,lr,chunk_size=4,use_muon=False,update_first=False)
    for got,ref in zip(actual[1:],expected):torch.testing.assert_close(got,ref,atol=1e-10,rtol=1e-10)


def test_sparse_kl_gradients_match_finite_difference():
    _,m,_=load('sparse_lightning_indexer_grad_kl_loss_enhance')
    q=torch.tensor([[[.2,.3]]]);k=torch.tensor([[[.1,.5]],[[.4,.2]],[[.3,.1]]])
    qi=torch.tensor([[[.5,.8]]]);ki=torch.tensor([[[.2,.4]],[[.7,.3]],[[.1,.8]]]);w=torch.tensor([[.6]])
    got=m.sparse_lightning_indexer_grad_kl_loss_enhance(q,k,qi,ki,w,torch.tensor([[[0,1,2]]]),
        actual_seq_lengths_query=[1],actual_seq_lengths_key=[3],sparse_mode=0)
    p=torch.softmax(q.double()[0]@k.double()[:,0].T,dim=-1).flatten()
    def loss(a,b,c):
        pred=torch.softmax((torch.relu(a[0]@b[:,0].T)*c[0,:,None]).sum(0),dim=-1)
        return (p*(p.log()-pred.log())).sum()
    args=[qi.double(),ki.double(),w.double()]
    for i,arg in enumerate(args):
        numerical=torch.empty_like(arg)
        for j in range(arg.numel()):
            high=[a.clone() for a in args];low=[a.clone() for a in args]
            high[i].view(-1)[j]+=1e-5;low[i].view(-1)[j]-=1e-5
            numerical.view(-1)[j]=(loss(*high)-loss(*low))/(2e-5)
        torch.testing.assert_close(got[i].double(),numerical,atol=2e-7,rtol=2e-4)
    assert abs(got[3].item()-loss(*args).item())<1e-7


def test_gdn_chunk_state_matches_token_recurrence_with_tail():
    _,m,_=load('gdn');torch.manual_seed(7)
    q,k,v=[torch.randn(1,3,1,2)*.2 for _ in range(3)]
    gate=torch.full((1,3,1),-.05);beta=torch.full((1,3,1),.4);initial=torch.ones(1,1,2,2)*.1
    got=m.Model()(q,k,v,gate,beta,torch.zeros(1),torch.zeros(1),initial,1.,2,
                  False,False,True,True,False,False,False,False)
    state=initial.double().clone();out=[]
    for t in range(3):
        state=state*gate[:,t].double().exp()[...,None,None]
        key=k[:,t].double();value=v[:,t].double()
        residual=value-torch.einsum('bhk,bhkv->bhv',key,state)
        state=state+beta[:,t,None,None].double()*key.unsqueeze(-1)*residual.unsqueeze(-2)
        out.append(torch.einsum('bhk,bhkv->bhv',q[:,t].double(),state))
    torch.testing.assert_close(got[0].double(),torch.stack(out,dim=1),atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(got[1].double(),state,atol=1e-7,rtol=1e-6)


def test_bsa_allmasked_and_bottom_right_causal_rows():
    _,m,_=load('bsa');torch.manual_seed(8)
    q=torch.randn(2,3,2);k=torch.randn(3,1,2);v=torch.randn(3,1,2)
    got=m.Model()(q,k,v,torch.tensor([0,2]),torch.tensor([0,3]),torch.tensor([0,1,-1]),
                  torch.tensor([3,1]*3),torch.zeros(1,1,1,1,dtype=torch.bool),1.,True,True)
    assert torch.equal(got[:,1],torch.zeros_like(got[:,1]))
    for row in range(2):
        for head in (0,2):
            scores=q[row,head].double()@k[:row+2,0].double().T
            expected=torch.softmax(scores,dim=0)@v[:row+2,0].double()
            torch.testing.assert_close(got[row,head].double(),expected,atol=1e-6,rtol=1e-6)


def test_tensor_vm_reads_current_registers_and_preserves_input():
    _,m,_=load('tensor_program_vm')
    program=torch.tensor([[0,0,1,2],[6,1,0,2],[5,2,1,0],[3,0,2,0],[7,2,0,0]],dtype=torch.int32)
    registers=torch.tensor([[0.,0.],[1.,2.],[3.,4.]])
    got=m.tensor_program_vm(program,registers)
    # ADD gives[4,6]; MULADD gives[13,26]; ROLL gives[26,13]; MEAN/COPY give19.5.
    torch.testing.assert_close(got,torch.tensor([[19.5,19.5],[13.,26.],[19.5,19.5]]))
    assert torch.equal(registers,torch.tensor([[0.,0.],[1.,2.],[3.,4.]]))


def run(workload):
    if sys.flags.optimize:
        raise RuntimeError('Independent assertions require non-optimized Python')
    checks = {'kimi_delta_attention': 'test_kimi_recurrence_matches_independent_numpy_final_state', 'lact_ttt_chunk': 'test_lact_closed_form_update_matches_independent_autograd', 'sparse_lightning_indexer_grad_kl_loss_enhance': 'test_sparse_kl_gradients_match_finite_difference', 'gdn': 'test_gdn_chunk_state_matches_token_recurrence_with_tail', 'bsa': 'test_bsa_allmasked_and_bottom_right_causal_rows', 'tensor_program_vm': 'test_tensor_vm_reads_current_registers_and_preserves_input'}
    if workload == 'fa':
        if execute_case('fa',1,'cpu',warmup=0,repeats=1)['maxAbsError'] >= 1e-5:
            raise ValueError('FA independent reference mismatch')
    else:
        globals()[checks[workload]]()
    return {'workload':workload,'passed':True,'checkerHash':file_hash(__file__)}
