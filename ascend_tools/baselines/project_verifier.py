"""Trusted full-case candidate correctness and paired performance verification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import types
import uuid
import torch
from . import describe, digest, file_hash, pack, validate_receipt
from .inputs import call, inputs, tensors
from .qualification import atomic, check, tensor_hash
from .comparison import native_reference


def load_candidate(path):
    path=Path(path).resolve();data=path.read_bytes()
    module=types.ModuleType('candidate_'+uuid.uuid4().hex);module.__file__=str(path)
    sys.path.insert(0,str(path.parent));sys.modules[module.__name__]=module
    exec(compile(data,str(path),'exec'),module.__dict__)
    if not callable(getattr(module,'run',None)):
        raise ValueError('Candidate must export run(*args, **attrs)')
    return module.run,hashlib.sha256(data).hexdigest()


def structure(value):
    if value is None:return None
    if isinstance(value,torch.Tensor):return 'tensor'
    if isinstance(value,(list,tuple)):return [structure(x) for x in value]
    raise ValueError('Output must contain only tensors and optional None states')


def output_check(workload,result,reference,args,device,native=None):
    if structure(result)!=structure(reference):raise ValueError('Output/state structure differs')
    actual=tensors(result);expected=tensors(reference)
    if any(t.device.type!=device.split(':')[0] for t in actual):raise ValueError('Candidate returned output on the wrong device')
    if workload=='gdn':dtypes=[args[2].dtype]+([torch.float32] if reference[1] is not None else [])
    elif workload=='kimi_delta_attention':dtypes=[args[0].dtype,torch.float32]
    elif workload=='tensor_program_vm':dtypes=[args[1].dtype]
    elif workload=='sparse_lightning_indexer_grad_kl_loss_enhance':dtypes=[torch.float32]*4
    else:dtypes=[args[0].dtype]*len(actual)
    if [a.dtype for a in actual]!=dtypes:raise ValueError('Candidate output dtype differs from contract')
    return check(workload,actual,expected,native)


def evaluate(workload,candidate,*,device='cpu',performance=False,case_ids=None,build=inputs,
             measure=None,warmup=5,repeats=20,progress=None):
    _,_,cases=pack(workload);all_ids=[c['case_id'] for c in cases]
    selected=all_ids if case_ids is None else case_ids
    if len(set(selected))!=len(selected) or any(i not in all_ids for i in selected):raise ValueError('Invalid case selection')
    sync=torch.npu.synchronize if device.startswith('npu') else lambda:None
    def measured(_label,operation):
        sync();start=time.perf_counter();result=operation();sync();return result,time.perf_counter()-start
    measure=measure or measured;rows=[]
    for case in cases:
        if case['case_id'] not in selected:continue
        if progress:progress({'cases':rows,'runningCase':case['case_id']})
        module,args,kwargs=build(workload,case)
        # Qualification already binds the original CPU reference and independent checks.
        original=tensor_hash([a for a in args if isinstance(a,torch.Tensor)])
        ref_args=[a.double() if isinstance(a,torch.Tensor) and a.is_floating_point() else a for a in args]
        if workload in ('gdn','bsa','sparse_lightning_indexer_grad_kl_loss_enhance'):ref_args=args
        reference=call(workload,module,ref_args,kwargs,oracle=workload not in ('gdn','bsa'))
        native=native_reference(workload,module,args,kwargs)
        if original!=tensor_hash([a for a in args if isinstance(a,torch.Tensor)]):raise ValueError('Reference mutated input')
        placed=[a.to(device) if isinstance(a,torch.Tensor) else a for a in args]
        before=tensor_hash([a for a in placed if isinstance(a,torch.Tensor)])
        operation=lambda:candidate(*placed,**kwargs)
        result=operation();sync()
        if before!=tensor_hash([a for a in placed if isinstance(a,torch.Tensor)]):raise ValueError('Candidate mutated input')
        error=output_check(workload,result,reference,placed,device,native)
        baseline=lambda:call(workload,module,placed,kwargs)
        row={'id':case['case_id'],'passed':True,'maxAbsError':error,'inputHash':before,
             'referenceHash':tensor_hash(tensors(reference)),'outputHash':tensor_hash(tensors(result))}
        if performance:
            for _ in range(warmup):baseline();operation()
            baseline_times=[];candidate_times=[]
            for repetition in range(repeats):
                order=[('baseline',baseline),('candidate',operation)]
                if repetition%2:order.reverse()
                for label,function in order:
                    output,duration=measure(label,function)
                    if not math.isfinite(duration) or duration<=0:raise ValueError('Nonpositive or nonfinite timing')
                    (baseline_times if label=='baseline' else candidate_times).append(duration)
                    if label=='candidate':output_check(workload,output,reference,placed,device,native)
            row.update(baselineSeconds=baseline_times,candidateSeconds=candidate_times,
                       speedup=statistics.median(baseline_times)/statistics.median(candidate_times))
        if before!=tensor_hash([a for a in placed if isinstance(a,torch.Tensor)]):raise ValueError('Candidate or baseline mutated input')
        rows.append(row)
        if progress:progress({'cases':rows,'runningCase':None})
    speedup=math.exp(sum(math.log(r['speedup']) for r in rows)/len(rows)) if performance and rows else None
    complete=sorted(selected)==sorted(all_ids)
    return {'passed':complete and (not performance or speedup>1),'complete':complete,'cases':rows,
            'geometricMeanSpeedup':speedup,'targetEvidence':device.startswith('npu'),
            'timingMethod':'paired-synchronized-call-wall-latency' if performance else None}


def validate_contract(contract):
    workload=contract['workload'];expected={**describe(workload),'target':contract['target'],
                                         'runtimeFingerprint':contract['identity']['fingerprint']}
    if contract.get('format')!=1 or contract.get('expected')!=expected:raise ValueError('Frozen project contract changed')
    language={'a3':'Triton-Ascend','a5':'Catlass DSL'}[contract['target']]
    if contract.get('language')!=language:raise ValueError('Target language contract changed')
    validate_receipt(contract['qualification'],expected)
    if contract['qualification']['runtime']!=contract['identity']:raise ValueError('Qualification runtime differs')
    if contract.get('warmup')!=5 or contract.get('repeats')!=20:raise ValueError('Timing protocol changed')
    return expected


def verify(contract,candidate,artifacts,mode):
    validate_contract(contract)
    if contract['identity'].get('torch')!=torch.__version__:raise ValueError('Observed NPU runtime differs')
    import torch_npu
    observed={'torch':torch.__version__,'torchNpu':torch_npu.__version__,'deviceName':torch.npu.get_device_name(0)}
    if any(contract['identity'].get(k)!=v for k,v in observed.items()):raise ValueError('Observed NPU runtime differs')
    function,source_hash=load_candidate(candidate)
    directory=Path(artifacts)/uuid.uuid4().hex;directory.mkdir(parents=True,exist_ok=False)
    common=dict(candidateHash=source_hash,workload=contract['workload'],target=contract['target'],
                sourceHash=contract['expected']['sourceHash'],caseHash=contract['expected']['caseHash'],
                verifierHash=contract['expected']['verifierHash'],qualificationHash=digest(contract['qualification']),
                runtimeFingerprint=contract['identity']['fingerprint'],language=contract['language'],languageAttested=False)
    partial={}
    def progress(value):
        partial.update(value);atomic(directory/'result.json',dict(common,passed=False,complete=False,**value))
    try:
        verdict=evaluate(contract['workload'],function,device='npu:0',performance=mode=='performance',
                         warmup=contract['warmup'],repeats=contract['repeats'],progress=progress)
        if file_hash(candidate)!=source_hash:raise ValueError('Candidate source changed during verification')
    except Exception as error:
        verdict=dict(partial,passed=False,complete=False,targetEvidence=bool(partial.get('cases')),
                     error=f'{type(error).__name__}: {error}')
    verdict.update(common);atomic(directory/'result.json',verdict);verdict['evidence']=str(directory/'result.json')
    return verdict


def main(contract_path):
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True)
    parser.add_argument('--artifacts',required=True);parser.add_argument('--mode',choices=['correctness','performance'],required=True)
    args=parser.parse_args()
    try:result=verify(json.loads(Path(contract_path).read_text()),args.candidate,args.artifacts,args.mode)
    except Exception as error:
        result={'passed':False,'targetEvidence':False,'error':f'{type(error).__name__}: {error}'}
        directory=Path(args.artifacts)/uuid.uuid4().hex;directory.mkdir(parents=True,exist_ok=False)
        atomic(directory/'result.json',result);result['evidence']=str(directory/'result.json')
    print(json.dumps(result,allow_nan=False))
    return 0 if result['passed'] else 1
