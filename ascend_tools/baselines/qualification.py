"""Serial qualification: CPU references, actual target outputs, isolated latency samples."""
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from . import describe, digest, pack, validate_receipt
from .inputs import call, inputs, tensors
from .comparison import finite_comparison, native_reference


def tensor_hash(values):
    return digest([{'shape':list(t.shape), 'dtype':str(t.dtype), 'sha256':hashlib.sha256(
        t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()} for t in values])


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def check(workload, actual, expected, native=None):
    spec, _, _ = pack(workload)
    if len(actual) != len(expected):
        raise ValueError('Output count changed')
    if native is not None and len(native) != len(expected):
        raise ValueError('Native reference output count changed')
    maximum = 0.
    for i, (a, e) in enumerate(zip(actual, expected)):
        if a.shape != e.shape or not torch.isfinite(a).all() or not torch.isfinite(e).all():
            raise ValueError('Invalid output shape or nonfinite value')
        a, e = a.detach().cpu().double(), e.detach().cpu().double()
        error = (a-e).abs(); maximum = max(maximum, error.max().item() if error.numel() else 0.)
        dtype = str(actual[i].dtype).split('.')[-1]
        threshold = spec['thresholds'][dtype]
        if workload == 'kimi_delta_attention':
            atol, rtol = ((2e-5, 2e-4) if i == 1 else
                          {'float32':(1e-5,1e-4), 'float16':(3e-4,3e-3), 'bfloat16':(2e-3,2e-2)}[dtype])
            passed = torch.allclose(a,e,atol=atol,rtol=rtol)
        elif workload in ('fa','gdn','bsa'):
            passed = torch.allclose(a,e,atol=spec.get('atols',{}).get(dtype,threshold),rtol=threshold)
        else:
            comparison = finite_comparison(actual[i], expected[i], threshold,
                                           None if native is None else native[i])
            passed = comparison['passed']
        if not passed:
            detail = f'; comparison={comparison}' if workload not in ('fa','gdn','bsa','kimi_delta_attention') else ''
            raise ValueError(f'Numerical mismatch output {i}, max absolute error {maximum}{detail}')
    return maximum


def execute_case(workload, case_id, device, warmup=5, repeats=20):
    spec, _, cases = pack(workload)
    case = next(c for c in cases if c['case_id'] == case_id)
    module, args, kwargs = inputs(workload, case)
    expected_args = [a.double() if isinstance(a,torch.Tensor) and a.is_floating_point() else a for a in args]
    if workload in ('gdn','bsa','sparse_lightning_indexer_grad_kl_loss_enhance'):
        expected_args = args # Their original reference explicitly computes FP32; independent small checks are separate.
    reference_before=tensor_hash([a for a in args if isinstance(a,torch.Tensor)])
    expected = tensors(call(workload,module,expected_args,kwargs,oracle=workload not in ('gdn','bsa')))
    native = native_reference(workload,module,args,kwargs)
    if reference_before != tensor_hash([a for a in args if isinstance(a,torch.Tensor)]):
        raise ValueError('CPU reference mutated input')
    actual_args = [a.to(device) if isinstance(a,torch.Tensor) else a for a in args]
    before = tensor_hash([a for a in actual_args if isinstance(a,torch.Tensor)])
    synchronize = torch.npu.synchronize if device.startswith('npu') else lambda: None
    actual = tensors(call(workload,module,actual_args,kwargs)); synchronize()
    if device.startswith('npu') and any(t.device.type != 'npu' for t in actual):
        raise ValueError('Target baseline returned CPU output')
    if before != tensor_hash([a for a in actual_args if isinstance(a,torch.Tensor)]):
        raise ValueError('Baseline mutated input')
    maximum = check(workload,actual,expected,native)
    for _ in range(warmup):
        call(workload,module,actual_args,kwargs)
    synchronize(); samples=[]
    for _ in range(repeats):
        started=time.perf_counter(); call(workload,module,actual_args,kwargs); synchronize()
        samples.append(time.perf_counter()-started)
    return {'id':case_id,'passed':True,'referenceHash':tensor_hash(expected),'outputHash':tensor_hash(actual),
            'inputHash':before,'maxAbsError':maximum,'seconds':samples}


def child_owner(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start':fields[19],'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'state':fields[0]}
    except FileNotFoundError:
        return None


def _qualify(workload, target, identity, output, seconds, device='npu:0'):
    if target not in ('a3','a5') or device != 'npu:0' or not identity.get('fingerprint'):
        raise ValueError('Target NPU and frozen runtime identity required for qualification')
    import torch_npu
    observed={'torch':torch.__version__, 'torchNpu':torch_npu.__version__,
              'deviceName':torch.npu.get_device_name(0)}
    for key,value in observed.items():
        if identity.get(key) != value:
            raise ValueError('Observed runtime identity mismatch: '+key)
    if not math.isfinite(seconds) or seconds <= 0 or seconds > 7200:
        raise ValueError('Workload elapsed allocation must be at most two hours')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    expected={**describe(workload),'target':target,'runtimeFingerprint':identity['fingerprint']}
    journal=output/'qualification.json'
    state=json.loads(journal.read_text()) if journal.exists() else dict(format=1,**expected,runtime=identity,
        allocatedSeconds=seconds,elapsedSeconds=0.,cases=[],qualified=False)
    for key in (*expected, 'allocatedSeconds'):
        if state.get(key) != (seconds if key=='allocatedSeconds' else expected[key]):
            raise ValueError('Qualification inputs/allocation changed on resume')
    if state.get('qualified') is True:
        return validate_receipt(state,expected)
    if state.get('runningAt') is not None:
        state['elapsedSeconds']+=max(0.,time.time()-state.pop('runningAt'))
        if state.get('dispatching'):
            raise RuntimeError('Prior dispatch has unknown child identity; reconcile before resume')
        prior=state.get('owner'); current=child_owner(prior['pid']) if prior else None
        if current and current['state'] != 'Z' and all(current[k] == prior[k] for k in ('pid','start','boot')):
            raise RuntimeError('Prior qualification child may remain live; observe before resume')
    from .checks import run as independent_checks
    if not state.get('independentChecks'):
        started=time.monotonic();state['runningAt']=time.time();state['owner']=child_owner(os.getpid());atomic(journal,state)
        try: state['independentChecks']=independent_checks(workload)
        finally:
            state['elapsedSeconds']+=time.monotonic()-started
            state.pop('runningAt',None);state.pop('owner',None);atomic(journal,state)
    for case_id in expected['caseIds']:
        if any(c['id']==case_id and c['passed'] for c in state['cases']):
            continue
        remaining=seconds-state['elapsedSeconds']
        if remaining<=0:
            atomic(journal,state); raise TimeoutError('Qualification elapsed allocation exhausted')
        result=output/f'case-{case_id}.json'
        command=[sys.executable,'-m','ascend_tools.baselines','case',workload,str(case_id),'--device',device,'--output',str(result)]
        state['runningAt']=time.time();state['dispatching']=True;atomic(journal,state);started=time.monotonic()
        child=subprocess.Popen(command,start_new_session=True)
        state['owner']=child_owner(child.pid);state['dispatching']=False;atomic(journal,state)
        try:
            child.wait(timeout=remaining)
            if child.returncode: raise RuntimeError(f'Baseline case {case_id} failed; see retained output')
            state['cases'].append(json.loads(result.read_text()))
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            state['elapsedSeconds']+=time.monotonic()-started
            state.pop('runningAt',None);state.pop('owner',None);state.pop('dispatching',None);atomic(journal,state)
    state['timing']={'passed':True,'method':'synchronized-call-wall-latency','warmup':5,'repeats':20,
                     'samples':[{'id':c['id'],'seconds':c.pop('seconds')} for c in state['cases']]}
    state['qualified']=True;validate_receipt(state,expected);atomic(journal,state)
    return state


def qualify(workload,target,identity,output,seconds,device='npu:0'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'.qualification.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Qualification already owned; observe instead of resubmitting')
        return _qualify(workload,target,identity,output,seconds,device)
