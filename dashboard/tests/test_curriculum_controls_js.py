"""Run the actual browser controller through uncertain request lifecycles."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_curriculum_reconciles_exact_requests_across_reload_and_settled_failure():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the browser controller runtime test')
    script = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = require('node:fs').readFileSync(process.argv[1], 'utf8');
const saved = new Map(); let number = 0;
const storage = {getItem:k=>saved.get(k), setItem:(k,v)=>saved.set(k,v), removeItem:k=>saved.delete(k)};
function harness(sequence, project='') {
  let submit, retry, approve, reloads=0;
  const calls=[], button={}, status={}, reconcile={}, field={value:'first prompt'};
  const approval={dataset:{curriculumApprove:'proposal',restart:'yes'},
    closest:()=>({querySelector:()=>({value:'60'})}), addEventListener:(_,fn)=>approve=fn};
  const form={querySelector:()=>button,addEventListener:(_,fn)=>submit=fn};
  const section={dataset:{curriculum:'campaign',curriculumProject:project},
    querySelector:s=>s.includes('status')?status:s.includes('reconcile')?reconcile:form,
    querySelectorAll:s=>s==='[data-curriculum-approve]'?[approval]:[button,field,approval,reconcile]};
  reconcile.addEventListener=(_,fn)=>retry=fn;
  vm.runInNewContext(source,{document:{body:{dataset:{csrf:'token'}},
    querySelectorAll:s=>s==='[data-curriculum]'?[section]:[]},sessionStorage:storage,
    crypto:{randomUUID:()=>`key-${++number}`}, confirm:()=>true,
    FormData:class{*[Symbol.iterator](){yield ['kind','request-project'];
      yield ['instructions',field.value];yield ['format_seconds','10'];}},
    location:{reload:()=>reloads++},setTimeout:fn=>fn(),fetch:async(url,options)=>{
      calls.push({url,options});assert.ok(sequence.length,'Unexpected request');
      const next=sequence.shift();if(next instanceof Error)throw next;
      return {ok:next.ok!==false,status:next.status||200,json:async()=>next.body};
    }});
  return {calls,field,reconcile,button,submit:()=>submit({preventDefault(){}}),
    retry:()=>retry(),approve:()=>approve(),reloads:()=>reloads};
}
(async()=>{
  // Lost POST response: retain its exact body/key across reload and edited fields.
  const first=harness([new Error('response lost')]);await first.submit();
  assert.equal(first.field.disabled,true);assert.equal(first.reconcile.hidden,false);
  const reloaded=harness([{body:{operation_id:'accepted'}},{body:{status:'succeeded'}}]);
  reloaded.field.value='different prompt';await reloaded.retry();
  assert.equal(first.calls[0].options.body,reloaded.calls[0].options.body);
  assert.equal(first.calls[0].options.headers['Idempotency-Key'],reloaded.calls[0].options.headers['Idempotency-Key']);
  assert.equal(reloaded.reloads(),1);assert.equal(saved.size,0);

  // Lost poll: reconciliation reads the known operation without another POST.
  const poll=harness([{body:{operation_id:'retained'}},new Error('poll lost')]);await poll.submit();
  const recovered=harness([{body:{status:'failed',error:'stale base'}},
    {body:{operation_id:'fresh'}},{body:{status:'succeeded'}}]);
  await recovered.retry();assert.equal(recovered.calls[0].url,'/api/operations/retained');
  assert.equal(recovered.field.disabled,false);recovered.field.value='review again';
  await recovered.submit();
  assert.notEqual(poll.calls[0].options.headers['Idempotency-Key'],recovered.calls[1].options.headers['Idempotency-Key']);
  assert.equal(JSON.parse(recovered.calls[1].options.body).instructions,'review again');

  // HTTP 500 is uncertain, whereas a definitive 409 permits a fresh request.
  const server=harness([{ok:false,status:500,body:{error:'gateway'}}]);await server.submit();
  assert.equal(server.field.disabled,true);
  const reject=harness([{ok:false,status:409,body:{error:'busy'}}]);await reject.retry();
  assert.equal(reject.field.disabled,false);assert.equal(saved.size,0);

  // Approvals retain their original elapsed allowance through response loss too.
  const approval=harness([new Error('lost')],'project');await approval.approve();
  const resume=harness([{body:{operation_id:'approved'}},{body:{status:'succeeded'}}],'project');
  await resume.retry();assert.equal(approval.calls[0].options.body,resume.calls[0].options.body);
  assert.equal(JSON.parse(resume.calls[0].options.body).additional_seconds,60);
  assert.equal(saved.size,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    source = Path(__file__).parents[1] / 'static' / 'curriculum_controls.js'
    subprocess.run([node, '-e', script, str(source)], check=True)
