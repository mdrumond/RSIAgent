"""Execute the browser controller against network and operation lifecycles."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_memory_form_retries_settled_failures_and_reconciles_uncertain_requests():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the browser controller runtime test')
    script = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = require('node:fs').readFileSync(process.argv[1], 'utf8');
function harness(sequence) {
  let submit, number = 0, reloads = 0;
  const calls = [], reason = {value:'original reason', disabled:false}, button = {}, output = {};
  const form = {dataset:{project:'p',forceMemory:'/force',csrf:'token'}, elements:{reason},
    querySelector: selector => selector === 'button' ? button : output,
    addEventListener: (name, callback) => { submit = callback; }};
  vm.runInNewContext(source, {document:{querySelectorAll:()=>[form]}, confirm:()=>true,
    crypto:{randomUUID:()=>`request-${++number}`}, location:{reload:()=>reloads++},
    setTimeout:callback=>callback(), fetch:async (url, options) => {
      calls.push({url, options});
      assert.ok(sequence.length, 'Unexpected network request');
      const next = sequence.shift();
      if (next instanceof Error) throw next;
      return {ok:next.ok !== false, status:next.status || 200, json:async()=>next.body};
    }});
  return {calls, reason, button, submit:()=>submit({preventDefault(){}}), reloads:()=>reloads};
}
(async()=>{
  // A settled backend failure permits a fresh request and edited reason.
  const settled = harness([{body:{operation_id:'first'}},
    {body:{status:'failed',error:'stale base'}}, {body:{operation_id:'second'}},
    {body:{status:'succeeded',result:{admission:{version:'accepted'}}}}]);
  await settled.submit();
  assert.equal(settled.reason.disabled, false);
  settled.reason.value = 'reconciled reason';
  await settled.submit();
  const posts = settled.calls.filter(call=>call.options?.method === 'POST');
  assert.notEqual(posts[0].options.headers['Idempotency-Key'], posts[1].options.headers['Idempotency-Key']);
  assert.equal(JSON.parse(posts[1].options.body).reason, 'reconciled reason');
  assert.equal(settled.reloads(), 1);

  // A lost POST response keeps its exact key and payload even if a script edits the field.
  const uncertain = harness([new Error('connection lost'), {body:{operation_id:'retained'}},
    {body:{status:'succeeded',result:{admission:{version:'accepted'}}}}]);
  await uncertain.submit();
  assert.equal(uncertain.reason.disabled, true);
  uncertain.reason.value = 'changed while uncertain';
  await uncertain.submit();
  assert.equal(uncertain.calls[0].options.headers['Idempotency-Key'], uncertain.calls[1].options.headers['Idempotency-Key']);
  assert.equal(uncertain.calls[0].options.body, uncertain.calls[1].options.body);
  assert.equal(uncertain.reloads(), 1);

  // Once an operation ID is known, reconcile it rather than submitting another POST.
  const polling = harness([{body:{operation_id:'retained'}}, new Error('poll disconnected'),
    {body:{status:'failed',error:'index unavailable'}}, {body:{operation_id:'fresh'}},
    {body:{status:'succeeded',result:{admission:{version:'accepted'}}}}]);
  await polling.submit();
  assert.equal(polling.reason.disabled, true);
  await polling.submit();
  assert.equal(polling.calls.length, 3);
  assert.equal(polling.calls[2].url, '/api/operations/retained');
  assert.equal(polling.reason.disabled, false);
  polling.reason.value = 'retry after recovery';
  await polling.submit();
  assert.notEqual(polling.calls[0].options.headers['Idempotency-Key'], polling.calls[3].options.headers['Idempotency-Key']);
  assert.equal(JSON.parse(polling.calls[3].options.body).reason, 'retry after recovery');
  assert.equal(polling.reloads(), 1);

  // A synchronous validation rejection has no pending operation to reconcile.
  const rejected = harness([{ok:false,status:409,body:{error:'another operation active'}},
    {body:{operation_id:'fresh'}}, {body:{status:'succeeded',result:{admission:{version:'accepted'}}}}]);
  await rejected.submit();
  assert.equal(rejected.reason.disabled, false);
  await rejected.submit();
  assert.notEqual(rejected.calls[0].options.headers['Idempotency-Key'], rejected.calls[1].options.headers['Idempotency-Key']);
  assert.equal(rejected.reloads(), 1);
})().catch(error=>{ console.error(error); process.exitCode=1; });
'''
    source = Path(__file__).parents[1] / 'static' / 'memory_controls.js'
    subprocess.run([node, '-e', script, str(source)], check=True)
