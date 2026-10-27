import shutil
import subprocess
from pathlib import Path

import pytest


def test_uncertain_post_retries_original_feedback_identity_after_reload():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed for the actual frontend behavior test')
    script = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = require('node:fs').readFileSync(process.argv[1], 'utf8');
const retained = new Map(), posts = [];
let counter = 0;
function mount(text) {
  let submit;
  const form = {elements:{text:{value:text},additional_seconds:{value:'15'}},
    querySelector(){return {disabled:false}},
    addEventListener(name, handler){submit=handler}};
  const section = {dataset:{actorFeedback:'/api/campaigns/campaign/projects/one/feedback'},
    querySelector(selector){if(selector==='form')return form;
      if(selector==='[data-feedback-receipts]')return {replaceChildren(){}};
      return {textContent:''};}};
  vm.runInNewContext(source, {
    document:{querySelectorAll(){return [section]},createElement(){return {}}},
    location:{pathname:'/campaign/campaign/project/one'},
    sessionStorage:{getItem:key=>retained.get(key)||null,
      setItem:(key,value)=>retained.set(key,value),removeItem:key=>retained.delete(key)},
    crypto:{randomUUID:()=>String(++counter)},setInterval(){},
    fetch:async (url,options)=>{
      if(options?.method==='POST') {
        posts.push({key:options.headers['Idempotency-Key'],body:JSON.parse(options.body)});
        if(posts.length===1)throw new Error('response lost after acceptance');
        return {ok:true,status:202,json:async()=>({request:{id:'retained-request'}})};
      }
      if(url==='/api/csrf')return {ok:true,json:async()=>({token:'csrf'})};
      if(url.includes('manual-requests'))return {ok:true,json:async()=>({status:'succeeded'})};
      return {ok:true,json:async()=>({messages:[]})};
    }
  });
  return ()=>submit({preventDefault(){}});
}
(async()=>{
  await mount('original feedback')();
  assert.equal(retained.size,1);
  const retry=mount('deliberately edited after reload');
  await retry();
  assert.deepEqual(posts[1],posts[0]);
  assert.equal(new Set(posts.map(row=>row.key)).size,1);
  assert.equal(retained.size,0);
  await retry();
  assert.notEqual(posts[2].key,posts[0].key);
  assert.equal(posts[2].body.text,'deliberately edited after reload');
})().catch(error=>{console.error(error);process.exit(1)});
'''
    source = Path(__file__).parents[1] / 'static/actor_controls.js'
    subprocess.run([node, '-e', script, str(source)], check=True)
