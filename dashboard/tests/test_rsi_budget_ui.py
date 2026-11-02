"""Execute the real ledger renderer with effective, unlimited and unknown budgets."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_budget_cells_distinguish_effective_ceiling_unlimited_and_unknown():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser runtime test')
    script = r'''
const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
class Element {
  constructor(tag='div') { this.tag=tag;this.children=[];this.handlers={};this.dataset={};this.value=''; }
  append(child){this.children.push(child);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,fn){this.handlers[name]=fn;}
  querySelector(){return new Element();}
}
const nodes=new Map();
const get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
get('rsi-execution').dataset.enabled='false';
get('phase-form').elements={operation:new Element()};
const projects=[
 {project_id:'finite',operation_id:'one',actor_seconds:3600,actor_allocation:3593,
  remaining_actor_seconds:0,wave_actor_ceiling:6300},
 {project_id:'unlimited',operation_id:'two',actor_seconds:5,actor_allocation:null,
  remaining_actor_seconds:null},
 {project_id:'not-started',operation_id:'three',actor_seconds:null}
];
const detail={inspection:{available:true,waves:[{wave_id:'wave',projects}],tasks:[{
 task_id:'runner',actor_allocation:3593,actor_union_seconds:3600,remaining:{actor_seconds:0}}]}};
const calls=[];
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{
 document:{getElementById:get,createElement:tag=>new Element(tag),body:{dataset:{}}},
 Option:class extends Element {constructor(text,value){super('option');this.textContent=text;this.value=value;}},
 fetch:async url=>{calls.push(url);return {ok:true,json:async()=>url==='/api/rsi/capabilities'?{}:
  url==='/api/rsi/jobs'?{jobs:[{job_id:'job'}]}:detail};}
});
(async()=>{
 for(let i=0;i<5;i++)await new Promise(setImmediate);
 await get('jobs-list').children[0].handlers.click();
 const ledger=get('time-ledger'), table=ledger.children.find(n=>n.tag==='table');
 const rows=table.children.slice(1).map(row=>row.children.map(n=>n.textContent));
 assert.deepEqual(rows[0].slice(3,7),['3600 seconds','3593 seconds','0 seconds','6300 seconds']);
 assert.deepEqual(rows[1].slice(3,7),['5 seconds','unlimited','unlimited','unknown']);
 assert.deepEqual(rows[2].slice(3,7),['unknown','unknown','unknown','unknown']);
 assert.equal(calls.filter(url=>url==='/api/rsi/jobs/job').length,1);
 assert.equal(get('execution-status').textContent,undefined);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    source = Path(__file__).parents[1]/'static/rsi_execution.js'
    subprocess.run([node, '-e', script, str(source)], check=True)
