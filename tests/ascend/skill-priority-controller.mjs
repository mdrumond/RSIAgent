import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile,readdir,stat} from 'node:fs/promises';
import {join} from 'node:path';
import {setup,rows} from './driver_setup.mjs';

test('public controller analyzes a failed preplanned Phase 1 project before completing the campaign',{timeout:90000},async t=>{
 let instance,reviews=0;
 const f=await setup(t,'freeze',['preplanned-arithmetic'],{providerOptions:{reviewPlan:async()=>{
  const inputs=[];
  const visit=async dir=>{for(const e of await readdir(dir,{withFileTypes:true})){const path=join(dir,e.name);if(e.isDirectory())await visit(path);else if(e.name==='input.json')inputs.push({path,mtime:(await stat(path)).mtimeMs});}};
  await visit(join(instance,'skills','priority','analysis'));inputs.sort((a,b)=>b.mtime-a.mtime);
  const input=JSON.parse(await readFile(inputs[0].path,'utf8')),groups=input.groups??[];
  const plan=[{name:'read',args:{path:'input.json'}}];
  for(const group of groups)for(const path of [group.diagnosticsFile,group.linkedDiagnosticsFile].filter(Boolean))plan.push({name:'read',args:{path,limit:200}});
  const result=input.task.startsWith('Classify')?{classifications:groups.map(g=>({groupId:g.id,cause:'unknown',bucket:'correctness',reason:'The observed numeric failure requires evidence before proposing an instruction',coverage:'unknown',guidanceIds:[],evidenceRefs:[g.occurrences[0].reference]}))}:
   {action:'no-change',reason:'The fixture records the failure but supplies no grounded new instruction',groupIds:groups.map(g=>g.id)};
  reviews++;plan.push({name:'submit_review',args:{result}});return plan;
 }}});instance=f.root;
 const environment=join(f.root,'two-slots.mjs');
 await writeFile(environment,`import {createEnvironment as base} from './machinery/integrations/environments/index.mjs';
export const createEnvironment=(options,context)=>base(options,{...context,resources:async()=>[{id:'foreground',available:true,metadata:{}},{id:'background',available:true,metadata:{backgroundSlot:true}}]});\n`);
 const config=join(f.root,'priority-config.json');await writeFile(config,JSON.stringify({campaign_seconds:180,concurrency:2,recovery:false,
  environment:{module:environment,options:{mode:'local'}},skills:{activeVersion:'baseline',background:true,tuningDisabled:false,priority:{enabled:true}}}));
 await f.run('config','--config',config);await f.run('campaign','--no-recovery');
 assert(reviews>0);
 assert(rows(f.root,"SELECT id FROM skill_backlog_failures").length>0);
 assert.equal(rows(f.root,"SELECT id FROM skill_backlog_failures WHERE dirty=1").length,0);
 assert(rows(f.root,"SELECT sequence FROM events WHERE type='skill-priority-evidence-ready' AND json_extract(data,'$.phase')='phase1'").length>0);
 assert.equal(rows(f.root,"SELECT sequence FROM events WHERE type='skill-update-error'").length,0);
 f.pass();
});
