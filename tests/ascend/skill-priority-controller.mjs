import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile,readdir,stat} from 'node:fs/promises';
import {join} from 'node:path';
import {setup,rows,until} from './driver_setup.mjs';
import {firstPromptObject} from './driver_fixture.mjs';
import {State} from '../../controller/state.mjs';
import {enqueueLearningRequest} from '../../controller/skill-review.mjs';

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
  const proof=group=>({reference:group.occurrences[0].reference,startLine:1,endLine:1});
  result.claims=result.classifications?result.classifications.map((row,i)=>({path:`classifications.${i}.reason`,kind:'hypothesis',proofs:[proof(groups[i])]})):[{path:'reason',kind:'hypothesis',proofs:[proof(groups[0])]}];
  reviews++;plan.push({name:'submit_review',args:{result}});return plan;
 }}});instance=f.root;
 const environment=join(f.root,'two-slots.mjs');
 await writeFile(environment,`import {createEnvironment as base} from './machinery/integrations/environments/index.mjs';
export const createEnvironment=(options,context)=>base(options,{...context,resources:async()=>[{id:'foreground',available:true,metadata:{}},{id:'background',available:true,metadata:{backgroundSlot:true}}]});\n`);
 const config=join(f.root,'priority-config.json');await writeFile(config,JSON.stringify({campaign_seconds:180,concurrency:2,recovery:false,
  environment:{module:environment,options:{mode:'local'}},skills:{activeVersion:'baseline',background:true,tuningDisabled:false,priority:{enabled:true}}}));
 await f.run('config','--config',config);
 const state=new State(join(f.root,'state.sqlite')),project=state.project('preplanned-arithmetic');
 const request=enqueueLearningRequest(state,{target:'cpu',question:'Observe the preplanned correctness failure',requiredEvidence:'Actual attempt outcome'});
 state.db.prepare("UPDATE skill_learning_requests SET status='claimed',draft='preplanned' WHERE id=?").run(request);
 state.db.prepare('UPDATE projects SET spec=? WHERE id=?').run(JSON.stringify({...project.spec,generation:{draft:'preplanned',learningRequest:{id:request}}}),project.id);state.close();
 const running=f.start('campaign','--no-recovery');
 try{await until(()=>rows(f.root,"SELECT 1 FROM sqlite_master WHERE name='skill_backlog_failures'").length>0&&rows(f.root,"SELECT id FROM skill_backlog_failures").length>0&&rows(f.root,"SELECT id FROM skill_backlog_failures WHERE dirty=1").length===0&&rows(f.root,"SELECT sequence FROM events WHERE type='skill-priority-evidence-ready' AND json_extract(data,'$.phase')='phase1'").length>0,30000);}finally{await f.run('stop');await running.output;}
 const learning=rows(f.root,'SELECT status,data FROM skill_learning_requests')[0];assert.equal(learning.status,'inconclusive');
 assert(rows(f.root,'SELECT id FROM attempts').some(a=>a.id===JSON.parse(learning.data).findings.attemptId));
 assert(reviews>0);
 assert(rows(f.root,"SELECT id FROM skill_backlog_failures").length>0);
 assert.equal(rows(f.root,"SELECT id FROM skill_backlog_failures WHERE dirty=1").length,0);
 assert(rows(f.root,"SELECT sequence FROM events WHERE type='skill-priority-evidence-ready' AND json_extract(data,'$.phase')='phase1'").length>0);
 assert.equal(rows(f.root,"SELECT sequence FROM events WHERE type='skill-update-error'").length,0);
 f.pass();
});

test('summary fixture decodes the first nested JSON context before trailing output examples',()=>{
 const context={events:[{data:{text:'A quoted brace } and escaped quote " remain data'}}]},prompt='Inspect the native session. '+JSON.stringify(context)+'\nReturn '+JSON.stringify({findings:'different result'});
 assert.deepEqual(firstPromptObject(prompt),context);assert.deepEqual(firstPromptObject('Project facts '+JSON.stringify({status:'failed'})+'\nExample {findings, paths}\n'+prompt,value=>Array.isArray(value.events)),context);assert.throws(()=>firstPromptObject('Inspect {"events":['),/no complete JSON/);
});
