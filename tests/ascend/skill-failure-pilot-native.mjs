import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {runFailurePilot} from '../../cli/skill-failure-pilot.mjs';
import {nativeSample,check} from '../../cli/skill-microeval.mjs';
import {scriptedReview} from './helpers/scripted-review.mjs';
const hash=text=>createHash('sha256').update(text).digest('hex');
const code=value=>'def answer(): return '+value+'\n';
const task=id=>({id,caseIds:['answer','callable'],controlCaseIds:['callable'],files:{'kernel.py':code(0)}});
const candidates=['principle-a','principle-b','principle-c'].map((id,i)=>({id,text:'Independent broad principle '+i,evidence:[{reference:'retained-failure',startLine:1,endLine:1}]}));

test('native Pi parallel pilot executes independent CPU oracles across45 fresh sessions', {timeout:120000},async t=>{
 const directory=await mkdtemp(join(tmpdir(),'native-failure-pilot-'));t.after(()=>rm(directory,{recursive:true,force:true}));
 const privateDirectory=join(directory,'private');await mkdir(privateDirectory);const oracle=join(privateDirectory,'oracle.py');
 const trusted="import sys,json,runpy\nfrom pathlib import Path\nm=runpy.run_path(str(Path(sys.argv[1])/'kernel.py'))\nf=m['answer']; actual=f(); rows=[{'id':'answer','correct':actual==sum(6 for _ in range(7))},{'id':'callable','correct':callable(f)}]\nprint(json.dumps({'caseResults':rows}))\nsys.exit(0 if all(r['correct'] for r in rows) else 1)\n";
 await writeFile(oracle,trusted);let blockedProbe=false;
 const provider=await scriptedReview(t,directory,body=>{
  const first=body.messages.find(m=>m.role==='user').content,text=typeof first==='string'?first:first.map(c=>c.text??'').join('\n');
  assert(!text.includes('PRIVATE-TRANSFER'));assert(!text.includes(oracle));
  const marker=JSON.parse(text.match(/PILOT_FIXTURE=(\{[^\n]+\})/)[1]);
  const calls=body.messages.flatMap(m=>m.tool_calls??[]).map(c=>({name:c.function.name,args:JSON.parse(c.function.arguments)}));
  if(marker.phase==='screen'&&marker.repetition===0&&!calls.some(c=>c.args.path===oracle))return {name:'write',args:{path:oracle,content:'corrupt oracle'}};
  if(calls.some(c=>c.args.path===oracle)&&!calls.some(c=>c.args.path==='kernel.py')){assert(body.messages.some(m=>m.role==='tool'));blockedProbe=true;}
  if(!calls.some(c=>c.args.path==='kernel.py')){
   const correct=marker.arm==='baseline'?marker.repetition<2:marker.arm==='principle-a'?true:marker.arm==='principle-b'?marker.repetition<3:marker.repetition<4;
   return {name:'write',args:{path:'kernel.py',content:code(correct?42:0)}};
  }
  if(!calls.some(c=>c.name==='bash'))return {name:'bash',args:{command:'python public_check.py'}};
  return null;
 });
 const python=process.env.DRIVER_PYTHON??'python3',sessions=new Set(),verified=[];let active=0,peak=0;
 const manifest={baseline:'Current guidance',identities:{model:provider.profile.metadata,runtime:{target:'cpu'},memory:'frozen-test-snapshot'},families:[{id:'arithmetic',development:task('development'),evidence:[{reference:'retained-failure'}],transfer:[{...task('transfer-a'),private:'PRIVATE-TRANSFER-A'},{...task('transfer-b'),private:'PRIVATE-TRANSFER-B'}]}]};
 const report=await runFailurePilot({directory:join(directory,'pilot'),manifest,seconds:100,trialSeconds:15,concurrency:4},{
  qualify:async()=>{
   const rows=[];
   for(const [id,value]of [['reference',42],['negative',0]]){const cwd=join(privateDirectory,id);await mkdir(cwd);await writeFile(join(cwd,'kernel.py'),code(value));rows.push(await check(python,oracle,cwd,10));}
   return {qualified:rows[0].correct&&!rows[1].correct,referencePassed:rows[0].correct,negativeFailed:!rows[1].correct};
  },
  design:async input=>{assert(!JSON.stringify(input).includes('PRIVATE-TRANSFER'));assert.equal(input.baselineResults.filter(r=>!r.correct&&r.mechanismMatch).length,3);return candidates;},
  trial:async input=>{
   active++;peak=Math.max(peak,active);
   try{
    const cwd=join(input.directory,'public');await mkdir(cwd);await writeFile(join(cwd,'kernel.py'),input.task.files['kernel.py']);
    await writeFile(join(cwd,'public_check.py'),"from kernel import answer\nassert answer()==sum(6 for _ in range(7))\n");
    const marker=JSON.stringify({phase:input.phase,arm:input.arm,repetition:input.repetition});
    const actor=await nativeSample({cwd,guidance:input.guidance,prompt:'Implement answer() returning six times seven. Use public_check.py for feedback.\nPILOT_FIXTURE='+marker,profile:provider.profile,seconds:Math.min(10,input.remainingSeconds),signal:input.signal,sessionDirectory:join(input.directory,'session')});
    assert.equal(actor.error,null);assert.equal(actor.timedOut,false);assert(!sessions.has(actor.session));sessions.add(actor.session);
    const verification=await check(python,oracle,cwd,Math.min(5,input.remainingSeconds),input.signal),parsed=JSON.parse(verification.output.trim()),sourceSha256=hash(await readFile(join(cwd,'kernel.py')));
    const file=join(input.directory,'oracle-result.json');await writeFile(file,JSON.stringify({...verification,sourceSha256}));verified.push({phase:input.phase,arm:input.arm,correct:verification.correct});
    return {correct:verification.correct,mechanismMatch:!parsed.caseResults[0].correct,caseResults:parsed.caseResults,sourceSha256,artifacts:[{path:file,sha256:hash(await readFile(file))},{path:actor.session,sha256:hash(await readFile(actor.session))}],handles:[]};
   }finally{active--;}
  }
 });
 assert.equal(report.status,'complete');assert.equal(report.promising,true);assert.equal(report.winner.id,'principle-a');assert.equal(sessions.size,45);assert.equal(peak,4);assert.equal(active,0);assert(blockedProbe);
 assert.equal(await readFile(oracle,'utf8'),trusted);assert.equal(verified.filter(r=>r.phase==='screen'&&r.correct).length,2);
 for(const task of Object.values(report.confirmation)){assert.equal(task.baseline.filter(r=>r.correct).length,2);assert.equal(task['principle-a'].filter(r=>r.correct).length,5);}
 assert.equal(Object.values(report.operations).filter(op=>op.kind==='trial'&&op.result.artifacts.length===2).length,45);
});
