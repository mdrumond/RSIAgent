/** CPU-only production-entrypoint smoke: fake inference, real Pi/MCP/Python verification. */
import test from 'node:test';import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';import {join,resolve} from 'node:path';import {homedir} from 'node:os';import {execFile} from 'node:child_process';import {promisify} from 'node:util';import {randomUUID,createHash} from 'node:crypto';
import {State} from '../../controller/state.mjs';import {SkillBacklog} from '../../controller/skill-backlog.mjs';import {stages} from '../../controller/workflow.mjs';
import {main} from '../../cli/main.mjs';import {runCampaignFailureStep} from '../../cli/skill-failure-campaign.mjs';
import {createFailureRuntime} from '../../cli/skill-failure-runtime.mjs';import {prepareFailureMemory} from '../../cli/skill-failure-memory.mjs';
import {hypothesisTarget} from '../../cli/skill-hypothesis-evaluation.mjs';import {nativeSample,skillInput} from '../../cli/skill-microeval.mjs';import {qaTopics} from '../../cli/skill-review-qa.mjs';import {scriptedReview} from './helpers/scripted-review.mjs';
const execute=promisify(execFile),hash=x=>createHash('sha256').update(x).digest('hex');

test('actual skills-ab CLI completes all stages, native trials, independent transfers and publication', {timeout:120000},async t=>{
 const abort=new AbortController();t.after(()=>abort.abort());
 const parent=join(homedir(),'.cache/rsi-campaign-native');await mkdir(parent,{recursive:true});const root=await mkdtemp(join(parent,'run-'));t.after(()=>rm(root,{recursive:true,force:true}));const providerDir=join(root,'provider');await mkdir(providerDir);
 let sessions=0,qaSessions=0;const reviewSessions={};
 const provider=await scriptedReview(t,providerDir,body=>{
  const calls=body.messages.flatMap(m=>m.tool_calls??[]),raw=body.messages.find(m=>m.role==='user').content,text=typeof raw==='string'?raw:raw.map(p=>p.text??'').join('\n');
  const review=body.tools.find(tool=>tool.function.name==='submit_review');
  if(review){
   const shape=review.function.parameters.properties.result.properties,kind=shape.classifications?'classify':shape.family?'isolate':shape.memoryRefs&&shape.scope?'write':shape.answers?'answer':'review';
   const tools=body.messages.filter(m=>m.role==='tool');
   if(!tools.length){reviewSessions[kind]=(reviewSessions[kind]??0)+1;if(['answer','review'].includes(kind))qaSessions++;return {name:'read',args:{path:'input.json',limit:100}};}
   if(String(tools.at(-1).content).includes('Accepted. Finish'))return null;
   const inputCalls=calls.filter(c=>c.function.name==='read'&&JSON.parse(c.function.arguments).path==='input.json'),pages=inputCalls.map(c=>tools.find(t=>t.tool_call_id===c.id)).filter(Boolean),lastPage=String(pages.at(-1).content),next=lastPage.match(/Use offset=(\d+) to continue/);
   if(next&&!inputCalls.some(c=>JSON.parse(c.function.arguments).offset===Number(next[1])))return {name:'read',args:{path:'input.json',offset:Number(next[1]),limit:100}};
   const input=JSON.parse(pages.map(p=>String(p.content).replace(/\n\n\[(?:Showing lines|\d+ more lines)[\s\S]*$/,'')).join('\n')),readPaths=new Set(calls.filter(c=>c.function.name==='read').map(c=>JSON.parse(c.function.arguments).path));
   const missing=(input.groups??[]).flatMap(g=>[g.diagnosticsFile,g.linkedDiagnosticsFile]).filter(Boolean).find(path=>!readPaths.has(path));
   if(missing)return {name:'read',args:{path:missing}};
   const selector=reference=>{const row=input.proofSelectors.find(s=>s.reference===reference);assert(row,'Missing proof source '+reference);return row.selectors.includes('reference')?{reference}:{reference,startLine:1,endLine:1,...(row.segments.length>1?{segment:1}:{})};};
   const claim=(path,proof)=>({path,kind:'hypothesis',proofs:[proof]});let value;
   if(kind==='classify')value={classifications:input.groups.map(g=>({groupId:g.id,cause:'actor',bucket:'correctness',reason:'The fixture records an actor correctness failure.',coverage:'uncovered',guidanceIds:[],evidenceRefs:g.occurrences.map(o=>o.reference)})),claims:input.groups.map((g,i)=>claim('classifications.'+i+'.reason',selector(g.occurrences[0].reference)))};
   else if(kind==='isolate')value={status:'ready',reason:'A retained failure admits controlled replay fixtures.',family,files:bundle,claims:[claim('reason',selector(reference))]};
   else if(kind==='write'){
    if(!calls.some(c=>c.function.name==='mcp__kdb__search'))return {name:'mcp__kdb__search',args:{query:'kernel answer',limit:3}};
    const retrieved=[];const visit=x=>{if(!x||typeof x!=='object')return;if(x.collection&&x.chunk_id)retrieved.push(x);for(const v of Object.values(x))visit(v);};for(const tool of tools)try{visit(JSON.parse(String(tool.content)));}catch{}
    assert(retrieved.length,'Writer must use real MCP results');assert.equal(models,5);writes++;const hit=retrieved[0];
    value={text:'A bounded representation can preserve the mathematical contract across workload variations. Principle '+writes+'.',scope:'implementation',memoryRefs:[hit.collection+':'+hit.chunk_id],claims:[claim('text',selector('baseline-0'))]};
   }else{const proof={reference:'input:proposal',field:'/reason'};value=kind==='answer'?{answers:Object.fromEntries(qaTopics.map(topic=>[topic,{text:'A bounded principle can be evaluated as an untested correctness instruction.',refs:[reference]}])),claims:qaTopics.map(topic=>claim('answers.'+topic+'.text',proof))}:{verdict:'approved',reason:'The instruction is admitted to this controlled CPU workflow test.',claims:[claim('reason',proof)]};}
   return {name:'submit_review',args:{result:value}};
  }
  if(!calls.some(c=>c.function.name==='mcp__kdb__search')){sessions++;return {name:'mcp__kdb__search',args:{query:'kernel answer',limit:3}};}
  if(!calls.some(c=>c.function.name==='write'))return {name:'write',args:{path:'kernel.py',content:'answer='+(text.includes('A bounded representation')?Number(text.match(/Target=(\d+)/)[1]):0)+'\n'}};
  if(!calls.some(c=>c.function.name==='bash'))return {name:'bash',args:{command:'python3 public_verify.py .'}};return null;
 });
 const python=(await execute(process.env.DRIVER_PYTHON??'python3',['-c','import sys;print(sys.executable)'])).stdout.trim(),packageRoot=resolve(new URL('../..',import.meta.url).pathname);
 await writeFile(join(root,'memory_fixture.py'),"import hashlib\nclass Embeddings:\n model='deterministic-cpu';revision='1'\n def embed(self,texts):return [[v/255 for v in hashlib.sha256(t.encode()).digest()]for t in texts]\n");
 const wrapper=join(root,'python-wrapper');await writeFile(wrapper,`#!${python}\nimport sys,runpy\nsys.path[:0]=[${JSON.stringify(root)},${JSON.stringify(packageRoot)}]\nif sys.argv[1]=='-c':\n code=sys.argv[2];sys.argv=['-c',*sys.argv[3:]];exec(code)\nelse:\n name=sys.argv[2];sys.argv=[name,*sys.argv[3:]];runpy.run_module(name,run_name='__main__')\n`,{mode:0o700});
 const config={format:1,revision:0,root,target:'a3',mode:'phase1',projects:[],python:wrapper,embedding_factory:'memory_fixture:Embeddings',actor:{...provider.profile,metadata:{...provider.profile.metadata,contextWindow:200000}},environment:{options:{mode:'local',python}},skills:{activeVersion:'base',background:true,failureFirst:{enabled:true,sourceRevision:'a'.repeat(40)}}};
 const version=join(root,'skills/versions/base'),map={},entries=['global-kdb',...stages].map(scope=>({id:scope+'-base',scope,text:'Base '+scope,critical:true,negative:false}));await mkdir(version,{recursive:true});for(const e of entries){map[e.scope]=e.scope+'.md';await writeFile(join(version,map[e.scope]),e.text);}await writeFile(join(version,'skill-map.json'),JSON.stringify(map));await writeFile(join(version,'guidance.json'),JSON.stringify(entries));await writeFile(join(root,'instance.json'),JSON.stringify(config));
 const source=join(root,'source');await mkdir(source);const starter='factor=1\nanswer=0\n';await writeFile(join(source,'kernel.py'),starter);await writeFile(join(root,'retained.py'),starter);await execute(wrapper,['-c',`from pathlib import Path\nfrom ascend_tools.kdb import KDB\nfrom memory_fixture import Embeddings\nk=KDB(Path(${JSON.stringify(join(root,'knowledge.sqlite'))}),Embeddings());k.add_repo(${JSON.stringify(source)},'fixture','python');k.close()`]);
 const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());state.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');for(const [key,value]of Object.entries({pause:true,'campaign-budget':{seconds:10000,elapsed:70,cycle:'original',boot:'prior',monotonic:0,running:false}}))state.db.prepare('INSERT INTO control VALUES(?,?)').run(key,JSON.stringify(value));
 state.add({source,id:'project',phase:'phase1',actor_seconds:180,goal:'CPU workflow fixture'});const attempt=state.admit('project',{pid:1,start:'fixture',boot:'fixture'});state.db.prepare("UPDATE attempts SET session='fixture.jsonl',outcome='failed' WHERE id=?").run(attempt.id);state.event(attempt.id,'job-admitted',{kind:'original',mode:'phase1'});state.event(attempt.id,'verification',{actor_elapsed:45,correctness:{passed:false,exitCode:1,receipt:{correct:false,kernelSha256:hash(starter)}}});const reference='events:'+state.db.prepare('SELECT max(sequence) n FROM events').get().n;
 const bundle={},task=(id,factor,changeType)=>{
  bundle[id+'/starter.py']=`factor=${factor}\nanswer=0\n`;bundle[id+'/reference.py']=`answer=${42*factor}\n`;
  bundle[id+'/public.py']=`from kernel import answer\nassert answer==${42*factor}\n`;
  bundle[id+'/private.py']=`import pathlib,runpy,hashlib,json,sys\np=pathlib.Path(sys.argv[1])/'kernel.py';x=runpy.run_path(str(p))['answer'];control=isinstance(x,int);ok=control and x==sum([6]*7)*${factor}\nprint(json.dumps({'correct':ok,'kernelSha256':hashlib.sha256(p.read_bytes()).hexdigest(),'evidence':{'checkedCases':['control','value'],'caseResults':[{'id':'control','correct':control},{'id':'value','correct':ok}],'phase':'numeric','type':'Mismatch'}}));sys.exit(0 if ok else 1)\n`;
  return {id,prompt:`Implement answer and preserve integer output. Target=${42*factor}`,caseIds:['value','control'],controlCaseIds:['control'],files:{'kernel.py':id+'/starter.py'},checker:id+'/private.py',publicChecker:id+'/public.py',controls:{reference:{files:{'kernel.py':id+'/reference.py'}},negatives:[{files:{'kernel.py':id+'/starter.py'}}]},...(changeType?{changeType,relation:'A distinct independent CPU contract for this workflow test'}:{})};
 };
 const family={format:2,id:'fixture-family',target:'a3',objective:'correctness',mechanism:{phase:'numeric',type:'Mismatch'},provenance:{reference,source:{path:'retained.py'}},development:task('dev',1),transfer:[task('layout',2,'layout'),task('aggregate',3,'operation')]};
 let writes=0,models=0,privateChecks=0,peak=0,active=0;
 const localTarget=args=>{const target=hypothesisTarget(args,{nativeSample:async input=>{active++;peak=Math.max(peak,active);try{return await nativeSample(input);}finally{active--; }},createEnvironment:async()=>({prepare:async()=>{},reconcileLease:async()=>true,instructions:'CPU workflow fixture; use rsi_target only for CPU checks.',execute:async(command,cwd,options)=>{try{const out=await execute('bash',['-c',command],{cwd,signal:options.signal,maxBuffer:1048576});options.onData?.(out.stdout);return {settled:true,exitCode:0,handle:'local:'+randomUUID()};}catch(e){if(!Number.isInteger(e.code))throw e;options.onData?.(e.stdout);return {settled:true,exitCode:e.code,handle:'local:'+randomUUID()};}}})});const sample=target.sample,qualify=target.qualifyBatch;target.sample=async args=>{models++;return sample(args);};target.qualifyBatch=async args=>{privateChecks++;return qualify(args);};return target;};
 const callbacks={runtime:args=>createFailureRuntime({...args,config:{...args.config,target:'cpu',environment:{options:{mode:'local',python}}}},{target:localTarget,settled:async()=>true})};
 const slots=Array.from({length:4},(_,physical)=>({resource:{id:'slot-'+physical,available:true,metadata:{kind:'host-slot',pool:'fixture',hostSlot:physical}},physical}));
 const out=await main(['skills-ab','--instance',root,'--once'],{signal:abort.signal,failureDependencies:{settled:async()=>true,slots:async()=>slots,memory:prepareFailureMemory,step:(config,values)=>runCampaignFailureStep(config,values,callbacks)}});
 assert.equal(out.status,'accepted',JSON.stringify(out));assert.equal(models,45);assert.equal(sessions,45);assert.equal(writes,3);assert.equal(qaSessions,6);assert(peak>1&&peak<=4);assert.equal(privateChecks,48);
 assert.deepEqual(reviewSessions,{classify:1,isolate:1,write:3,answer:3,review:3});
 const updated=JSON.parse(await readFile(join(root,'instance.json'),'utf8'));assert.notEqual(updated.skills.activeVersion,'base');const effective=await skillInput(join(root,'skills/versions',updated.skills.activeVersion));assert(effective.text.includes('A bounded representation'));
 const backlog=new SkillBacklog(state,{target:'a3'}),item=backlog.list({status:'accepted'})[0];assert(item);assert.equal(backlog.meta('failure-first-current'),null);assert.equal(JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='pause'").get().value),true);
});
