import test from 'node:test';
import {Check} from 'typebox/value';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,symlink} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {reviewWritePathAllowed} from '../../cli/skill-review-native.mjs';
import {State} from '../../controller/state.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';
import {fingerprint} from '../../controller/evidence.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {stages} from '../../controller/workflow.mjs';
import {analyzeBacklog} from '../../cli/skill-backlog-analysis.mjs';
import {loadFailureFamily,failureWitness,failureSourceInventory,prepareFailureFamily} from '../../cli/skill-failure-families.mjs';
const sha=s=>createHash('sha256').update(s).digest('hex');
const emit=(state,...args)=>{state.event(...args);return state.db.prepare('SELECT MAX(sequence) sequence FROM events').get();};
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'failure-family-'));t.after(()=>rm(root,{recursive:true,force:true}));const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
 const version=join(root,'skills/versions/base');await mkdir(version,{recursive:true});const entries=['global-kdb',...stages].map(scope=>({id:scope,scope,text:scope,critical:true,negative:false})),map={};
 for(const e of entries){map[e.scope]=e.scope+'.md';await writeFile(join(version,map[e.scope]),e.text);}await writeFile(join(version,'skill-map.json'),JSON.stringify(map));await writeFile(join(version,'guidance.json'),JSON.stringify(entries));
 const config={format:1,root,target:'a3',skills:{activeVersion:'base'}},code='def run(x): return x + 1\n';await writeFile(join(root,'retained.py'),code);await writeFile(join(root,'kernel.py'),code);state.add({source:root,id:'project',phase:'phase1',actor_seconds:180,goal:'A kernel'});const attempt=state.admit('project',{pid:1,start:'test',boot:'test'});state.db.prepare("UPDATE attempts SET session='native.jsonl',outcome='failed' WHERE id=?").run(attempt.id);state.event(attempt.id,'job-admitted',{kind:'original',mode:'phase1'});
 const event=emit(state,attempt.id,'verification',{actor_elapsed:45,correctness:{passed:false,exitCode:1,receipt:{correct:false,kernelSha256:sha(code)}}});const reference='events:'+event.sequence;
 const backlog=new SkillBacklog(state,{target:'a3'});backlog.ingest();const group=backlog.failures()[0],item={id:'item',base:'base',objective:'correctness',groupIds:[group.id],evidence:{groups:[group]}};
 const directory=join(root,'family');await mkdir(directory);const files={};const task=(id,changeType)=>{
  const prefix=id+'/';files[prefix+'starter.py']=id==='dev'?code:'def run(x): return x + '+(id==='layout'?2:3)+'\n';files[prefix+'reference.py']='def run(x): return x\n';files[prefix+'private.py']='print("private")\n';files[prefix+'public.py']='print("public")\n';files[prefix+'helper.py']='VALUE=1\n';
  return {id,prompt:'Implement run; preserve the public contract',caseIds:['ordinary','edge'],files:{'kernel.py':prefix+'starter.py'},checker:prefix+'private.py',publicChecker:prefix+'public.py',checkerFiles:{'helper.py':prefix+'helper.py'},controls:{reference:{files:{'kernel.py':prefix+'reference.py'}},negatives:[{files:{'kernel.py':prefix+'starter.py'}}]},...(changeType?{changeType,relation:'Adapt a distinct data contract'}:{})};
 };
 const family={format:2,id:'family',target:'a3',objective:'correctness',mechanism:{phase:'candidate-call',type:'AssertionError'},provenance:{reference,source:{path:'retained.py'}},development:task('dev'),transfer:[task('layout','layout'),task('reduce','operation')]};
 const write=async(value=family)=>{for(const [p,s]of Object.entries(files)){await mkdir(join(directory,p,'..'),{recursive:true});await writeFile(join(directory,p),s);}const file=join(directory,'family.json');await writeFile(file,JSON.stringify(value));return file;};
 return {root,state,config,backlog,item,attempt,directory,code,reference,group,family,files,write};
}
async function submitDraft(validate,draft,{files,...value}){for(const [name,content]of Object.entries(files)){await mkdir(join(draft,name,'..'),{recursive:true});await writeFile(join(draft,name),content);}return validate(value);}
const classify=bucket=>async({groups,validate})=>validate({classifications:groups.map(g=>({groupId:g.id,cause:bucket==='harness'?'harness':'actor',bucket,coverage:'uncovered',guidanceIds:[],evidenceRefs:[g.occurrences[0].reference],reason:'Observed verification result'}))});

test('classification-only queues facts without proposals, knowledge-gap requests or QA',async t=>{
 const f=await fixture(t);let reviews=0;
 const result=await analyzeBacklog({...f,directory:join(f.root,'analysis'),classificationOnly:true},{classify:classify('correctness'),review:async()=>{reviews++;throw Error('Must not write skills');}});
 assert.equal(reviews,0);assert.equal(result.items[0].status,'queued');assert.equal(result.items[0].proposal,null);assert.equal(result.items[0].workflow,'failure-first-v1');assert.deepEqual(result.items[0].groupIds,[f.group.id]);assert.equal(f.backlog.failures().length,0);assert.equal(f.state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get(),undefined);
});
for(const [bucket,status]of [['optimization','deferred'],['harness','user-required']])test(bucket+' classification stays outside the correctness actor queue',async t=>{
 const f=await fixture(t);const result=await analyzeBacklog({...f,directory:join(f.root,'analysis'),classificationOnly:true},{classify:classify(bucket),review:()=>assert.fail('No speculative writing')});assert.equal(result.items[0].status,status);assert.equal(result.items[0].proposal,null);assert.equal(f.backlog.claim({base:'base'}),null);
});

test('v2 family freezes exact failed code, helpers and explicit transfer roles',async t=>{
 const f=await fixture(t),file=await f.write(),value=await loadFailureFamily({file,config:f.config,item:f.item});assert.equal(value.development.files['kernel.py'],f.code);assert.equal(value.provenance.sourceSha256,sha(f.code));assert.equal(value.provenance.attemptId,f.attempt.id);assert.equal(value.development.checkerFiles['helper.py'],'VALUE=1\n');assert.equal(value.transfer.length,2);assert.match(value.development.checker,/dev\/private.py$/);assert.equal(value.development.controls.negatives[0].files['kernel.py'],f.code);
 await writeFile(join(f.directory,'dev/private.py'),'print("changed")');const changed=await loadFailureFamily({file,config:f.config,item:f.item});assert.notEqual(changed.identity,value.identity);
});

test('a failed verifier within an eventually successful attempt remains valid historical evidence',async t=>{
 const f=await fixture(t);f.state.db.prepare("UPDATE attempts SET outcome='passed' WHERE id=?").run(f.attempt.id);const value=await failureWitness({...f,provenance:f.family.provenance});assert.equal(value.sourceSha256,sha(f.code));
});

for(const [name,change]of [['verifier infrastructure failure',f=>f.state.db.prepare("UPDATE events SET data=? WHERE sequence=?").run(JSON.stringify({actor_elapsed:2065.8407118270006,correctness:{passed:false,exitCode:125},fingerprint:'6a2bb6c469f2411aaec5d78349395eb0b5d3ad5875638371254c04c7b696c86d'}),Number(f.reference.split(':')[1]))],['pre-actor failure',f=>f.state.db.prepare('UPDATE attempts SET session=NULL WHERE id=?').run(f.attempt.id)],['performance-only failure',f=>f.state.db.prepare("UPDATE events SET data=? WHERE sequence=?").run(JSON.stringify({actor_elapsed:40,correctness:{passed:true},performance:{passed:false}}),Number(f.reference.split(':')[1]))],['unbound source',async f=>writeFile(join(f.root,'retained.py'),'def run(x): return x - 1\n')],['unrelated evidence',f=>{f.item.evidence.groups=[];}]])test(name+' cannot seed a skill experiment',async t=>{const f=await fixture(t);await change(f);await assert.rejects(failureWitness({...f,provenance:f.family.provenance}));});

test('retained source can be recovered from the exact controller source tree',async t=>{
 const f=await fixture(t),repo=join(f.root,'history');await mkdir(repo);execFileSync('git',['init','-q',repo]);await writeFile(join(repo,'kernel.py'),f.code);execFileSync('git',['-C',repo,'add','kernel.py']);execFileSync('git',['-C',repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','retained']);const tree=execFileSync('git',['-C',repo,'rev-parse','HEAD^{tree}'],{encoding:'utf8'}).trim();
 // A source snapshot must precede the specific failed verifier, not merely name any tree.
 const snap=emit(f.state,f.attempt.id,'source-snapshot',{tree_oid:tree}),verification=emit(f.state,f.attempt.id,'verification',{actor_elapsed:50,correctness:{passed:false,receipt:{kernelSha256:sha(f.code)}}});const reference='events:'+verification.sequence;f.item.evidence.groups[0].occurrences.push({reference});const provenance={reference,source:{snapshotReference:'events:'+snap.sequence,repository:'history',file:'kernel.py'}};const value=await failureWitness({...f,provenance});assert.equal(value.code,f.code);assert.equal(value.source.tree,tree);
 provenance.source.snapshotReference=f.reference;await assert.rejects(failureWitness({...f,provenance}),/snapshot/);
});

for(const [name,edit]of [['legacy suite',f=>{f.family.format=1;}],['seed-only transfer',f=>{f.family.transfer[0].changeType='seed';}],['private checker leaked into editable files',f=>{f.family.development.files['oracle.py']='dev/private.py';}],['public checker helper alias',f=>{f.family.development.checkerFiles={'hidden.py':'dev/./public.py'};}],['private helper path alias',f=>{f.family.development.files['helper.py']='dev/./helper.py';}],['duplicate transfer adaptation',f=>{f.family.transfer[1].changeType='layout';}],['unexplained replacement starter',f=>{f.files['dev/starter.py']='def run(x): return x\n';}]])test(name+' is rejected before any model evaluation',async t=>{const f=await fixture(t);edit(f);await assert.rejects(loadFailureFamily({file:await f.write(),config:f.config,item:f.item}));});

test('native isolation validates its submission, never calls a curriculum, and reuses frozen output',async t=>{
 const f=await fixture(t);let calls=0;const input={...f,directory:join(f.root,'isolated'),seconds:90};
 const dependencies={session:async({seconds,validate,writableDirectory})=>{calls++;assert(seconds<=90);return {value:await submitDraft(validate,writableDirectory,{status:'ready',reason:'A replay is recoverable',family:f.family,files:f.files,claims:[]})};}};
 const result=await prepareFailureFamily(input,dependencies);assert.equal(result.status,'ready');assert.equal(result.family.provenance.sourceSha256,sha(f.code));assert.equal(calls,1);const again=await prepareFailureFamily(input,{session:()=>assert.fail('Must not rerun accepted research')});assert.equal(again.family.identity,result.family.identity);await writeFile(join(input.directory,'dev/private.py'),'changed');await assert.rejects(prepareFailureFamily(input,dependencies),/Frozen family changed/);
});

test('missing evidence is terminal without a speculative proposal; unknown interrupted spend cannot renew',async t=>{
 const f=await fixture(t),input={...f,directory:join(f.root,'missing'),seconds:30};let calls=0;const session=async()=>{calls++;return {value:{status:'needs-evidence',reason:'No failed source retained'}};};const value=await prepareFailureFamily(input,{session});assert.equal(value.status,'needs-evidence');await prepareFailureFamily(input,{session});assert.equal(calls,1);
 const file=join(input.directory,'isolation.json'),ledger=JSON.parse(await readFile(file,'utf8'));delete ledger.status;ledger.running=true;await writeFile(file,JSON.stringify(ledger));await assert.rejects(prepareFailureFamily(input,{session}),/unknown/);assert.equal(calls,1);
});


test('source-history Git metadata is materialized for read-only native research',async t=>{
 const f=await fixture(t),repo=join(f.root,'work');await mkdir(repo);execFileSync('git',['init','-q',repo]);await writeFile(join(repo,'kernel.py'),f.code);execFileSync('git',['-C',repo,'add','kernel.py']);execFileSync('git',['-C',repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','retained']);const tree=execFileSync('git',['-C',repo,'rev-parse','HEAD^{tree}'],{encoding:'utf8'}).trim();
 const history=join(f.root,'source-history/retained');await mkdir(join(f.root,'source-history'));execFileSync('git',['clone','--bare','-q',repo,history]);execFileSync('git',['--git-dir',history,'config','core.bare','false']);
 const snapshot=emit(f.state,f.attempt.id,'source-snapshot',{tree_oid:tree}),event=emit(f.state,f.attempt.id,'verification',{actor_elapsed:60,fingerprint:await fingerprint(repo),correctness:{passed:false}}),reference='events:'+event.sequence;f.item.evidence.groups[0].occurrences=[{reference}];
 const rows=await failureSourceInventory({...f,directory:join(f.root,'source-index')});const recovered=rows.find(r=>r.reference===reference&&r.source.snapshotReference);assert.ok(recovered);assert.equal(await readFile(recovered.path,'utf8'),f.code);assert.equal(recovered.source.snapshotReference,'events:'+snapshot.sequence);assert.equal(recovered.source.repository,'source-history/retained');
});

test('no actual failed actor evidence avoids inference entirely',async t=>{
 const f=await fixture(t);f.state.db.prepare('UPDATE attempts SET session=NULL WHERE id=?').run(f.attempt.id);const result=await prepareFailureFamily({...f,directory:join(f.root,'no-actor'),seconds:30},{session:()=>assert.fail('No actor evidence must not start research')});assert.equal(result.status,'needs-evidence');
});

test('isolation checkpoints elapsed work while native research is still running',async t=>{
 const f=await fixture(t),directory=join(f.root,'checkpoint');let observed;
 const result=await prepareFailureFamily({...f,directory,seconds:5},{session:async()=>{await new Promise(r=>setTimeout(r,1100));observed=JSON.parse(await readFile(join(directory,'isolation.json'),'utf8'));return {value:{status:'needs-evidence',reason:'Reference source unavailable'}};}});
 assert.equal(observed.running,true);assert(observed.elapsedSeconds>=.9);assert.equal(result.running,false);assert(result.elapsedSeconds>=observed.elapsedSeconds);
});


test('debug edits after a stage snapshot bind only to the source actually verified',async t=>{
 const f=await fixture(t),repo=join(f.root,'history');await mkdir(repo);execFileSync('git',['init','-q',repo]);await writeFile(join(repo,'kernel.py'),f.code);await mkdir(join(repo,'a'));await writeFile(join(repo,'a/x.txt'),'nested');await writeFile(join(repo,'a.b'),'sibling');await writeFile(join(repo,'payload.dat'),Buffer.from([255,0,1]));execFileSync('git',['-C',repo,'add','.']);const treeA=execFileSync('git',['-C',repo,'write-tree'],{encoding:'utf8'}).trim(),stage=emit(f.state,f.attempt.id,'source-snapshot',{tree_oid:treeA});
 const codeB='def run(x): return x - 2\n';await writeFile(join(repo,'kernel.py'),codeB);execFileSync('git',['-C',repo,'add','kernel.py']);const treeB=execFileSync('git',['-C',repo,'write-tree'],{encoding:'utf8'}).trim();
 const storage=new CampaignStorage(f.state,f.root);await storage.initialize();storage.iteration(f.attempt.id,{stage:'validation',sourceHash:treeB,reflection:JSON.stringify({tree_oid:treeB})});const iterationId=f.state.db.prepare("SELECT id FROM storage_iterations WHERE attempt=? AND stage='validation'").get(f.attempt.id).id;
 const data={actor_elapsed:50,fingerprint:await fingerprint(repo),correctness:{passed:false}},event=emit(f.state,f.attempt.id,'verification',data),reference='events:'+event.sequence;f.item.evidence.groups[0].occurrences=[{reference}];f.item.evidence.groups[0].linkedDiagnostics=[];const spec=JSON.parse(f.state.db.prepare('SELECT spec FROM projects WHERE id=?').get('project').spec);f.state.db.prepare('UPDATE projects SET spec=? WHERE id=?').run(JSON.stringify({...spec,source:repo}),'project');
 const stale={reference,source:{snapshotReference:'events:'+stage.sequence,repository:'history',file:'kernel.py'}},current={reference,source:{iterationId,repository:'history',file:'kernel.py'}};
 await assert.rejects(failureWitness({...f,provenance:stale}),/not bound/);const witness=await failureWitness({...f,provenance:current});assert.equal(witness.code,codeB);assert.equal(witness.source.tree,treeB);
 const rows=await failureSourceInventory({...f,directory:join(f.root,'matching-index')});assert(rows.some(r=>r.source.iterationId===iterationId&&r.sourceSha256===sha(codeB)));assert(!rows.some(r=>r.sourceSha256===sha(f.code)));
 // A kernel receipt is an alternative exact binding when a full tree fingerprint is absent.
 delete data.fingerprint;data.correctness.receipt={kernelSha256:sha(codeB)};f.state.db.prepare('UPDATE events SET data=? WHERE sequence=?').run(JSON.stringify(data),event.sequence);await assert.rejects(failureWitness({...f,provenance:stale}),/not bound/);assert.equal((await failureWitness({...f,provenance:current})).code,codeB);
 delete data.correctness.receipt;f.state.db.prepare('UPDATE events SET data=? WHERE sequence=?').run(JSON.stringify(data),event.sequence);await assert.rejects(failureWitness({...f,provenance:current}),/not bound/);assert.deepEqual(await failureSourceInventory({...f,directory:join(f.root,'unbound-index')}),[]);
});


test('A3 investigation actor failures retain exact source admission without admitting pre-actor or unrelated work',async t=>{
 const f=await fixture(t),admission={kind:'investigation',wave:16,actor_seconds:9600,mode:'phase1',campaign_entry:'fresh-20261108-r4-a3-r04-p1'};
 f.state.db.prepare("UPDATE events SET data=? WHERE attempt=? AND type='job-admitted'").run(JSON.stringify(admission),f.attempt.id);f.state.db.prepare("UPDATE attempts SET outcome='passed' WHERE id=?").run(f.attempt.id);
 const witness=await failureWitness({...f,provenance:f.family.provenance});assert.equal(witness.sourceSha256,sha(f.code));assert.equal((await loadFailureFamily({file:await f.write(),config:f.config,item:f.item})).provenance.attemptId,f.attempt.id);
 await writeFile(join(f.root,'retained.py'),'def run(x): return x - 1\n');await assert.rejects(failureWitness({...f,provenance:f.family.provenance}),/not bound/);await writeFile(join(f.root,'retained.py'),f.code);
 f.state.db.prepare("UPDATE attempts SET session=NULL WHERE id=?").run(f.attempt.id);await assert.rejects(failureWitness({...f,provenance:f.family.provenance}),/actual failed actor/);f.state.db.prepare("UPDATE attempts SET session='native.jsonl' WHERE id=?").run(f.attempt.id);
 f.state.db.prepare("UPDATE events SET data=? WHERE attempt=? AND type='job-admitted'").run(JSON.stringify({...admission,kind:'curriculum'}),f.attempt.id);await assert.rejects(failureWitness({...f,provenance:f.family.provenance}),/admitted actor/);
});


test('native isolation retries syntax and reference failures within one session before accepting corrected files',async t=>{
 const f=await fixture(t),directory=join(f.root,'retry-isolation');let checks=0,sessions=0;
 const checkReferences=async({family,directory:received,seconds,signal})=>{checks++;assert.equal(received,directory);assert(seconds>0&&seconds<=30);assert.equal(signal.aborted,false);assert.equal(family.development.files['kernel.py'],f.code);if(checks===1)throw Error('Reference failed ordinary case');};
 const result=await prepareFailureFamily({...f,directory,seconds:30,checkReferences},{session:async({validate:check,writableDirectory})=>{
  const validate=value=>submitDraft(check,writableDirectory,value);sessions++;const submission={status:'ready',reason:'Recovered failure',family:f.family,files:{...f.files},claims:[]};submission.files['dev/private.py']='def broken(:\n';
  await assert.rejects(validate(submission),/syntax validation failed/);assert.equal(checks,0);
  submission.files['dev/private.py']=f.files['dev/private.py'];for(const path of ['dev/public.py','dev/helper.py']){submission.files[path]='def broken(:\n';await assert.rejects(validate(submission),/syntax validation failed/);assert.equal(checks,0);submission.files[path]=f.files[path];}
  submission.files['dev/private.py']='raise RuntimeError("syntax must not execute checker")\n';await assert.rejects(validate(submission),/Reference failed ordinary case/);assert.equal(checks,1);
  return {value:await validate(submission)};
 }});assert.equal(result.status,'ready');assert.equal(sessions,1);assert.equal(checks,2);
});


test('draft native writes are scoped through existing ancestors and accepted bytes are independent',async t=>{
 const f=await fixture(t),draft=join(f.root,'draft'),outside=join(f.root,'outside');await mkdir(draft);await mkdir(outside);await symlink(outside,join(draft,'link'));
 assert.equal(await reviewWritePathAllowed(join(draft,'new/sub/kernel.py'),draft),true);assert.equal(await reviewWritePathAllowed(join(draft,'link/kernel.py'),draft),false);assert.equal(await reviewWritePathAllowed(join(outside,'kernel.py'),draft),false);assert.equal(await reviewWritePathAllowed(join(draft,'kernel.py')),false);
 const directory=join(f.root,'incremental');let manifest;const result=await prepareFailureFamily({...f,directory,seconds:30},{session:async({writableDirectory,validate,resultSchema})=>{
  const wire={status:'ready',reason:'Isolated source',family:f.family,claims:[{path:'reason',kind:'observation',proofs:[{reference:f.reference}]}]};assert(Check(resultSchema,wire));assert.equal(Check(resultSchema,{status:wire.status,reason:wire.reason,claims:wire.claims}),false);await assert.rejects(validate(wire),/Missing draft file: dev\//);assert.equal(Check(resultSchema,{...wire,files:f.files}),false);assert.equal(Check(resultSchema,{...wire,family:{...f.family,development:{...f.family.development,checker:42}}}),false);manifest=await submitDraft(validate,writableDirectory,{status:'ready',reason:'Isolated source',family:f.family,files:f.files,claims:[]});await writeFile(join(writableDirectory,'dev/starter.py'),'changed after acceptance');return {value:manifest};
 }});assert.equal(result.family.development.files['kernel.py'],f.code);assert.equal(await readFile(join(directory,'dev/starter.py'),'utf8'),f.code);
});


for(const [name,edit]of [
 ['transfer checker in development editable',f=>{f.family.development.files['oracle.py']='layout/./private.py';}],
 ['transfer helper in development editable',f=>{f.family.development.files['oracle.py']='layout/./helper.py';}],
 ['transfer checker in development public',f=>{f.family.development.publicChecker='layout/./private.py';}],
 ['development helper in transfer public',f=>{f.family.transfer[0].publicChecker='dev/./helper.py';}]
])test(name+' is rejected across family task boundaries',async t=>{const f=await fixture(t);edit(f);await assert.rejects(loadFailureFamily({file:await f.write(),config:f.config,item:f.item}),/any family task/);});


test('A5 family isolation freezes A5 tasks and refuses an A3 manifest or schema submission',async t=>{
 const f=await fixture(t);f.config.target='a5';f.family.target='a5';const file=await f.write(),loaded=await loadFailureFamily({file,config:f.config,item:f.item});assert.deepEqual([loaded.development,...loaded.transfer].map(task=>task.target),['a5','a5','a5']);
 await assert.rejects(loadFailureFamily({file,config:{...f.config,target:'a3'},item:f.item}),/matching A3.A5/);
 const directory=join(f.root,'a5-isolation'),result=await prepareFailureFamily({...f,directory,seconds:30},{session:async({validate,writableDirectory,prompt,resultSchema})=>{
  assert.equal(prompt.target,'a5');assert.equal(prompt.contract.target,'a5');const wire={status:'ready',reason:'Retained source',family:f.family,claims:[{path:'reason',kind:'observation',proofs:[{reference:f.reference}]}]};assert(Check(resultSchema,wire));assert.equal(Check(resultSchema,{...wire,family:{...f.family,target:'a3'}}),false);
  return {value:await submitDraft(validate,writableDirectory,{...wire,files:f.files})};
 }});assert.equal(result.family.target,'a5');await assert.rejects(prepareFailureFamily({...f,directory,config:{...f.config,target:'a3'},seconds:30}),/Isolation inputs changed/);
});
test('A3 family cannot enter A5 isolation and unsupported targets stop before inference',async t=>{
 const f=await fixture(t);await assert.rejects(loadFailureFamily({file:await f.write(),config:{...f.config,target:'a5'},item:f.item}),/matching A3.A5/);await assert.rejects(prepareFailureFamily({...f,directory:join(f.root,'unsupported'),config:{...f.config,target:'cpu'}},{session:()=>assert.fail('Unsupported target must not invoke model')}),/A3 or A5/);
});
