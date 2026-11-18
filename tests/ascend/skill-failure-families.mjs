import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {State} from '../../controller/state.mjs';
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

for(const [name,change]of [['pre-actor failure',f=>f.state.db.prepare('UPDATE attempts SET session=NULL WHERE id=?').run(f.attempt.id)],['performance-only failure',f=>f.state.db.prepare("UPDATE events SET data=? WHERE sequence=?").run(JSON.stringify({actor_elapsed:40,correctness:{passed:true},performance:{passed:false}}),Number(f.reference.split(':')[1]))],['unbound source',async f=>writeFile(join(f.root,'retained.py'),'def run(x): return x - 1\n')],['unrelated evidence',f=>{f.item.evidence.groups=[];}]])test(name+' cannot seed a skill experiment',async t=>{const f=await fixture(t);await change(f);await assert.rejects(failureWitness({...f,provenance:f.family.provenance}));});

test('retained source can be recovered from the exact controller source tree',async t=>{
 const f=await fixture(t),repo=join(f.root,'history');await mkdir(repo);execFileSync('git',['init','-q',repo]);await writeFile(join(repo,'kernel.py'),f.code);execFileSync('git',['-C',repo,'add','kernel.py']);execFileSync('git',['-C',repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','retained']);const tree=execFileSync('git',['-C',repo,'rev-parse','HEAD^{tree}'],{encoding:'utf8'}).trim();
 // A source snapshot must precede the specific failed verifier, not merely name any tree.
 const snap=emit(f.state,f.attempt.id,'source-snapshot',{tree_oid:tree}),verification=emit(f.state,f.attempt.id,'verification',{actor_elapsed:50,correctness:{passed:false}});const reference='events:'+verification.sequence;f.item.evidence.groups[0].occurrences.push({reference});const provenance={reference,source:{snapshotReference:'events:'+snap.sequence,repository:'history',file:'kernel.py'}};const value=await failureWitness({...f,provenance});assert.equal(value.code,f.code);assert.equal(value.source.tree,tree);
 provenance.source.snapshotReference=f.reference;await assert.rejects(failureWitness({...f,provenance}),/snapshot/);
});

for(const [name,edit]of [['legacy suite',f=>{f.family.format=1;}],['seed-only transfer',f=>{f.family.transfer[0].changeType='seed';}],['private checker leaked into editable files',f=>{f.family.development.files['oracle.py']='dev/private.py';}],['private helper path alias',f=>{f.family.development.files['helper.py']='dev/./helper.py';}],['duplicate transfer adaptation',f=>{f.family.transfer[1].changeType='layout';}],['unexplained replacement starter',f=>{f.files['dev/starter.py']='def run(x): return x\n';}]])test(name+' is rejected before any model evaluation',async t=>{const f=await fixture(t);edit(f);await assert.rejects(loadFailureFamily({file:await f.write(),config:f.config,item:f.item}));});

test('native isolation validates its submission, never calls a curriculum, and reuses frozen output',async t=>{
 const f=await fixture(t);let calls=0;const input={...f,directory:join(f.root,'isolated'),seconds:90};
 const dependencies={session:async({seconds,validate})=>{calls++;assert(seconds<=90);return {value:await validate({status:'ready',reason:'A replay is recoverable',family:f.family,files:f.files,claims:[]})};}};
 const result=await prepareFailureFamily(input,dependencies);assert.equal(result.status,'ready');assert.equal(result.family.provenance.sourceSha256,sha(f.code));assert.equal(calls,1);const again=await prepareFailureFamily(input,{session:()=>assert.fail('Must not rerun accepted research')});assert.equal(again.family.identity,result.family.identity);await writeFile(join(input.directory,'dev/private.py'),'changed');await assert.rejects(prepareFailureFamily(input,dependencies),/Frozen family changed/);
});

test('missing evidence is terminal without a speculative proposal; unknown interrupted spend cannot renew',async t=>{
 const f=await fixture(t),input={...f,directory:join(f.root,'missing'),seconds:30};let calls=0;const session=async()=>{calls++;return {value:{status:'needs-evidence',reason:'No failed source retained'}};};const value=await prepareFailureFamily(input,{session});assert.equal(value.status,'needs-evidence');await prepareFailureFamily(input,{session});assert.equal(calls,1);
 const file=join(input.directory,'isolation.json'),ledger=JSON.parse(await readFile(file,'utf8'));delete ledger.status;ledger.running=true;await writeFile(file,JSON.stringify(ledger));await assert.rejects(prepareFailureFamily(input,{session}),/unknown/);assert.equal(calls,1);
});


test('source-history Git metadata is materialized for read-only native research',async t=>{
 const f=await fixture(t),repo=join(f.root,'work');await mkdir(repo);execFileSync('git',['init','-q',repo]);await writeFile(join(repo,'kernel.py'),f.code);execFileSync('git',['-C',repo,'add','kernel.py']);execFileSync('git',['-C',repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','retained']);const tree=execFileSync('git',['-C',repo,'rev-parse','HEAD^{tree}'],{encoding:'utf8'}).trim();
 const history=join(f.root,'source-history/retained');await mkdir(join(f.root,'source-history'));execFileSync('git',['clone','--bare','-q',repo,history]);execFileSync('git',['--git-dir',history,'config','core.bare','false']);
 const snapshot=emit(f.state,f.attempt.id,'source-snapshot',{tree_oid:tree}),event=emit(f.state,f.attempt.id,'verification',{actor_elapsed:60,correctness:{passed:false}}),reference='events:'+event.sequence;f.item.evidence.groups[0].occurrences=[{reference}];
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
