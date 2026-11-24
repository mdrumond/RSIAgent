import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {publicPacket,validateDesign,checkControls,runFixtureDesign,scoreControl} from '../../cli/fixture-development-replay.mjs';

const code='import catlass.tla as tla\ndef run():\n return tla.compile(fn)()';
const input={operation:'update each valid output lane',publicTask:'cover all lanes',records:[['public-task','cover all lanes'],['failing-source',code],['observed-output','error: duplicated index'],['inherited-guidance','historical guidance'],['private-evaluator','secret verifier']].map(([reference,text])=>({reference,text}))};
const design={decision:'test',reason:'inspect the values beyond one tile',prompt:'cover lanes',starter:code,good:code,bad:code,checker:'check',diagnostic:'duplicated index',mathFailure:'tail mismatch'};
const outputs={starter:{exit:1,output:'SyntaxError: duplicated index'},good:{exit:0,output:'VERIFIED_NATIVE_ROUTE 1 1'},bad:{exit:1,output:'VERIFIED_NATIVE_ROUTE 1 1\nAssertionError: tail mismatch'}};

test('missing operation or unbound original contract stops admission before model or scoring',async()=>{
 const run=input=>runFixtureDesign({config:{},input,arm:'control',dir:'/unreachable',seconds:1},{session:()=>assert.fail(),score:()=>assert.fail()});
 for(const operation of [undefined,'','  '])await assert.rejects(run({...input,operation}),/bounded operation/);
 await assert.rejects(run({...input,publicTask:undefined}),/original project contract/);
 await assert.rejects(run({...input,originalContract:'incorrect contract'}),/differs/);
 assert.deepEqual(publicPacket(input),publicPacket({...input,publicTask:undefined,originalContract:input.publicTask}));
});
test('host-only proposed controls are rejected and failed wrong-control must exercise native route',()=>{
 const packet=publicPacket(input);assert.throws(()=>validateDesign({...design,good:'def run(x): return x+1'},packet),/native/);
 assert.throws(()=>validateDesign({...design,diagnostic:'SyntaxError'},{...packet,'observed-output':'SyntaxError: duplicated index'}),/concrete cause/);
 assert.equal(checkControls(design,{...outputs,starter:{exit:1,output:'Traceback\n    raise Exception("duplicated index")\nRuntimeError: missing module'}})[0].role,'starter');
 assert.deepEqual(checkControls(design,outputs),[]);assert.equal(checkControls(design,{...outputs,bad:{exit:1,output:'AssertionError: tail mismatch'}})[0].role,'bad');
});
test('paired public input stays identical; restore completed scores and apply reviewer feedback once',async t=>{
 const root=await mkdtemp(join(tmpdir(),'fixture-dev-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const sdk=join(root,'sdk');const {mkdir}=await import('node:fs/promises');await mkdir(sdk);await writeFile(join(sdk,'core.py'),'x=1');
 const probe=join(root,'run.sh');await writeFile(probe,'test');const prompts={};
 for(const arm of ['control','designer']){
  const launches=[],sessions=[];let interrupted=false;
  const args={config:{},input,arm,dir:join(root,arm),probe,device:0,sdk,runtime:'test',seconds:20};
  const skill=join(root,arm+'-skill.md');await writeFile(skill,'skill bytes');
  const deps={modelIdentity:async()=>({model:'one'}),designerSkill:skill,score:async({role,dir})=>{if(role==='good'&&!interrupted){interrupted=true;throw Error('infrastructure');}launches.push(dir);return outputs[role];},
   session:async args=>{assert(!JSON.stringify(args.prompt).includes('secret verifier'));sessions.push(args);assert.equal(args.prompt.operation,input.operation);assert.equal(args.prompt.records.originalContract,input.publicTask);
    if(args.directory.endsWith('/designer'))return {value:args.validate(design)};
    return {value:args.validate({verdict:sessions.length===2?'revise':'approve',reason:'protect tail values'})};}};
  await assert.rejects(runFixtureDesign(args,deps),/infrastructure/);
  assert.equal((await runFixtureDesign(args,deps)).status,'qualified');
  assert.equal((await runFixtureDesign(args,deps)).status,'qualified');
  assert.equal(new Set(launches).size,6);assert.equal(launches.length,6);assert.equal(sessions.length,4);
  assert.equal(sessions[0].fixtureDesigner,arm==='designer');assert.equal(sessions[1].fixtureDesigner,undefined);
  assert.deepEqual(sessions[2].prompt.feedback.review,{verdict:'revise',reason:'protect tail values'});assert.deepEqual(sessions[2].prompt.feedback.previousFixture,design);assert.deepEqual(sessions[2].prompt.feedback.measuredControls.bad,outputs.bad);prompts[arm]=sessions[0].prompt;
  await assert.rejects(runFixtureDesign({...args,device:1},deps),/inputs changed/);
  await assert.rejects(runFixtureDesign(args,{...deps,modelIdentity:async()=>({model:'two'})}),/inputs changed/);
  await writeFile(join(sdk,'core.py'),'x=2');await assert.rejects(runFixtureDesign(args,deps),/inputs changed/);await writeFile(join(sdk,'core.py'),'x=1');
  await writeFile(skill,'changed bytes');if(arm==='control')assert.equal((await runFixtureDesign(args,deps)).status,'qualified');else await assert.rejects(runFixtureDesign(args,deps),/inputs changed/);
 }
 assert.deepEqual(prompts.control,prompts.designer);
});
test('native route keeps oracle failure for incomplete second tile, rejects no launch',async t=>{
 const root=await mkdtemp(join(tmpdir(),'fixture-route-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const {mkdir}=await import('node:fs/promises');await mkdir(join(root,'catlass'));await writeFile(join(root,'catlass','__init__.py'),'');
 await writeFile(join(root,'catlass','tla.py'),'def compile(fn): return fn\n');
 const route=fileURLToPath(new URL('../../cli/fixture-native-route.py',import.meta.url));
 const launch=async body=>{await writeFile(join(root,'checker.py'),body);return spawnSync('python3',[route,'checker.py','kernel.py'],{cwd:root,env:{...process.env,PYTHONPATH:root},encoding:'utf8'});};
 await writeFile(join(root,'kernel.py'),'def fill(a,n):\n for i in range(n): a[i]=1\n');
 const check=limit=>'import catlass.tla as tla,runpy\na=[0]*129\nfill=runpy.run_path("kernel.py")["fill"]\ntla.compile(fill)(a,'+limit+')\nassert a==[1]*129,"tail mismatch"';
 const good=await launch(check(129)),bad=await launch(check(128)),host=await launch('import catlass.tla as tla\nfrom kernel import fill\nfill([0],1)\ntla.compile(lambda: 1)()');
 assert.equal(good.status,0);assert.match(good.stdout,/VERIFIED_NATIVE_ROUTE 1 1/);assert.equal(bad.status,1);assert.match(bad.stderr,/tail mismatch/);assert.equal(host.status,1);assert.match(host.stderr,/no native Catlass launch/);
});

test('cancellation closes a live native control and its subprocess before returning',async t=>{
 const root=await mkdtemp(join(tmpdir(),'fixture-cancel-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const probe=join(root,'runner.sh'),pidFile=join(root,'child.pid');await writeFile(probe,'sleep 60 &\necho $! > "$1/child.pid"\nwait\n');
 const abort=new AbortController(),pending=scoreControl({dir:root,role:'good',design,probe,device:0,remainingSeconds:30,signal:abort.signal});
 let pid;for(let i=0;i<100&&!pid;i++){try{pid=Number(await readFile(pidFile,'utf8'));}catch{await new Promise(r=>setTimeout(r,10));}}
 assert(pid);abort.abort(Error('operator cancelled'));await assert.rejects(pending,/operator cancelled/);
 try{const stat=await readFile('/proc/'+pid+'/stat','utf8');assert.equal(stat.split(' ')[2],'Z');}catch(e){if(e.code!=='ENOENT')throw e;}
});
