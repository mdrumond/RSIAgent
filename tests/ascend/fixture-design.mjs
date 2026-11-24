import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {publicPacket,validateDesign,checkControls,runFixtureDesign,scoreControl} from '../../cli/fixture-design.mjs';

const code='import catlass.tla as tla\ndef run():\n return tla.compile(fn)()';
const input={operation:'update each valid output lane',publicTask:'cover all lanes',records:[['public-task','cover all lanes'],['failing-source',code],['observed-output','error: duplicated index'],['inherited-guidance','historical guidance'],['private-evaluator','secret verifier']].map(([reference,text])=>({reference,text}))};
const design={decision:'test',reason:'inspect the values beyond one tile',prompt:'cover lanes',starter:code+' # starter',good:code,bad:code+' # mutant',checker:'check',diagnosticRef:'observed-output:L1',checkId:'output-values'};
const outputs={starter:{exit:1,output:'error: duplicated index'},good:{exit:0,output:'VERIFIED_NATIVE_ROUTE 1 1'},bad:{exit:1,output:'VERIFIED_NATIVE_ROUTE 1 1\nVERIFIED_CHECK_FAILURE {"id":"output-values","details":"29 wrong lanes"}\nAssertionError: output-values: 29 wrong lanes'}};

test('missing operation or unbound original contract stops admission before model or scoring',async()=>{
 const run=input=>runFixtureDesign({config:{},input,dir:'/unreachable',seconds:1},{session:()=>assert.fail(),score:()=>assert.fail()});
 for(const operation of [undefined,'','  '])await assert.rejects(run({...input,operation}),/bounded operation/);
 await assert.rejects(run({...input,publicTask:undefined}),/original project contract/);
 await assert.rejects(run({...input,originalContract:'incorrect contract'}),/differs/);
 assert.deepEqual(publicPacket(input),publicPacket({...input,publicTask:undefined,originalContract:input.publicTask}));
});
test('host-only proposed controls are rejected and failed wrong-control must exercise native route',()=>{
 const packet=publicPacket(input);assert.throws(()=>validateDesign({...design,good:'def run(x): return x+1'},packet),/native/);
 assert.throws(()=>validateDesign({...design,diagnosticRef:'SyntaxError'},packet),/select an observed/);
 assert.equal(checkControls(design,{...outputs,starter:{exit:1,output:'Traceback\n    raise Exception("duplicated index")\nRuntimeError: missing module'}},packet)[0].role,'starter');
 assert.deepEqual(checkControls(design,outputs,packet),[]);assert.equal(checkControls(design,{...outputs,bad:{exit:1,output:'AssertionError: output-values: 29 wrong lanes'}},packet)[0].role,'bad');
});
test('ordinary fixture design restores completed scores and applies reviewer feedback once',async t=>{
 const root=await mkdtemp(join(tmpdir(),'fixture-dev-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const sdk=join(root,'sdk');const {mkdir}=await import('node:fs/promises');await mkdir(sdk);await writeFile(join(sdk,'core.py'),'x=1');
 const probe=join(root,'run.sh');await writeFile(probe,'test');
  const launches=[],sessions=[];let interrupted=false;
  const args={config:{},input,dir:join(root,'trial'),probe,device:0,sdk,runtime:'test',seconds:20};
  const deps={modelIdentity:async()=>({model:'one'}),score:async({role,dir})=>{if(role==='good'&&!interrupted){interrupted=true;throw Error('infrastructure');}launches.push(dir);return outputs[role];},
   session:async args=>{assert(!JSON.stringify(args.prompt).includes('secret verifier'));assert(args.prompt.workedExample?.accepted);sessions.push(args);assert.equal(args.prompt.operation,input.operation);assert.equal(args.prompt.records.originalContract,input.publicTask);
    if(args.directory.endsWith('/designer'))return {value:args.validate(sessions.length===1?design:{...design,bad:design.bad+' changed'})};
    return {value:args.validate({verdict:sessions.length===2?'revise':'approve',reason:'protect tail values',reference:'bad'})};}};
  await assert.rejects(runFixtureDesign(args,deps),/infrastructure/);
  assert.equal((await runFixtureDesign(args,deps)).status,'qualified');
  assert.equal((await runFixtureDesign(args,deps)).status,'qualified');
  assert.equal(new Set(launches).size,4);assert.equal(launches.length,4);assert.equal(sessions.length,4);
  assert.deepEqual(sessions[2].prompt.feedback.review,{verdict:'revise',reason:'protect tail values',reference:'bad'});assert.deepEqual(sessions[2].prompt.feedback.previousFixture,design);assert.deepEqual(sessions[2].prompt.feedback.measuredControls.bad,outputs.bad);
  await assert.rejects(runFixtureDesign({...args,device:1},deps),/inputs changed/);
  await assert.rejects(runFixtureDesign(args,{...deps,modelIdentity:async()=>({model:'two'})}),/inputs changed/);
  await writeFile(join(sdk,'core.py'),'x=2');await assert.rejects(runFixtureDesign(args,deps),/inputs changed/);await writeFile(join(sdk,'core.py'),'x=1');
});
test('native route keeps oracle failure for incomplete second tile, rejects no launch',async t=>{
 const root=await mkdtemp(join(tmpdir(),'fixture-route-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const {mkdir}=await import('node:fs/promises');await mkdir(join(root,'catlass'));await writeFile(join(root,'catlass','__init__.py'),'');
 await writeFile(join(root,'catlass','tla.py'),'def compile(fn): return fn\n');
 const route=fileURLToPath(new URL('../../cli/fixture-native-route.py',import.meta.url));
 const launch=async body=>{await writeFile(join(root,'checker.py'),body);return spawnSync('python3',[route,'checker.py','kernel.py'],{cwd:root,env:{...process.env,PYTHONPATH:root},encoding:'utf8'});};
 await writeFile(join(root,'kernel.py'),'def fill(a,n):\n for i in range(n): a[i]=1\n');
 const check=limit=>'import catlass.tla as tla,runpy\na=[0]*129\nfill=runpy.run_path("kernel.py")["fill"]\ntla.compile(fill)(a,'+limit+')\nassert a==[1]*129,"output-values: tail mismatch"';
 const good=await launch(check(129)),bad=await launch(check(128)),host=await launch('import catlass.tla as tla\nfrom kernel import fill\nfill([0],1)\ntla.compile(lambda: 1)()');
 assert.equal(good.status,0);assert.match(good.stdout,/VERIFIED_NATIVE_ROUTE 1 1/);assert.equal(bad.status,1);assert.match(bad.stderr,/tail mismatch/);assert.match(bad.stdout,/VERIFIED_CHECK_FAILURE .*output-values/);assert.equal(host.status,1);assert.match(host.stderr,/no native Catlass launch/);
});

test('cancellation closes a live native control and its subprocess before returning',async t=>{
 const root=await mkdtemp(join(tmpdir(),'fixture-cancel-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const probe=join(root,'runner.sh'),pidFile=join(root,'child.pid');await writeFile(probe,'sleep 60 &\necho $! > "$1/child.pid"\nwait\n');
 const abort=new AbortController(),pending=scoreControl({dir:root,role:'good',design,probe,device:0,remainingSeconds:30,signal:abort.signal});
 let pid;for(let i=0;i<100&&!pid;i++){try{pid=Number(await readFile(pidFile,'utf8'));}catch{await new Promise(r=>setTimeout(r,10));}}
 assert(pid);abort.abort(Error('operator cancelled'));await assert.rejects(pending,/operator cancelled/);
 try{const stat=await readFile('/proc/'+pid+'/stat','utf8');assert.equal(stat.split(' ')[2],'Z');}catch(e){if(e.code!=='ENOENT')throw e;}
});

test('retained rejection patterns produce targeted feedback before device allocation',()=>{
 const packet=publicPacket(input);
 assert.throws(()=>validateDesign({...design,diagnosticRef:'The failure is '+packet['observed-output']},packet),/observed-output:L1/);
 assert.throws(()=>validateDesign({...design,starter:design.good},packet),/must differ/);
 assert.throws(()=>validateDesign({...design,reason:'new prose only'},packet,{previousFixture:design}),/unchanged fixture/);
 assert.throws(()=>validateDesign({...design,checker:'import subprocess\nsubprocess.run(["python", "kernel.py"])'},packet),/in this Python process/);
 assert.throws(()=>validateDesign({decision:'withhold',reason:'PLACEHOLDER',missing:'checker',basis:'public-task'},packet),/concrete reason/);
 assert.throws(()=>validateDesign({decision:'withhold',reason:'need data',missing:'outputs',basis:'hidden-transfer'},packet),/available evidence/);
 const changed={...outputs,bad:{...outputs.bad,output:outputs.bad.output.replace('29 wrong lanes','actual 31.5, expected 28.0')}};
 assert.deepEqual(checkControls(design,changed,packet),[]);
 const [failure]=checkControls(design,{...outputs,bad:{exit:1,output:'VERIFIED_NATIVE_ROUTE 1 1\nAssertionError: input-preserved: modified index 3'}},packet);
 assert.deepEqual(failure.expected,{exit:'nonzero',routeVerified:true,checkId:'output-values'});assert.equal(failure.observed.numerical,null);assert.equal(failure.observed.routeVerified,true);
});

test('withhold dispute can obtain an actual probe observation and reach an independently grounded disposition',async t=>{
 const root=await mkdtemp(join(tmpdir(),'fixture-disposition-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const sdk=join(root,'sdk');await (await import('node:fs/promises')).mkdir(sdk);await writeFile(join(sdk,'sdk.py'),'x=1');
 const probe=join(root,'adapter.sh');await writeFile(probe,'cd "$1" && python3 probe.py');
 const args={config:{},input,dir:join(root,'trial'),probe,device:0,sdk,runtime:'local',seconds:30};
 const disposition={decision:'withhold',reason:'historical arguments are not observable',missing:'the argument binding',basis:'observed-output:L1'};
 let turns=0,reviews=0;
 const result=await runFixtureDesign(args,{modelIdentity:async()=>({model:'mock'}),session:async o=>{
  assert(!JSON.stringify(o.prompt).includes('secret verifier'));
  if(o.directory.endsWith('/withhold-review')){
   reviews++;return {value:o.validate({verdict:reviews===1?'revise':'withhold',reference:reviews===1?'observed-output:L1':'probe',reason:reviews===1?'inspect the public argument count':'probe shows the binding is unavailable'})};
  }
  turns++;if(turns===2){assert.equal(o.prompt.feedback.review.verdict,'revise');return {value:o.validate({decision:'probe',reason:'measure argument count',code:"print('observed count=2')"})};}
  if(turns===3){assert.match(o.prompt.feedback.probe.output,/observed count=2/);return {value:o.validate({...disposition,basis:'probe'})};}
  return {value:o.validate(disposition)};
 }});
 assert.equal(result.status,'withheld');assert.equal(result.basis,'probe');assert.equal(result.review.reference,'probe');assert.equal(turns,3);assert.equal(reviews,2);
});

test('submitted exceptions cannot impersonate an oracle assertion',async t=>{
 const root=await mkdtemp(join(tmpdir(),'fixture-wrong-origin-'));t.after(()=>rm(root,{recursive:true,force:true}));
 await (await import('node:fs/promises')).mkdir(join(root,'catlass'));await writeFile(join(root,'catlass','__init__.py'),'');
 await writeFile(join(root,'catlass','tla.py'),'def compile(fn): return fn\n');
 await writeFile(join(root,'kernel.py'),'import catlass.tla as tla\ndef run():\n tla.compile(lambda: 1)()\n raise AssertionError("output-values: fake result")');
 await writeFile(join(root,'checker.py'),'import runpy\nrunpy.run_path("kernel.py")["run"]()');
 const script=fileURLToPath(new URL('../../cli/fixture-native-route.py',import.meta.url));
 const run=spawnSync('python3',[script,'checker.py','kernel.py'],{cwd:root,env:{...process.env,PYTHONPATH:root},encoding:'utf8'});
 assert.equal(run.status,1);assert.match(run.stdout,/VERIFIED_NATIVE_ROUTE/);assert.doesNotMatch(run.stdout,/VERIFIED_CHECK_FAILURE/);
 assert.equal(checkControls(design,{...outputs,bad:{exit:1,output:run.stdout+run.stderr}},publicPacket(input))[0].role,'bad');
});

test('packaged row-sum handoffs obey real replay admission and independently measured calibration',()=>{
 const example=fileURLToPath(new URL('../../examples/fixture-design/row-sum/validate.mjs',import.meta.url));
 const consumer=fileURLToPath(new URL('../../cli/fixture-design.mjs',import.meta.url));
 const checked=spawnSync(process.execPath,[example,consumer],{encoding:'utf8'});
 assert.equal(checked.status,0,checked.stderr);
});

test('a live control exceeding the assignment deadline immediately settles elapsed',async t=>{
 const root=await mkdtemp(join(tmpdir(),'fixture-expiry-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const sdk=join(root,'sdk');await (await import('node:fs/promises')).mkdir(sdk);await writeFile(join(sdk,'x.py'),'x=1');
 const probe=join(root,'run.sh');await writeFile(probe,'sleep 10');
 const args={config:{},input,dir:join(root,'trial'),probe,device:0,sdk,runtime:'local',seconds:0.25};
 const deps={modelIdentity:async()=>({model:'mock'}),session:async o=>({value:o.validate(design)})};
 assert.equal((await runFixtureDesign(args,deps)).status,'elapsed');
 assert.equal(JSON.parse(await readFile(join(args.dir,'outcome.json'))).status,'elapsed');
 assert.equal((await runFixtureDesign(args,{...deps,session:()=>assert.fail('completed')})).status,'elapsed');
});

test('ordinary skill action refuses archived paired requests without touching their outputs',async t=>{
 const {run}=await import('../../cli/skill-input.mjs');
 const dir=await mkdtemp(join(tmpdir(),'archived-pair-'));t.after(()=>rm(dir,{recursive:true,force:true}));
 const prior='{"status":"qualified","round":2}';await writeFile(join(dir,'outcome.json'),prior);
 await assert.rejects(run({target:'a5'},{action:'fixture-design',arm:'skill',output:dir}),/original frozen runner/);
 assert.equal(await readFile(join(dir,'outcome.json'),'utf8'),prior);
});
