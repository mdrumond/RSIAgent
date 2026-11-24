import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {publicPacket,validateDesign,checkControls,runFixtureDesign} from '../../cli/fixture-development-replay.mjs';

const code='import catlass.tla as tla\ndef run():\n return tla.compile(fn)()';
const input={records:[['public-task','cover all lanes'],['failing-source',code],['observed-output','error: duplicated index'],['inherited-guidance','historical guidance'],['private-evaluator','secret verifier']].map(([reference,text])=>({reference,text}))};
const design={decision:'test',reason:'inspect the values beyond one tile',prompt:'cover lanes',starter:code,good:code,bad:code,checker:'check',diagnostic:'duplicated index',mathFailure:'tail mismatch'};
const outputs={starter:{exit:1,output:'duplicated index'},good:{exit:0,output:'VERIFIED_NATIVE_ROUTE 1 1'},bad:{exit:1,output:'VERIFIED_NATIVE_ROUTE 1 1\nAssertionError: tail mismatch'}};

test('host-only proposed controls are rejected and failed wrong-control must exercise native route',()=>{
 const packet=publicPacket(input);assert.throws(()=>validateDesign({...design,good:'def run(x): return x+1'},packet),/native/);
 assert.deepEqual(checkControls(design,outputs),[]);assert.equal(checkControls(design,{...outputs,bad:{exit:1,output:'AssertionError: tail mismatch'}})[0].role,'bad');
});
test('paired public input stays identical; restore completed scores and apply reviewer feedback once',async t=>{
 const root=await mkdtemp(join(tmpdir(),'fixture-dev-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const probe=join(root,'run.sh');await writeFile(probe,'test');const prompts={};
 for(const arm of ['control','designer']){
  const launches=[],sessions=[];let interrupted=false;
  const args={config:{},input,arm,dir:join(root,arm),probe,device:0,sdk:root,runtime:'test',seconds:20};
  const skill=join(root,arm+'-skill.md');await writeFile(skill,'skill bytes');
  const deps={designerSkill:skill,score:async({role,dir})=>{if(role==='good'&&!interrupted){interrupted=true;throw Error('infrastructure');}launches.push(dir);return outputs[role];},
   session:async args=>{assert(!JSON.stringify(args.prompt).includes('secret verifier'));sessions.push(args);
    if(args.directory.endsWith('/designer'))return {value:args.validate(design)};
    return {value:args.validate({verdict:sessions.length===2?'revise':'approve',reason:'protect tail values'})};}};
  await assert.rejects(runFixtureDesign(args,deps),/infrastructure/);
  assert.equal((await runFixtureDesign(args,deps)).status,'qualified');
  assert.equal((await runFixtureDesign(args,deps)).status,'qualified');
  assert.equal(new Set(launches).size,6);assert.equal(launches.length,6);assert.equal(sessions.length,4);
  assert.equal(sessions[0].fixtureDesigner,arm==='designer');assert.equal(sessions[1].fixtureDesigner,undefined);
  assert.deepEqual(sessions[2].prompt.feedback.review,{verdict:'revise',reason:'protect tail values'});assert.deepEqual(sessions[2].prompt.feedback.previousFixture,design);assert.deepEqual(sessions[2].prompt.feedback.measuredControls.bad,outputs.bad);prompts[arm]=sessions[0].prompt;
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
 const check=limit=>'import catlass.tla as tla\na=[0]*129\ndef fill():\n for i in range('+limit+'): a[i]=1\ntla.compile(fill)()\nassert a==[1]*129,"tail mismatch"';
 const good=await launch(check(129)),bad=await launch(check(128)),host=await launch('pass');
 assert.equal(good.status,0);assert.match(good.stdout,/VERIFIED_NATIVE_ROUTE 1 1/);assert.equal(bad.status,1);assert.match(bad.stderr,/tail mismatch/);assert.equal(host.status,1);assert.match(host.stderr,/no native Catlass launch/);
});
