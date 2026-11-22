import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {homedir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {nativeSample,check} from '../../cli/skill-microeval.mjs';
import {scriptedReview} from './helpers/scripted-review.mjs';
const quote=text=>"'"+text.replaceAll("'","'\\''")+"'";
async function fixture(t){
 // Outside /tmp so the old blanket /tmp mask cannot accidentally satisfy this test.
 const cache=join(homedir(),'.cache','rsi-pilot-sandbox-tests');await mkdir(cache,{recursive:true});
 const root=await mkdtemp(join(cache,'case-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const privateRoot=join(root,'private'),pool=join(root,'public'),cwd=join(pool,'current'),sibling=join(pool,'other');
 for(const path of [privateRoot,cwd,sibling])await mkdir(path,{recursive:true});
 const sentinel=join(privateRoot,'sentinel.txt'),siblingFile=join(sibling,'kernel.py'),oracle=join(privateRoot,'oracle.py');
 await writeFile(sentinel,'supervisor-secret\n');await writeFile(siblingFile,'sibling-secret\n');await writeFile(join(cwd,'kernel.py'),'def answer(): return 0\n');
 const trusted="import sys,runpy\nfrom pathlib import Path\nm=runpy.run_path(str(Path(sys.argv[1])/'kernel.py'))\nassert m['answer']()==sum(6 for _ in range(7))\n";
 await writeFile(oracle,trusted);await writeFile(join(cwd,'public_check.py'),"from kernel import answer\nassert answer()==42\n");
 return {root,privateRoot,pool,cwd,sentinel,siblingFile,oracle,trusted};
}
test('native Pi masked Bash cannot read supervisor or sibling data while own edits persist',async t=>{
 const f=await fixture(t),base=['--die-with-parent','--new-session','--unshare-pid','--ro-bind','/','/','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--bind',f.cwd,f.cwd,'--chdir',f.cwd,'--','bash','-c'];
 const probe=spawnSync('bwrap',[...base,'true'],{encoding:'utf8',timeout:5000});
 if(probe.status!==0){t.skip('Bubblewrap namespaces unavailable: '+(probe.stderr??probe.error?.message));return;}
 // This visibility control must pass before the new API is exercised.
 const visible=spawnSync('bwrap',[...base,'test -r '+quote(f.sentinel)+' && test -r '+quote(f.siblingFile)],{encoding:'utf8',timeout:5000});assert.equal(visible.status,0,visible.stderr);
 const commands={legacy:'cat '+quote(f.sentinel)+' '+quote(f.siblingFile)+' > visibility-control.txt',masked:'if cat '+quote(f.sentinel)+' >/dev/null 2>&1; then exit 17; fi; if cat '+quote(f.siblingFile)+' >/dev/null 2>&1; then exit 18; fi; printf \'def answer(): return 42\\n\' > kernel.py; python public_check.py'};
 const provider=await scriptedReview(t,f.root,body=>{const raw=body.messages.find(m=>m.role==='user').content,prompt=typeof raw==='string'?raw:raw.map(part=>part.text??'').join('\n');return body.messages.some(m=>m.role==='tool')?null:{name:'bash',args:{command:commands[prompt.includes('fixture-masked')?'masked':'legacy']}};});
 const common={cwd:f.cwd,profile:provider.profile,seconds:10,guidance:'fixture'};
 const legacy=await nativeSample({...common,prompt:'fixture-legacy',sessionDirectory:join(f.root,'legacy-session'),localFilesystemSandbox:true});assert.equal(legacy.error,null);assert.equal(await readFile(join(f.cwd,'visibility-control.txt'),'utf8'),'supervisor-secret\nsibling-secret\n');
 const actor=await nativeSample({...common,prompt:'fixture-masked',sessionDirectory:join(f.root,'masked-session'),localFilesystemSandbox:{hiddenPaths:[f.privateRoot,f.pool]}});
 assert.equal(actor.error,null);assert.equal(actor.timedOut,false);assert.equal(actor.toolCalls,1);
 const verified=await check(process.env.DRIVER_PYTHON??'python3',f.oracle,f.cwd,5);assert.equal(verified.correct,true,JSON.stringify(verified));
 assert.equal(await readFile(f.oracle,'utf8'),f.trusted);assert.equal(await readFile(f.sentinel,'utf8'),'supervisor-secret\n');assert.equal(await readFile(f.siblingFile,'utf8'),'sibling-secret\n');
});
test('hidden-path validation fails before any provider or target dispatch',async t=>{
 const f=await fixture(t),common={cwd:f.cwd,sessionDirectory:join(f.root,'session'),profile:{},seconds:5,guidance:'fixture',prompt:'fixture'};
 for(const hiddenPaths of [['relative'],[f.sentinel],['/']])await assert.rejects(nativeSample({...common,localFilesystemSandbox:{hiddenPaths}}),/absolute directories|non-root directories/);
 await assert.rejects(nativeSample({...common,localFilesystemSandbox:{hiddenPaths:[join(f.cwd,'missing')]}}),{code:'ENOENT'});
 const inside=join(f.cwd,'private');await mkdir(inside);await assert.rejects(nativeSample({...common,localFilesystemSandbox:{hiddenPaths:[inside]}}),/inside the actor workspace/);
 let dispatches=0;await assert.rejects(nativeSample({...common,targetBash:true,targetEnvironment:{execute:async()=>{dispatches++;}},localFilesystemSandbox:{hiddenPaths:[f.privateRoot]}}),/cannot protect target Bash/);assert.equal(dispatches,0);
});
