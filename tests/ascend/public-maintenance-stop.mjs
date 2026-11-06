import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {campaignFixture,waitFor} from './resilience/campaign-fixture.mjs';

async function inactive(pid) {
  try {
    const stat=await readFile('/proc/'+pid+'/stat','utf8');
    return ['Z','X'].includes(stat.slice(stat.lastIndexOf(')')+2).split(' ')[0]);
  }catch(error){if(error.code==='ENOENT'||error.code==='ESRCH')return true;throw error;}
}

test('public STOP cancels foreground maintenance and its subprocesses',{timeout:30000},async t=>{
  const directory=await mkdtemp(join(tmpdir(),'foreground-maintenance-')),cleanups=[];
  let passed=false;
  t.after(async()=>{
    for(const cleanup of cleanups.reverse())await cleanup({remainingSeconds:10});
    if(passed)await rm(directory,{recursive:true,force:true});else t.diagnostic(directory);
  });
  const f=await campaignFixture({directory,onCleanup:fn=>cleanups.push(fn)},'recover',['unrelated']);
  const config=JSON.parse(await readFile(join(f.instance,'instance.json'),'utf8'));
  const {stdout}=await promisify(execFile)(config.python,['-c','import sys;print(sys.executable)']);
  const python=stdout.trim(),wrapper=join(directory,'held-python'),action=join(directory,'action.json');
  await writeFile(wrapper,`#!${python}
import os,sys,time,subprocess
from pathlib import Path
if sys.argv[1:2]==['-m'] and sys.argv[2] in ('ascend_tools.kdb_client','ascend_tools.storage_cli'):
 kind='kdb' if sys.argv[2].endswith('kdb_client') else 'storage'
 Path(${JSON.stringify(directory)}+'/'+kind+'.pid').write_text(str(os.getpid()))
 code="import os,time; from pathlib import Path; Path("+repr(${JSON.stringify(directory)}+'/'+kind+'.child')+").write_text(str(os.getpid())); time.sleep(60)"
 subprocess.Popen([sys.executable,'-c',code])
 time.sleep(60)
os.execv(${JSON.stringify(python)},[${JSON.stringify(python)},*sys.argv[1:]])
`,{mode:0o700});
  await f.config({python:wrapper});
  for(const kind of ['kdb','storage']) {
    await writeFile(action,JSON.stringify(kind==='kdb'?{tool:'search',arguments:{query:'held'}}:{action:'compress'}));
    const held=f.start(kind,'--config',action);void held.output.catch(()=>{});
    await waitFor(async()=>{
      try{return Number(await readFile(join(directory,kind+'.child'),'utf8'));}
      catch(error){if(error.code==='ENOENT')return false;throw error;}
    });
    const [owner]=f.rows("SELECT * FROM requests WHERE kind=? AND status='running'",'state.sqlite',[kind]);
    assert.ok(owner?.owner);
    const client=Number(await readFile(join(directory,kind+'.pid'),'utf8'));
    const child=Number(await readFile(join(directory,kind+'.child'),'utf8'));
    assert.equal(await inactive(client),false);assert.equal(await inactive(child),false);
    await f.run('stop');await assert.rejects(held.output);
    await waitFor(async()=>await inactive(client)&&await inactive(child),undefined,5);
    assert.equal(f.rows('SELECT status FROM requests WHERE id=?','state.sqlite',[owner.id])[0].status,'failed');
  }
  await f.config({python});passed=true;
});
