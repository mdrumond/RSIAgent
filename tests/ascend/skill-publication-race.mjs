import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm,access} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {tmpdir} from 'node:os';
import {spawn} from 'node:child_process';
import {State} from '../../controller/state.mjs';
import {publishSkillVersion} from '../../cli/skills.mjs';

const until=async check=>{const end=Date.now()+10000;while(Date.now()<end){if(await check())return;await new Promise(r=>setTimeout(r,10));}throw Error('publication race deadline');};
test('promotion waits for concurrent curriculum config commit and preserves its new projects',async t=>{
  const root=await mkdtemp(join(tmpdir(),'skill-publication-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const state=new State(join(root,'state.sqlite'));state.close();
  await writeFile(join(root,'instance.json'),JSON.stringify({format:1,revision:0,mode:'phase1',projects:[],skills:{activeVersion:'baseline'}}));
  const start=script=>{
    const child=spawn(process.execPath,['--input-type=module','-e',script],{stdio:['ignore','pipe','pipe']});
    let stderr='';child.stderr.on('data',data=>stderr+=data);
    const done=new Promise((resolve,reject)=>{child.on('error',reject);child.on('close',code=>code?reject(Error(stderr)):resolve());});
    t.after(()=>{if(child.exitCode===null)child.kill('SIGTERM');});
    return done;
  };
  const locked=join(root,'locked'),release=join(root,'release');
  const writer=start(`import {updateConfig} from ${JSON.stringify(pathToFileURL(resolve('cli/instance.mjs')).href)};
import {writeFile,access} from 'node:fs/promises';
await updateConfig(${JSON.stringify(root)},async current=>{
 await writeFile(${JSON.stringify(locked)},'locked');
 while(true){try{await access(${JSON.stringify(release)});break;}catch{}await new Promise(r=>setTimeout(r,10));}
 return {...current,projects:[...current.projects,{id:'new-project',goal:'New curriculum original',actor_seconds:60}]};
});`);
  await until(async()=>{try{await access(locked);return true;}catch{return false;}});
  const promoted=start(`import {publishSkillVersion} from ${JSON.stringify(pathToFileURL(resolve('cli/skills.mjs')).href)};
const result=await publishSkillVersion(${JSON.stringify(root)},{active:'baseline',candidate:'candidate'});if(!result)throw Error('not promoted');`);
  await writeFile(release,'release');await Promise.all([writer,promoted]);
  const config=JSON.parse(await readFile(join(root,'instance.json'),'utf8'));
  assert.equal(config.skills.activeVersion,'candidate');assert.equal(config.projects[0].id,'new-project');assert.equal(config.revision,2);
  assert.equal(await publishSkillVersion(root,{active:'baseline',candidate:'stale'}),null);
  assert.deepEqual(JSON.parse(await readFile(join(root,'instance.json'),'utf8')),config);
});
