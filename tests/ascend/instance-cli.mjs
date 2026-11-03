import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';
import { bootstrap, readConfig, updateConfig } from '../../cli/instance.mjs';
import { main } from '../../cli/main.mjs';

test('bootstrap snapshots own memory, skills and machinery; CLI changes persist independently',async()=>{
  const root=await mkdtemp(join(tmpdir(),'ascend-instance-'));
  try {
    const baseline=join(root,'baseline.sqlite'),db=new DatabaseSync(baseline);
    db.exec('PRAGMA journal_mode=WAL; CREATE TABLE knowledge (text TEXT); INSERT INTO knowledge VALUES (\'baseline\')');
    const skills=join(root,'baseline-skills');await mkdir(skills);
    await writeFile(join(skills,'skill-map.json'),'{}');
    const machinery=join(root,'release');await mkdir(machinery);await mkdir(join(machinery,'controller'));
    await writeFile(join(machinery,'controller','marker.mjs'),'export const marker=1');
    const instance=join(root,'instance');
    await bootstrap(instance,{baseline,skills,machinery,environment:{module:'./local.mjs',options:{mode:'local'}}});
    const memory=new DatabaseSync(join(instance,'knowledge.sqlite'));
    assert.equal(memory.prepare('SELECT text FROM knowledge').get().text,'baseline');memory.close();
    db.exec("UPDATE knowledge SET text='changed'");db.close();
    assert.equal((await readConfig(instance)).skills.activeVersion,'baseline');
    await updateConfig(instance,c=>({...c,mode:'phase2',concurrency:2}));
    assert.equal((await main(['config','--instance',instance])).mode,'phase2');
    assert.equal(await readFile(join(instance,'machinery','controller','marker.mjs'),'utf8'),'export const marker=1');
    const spec=join(root,'project.json');await writeFile(spec,JSON.stringify({id:'python',goal:'numeric',actor_seconds:240}));
    await main(['project','--instance',instance,'--project',spec]);
    const status=await main(['status','--instance',instance]);
    assert.equal(status.projects[0].status,'queued');
    assert.equal(status.projects[0].id,'python');
    await writeFile(spec,JSON.stringify({id:'python',goal:'edited numeric goal',actor_seconds:240}));
    await main(['project','--instance',instance,'--project',spec]);
    assert.equal((await readConfig(instance)).projects.length,1);
    assert.ok((await main(['browse','--instance',instance])).entries.includes('machinery'));
    await assert.rejects(bootstrap(join(machinery,'nested'),{baseline,skills,machinery,environment:{}}));
  }finally{await rm(root,{recursive:true,force:true});}
});
