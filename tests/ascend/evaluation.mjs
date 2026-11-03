import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,mkdir,writeFile,readFile,rm,cp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { DatabaseSync } from 'node:sqlite';
import { bootstrap as instance,readConfig } from '../../cli/instance.mjs';
import { bootstrap as sample } from '../../cli/evaluation.mjs';
import { stages } from '../../controller/workflow.mjs';
import { State } from '../../controller/state.mjs';

test('A/B samples freeze memory and source once, relocate both checks and block unknown prior work',async()=>{
  const root=await mkdtemp(join(tmpdir(),'ascend-evaluation-'));
  try {
    const baseline=join(root,'baseline.sqlite'),db=new DatabaseSync(baseline);
    db.exec("CREATE TABLE knowledge(value); INSERT INTO knowledge VALUES ('baseline')");db.close();
    const skills=join(root,'skills');await mkdir(skills);
    const map=Object.fromEntries([...stages,'global-kdb'].map(stage=>[stage,stage+'.md']));
    for(const file of Object.values(map))await writeFile(join(skills,file),'Retain independent correctness evidence.');
    await writeFile(join(skills,'skill-map.json'),JSON.stringify(map));
    const source=join(root,'source');await mkdir(source);await writeFile(join(source,'answer.py'),'value=42\n');
    const verifier=join(root,'verify.py'),performance=join(root,'performance.py');
    await writeFile(verifier,'from answer import value\nassert value==42\n');
    await writeFile(performance,'from answer import value\nassert value>40\n');
    const python=process.env.DRIVER_PYTHON??'python3.12';
    const quote=value=>"'"+value.replaceAll("'","'\\''")+"'";
    const project={id:'p',goal:'answer',source,actor_seconds:120,performance_goal:'frozen threshold',
      verifier:{files:[verifier,performance],command:`PYTHONPATH=. ${quote(python)} ${quote(verifier)}`,
        performance_command:`PYTHONPATH=. ${quote(python)} ${quote(performance)}`}};
    const parent=join(root,'parent');
    await instance(parent,{baseline,skills,projects:[project],environment:{module:'machinery/integrations/environments/index.mjs',options:{mode:'local'}},install:false});
    const config=await readConfig(parent),round=join(parent,'round');await mkdir(round);
    await cp(join(parent,'skills/versions/baseline'),join(parent,'skills/versions/candidate'),{recursive:true});
    const request=arm=>({sourceInstanceRoot:parent,instanceRoot:join(round,arm),config,
      projectId:'p',projectIds:['p'],skillVersion:arm==='a'?'baseline':'candidate'});
    await sample(request('a'));
    const memory=new DatabaseSync(join(parent,'knowledge.sqlite'));
    memory.exec("INSERT INTO knowledge VALUES ('later lesson')");memory.close();
    await writeFile(join(config.projects[0].source,'answer.py'),'value=7\n');
    await sample(request('b'));
    for(const arm of ['a','b']) {
      const own=await readConfig(join(round,arm)),p=own.projects[0];
      assert.notEqual(p.source,config.projects[0].source);
      assert.equal(await readFile(join(p.source,'answer.py'),'utf8'),'value=42\n');
      for(const command of [p.verifier.command,p.verifier.performance_command])
        execFileSync('bash',['-c',command],{cwd:p.source,stdio:'pipe'});
      const isolated=new DatabaseSync(join(own.root,'knowledge.sqlite'));
      assert.equal(isolated.prepare('SELECT count(*) AS n FROM knowledge').get().n,1);isolated.close();
      assert.equal(own.skills.tuningDisabled,true);
    }
    const state=new State(join(round,'a','state.sqlite'));
    assert.ok(state.project('p'));state.admit('p',{});state.close();
    await assert.rejects(sample(request('c')),/unresolved evaluation instance/);
  }finally{await rm(root,{recursive:true,force:true});}
});
