import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {setup,rows,until} from './driver_setup.mjs';

test('public campaign admits a fully imported original before the next curriculum draft finishes',{timeout:90000},async t=>{
  let starts=0,release;
  const held=new Promise(resolve=>{release=resolve;});
  t.after(()=>release());
  const f=await setup(t,'phase2',['supplied'],{providerOptions:{resolveCall:async(call,body)=>{
    if(call?.name==='mcp__kdb__search'&&body.tools.some(tool=>tool.function.name==='submit_project')) {
      if(++starts===2)await held;
    }
    return call;
  }}});
  const input=join(f.root,'streaming-config.json');
  await writeFile(input,JSON.stringify({campaign_plan:[{id:'generated',phase:'phase1',new_projects:2,
    generation:{topics:['arithmetic','layout']}}],
    curriculum:{phase1:[],generation:{enabled:true,topics:['arithmetic','layout']}}}));
  await f.run('config','--config',input);
  const running=f.start('campaign');
  try {
    await until(()=>starts===2,30000);
    await until(()=>rows(f.root,"SELECT id FROM attempts WHERE project LIKE 'generated-%'").length===1,5000);
    const [project]=rows(f.root,"SELECT spec FROM projects WHERE id LIKE 'generated-%'");
    const spec=JSON.parse(project.spec);
    assert.ok(spec.source.startsWith(join(f.root,'projects',spec.id)));
    assert.deepEqual(spec.generation.checks.map(check=>check.exitCode),[0,1,0]);
    const draft=JSON.parse(rows(f.root,"SELECT data FROM generation_drafts WHERE id='"+spec.id+"'")[0].data);
    assert.equal(draft.status,'imported');assert.ok(draft.elapsed>0);assert.ok(draft.spec.bundle.seconds<draft.seconds);
    const config=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
    assert.equal(config.projects.filter(project=>project.id===spec.id).length,1);
    assert.equal(config.curriculum.phase1.filter(id=>id===spec.id).length,1);
    release();
    await running.output;
    assert.equal(rows(f.root,"SELECT COUNT(*) n FROM attempts WHERE project LIKE 'generated-%'")[0].n,2);
    assert.equal(rows(f.root,"SELECT COUNT(*) n FROM events WHERE type='generation-error'")[0].n,0);
    f.pass();
  }finally{release();await f.run('stop');await running.output;}
});
