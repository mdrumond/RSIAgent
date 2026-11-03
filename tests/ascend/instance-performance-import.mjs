import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,mkdir,writeFile,rm,stat,cp } from 'node:fs/promises';
import { join,relative } from 'node:path';
import { tmpdir } from 'node:os';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { importProject } from '../../cli/instance.mjs';

for(const mode of ['absolute','relative'])test(`performance project import: ${mode}`,async()=>{
  const root=await mkdtemp(join(tmpdir(),'ascend-import-performance-'));
  try {
    const original=join(root,'original'),source=join(original,'source'),checks=join(original,'checks');
    await mkdir(source,{recursive:true});await mkdir(checks);
    await writeFile(join(source,'answer.py'),'answer=42\n');
    const files=[join(checks,'numeric.py'),join(checks,'performance.py')];
    for(const file of files)await writeFile(file,"import runpy,sys\nvalue=runpy.run_path(sys.argv[1])['answer']\nassert value==42\nprint(value)\n");
    const command=index=>mode==='absolute'
      ? `python3.12 '${files[index]}' '${join(source,'answer.py')}'`
      : `python3.12 ../checks/${index?'performance':'numeric'}.py answer.py`;
    const spec={id:'p',source,goal:'correct answer',actor_seconds:30,performance_goal:'trusted frozen check',
      verifier:{files,command:command(0),performance_command:command(1)}};
    const instance=join(root,'instance');
    await assert.rejects(importProject(instance,{...spec,verifier:{...spec.verifier,performance_command:undefined}}),/performance_command/);
    await assert.rejects(stat(join(instance,'projects','p')),{code:'ENOENT'});
    const imported=await importProject(instance,spec);
    assert.equal(imported.performance_goal,spec.performance_goal);
    assert.equal(imported.goal,spec.goal);
    await rm(original,{recursive:true});
    const relocated=join(root,'relocated');await cp(instance,relocated,{recursive:true});
    await rm(instance,{recursive:true});
    const relocatedSource=join(relocated,relative(instance,imported.source));
    for(const check of ['command','performance_command']) {
      const result=await promisify(execFile)('bash',['-c',imported.verifier[check]],{cwd:relocatedSource});
      assert.equal(Number(result.stdout.trim()),42);
    }
  }finally{await rm(root,{recursive:true,force:true});}
});
