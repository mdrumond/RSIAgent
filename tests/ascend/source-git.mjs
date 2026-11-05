import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,readdir,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {prepareSourceGit} from '../../controller/source-git.mjs';
import {fingerprint} from '../../controller/evidence.mjs';

async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'rsi-source-git-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const source=join(root,'source'),gitDir=join(root,'metadata','project.git');
  await mkdir(source);await writeFile(join(source,'kernel.py'),'answer = 42\n');
  return {root,source,gitDir};
}
async function ok(repo,argv,options) {
  const result=await repo.execute(argv,options);
  assert.equal(result.exitCode,0,result.stderr);
  return result.stdout;
}

test('real Git history is actor controlled and metadata never changes source fingerprint',async t=>{
  const paths=await fixture(t),before=await fingerprint(paths.source);
  const repo=await prepareSourceGit({...paths,attemptId:'first'});
  assert.equal(await fingerprint(paths.source),before);
  assert.deepEqual(await readdir(paths.source),['kernel.py']);
  let state=await repo.readState();
  assert.equal(state.head,null);assert.equal(state.dirty,true);
  assert.equal(state.branch,repo.attemptBranch);
  await ok(repo,['add','kernel.py']);
  await ok(repo,['commit','-m','Implement kernel']);
  state=await repo.readState();assert.match(state.head,/^[0-9a-f]{40,64}$/);
  assert.equal(state.dirty,false);
  assert.equal(await fingerprint(paths.source),before);
  await writeFile(join(paths.source,'kernel.py'),'answer = 43\n');
  const diff=await ok(repo,['diff']);
  assert.match(diff,/-answer = 42/);assert.match(diff,/\+answer = 43/);
  assert.equal((await repo.readState()).dirty,true);
  const same=await prepareSourceGit({...paths,attemptId:'first'});
  assert.equal((await same.readState()).head,state.head);
  assert.equal(await readFile(join(paths.source,'kernel.py'),'utf8'),'answer = 43\n');
  const next=await prepareSourceGit({...paths,attemptId:'second'});
  assert.notEqual(next.attemptBranch,repo.attemptBranch);
  assert.equal((await next.readState()).head,state.head);
  await ok(next,['add','.']);await ok(next,['commit','-m','Optimise kernel']);
  assert.notEqual((await next.readState()).head,state.head);
  assert.match(await ok(next,['log','--format=%s']),/Optimise kernel\nImplement kernel/);
  assert.equal((await next.readState()).dirty,false);
});

test('ordinary add excludes runtime, dependency, generated and credential assets',async t=>{
  const paths=await fixture(t);
  for(const path of ['node_modules','build','__pycache__','.venv','.cache','artifacts']) {
    await mkdir(join(paths.source,path));
    await writeFile(join(paths.source,path,'payload.py'),'generated');
  }
  for(const path of ['kernel.so','tensor.bin','.env','private.key'])
    await writeFile(join(paths.source,path),'not source');
  await writeFile(join(paths.source,'.gitignore'),'ignored.txt\n');
  await writeFile(join(paths.source,'ignored.txt'),'project exclusion');
  const repo=await prepareSourceGit({...paths,attemptId:'assets'});
  await ok(repo,['add','.']);
  assert.deepEqual((await ok(repo,['ls-files'])).trim().split('\n'),['.gitignore','kernel.py']);
  await ok(repo,['commit','-m','Source only']);
  assert.equal((await repo.readState()).dirty,false);
});

test('Pi extension executes actual local argv and reports Git failures',async t=>{
  const paths=await fixture(t),repo=await prepareSourceGit({...paths,attemptId:'tool'});
  let tool;await repo.extension({registerTool(value){tool=value;}});
  assert.equal(tool.name,'rsi_git');
  let result=await tool.execute('one',{argv:['status','--short']});
  assert.equal(result.isError,false);assert.match(JSON.parse(result.content[0].text).stdout,/kernel.py/);
  await tool.execute('two',{argv:['add','kernel.py']});
  result=await tool.execute('three',{argv:['commit','-m','Actor milestone']});
  assert.equal(result.isError,false);assert.ok((await repo.readState()).head);
  result=await tool.execute('four',{argv:['show','HEAD:missing.py']});
  assert.equal(result.isError,true);
  for(const argv of [['push'],['fetch'],['remote','add','origin','https://example.invalid'],
    ['config','credential.helper','store'],['-c','alias.x=!sh','x'],['status','--work-tree=/tmp']])
    await assert.rejects(repo.execute(argv));
});

test('output cap, deadline and cancellation bound actor subprocesses',async t=>{
  const paths=await fixture(t),repo=await prepareSourceGit({...paths,attemptId:'bounded'});
  await writeFile(join(paths.source,'large.txt'),'a\n'.repeat(1000000));
  await ok(repo,['add','.']);await ok(repo,['commit','-m','Large fixture']);
  const output=await repo.execute(['show','HEAD:large.txt']);
  assert.equal(output.truncated,true);
  assert.ok(Buffer.byteLength(output.stdout)+Buffer.byteLength(output.stderr)<=64*1024);
  assert.equal(output.exitCode,0);
  await assert.rejects(repo.execute(['status'],{timeout:0}),/deadline/);
  const abort=new AbortController();abort.abort();
  await assert.rejects(repo.execute(['status'],{signal:abort.signal}),{name:'AbortError'});
  // A real multi-megabyte diff outlasts the deadline without relying on process
  // startup racing a millisecond timer for an otherwise trivial Git command.
  await writeFile(join(paths.source,'large.txt'),'b\n'.repeat(1000000));
  const stopped=await repo.execute(['diff','--stat'],{timeout:1});
  assert.equal(stopped.timedOut,true);
});

test('metadata cannot be placed in source or rebound to another project',async t=>{
  const paths=await fixture(t);
  await assert.rejects(prepareSourceGit({...paths,gitDir:join(paths.source,'git'),attemptId:'one'}),/separate/);
  const other=join(paths.root,'other');await mkdir(other);
  await prepareSourceGit({...paths,attemptId:'one'});
  await assert.rejects(prepareSourceGit({...paths,source:other,attemptId:'one'}),/different source/);
});

test('resume preserves actor-selected branches and detached HEAD with dirty edits',async t=>{
  const paths=await fixture(t),repo=await prepareSourceGit({...paths,attemptId:'resume'});
  await ok(repo,['add','.']);await ok(repo,['commit','-m','Base']);
  const original=(await repo.readState()).head;
  await ok(repo,['switch','-c','experiment']);
  let resumed=await prepareSourceGit({...paths,attemptId:'resume'});
  assert.equal((await resumed.readState()).branch,'experiment');
  await ok(resumed,['checkout','--detach',original]);
  await writeFile(join(paths.source,'kernel.py'),'dirty = True\n');
  resumed=await prepareSourceGit({...paths,attemptId:'resume'});
  const state=await resumed.readState();
  assert.equal(state.branch,null);assert.equal(state.head,original);assert.equal(state.dirty,true);
  const next=await prepareSourceGit({...paths,attemptId:'from-detached'});
  assert.equal((await next.readState()).branch,next.attemptBranch);
  assert.equal(await readFile(join(paths.source,'kernel.py'),'utf8'),'dirty = True\n');
});

test('source repository metadata is left untouched and inherited Git overrides are ignored',async t=>{
  const paths=await fixture(t),dotgit=join(paths.source,'.git');
  await mkdir(dotgit);await writeFile(join(dotgit,'config'),'original project metadata');
  const original=process.env.GIT_DIR;
  process.env.GIT_DIR=join(paths.root,'must-not-be-used');
  try {
    const repo=await prepareSourceGit({...paths,attemptId:'isolated'});
    await ok(repo,['add','.']);await ok(repo,['commit','-m','Isolated history']);
    assert.deepEqual((await ok(repo,['ls-files'])).trim().split('\n'),['kernel.py']);
    assert.equal(await readFile(join(dotgit,'config'),'utf8'),'original project metadata');
    assert.ok((await repo.readState()).head);
  } finally {
    if(original===undefined)delete process.env.GIT_DIR;else process.env.GIT_DIR=original;
  }
});

test('boundary trees retain actual edits without changing actor HEAD, branch or index',async t=>{
  const paths=await fixture(t),repo=await prepareSourceGit({...paths,attemptId:'capture'});
  const unborn=await repo.captureTree();
  assert.equal(unborn.head,null);assert.equal(unborn.dirty,true);
  assert.equal(await ok(repo,['show',unborn.tree_oid+':kernel.py']),'answer = 42\n');
  await assert.rejects(readFile(join(paths.gitDir,'index')),{code:'ENOENT'});
  await ok(repo,['add','.']);await ok(repo,['commit','-m','Actor implementation']);
  await writeFile(join(paths.source,'kernel.py'),'staged = True\n');
  await ok(repo,['add','kernel.py']);
  await writeFile(join(paths.source,'kernel.py'),'failed_optimisation = True\n');
  await writeFile(join(paths.source,'new.py'),'untracked = True\n');
  await mkdir(join(paths.source,'build'));
  await writeFile(join(paths.source,'build','generated.py'),'excluded');
  const index=await readFile(join(paths.gitDir,'index')),state=await repo.readState();
  const fingerprintBefore=await fingerprint(paths.source);
  const first=await repo.captureTree();
  assert.equal(first.head,state.head);assert.equal(first.dirty,true);
  assert.notEqual(first.tree_oid,unborn.tree_oid);
  assert.equal(await ok(repo,['show',first.tree_oid+':kernel.py']),'failed_optimisation = True\n');
  assert.equal(await ok(repo,['show',first.tree_oid+':new.py']),'untracked = True\n');
  assert.notEqual((await repo.execute(['show',first.tree_oid+':build/generated.py'])).exitCode,0);
  assert.deepEqual(await readFile(join(paths.gitDir,'index')),index);
  assert.deepEqual(await repo.readState(),state);
  assert.equal(await fingerprint(paths.source),fingerprintBefore);
  await rm(join(paths.source,'new.py'));
  await writeFile(join(paths.source,'kernel.py'),'next_optimisation = True\n');
  const second=await repo.captureTree();
  assert.notEqual(second.tree_oid,first.tree_oid);
  assert.equal(second.head,state.head);
  assert.equal(await ok(repo,['show',second.tree_oid+':kernel.py']),'next_optimisation = True\n');
  assert.notEqual((await repo.execute(['show',second.tree_oid+':new.py'])).exitCode,0);
  assert.deepEqual(await readFile(join(paths.gitDir,'index')),index);
  assert.equal((await repo.readState()).branch,state.branch);
  assert.equal((await repo.readState()).head,state.head);
  assert.equal((await readdir(paths.gitDir)).some(name=>name.startsWith('capture-')),false);
  await assert.rejects(repo.captureTree({timeout:0}),/deadline/);
  assert.equal((await readdir(paths.gitDir)).some(name=>name.startsWith('capture-')),false);
});
