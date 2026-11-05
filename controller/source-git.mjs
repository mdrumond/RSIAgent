import {spawn} from 'node:child_process';
import {mkdir,mkdtemp,readFile,writeFile,realpath,rm} from 'node:fs/promises';
import {resolve,relative,isAbsolute,join} from 'node:path';
import {createHash} from 'node:crypto';
import {Type} from 'typebox';
import {StringDecoder} from 'node:string_decoder';

const MAX_OUTPUT=64*1024;
const COMMANDS=new Set(['status','diff','log','show','add','commit','branch','switch',
  'checkout','restore','reset','rm','mv','tag','rev-parse','ls-files','show-ref']);
// Git metadata never enters source snapshots. Also keep incidental runtime assets
// out of an actor's ordinary `git add .`; project .gitignore remains effective.
const EXCLUDES=`
.git/
node_modules/
deps/
dependencies/
.conda/
conda/
.runtime/
runtime/
.tox/
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.npm/
.cache/
.cache-*/
build/
dist/
artifacts/
*.o
*.so
*.a
*.bin
*.whl
*.dll
*.dylib
*.bc
*.safetensors
*.onnx
*.pt
*.pth
.env
.env.*
*.pem
*.key
credentials*.json
`;
const within=(parent,child)=>{const r=relative(parent,child);return !r||(!r.startsWith('../')&&r!=='..'&&!isAbsolute(r));};

function environment() {
  const env=Object.fromEntries(Object.entries(process.env).filter(([key])=>!key.startsWith('GIT_')));
  return {...env,GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:'/dev/null',
    GIT_TERMINAL_PROMPT:'0',GIT_EDITOR:'true',GIT_SEQUENCE_EDITOR:'true',
    GIT_PAGER:'cat',GIT_OPTIONAL_LOCKS:'0'};
}

// No shell, pager, global configuration, transport, or user-installed hooks.
function command(source,gitDir,argv,{timeout=30000,signal,indexFile}={}) {
  if(!Number.isFinite(timeout)||timeout<=0)throw new Error('Git deadline expired');
  signal?.throwIfAborted();
  return new Promise((resolveResult,reject)=>{
    const location=argv[0]==='init'?['--git-dir='+gitDir]:['--git-dir='+gitDir,'--work-tree='+source];
    const child=spawn('git',[...location,
      '-c','core.hooksPath=/dev/null','-c','core.fsmonitor=false',
      '-c','core.attributesFile=/dev/null','-c','diff.external=',
      '-c','protocol.allow=never','-c','commit.gpgSign=false','-c','tag.gpgSign=false',
      '-c','user.name=RSI Actor','-c','user.email=actor@rsi.local',...argv],
    {cwd:source,env:{...environment(),...(indexFile?{GIT_INDEX_FILE:indexFile}:{})},stdio:['ignore','pipe','pipe']});
    let size=0,truncated=false,timedOut=false,aborted=false;
    const stdout=[],stderr=[];
    const collect=target=>chunk=>{
      const keep=Math.max(0,MAX_OUTPUT-size);
      if(chunk.length>keep)truncated=true;
      if(keep)target.push(chunk.subarray(0,keep));
      size+=Math.min(keep,chunk.length);
    };
    child.stdout.on('data',collect(stdout));child.stderr.on('data',collect(stderr));
    const abort=()=>{aborted=true;child.kill('SIGKILL');};
    signal?.addEventListener('abort',abort,{once:true});
    const timer=setTimeout(()=>{timedOut=true;child.kill('SIGKILL');},Math.min(timeout,2147483647));
    child.once('error',error=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);reject(error);});
    child.once('close',(exitCode,exitSignal)=>{
      clearTimeout(timer);signal?.removeEventListener('abort',abort);
      resolveResult({exitCode,signal:exitSignal,stdout:new StringDecoder('utf8').write(Buffer.concat(stdout)),
        stderr:new StringDecoder('utf8').write(Buffer.concat(stderr)),truncated,timedOut,aborted});
    });
  });
}

/** Prepare outside actor time (caller excludes setup). Actor execute calls are
 * normal elapsed work. source must be the ledger's authoritative LOCAL source,
 * never an Ascend target path. The caller owns project gitDir selection. */
export async function prepareSourceGit({source,gitDir,attemptId}) {
  source=await realpath(source);
  gitDir=resolve(gitDir);
  if(within(source,gitDir)||within(gitDir,source))throw new Error('Git metadata and source must be separate trees');
  await mkdir(gitDir,{recursive:true});gitDir=await realpath(gitDir);
  if(within(source,gitDir)||within(gitDir,source))throw new Error('Git metadata and source must be separate trees');
  if(typeof attemptId!=='string'||!attemptId)throw new Error('attemptId is required');
  const run=(argv,options)=>command(source,gitDir,argv,options);
  const must=async argv=>{
    const response=await run(argv);
    if(response.exitCode!==0)throw new Error(`Source Git setup failed: ${response.stderr}`);
    return response.stdout.trim();
  };
  const binding=join(gitDir,'rsi-source.json');
  let identity;
  try {identity=JSON.parse(await readFile(binding,'utf8'));}
  catch(error) {if(error.code!=='ENOENT')throw error;}
  if(identity&&identity.source!==source)throw new Error('Git directory belongs to a different source');
  if(!identity) {
    // Empty template prevents host hooks/templates from contaminating the repo.
    await must(['init','--bare','--template=']);
    await writeFile(binding,JSON.stringify({source})+'\n');
  }
  await must(['config','core.bare','false']);
  await must(['config','core.logAllRefUpdates','true']);
  await mkdir(join(gitDir,'info'),{recursive:true});
  await writeFile(join(gitDir,'info','exclude'),EXCLUDES);
  const branch='attempt/'+createHash('sha256').update(attemptId).digest('hex').slice(0,24);
  const current=await run(['symbolic-ref','--quiet','HEAD']);
  const existing=await run(['show-ref','--verify','--quiet','refs/heads/'+branch]);
  if(current.stdout.trim()!=='refs/heads/'+branch&&existing.exitCode!==0) {
    const head=await run(['rev-parse','--verify','HEAD']);
    if(head.exitCode===0)await must(['branch',branch,'HEAD']);
    await must(['symbolic-ref','HEAD','refs/heads/'+branch]);
  }
  // An existing attempt link is retained even when the actor selected a different
  // branch or detached HEAD. Preparation never rewinds files or changes history.
  const execute=async(argv,options={})=>{
    if(!Array.isArray(argv)||!argv.length||argv.some(a=>typeof a!=='string'||a.includes('\0')))
      throw new Error('Git argv must be a nonempty string array');
    if(!COMMANDS.has(argv[0]))throw new Error('Only local source history commands are supported');
    if(argv.some(a=>['--git-dir','--work-tree','--config-env','--exec-path','--namespace'].some(flag=>a===flag||a.startsWith(flag+'='))))
      throw new Error('Git repository overrides are not supported');
    if(argv[0]==='diff'||argv[0]==='show'||argv[0]==='log')
      argv=[argv[0],'--no-ext-diff','--no-textconv',...argv.slice(1)];
    return run(argv,options);
  };
  const readState=async(options={})=>{
    const head=await run(['rev-parse','--verify','HEAD'],options);
    const status=await run(['status','--porcelain=v1','--untracked-files=normal'],options);
    const ref=await run(['symbolic-ref','--quiet','--short','HEAD'],options);
    if(status.exitCode!==0||[head,status,ref].some(r=>r.timedOut||r.aborted))
      throw new Error('Cannot read source Git status');
    return {source,gitDir,attemptId,attemptBranch:branch,
      head:head.exitCode===0?head.stdout.trim():null,
      branch:ref.exitCode===0?ref.stdout.trim():null,dirty:Boolean(status.stdout),
      status:status.stdout,statusTruncated:status.truncated};
  };
  // Boundary evidence is a tree object, not an actor commit. A fresh index
  // captures current source even when the actor's own index is partially staged.
  const captureTree=async(options={})=>{
    const temporary=await mkdtemp(join(gitDir,'capture-'));
    try {
      const captureOptions={...options,indexFile:join(temporary,'index')};
      for(const argv of [['add','-A'],['write-tree']]) {
        const response=await run(argv,captureOptions);
        if(response.exitCode!==0||response.timedOut||response.aborted)
          throw new Error(`Source Git capture failed: ${response.stderr}`);
        if(argv[0]==='write-tree') {
          const {head,dirty}=await readState(options);
          return {tree_oid:response.stdout.trim(),head,dirty};
        }
      }
    } finally {await rm(temporary,{recursive:true,force:true});}
  };
  const extension=async pi=>pi.registerTool({name:'rsi_git',label:'Local source Git',
    description:'Run local Git argv in authoritative controller-local source. Advisory history: inspect status/diff and commit meaningful changes. No remotes. Remote Bash is a different filesystem.',
    parameters:Type.Object({argv:Type.Array(Type.String(),{minItems:1})}),
    execute:async(_id,{argv},signal)=>{
      const response=await execute(argv,{signal});
      return {content:[{type:'text',text:JSON.stringify(response)}],details:response,
        isError:response.exitCode!==0};
    }});
  return {extension,readState,captureTree,execute,source,gitDir,attemptBranch:branch};
}
