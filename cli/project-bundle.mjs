import {readdir,lstat,realpath} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {dirname,join,relative,resolve,sep,isAbsolute} from 'node:path';
import {freezeFiles} from '../controller/evidence.mjs';
import {createEnvironment} from '../integrations/environments/index.mjs';

const inside=(root,path)=>path===root||path.startsWith(root+sep);
const ignored=new Set(['__pycache__','.pytest_cache','.git','.venv','node_modules']);

// Enumerate only explicitly declared trusted fixture trees, never their parent.
async function trustedFiles(directories,{exclude}={}) {
  const files=[];
  async function visit(path) {
    if(path===exclude)return;
    const info=await lstat(path);
    if(info.isSymbolicLink())throw new Error('Trusted bundle dependencies must be regular files: '+path);
    if(info.isDirectory()) {
      for(const name of (await readdir(path)).sort())if(!ignored.has(name))await visit(join(path,name));
    }else if(info.isFile())files.push(path);
    else throw new Error('Invalid trusted bundle dependency: '+path);
  }
  for(const directory of directories)await visit(directory);
  return files;
}

// Fixture bundles can contain tensor data; hash sequentially without retaining
// all dependency contents in the controller at once.
export async function bundleHashes(files) {
  const hashes={};
  for(const file of files) {
    const hash=createHash('sha256');for await(const chunk of createReadStream(file))hash.update(chunk);
    hashes[file]=hash.digest('hex');
  }
  return hashes;
}

export async function projectBundle(artifacts,command,seconds,generationRoot) {
  let root=artifacts.source;
  for(const path of [artifacts.reference,...artifacts.negatives,...artifacts.files])
    while(!inside(root,path))root=dirname(root);
  // A coherent project subdirectory can include verifier helper packages and
  // alternate reference/control fixtures. Never enumerate the generation session
  // root (which also owns transcripts and execution journals).
  const directories=root!==generationRoot&&inside(generationRoot,root)?[root]:[artifacts.reference,...artifacts.negatives];
  const files=[...new Set([...artifacts.files,...await trustedFiles(directories,{exclude:artifacts.source})])];
  return {files,bundle:{version:1,root,reference:artifacts.reference,negative_controls:artifacts.negatives,
    construction_command:command,seconds}};
}

export async function checkBundle(spec) {
  const bundle=spec.bundle;
  if(!bundle)return;
  if(bundle.version!==1||!isAbsolute(bundle.root)||!Array.isArray(bundle.negative_controls)||
      !bundle.negative_controls.length||bundle.negative_controls.length>8||
      typeof bundle.construction_command!=='string'||!bundle.construction_command.trim()||
      !Number.isFinite(bundle.seconds)||bundle.seconds<=0)
    throw new Error('Invalid project bundle contract');
  if(bundle.baseline_construction_command!==undefined&&(typeof bundle.baseline_construction_command!=='string'||!bundle.baseline_construction_command.trim()))
    throw new Error('Invalid CPU baseline selftest command');
  const root=await realpath(bundle.root),source=await realpath(spec.source);
  const fixtures=await Promise.all([bundle.reference,...bundle.negative_controls].map(path=>realpath(path)));
  if(new Set(fixtures).size!==fixtures.length)throw new Error('Project bundle reference and controls must be distinct');
  const paths=[source,...fixtures,...await Promise.all(spec.verifier.files.map(path=>realpath(path)))];
  if(paths.some(path=>!inside(root,path)))throw new Error('Project bundle dependency escapes its root');
  if(fixtures.some((path,index)=>[source,...fixtures.filter((_,i)=>i!==index)].some(other=>inside(path,other)||inside(other,path))))
    throw new Error('Project bundle source, reference and controls must be separate trees');
  const declared=new Set(spec.verifier.files.map(path=>resolve(path)));
  for(const file of await trustedFiles(fixtures))if(!declared.has(file))
    throw new Error('Trusted bundle dependency is not frozen: '+relative(root,file));
}

// One elapsed allocation covers import I/O and CPU proof, including failure and
// interruption before the first command can start.
export async function withBundleBudget(spec,{signal,checkpoint=()=>{}}={},action) {
  if(!spec.bundle)return action({signal});
  const allocation=spec.bundle.seconds;
  if(!Number.isFinite(allocation)||allocation<=0)throw new Error('Project bundle validation allocation exhausted');
  const bounded=AbortSignal.timeout(Math.max(1,Math.ceil(allocation*1000)));
  const active=signal?AbortSignal.any([signal,bounded]):bounded,started=performance.now();
  const remaining=()=>Math.max(0,allocation-(performance.now()-started)/1000);
  const save=()=>{spec.bundle={...spec.bundle,seconds:remaining()};checkpoint(spec.bundle.seconds);};
  const timer=setInterval(save,250);let result;
  try {active.throwIfAborted();result=await action({signal:active,remaining});active.throwIfAborted();return result;}
  finally{clearInterval(timer);save();if(result?.bundle)result.bundle.seconds=spec.bundle.seconds;}
}

// Original-tree construction evidence cannot prove the imported dependency closure.
export async function validateBundle(spec,instancePath,options={}) {
  if(!spec.bundle)return;
  if(!options.budget)return withBundleBudget(spec,options,budget=>validateBundle(spec,instancePath,{...options,budget}));
  const {signal:active,remaining}=options.budget;
  active.throwIfAborted();await checkBundle(spec);
  const unchanged=await freezeFiles(spec.verifier.files,spec.source,spec.verifier.hashes);
  active.throwIfAborted();
  const environment=await createEnvironment({mode:'local',localExecution:options.localExecution},{instancePath,signal:active,
    exclude:async(_label,fn)=>fn(),event:()=>{}});
  const cases=[{kind:'reference',cwd:spec.bundle.reference,expected:0},
    ...spec.bundle.negative_controls.map(cwd=>({kind:'negative control',cwd,expected:1})),
    ...(spec.bundle.validate_starter===true?[{kind:'starter',cwd:spec.source,expected:0}]:[]),
    ...(spec.bundle.baseline_construction_command?[{kind:'baseline',cwd:spec.source,expected:0,command:spec.bundle.baseline_construction_command}]:[])];
  for(const {kind,cwd,expected,command=spec.bundle.construction_command} of cases) {
    active.throwIfAborted();await unchanged();active.throwIfAborted();
    const result=await environment.execute(command,cwd,{signal:active,timeout:Math.max(.001,remaining())});
    active.throwIfAborted();await unchanged();
    if(!result.settled||result.exitCode!==expected)
      throw new Error('Imported bundle CPU '+kind+' failed: '+JSON.stringify({cwd,...result}));
  }
}
