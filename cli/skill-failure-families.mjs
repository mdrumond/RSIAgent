/** Protected replay families, derived from observed campaign failures before instruction writing. */
import {readFile,writeFile,mkdir,realpath,stat,rename,readdir} from 'node:fs/promises';
import {dirname,resolve,join,relative,isAbsolute} from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {DatabaseSync} from 'node:sqlite';
import {Type} from 'typebox';
import {reviewSession} from './skill-review-native.mjs';
import {freezeGroups} from './skill-backlog-analysis.mjs';

const canonical=x=>Array.isArray(x)?x.map(canonical):x&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,canonical(x[k])])):x;
const git=promisify(execFile),sha=x=>createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(canonical(x))).digest('hex');
const slug=x=>typeof x==='string'&&/^[a-z0-9_]+(?:-[a-z0-9_]+)*$/i.test(x),text=x=>typeof x==='string'&&x.trim(),hex=x=>/^[a-f0-9]{64}$/.test(x??'');
const inside=(root,path)=>{const p=relative(root,path);return !isAbsolute(p)&&p!=='..'&&!p.startsWith('../');};
const optional=async file=>{try{return JSON.parse(await readFile(file,'utf8'));}catch(e){if(e.code==='ENOENT')return null;throw e;}};
async function save(file,value){const tmp=file+'.tmp';await writeFile(tmp,JSON.stringify(value,null,2)+'\n');await rename(tmp,file);}
async function utf8(file){const info=await stat(file);if(!info.isFile()||info.size>1048576)throw Error('Family source must be a bounded UTF-8 file');const s=new TextDecoder('utf-8',{fatal:true}).decode(await readFile(file));if(s.includes('\0'))throw Error('Binary family source is unsupported');return s;}
async function scoped(root,name){if(!text(name)||isAbsolute(name))throw Error('Family paths must be relative');const path=await realpath(resolve(root,name));if(!inside(root,path))throw Error('Family path escapes protected root');return path;}
function kernelHashes(value,result=new Set()){
 if(value&&typeof value==='object')for(const [key,v]of Object.entries(value)){if(key==='kernelSha256'&&hex(v))result.add(v);else kernelHashes(v,result);}return result;
}

function failedRecord(db,item,provenance){
 if(item.objective!=='correctness'||item.evidence?.groups?.some(g=>g.classification?.bucket==='harness'))throw Error('Only non-harness correctness failures can seed a family');
 const match=/^events:(\d+)$/.exec(provenance?.reference??'');if(!match)throw Error('A controller verification reference is required');
 const row=db.prepare('SELECT sequence,attempt,type,data FROM events WHERE sequence=?').get(Number(match[1]));
 const attempt=row?.attempt&&db.prepare('SELECT * FROM attempts WHERE id=?').get(row.attempt),data=row&&JSON.parse(row.data);
 if(!attempt||row.type!=='verification'||!['failed','passed'].includes(attempt.outcome)||!attempt.session||!(data.actor_elapsed>0)||data.correctness?.passed!==false)throw Error('Witness must be an actual failed actor correctness verification');
 if(provenance.attemptId&&provenance.attemptId!==attempt.id||provenance.projectId&&provenance.projectId!==attempt.project)throw Error('Witness attempt/project identity changed');
 const refs=(item.evidence?.groups??[]).flatMap(g=>[...(g.occurrences??[]),...(g.linkedDiagnostics??[])].map(o=>o.reference));
 if(!refs.includes(provenance.reference))throw Error('Witness is outside the supplied failure evidence');
 if(data.correctness.exitCode===125)throw Error('Verifier infrastructure failure cannot seed a kernel experiment');
 const admitted=db.prepare("SELECT data FROM events WHERE attempt=? AND type='job-admitted' ORDER BY sequence LIMIT 1").get(attempt.id),kind=admitted&&JSON.parse(admitted.data).kind;
 if(!['original','retry','investigation'].includes(kind))throw Error('Only admitted actor work can establish a kernel failure');
 return {row,attempt,data};
}
async function gitRead(repository,args,signal,binary=false){
 const metadata=await stat(join(repository,'HEAD')).then(s=>s.isFile(),()=>false);
 return (await git('git',[...(metadata?['--git-dir',repository]:['-C',repository]),...args],{maxBuffer:1048576,timeout:10000,encoding:binary?'buffer':'utf8',...(signal?{signal}:{})})).stdout;
}

async function verificationKernelHashes(root,data){
 const expected=kernelHashes(data),artifact=data.correctness.evidence,artifactHash=data.correctness.evidence_sha256;
 if(text(artifact)&&hex(artifactHash))try{const actual=await realpath(artifact);if(inside(root,actual)){const bytes=Buffer.from(await utf8(actual));if(sha(bytes)===artifactHash)for(const line of bytes.toString('utf8').split('\n'))try{kernelHashes(JSON.parse(line),expected);}catch{}}}catch(error){if(!['ENOENT','EISDIR'].includes(error.code))throw error;}
 return expected;
}
// The controller fingerprints all source files, not merely the most recent stage snapshot.
const treeFingerprints=new Map();
async function treeFingerprint(repository,tree,signal){
 signal?.throwIfAborted();
 const key=repository+'\0'+tree;if(treeFingerprints.has(key))return treeFingerprints.get(key);
 const records=(await gitRead(repository,['ls-tree','-rz',tree],signal)).split('\0').filter(Boolean).map(line=>{const tab=line.indexOf('\t');return {mode:line.slice(0,6),path:line.slice(tab+1)};}).filter(r=>!r.path.split('/').some(p=>['.git','node_modules','__pycache__','.pytest_cache'].includes(p)));
 if(records.length>128||records.some(r=>!['100644','100755'].includes(r.mode)))return null;
 records.sort((a,b)=>{const left=a.path.split('/'),right=b.path.split('/');for(let i=0;i<Math.min(left.length,right.length);i++)if(left[i]!==right[i])return left[i]<right[i]?-1:1;return left.length-right.length;});
 const hash=createHash('sha256');let size=0;for(const r of records){signal?.throwIfAborted();const code=await gitRead(repository,['show',tree+':'+r.path],signal,true);size+=Buffer.byteLength(code);if(size>8388608)return null;hash.update(JSON.stringify([r.path,sha(code)])+'\n');}
 const result=hash.digest('hex');if(treeFingerprints.size>=16)treeFingerprints.delete(treeFingerprints.keys().next().value);treeFingerprints.set(key,result);return result;
}
function sourceTrees(db,attempt,sequence){
 const snapshots=db.prepare("SELECT sequence,data FROM events WHERE attempt=? AND type='source-snapshot' AND sequence<? ORDER BY sequence DESC").all(attempt,sequence).map(s=>({tree:JSON.parse(s.data).tree_oid,snapshotReference:'events:'+s.sequence}));
 if(db.prepare("SELECT 1 FROM sqlite_master WHERE name='storage_iterations'").get())for(const row of db.prepare("SELECT id,source_hash FROM storage_iterations WHERE attempt=? AND stage='validation' ORDER BY ordinal DESC").all(attempt))snapshots.unshift({tree:row.source_hash,iterationId:row.id});
 const seen=new Set();return snapshots.filter(s=>{if(!/^[a-f0-9]{40}$/.test(s.tree??'')||seen.has(s.tree))return false;seen.add(s.tree);return true;});
}

/** Validate the replay witness against controller records, not a model summary. */
export async function failureWitness({config,item,provenance,signal}){
 const root=await realpath(config.project_evidence_root??config.root),db=new DatabaseSync(join(root,'state.sqlite'),{readOnly:true});
 try{
  const {row,attempt,data}=failedRecord(db,item,provenance);
  const source=provenance.source;if(!source)throw Error('Recoverable failed source is required');let code,sourceIdentity;
  if(source.snapshotReference||source.iterationId){
   let tree;
   if(source.iterationId){const entry=db.prepare("SELECT source_hash FROM storage_iterations WHERE id=? AND attempt=? AND stage='validation'").get(source.iterationId,attempt.id);tree=entry?.source_hash;}
   else{const snapshot=/^events:(\d+)$/.exec(source.snapshotReference),entry=snapshot&&db.prepare("SELECT data FROM events WHERE sequence=? AND attempt=? AND type='source-snapshot' AND sequence<?").get(Number(snapshot[1]),attempt.id,row.sequence);tree=entry&&JSON.parse(entry.data).tree_oid;}
   if(!/^[a-f0-9]{40}$/.test(tree??''))throw Error('Source snapshot does not belong to this verification');
   const record={tree_oid:tree};
   const repository=await scoped(root,source.repository);if(!text(source.file)||isAbsolute(source.file)||!inside('/tree',resolve('/tree',source.file)))throw Error('Snapshot file must remain inside the source tree');
   code=await gitRead(repository,['show',record.tree_oid+':'+source.file],signal);sourceIdentity={repository:source.repository,tree:record.tree_oid,file:source.file,...(source.iterationId?{iterationId:source.iterationId}:{snapshotReference:source.snapshotReference})};
   const hashes=await verificationKernelHashes(root,data);if(!hashes.has(sha(code))&&(!hex(data.fingerprint)||await treeFingerprint(repository,record.tree_oid,signal)!==data.fingerprint))throw Error('Source snapshot is not bound to this failed verification');
  }else{
   const path=await scoped(root,source.path);code=await utf8(path);const expected=await verificationKernelHashes(root,data);
   if(!expected.has(sha(code)))throw Error('Recovered source is not bound to the failed verification');sourceIdentity={path:relative(root,path)};
  }
  if(!text(code)||Buffer.byteLength(code)>1048576)throw Error('Retained kernel is empty or oversized');
  return {reference:provenance.reference,attemptId:attempt.id,projectId:attempt.project,sourceSha256:sha(code),verificationSha256:sha(row.data),source:sourceIdentity,code};
 }finally{db.close();}
}

/** Materialize retained Git objects for native read tools; no model needs to decode Git. */
export async function failureSourceInventory({config,item,directory,signal}){
 const root=await realpath(config.project_evidence_root??config.root),db=new DatabaseSync(join(root,'state.sqlite'),{readOnly:true}),candidates=[];
 try{
  const refs=[...new Set((item.evidence?.groups??[]).flatMap(g=>[...(g.occurrences??[]),...(g.linkedDiagnostics??[])].map(o=>o.reference)))];
  signal?.throwIfAborted();for(const reference of refs)try{candidates.push({...failedRecord(db,item,{reference}),reference});}catch{}
  if(!candidates.length)return [];
  const repositories=[];for(const entry of (await readdir(join(root,'source-history'),{withFileTypes:true}).catch(e=>{if(e.code==='ENOENT')return [];throw e;})).slice(0,256))if(entry.isDirectory())repositories.push(join(root,'source-history',entry.name));
  const sources=[];await mkdir(directory,{recursive:true});
  for(const {row,attempt,reference}of candidates.slice(0,6)){signal?.throwIfAborted();
   const spec=JSON.parse(db.prepare('SELECT spec FROM projects WHERE id=?').get(attempt.project)?.spec??'{}');
   for(const path of [spec.source&&join(spec.source,'kernel.py'),join(root,'attempts',attempt.id,'source/kernel.py')].filter(Boolean))try{const relativePath=relative(root,path),witness=await failureWitness({config,item,provenance:{reference,source:{path:relativePath}},signal});sources.push({reference,source:witness.source,sourceSha256:witness.sourceSha256,path});}catch{}
   if(sources.some(s=>s.reference===reference))continue;
   const snapshots=sourceTrees(db,attempt.id,row.sequence);
   for(const snapshot of snapshots){const tree=snapshot.tree;
    for(const repository of [...(spec.source?[spec.source]:[]),...repositories]){signal?.throwIfAborted();
     let names;try{names=(await gitRead(repository,['ls-tree','-r','--name-only',tree],signal)).trim().split('\n');}catch{continue;}
     for(const file of names.filter(n=>/(^|\/)(kernel|solution|main)\.(py|cpp|c|cce)$/.test(n)).slice(0,8)){
      const provenance={reference,source:{...(snapshot.iterationId?{iterationId:snapshot.iterationId}:{snapshotReference:snapshot.snapshotReference}),repository:relative(root,repository),file}};
      try{const witness=await failureWitness({config,item,provenance,signal}),path=join(directory,sha([reference,tree,file])+'.txt');await writeFile(path,witness.code);sources.push({reference,source:witness.source,sourceSha256:witness.sourceSha256,path});}catch{}
     }break;
    }
    if(sources.some(s=>s.reference===reference))break;
   }
  }
  await save(join(directory,'sources.json'),sources);return sources;
 }finally{db.close();}
}

/** v2 is distinct from legacy five-case suites: transfer roles must be authored explicitly. */
export async function loadFailureFamily({file,config,item,signal}){
 const root=await realpath(dirname(file)),manifest=JSON.parse(await utf8(file));
 if(manifest.format!==2||!slug(manifest.id)||manifest.target!==config.target||manifest.target!=='a3'||manifest.objective!=='correctness'||!text(manifest.mechanism?.phase)||!text(manifest.mechanism?.type)||!manifest.development||manifest.transfer?.length!==2)throw Error('Failure family v2 requires A3 correctness development and two transfers with a replay mechanism');
 const witness=await failureWitness({config,item,provenance:manifest.provenance,signal}),ids=new Set(),hashes={};
 const read=async name=>{const path=await scoped(root,name),s=await utf8(path);hashes[relative(root,path)]=sha(s);return {path,text:s};};
 async function task(input,transfer=false){
  if(!slug(input.id)||ids.has(input.id)||!text(input.prompt)||!Array.isArray(input.caseIds)||!input.caseIds.length||input.caseIds.some(id=>!text(id))||new Set(input.caseIds).size!==input.caseIds.length||input.controlCaseIds?.some(id=>!input.caseIds.includes(id)))throw Error('Family tasks require unique IDs and explicit case coverage');ids.add(input.id);
  if(transfer&&(!['layout','operation','dtype','boundary'].includes(input.changeType)||!text(input.relation)))throw Error('Transfers require a meaningful adaptation, not a new random seed');
  const checker=await read(input.checker),publicChecker=await read(input.publicChecker);if(checker.path===publicChecker.path)throw Error('Private and public checkers must be separate');
  const editablePaths=new Set();const materialize=async(mapping,editable=false)=>{if(!mapping||!Object.hasOwn(mapping,'kernel.py'))throw Error('Kernel file mapping required');const result={};for(const [name,path]of Object.entries(mapping)){if(!text(name)||isAbsolute(name)||!inside('/work',resolve('/work',name))||name==='.')throw Error('Editable files must remain within the task');const value=await read(path);if(value.path===checker.path)throw Error('Private checker cannot be editable');if(editable)editablePaths.add(value.path);result[name]=value.text;}return result;};
  const files=await materialize(input.files,true),reference={files:await materialize(input.controls?.reference?.files)},negative=input.controls?.negatives;
  if(!Array.isArray(negative)||!negative.length)throw Error('Every task requires reference and negative controls');const negatives=await Promise.all(negative.map(async n=>({files:await materialize(n.files)})));
  const checkerFiles={};for(const [name,path]of Object.entries(input.checkerFiles??{})){if(!text(name)||isAbsolute(name)||!inside('/check',resolve('/check',name))||name==='.')throw Error('Checker helpers must be relative');const value=await read(path);if(editablePaths.has(value.path)||value.path===publicChecker.path)throw Error('Private checker helper cannot alias editable or public checker files');checkerFiles[name]=value.text;}
  return {...input,kind:'target-kernel',target:'a3',bucket:'correctness',files,checker:checker.path,publicChecker:publicChecker.path,checkerFiles,controls:{reference,negatives}};
 }
 const development=await task(manifest.development),transfer=await Promise.all(manifest.transfer.map(t=>task(t,true)));
 if(new Set(transfer.map(t=>t.changeType)).size!==2)throw Error('The two transfers must exercise distinct adaptations');
 if(new Set([development,...transfer].map(t=>sha(t.files))).size!==3)throw Error('Development and transfer payloads must be distinct');
 if(!development.controls.negatives.some(n=>sha(n.files['kernel.py'])===sha(development.files['kernel.py'])))throw Error('Development negative must replay the isolated starter');
 const isolated=sha(development.files['kernel.py']);if(isolated!==witness.sourceSha256&&!text(manifest.provenance.isolationEdits))throw Error('Changes around retained failure require an isolation explanation');
 const provenance={...manifest.provenance,...witness,isolatedSourceSha256:isolated};delete provenance.code;
 const family={...manifest,development,transfer,provenance,evidence:[...(item.evidence?.groups??[]).flatMap(g=>g.occurrences??[]),{reference:'retained-source:'+witness.sourceSha256,text:witness.code}]};
 return {...family,identity:sha({manifest,hashes,provenance})};
}

async function checkPythonSyntax({config,family,manifest,directory,signal}){
 const paths=new Set();for(const [index,task]of [family.development,...family.transfer].entries()){
  paths.add(task.checker);paths.add(task.publicChecker);const original=[manifest.development,...manifest.transfer][index];
  for(const [name,path]of Object.entries(original.checkerFiles??{}))if(name.endsWith('.py'))paths.add(await scoped(await realpath(directory),path));
 }
 try{await git(config.python??'python3',['-c','import pathlib,sys\nfor name in sys.argv[1:]: compile(pathlib.Path(name).read_bytes(), name, "exec")',...paths],{timeout:10000,maxBuffer:65536,signal});}
 catch(error){throw Error('Family checker Python syntax validation failed: '+String(error.stderr||error.message).slice(-4000),{cause:error});}
}

const selector=Type.Object({reference:Type.String(),startLine:Type.Optional(Type.Integer({minimum:1})),endLine:Type.Optional(Type.Integer({minimum:1})),segment:Type.Optional(Type.Integer({minimum:1})),field:Type.Optional(Type.String())},{additionalProperties:false});
const resultSchema=Type.Object({status:Type.Union([Type.Literal('ready'),Type.Literal('needs-evidence')]),reason:Type.String({minLength:1}),family:Type.Optional(Type.Unknown()),files:Type.Optional(Type.Record(Type.String(),Type.String())),claims:Type.Array(Type.Object({path:Type.Literal('reason'),kind:Type.Union([Type.Literal('observation'),Type.Literal('hypothesis')]),proofs:Type.Array(selector,{minItems:1})},{additionalProperties:false}),{minItems:1})},{additionalProperties:false});
/** A bounded native research session may produce fixtures, never skill instructions or curriculum jobs. */
export async function prepareFailureFamily({config,backlog,item,directory,seconds=1800,signal,checkReferences},dependencies={}){
 if(!Number.isFinite(seconds)||seconds<=0||seconds>1800)throw Error('Isolation must fit 1800 elapsed seconds');await mkdir(directory,{recursive:true});
 const file=join(directory,'family.json'),stateFile=join(directory,'isolation.json'),prior=await optional(stateFile),identity=sha([item.id,item.base,item.evidence,config.target]);
 if(prior&&prior.identity!==identity)throw Error('Isolation inputs changed');
 if(prior?.status==='ready'){const family=await loadFailureFamily({file,config,item});if(prior.familyIdentity!==family.identity)throw Error('Frozen family changed');return {status:'ready',family,elapsedSeconds:prior.elapsedSeconds};}
 if(prior?.status==='needs-evidence')return prior;
 if(prior?.running)throw Error('Interrupted isolation elapsed custody is unknown');
 const elapsed=prior?.elapsedSeconds??0;if(elapsed>=1800)throw Error('Isolation elapsed allowance exhausted');
 const begun=performance.now(),bounded=AbortSignal.any([...(signal?[signal]:[]),AbortSignal.timeout(Math.max(1,Math.ceil(Math.min(seconds,1800-elapsed)*1000)))]);let ledger={identity,elapsedSeconds:elapsed,running:true};await save(stateFile,ledger);
 const checkpoint=async()=>{ledger.elapsedSeconds=elapsed+(performance.now()-begun)/1000;await save(stateFile,ledger);};let pending=Promise.resolve();const timer=setInterval(()=>{pending=pending.then(checkpoint);},1000);
 try{
  const sources=await failureSourceInventory({config,item,directory:join(directory,'sources'),signal:bounded});
  bounded.throwIfAborted();if(!sources.length){ledger.status='needs-evidence';ledger.reason='No recoverable source bound to an actual actor correctness failure';return ledger;}
  const groups=await freezeGroups(config.root,item.evidence?.groups??[],join(directory,'evidence'));bounded.throwIfAborted();
  const session=dependencies.session??reviewSession,result=await session({directory:join(directory,'research'),config,seconds:Math.max(.001,Math.min(seconds,1800-elapsed)-(performance.now()-begun)/1000),signal:bounded,knowledge:true,evidenceGroups:groups,allowedPaths:[config.project_evidence_root??config.root],proofRequired:true,selectorOnly:true,resultSchema,
   prompt:{task:'Isolate one actual failed actor correctness attempt from these campaign records. Inspect successful projects and KDB memories for a passing reference. Return needs-evidence if no retained failing source is recoverable. Do not write a skill instruction, propose changes to skills, or start a curriculum project. If ready, return a protected v2 family and its UTF-8 files map: one development task and two distinct transfer tasks; controls reference and negatives for each. Freeze all cases now, before any instruction writer. Transfers must change distinct layout/operation/dtype/boundary properties, never just seeds. Keep private checker/helper files out of editable and public files. Checker API: argv1 is candidate directory; enforce a 90-second elapsed deadline per case. Final JSON contains correct, kernelSha256 and evidence.caseResults with exactly one {id,correct:boolean} for every frozen case; overall correct is true only when all cases pass. checkedCases lists actual execution coverage only, including failed cases; unverified cases must not be marked successful. Include evidence.phase and type for failures. Public and private checkers are separate. The trusted runtime will qualify all controls on A3; no claim of execution from code alone.',contract:{format:2,target:'a3',objective:'correctness',id:'family-id',mechanism:{phase:'candidate-call-or-synchronize',type:'observed failure type'},provenance:{reference:'events:N',source:{path:'campaign-relative retained kernel OR snapshotReference OR iterationId, repository and file'},isolationEdits:'Only if adjusting orthogonal issues'},development:{id:'development',prompt:'fixed task contract',caseIds:['case'],files:{'kernel.py':'dev/starter.py'},checker:'dev/private.py',publicChecker:'dev/public.py',controls:{reference:{files:{'kernel.py':'dev/reference.py'}},negatives:[{files:{'kernel.py':'dev/starter.py'}}]}},transfer:'exactly two task records, each additionally changeType and relation'},groupIds:item.groupIds,sourceInventory:sources},
   validate:async value=>{
    if(value.status==='needs-evidence')return value;
    if(!value.family||!value.files||Object.keys(value.files).length>64||Object.values(value.files).reduce((n,s)=>n+Buffer.byteLength(s),0)>4194304)throw Error('Bounded family file bundle required');
    for(const [name,s]of Object.entries(value.files)){if(!text(name)||isAbsolute(name)||!inside(directory,resolve(directory,name))||['family.json','isolation.json'].includes(name)||s.includes('\0'))throw Error('Family output files must be relative UTF-8 payloads');await mkdir(dirname(join(directory,name)),{recursive:true});await writeFile(join(directory,name),s,{mode:0o600});}
    await save(file,value.family);const family=await loadFailureFamily({file,config,item,signal:bounded});
    await checkPythonSyntax({config,family,manifest:value.family,directory,signal:bounded});
    if(checkReferences){const remaining=Math.min(seconds,1800-elapsed)-(performance.now()-begun)/1000;bounded.throwIfAborted();if(remaining<=0)throw Error('Isolation elapsed allowance exhausted before reference checks');try{await checkReferences({family,directory,seconds:remaining,signal:bounded});}catch(error){throw Error('Family reference checks failed: '+error.message,{cause:error});}}
    bounded.throwIfAborted();return value;
   }});
  bounded.throwIfAborted();const value=result.value??result,family=value.status==='ready'?await loadFailureFamily({file,config,item,signal:bounded}):null;ledger={identity,status:value.status,reason:value.reason,...(family?{familyIdentity:family.identity}:{}),elapsedSeconds:elapsed+(performance.now()-begun)/1000,running:false};await save(stateFile,ledger);
  return family?{status:'ready',family,elapsedSeconds:ledger.elapsedSeconds}:ledger;
 }finally{clearInterval(timer);await pending;await checkpoint();ledger.running=false;await save(stateFile,ledger);}
}
