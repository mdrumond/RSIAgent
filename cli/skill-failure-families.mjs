/** Protected replay families, derived from observed campaign failures before instruction writing. */
import {setTimeout as delay} from 'node:timers/promises';
import {readFile,writeFile,mkdir,realpath,stat,rename,readdir,opendir} from 'node:fs/promises';
import {dirname,resolve,join,relative,isAbsolute} from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {DatabaseSync} from 'node:sqlite';
import {Type} from 'typebox';
import {reviewSession} from './skill-review-native.mjs';
import {freezeGroups} from './skill-backlog-analysis.mjs';
import {failureStageElapsed,failureTiming} from './skill-failure-protocol.mjs';

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
const fingerprintedFile=name=>!name.split('/').some(p=>['.git','node_modules','__pycache__','.pytest_cache'].includes(p));
async function treeFingerprint(repository,tree,signal){
 signal?.throwIfAborted();
 const key=repository+'\0'+tree;if(treeFingerprints.has(key))return treeFingerprints.get(key);
 const records=(await gitRead(repository,['ls-tree','-rlz',tree],signal)).split('\0').filter(Boolean).map(line=>{const tab=line.indexOf('\t');return {mode:line.slice(0,6),size:Number(line.slice(0,tab).trim().split(/\s+/).at(-1)),path:line.slice(tab+1)};}).filter(r=>fingerprintedFile(r.path));
 if(records.length>128||records.some(r=>r.size>1048576||!['100644','100755'].includes(r.mode)))return null;
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

// These inputs explain the historical check. Qualification still runs the authored controls.
async function isolationInputs({root,db,attempt,row,source,directory,signal}){
 const files=[],seen=new Set();let bytes=0;
 async function retain(name,role,content,provenance){
  if(bytes+content.length>8388608||files.length>=128)return;
  let code;try{code=new TextDecoder('utf-8',{fatal:true}).decode(content);}catch{return;}
  if(!code.trim()||code.includes('\0'))return;
  const digest=sha(content),path=join(directory,digest+'.txt');
  await writeFile(path,content);bytes+=content.length;files.push({name,role,path,sha256:digest,...provenance});
 }
 // A matching kernel hash by itself cannot vouch for helper files from a different revision.
 const tree=source.tree,expected=JSON.parse(row.data).fingerprint;
 if(tree&&hex(expected)){
  const repository=await scoped(root,source.repository);
  if(await treeFingerprint(repository,tree,signal)===expected){
   const names=(await gitRead(repository,['ls-tree','-rz','--name-only',tree],signal)).split('\0').filter(Boolean);
   for(const name of names){
    signal?.throwIfAborted();
    if(name!==source.file&&fingerprintedFile(name)&&files.length<128&&bytes<8388608)await retain(name,'verified-source-dependency',await gitRead(repository,['show',tree+':'+name],signal,true),{tree});
   }
  }
 }
 const record=db.prepare("SELECT sequence,data FROM events WHERE attempt=? AND type='curriculum-proposal' AND sequence<? ORDER BY sequence DESC LIMIT 1").get(attempt.id,row.sequence);
 if(record){
  const proposal=JSON.parse(record.data).proposal,base=resolve(root,'attempts',attempt.id,'investigation');
  // The controller accepted this proposal after checking its declared sibling fixture directories.
  if(inside(root,base)&&await realpath(base).catch(()=>null)===base&&proposal){
   const provenance={declaration:'events:'+record.sequence};
   const readArchived=async(name,role,details=provenance)=>{
    signal?.throwIfAborted();
    const path=resolve(base,name);
    if(!inside(base,path)||seen.has(path)||await realpath(path).catch(()=>null)!==path)return;
    seen.add(path);const info=await stat(path);
    if(info.isFile()&&info.size<=1048576&&bytes+info.size<=8388608)
     await retain(name,role,await readFile(path),details);
   };
   const localName=name=>text(name)&&inside(base,resolve(base,name))?relative(base,resolve(base,name)):null;
   const verifier=localName(proposal.verifier);
   if(verifier){
    const invocation={...provenance};
    if(text(proposal.verifier_command)&&proposal.verifier_command.length<=8192)invocation.command=proposal.verifier_command;
    if(Array.isArray(proposal.verifier_args)&&proposal.verifier_args.every(arg=>typeof arg==='string')&&JSON.stringify(proposal.verifier_args).length<=8192)
     invocation.args=proposal.verifier_args;
    await readArchived(verifier,'verifier',invocation);
   }
   const controls=proposal.controls??{},fixtures=[[controls.positive,'positive-control'],...(Array.isArray(controls.negative)?controls.negative.slice(0,8):[]).map(name=>[name,'negative-control'])];
   let inspected=0;
   async function visit(name,role){
    const path=resolve(base,name);
    if(await realpath(path).catch(()=>null)!==path)return;
    const info=await stat(path);if(!info.isDirectory())return;
    const dir=await opendir(path);
    for await(const entry of dir){
     signal?.throwIfAborted();
     if(++inspected>512||files.length>=128||bytes>=8388608)break;
     if(entry.name.startsWith('.')||['__pycache__','node_modules'].includes(entry.name))continue;
     const child=join(name,entry.name);
     if(entry.isDirectory())await visit(child,role);
     else if(entry.isFile())await readArchived(child,role);
    }
   }
   for(const [submitted,role] of fixtures){const name=localName(submitted);if(name&&name!=='source'&&dirname(name)==='.')await visit(name,role);}
  }
 }
 return files;
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
  for(const {row,attempt,reference,data}of candidates.slice(0,6)){signal?.throwIfAborted();
   const fallback=[],retained=[];
   const spec=JSON.parse(db.prepare('SELECT spec FROM projects WHERE id=?').get(attempt.project)?.spec??'{}');
   for(const path of [spec.source&&join(spec.source,'kernel.py'),join(root,'attempts',attempt.id,'source/kernel.py')].filter(Boolean))try{const relativePath=relative(root,path),witness=await failureWitness({config,item,provenance:{reference,source:{path:relativePath}},signal});fallback.push({reference,source:witness.source,sourceSha256:witness.sourceSha256,path});}catch{}
   if(fallback.length&&!hex(data.fingerprint)){sources.push(...fallback);continue;}
   const snapshots=sourceTrees(db,attempt.id,row.sequence);
   for(const snapshot of snapshots){const tree=snapshot.tree;
    for(const candidate of [...(spec.source?[spec.source]:[]),...repositories]){signal?.throwIfAborted();
     const repository=await realpath(candidate).catch(()=>null);if(!repository||!inside(root,repository))continue;
     let names;try{names=(await gitRead(repository,['ls-tree','-r','--name-only',tree],signal)).trim().split('\n');}catch{continue;}
     const matching=hex(data.fingerprint)&&await treeFingerprint(repository,tree,signal)===data.fingerprint;
     if(!matching&&fallback.length)break;
     for(const file of names.filter(n=>/(^|\/)(kernel|solution|main)\.(py|cpp|c|cce)$/.test(n)).slice(0,8)){
      const provenance={reference,source:{...(snapshot.iterationId?{iterationId:snapshot.iterationId}:{snapshotReference:snapshot.snapshotReference}),repository:relative(root,repository),file}};
      try{const witness=await failureWitness({config,item,provenance,signal}),path=join(directory,sha([reference,tree,file])+'.txt');await writeFile(path,witness.code);(matching?retained:fallback).push({reference,source:witness.source,sourceSha256:witness.sourceSha256,path});}catch{}
     }break;
    }
    if(retained.length||(!hex(data.fingerprint)&&fallback.length))break;
   }
   sources.push(...(retained.length?retained:fallback));
  }
  for(const entry of sources){
   const {attempt,row}=candidates.find(c=>c.reference===entry.reference);
   entry.replayInputs=await isolationInputs({root,db,attempt,row,source:entry.source,directory,signal});
  }
  await save(join(directory,'sources.json'),sources);return sources;
 }finally{db.close();}
}

/** v2 is distinct from legacy five-case suites: transfer roles must be authored explicitly. */
export async function loadFailureFamily({file,config,item,signal}){
 const root=await realpath(dirname(file)),manifest=JSON.parse(await utf8(file));
 if(manifest.format!==2||!slug(manifest.id)||manifest.target!==config.target||!['a3','a5'].includes(manifest.target)||manifest.objective!=='correctness'||!text(manifest.mechanism?.phase)||!text(manifest.mechanism?.type)||!manifest.development||manifest.transfer?.length!==2)throw Error('Failure family v2 requires matching A3/A5 correctness development and two transfers with a replay mechanism');
 const witness=await failureWitness({config,item,provenance:manifest.provenance,signal}),ids=new Set(),hashes={},privatePaths=new Set(),exposedPaths=new Set();
 const read=async name=>{const path=await scoped(root,name),s=await utf8(path);hashes[relative(root,path)]=sha(s);return {path,text:s};};
 async function task(input,transfer=false){
  if(!slug(input.id)||ids.has(input.id)||!text(input.prompt)||!Array.isArray(input.caseIds)||!input.caseIds.length||input.caseIds.some(id=>!text(id))||new Set(input.caseIds).size!==input.caseIds.length||input.controlCaseIds?.some(id=>!input.caseIds.includes(id)))throw Error('Family tasks require unique IDs and explicit case coverage');ids.add(input.id);
  if(transfer&&(!['layout','operation','dtype','boundary'].includes(input.changeType)||!text(input.relation)))throw Error('Transfers require a meaningful adaptation, not a new random seed');
  const checker=await read(input.checker),publicChecker=await read(input.publicChecker);privatePaths.add(checker.path);exposedPaths.add(publicChecker.path);if(checker.path===publicChecker.path)throw Error('Private and public checkers must be separate');
  const editablePaths=new Set();const materialize=async(mapping,editable=false)=>{if(!mapping||!Object.hasOwn(mapping,'kernel.py'))throw Error('Kernel file mapping required');const result={};for(const [name,path]of Object.entries(mapping)){if(!text(name)||isAbsolute(name)||!inside('/work',resolve('/work',name))||name==='.')throw Error('Editable files must remain within the task');const value=await read(path);if(value.path===checker.path)throw Error('Private checker cannot be editable');if(editable){editablePaths.add(value.path);exposedPaths.add(value.path);}result[name]=value.text;}return result;};
  const files=await materialize(input.files,true),reference={files:await materialize(input.controls?.reference?.files)},negative=input.controls?.negatives;
  if(!Array.isArray(negative)||!negative.length)throw Error('Every task requires reference and negative controls');const negatives=await Promise.all(negative.map(async n=>({files:await materialize(n.files)})));
  const calibration=input.controls.calibration?{files:await materialize(input.controls.calibration.files)}:null;
  const checkerFiles={};for(const [name,path]of Object.entries(input.checkerFiles??{})){if(!text(name)||isAbsolute(name)||!inside('/check',resolve('/check',name))||name==='.')throw Error('Checker helpers must be relative');const value=await read(path);if(editablePaths.has(value.path)||value.path===publicChecker.path)throw Error('Private checker helper cannot alias editable or public checker files');privatePaths.add(value.path);checkerFiles[name]=value.text;}
  return {...input,kind:'target-kernel',target:manifest.target,bucket:'correctness',files,checker:checker.path,publicChecker:publicChecker.path,checkerFiles,controls:{reference,negatives,...(calibration?{calibration}:{})}};
 }
 const development=await task(manifest.development),transfer=await Promise.all(manifest.transfer.map(t=>task(t,true)));
 if([...privatePaths].some(path=>exposedPaths.has(path)))throw Error('Private checker or helper cannot alias any family task editable or public file');
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
const nameSchema=Type.String({minLength:1}),fileMap=Type.Record(Type.String(),nameSchema),controlSchema=Type.Object({files:fileMap},{additionalProperties:false});
const taskSchema=target=>Type.Object({id:nameSchema,prompt:nameSchema,caseIds:Type.Array(nameSchema,{minItems:1}),controlCaseIds:Type.Optional(Type.Array(nameSchema)),files:fileMap,checker:nameSchema,publicChecker:nameSchema,checkerFiles:Type.Optional(fileMap),controls:Type.Object({reference:controlSchema,negatives:Type.Array(controlSchema,{minItems:1}),calibration:target==='a5'?controlSchema:Type.Optional(controlSchema)},{additionalProperties:false}),changeType:Type.Optional(Type.Union(['layout','operation','dtype','boundary'].map(Type.Literal))),relation:Type.Optional(nameSchema)},{additionalProperties:false});
const familySchema=target=>Type.Object({format:Type.Literal(2),id:nameSchema,target:Type.Literal(target),objective:Type.Literal('correctness'),mechanism:Type.Object({phase:nameSchema,type:nameSchema},{additionalProperties:false}),provenance:Type.Object({reference:nameSchema,attemptId:Type.Optional(nameSchema),projectId:Type.Optional(nameSchema),source:Type.Object({path:Type.Optional(nameSchema),snapshotReference:Type.Optional(nameSchema),iterationId:Type.Optional(Type.Integer({minimum:1})),repository:Type.Optional(nameSchema),file:Type.Optional(nameSchema)},{additionalProperties:false}),isolationEdits:Type.Optional(nameSchema)},{additionalProperties:false}),development:taskSchema(target),transfer:Type.Array(taskSchema(target),{minItems:2,maxItems:2})},{additionalProperties:false});
const claimSchema=Type.Array(Type.Object({path:Type.Literal('reason'),kind:Type.Union([Type.Literal('observation'),Type.Literal('hypothesis')]),proofs:Type.Array(selector,{minItems:1})},{additionalProperties:false}),{minItems:1});
const missingEvidence=['retained-source','failure-mechanism','passing-reference','missing-semantics','unreliable-oracle'];
const isOracleGap=blocker=>['missing-semantics','unreliable-oracle'].includes(blocker);
const resultSchema=target=>Type.Union([
 Type.Object({status:Type.Literal('ready'),reason:nameSchema,family:familySchema(target),claims:claimSchema},{additionalProperties:false}),
 Type.Object({status:Type.Literal('needs-evidence'),missing:Type.Union(missingEvidence.map(Type.Literal)),reason:nameSchema,verificationReference:nameSchema,
  nextQuestion:Type.Optional(nameSchema),claims:claimSchema},{additionalProperties:false})
]);
function authorInstructions(focusedCalibration){
 return [
  'Inspect the relevant replayInputs in sourceInventory with native read. Their roles and provenance identify retained checker dependencies and declared historical controls; use trusted qualification to establish their behavior. Use native write/edit to assemble the file-backed draft under draftDirectory. Call submit_review with status ready, the typed manifest and evidence claims to run syntax and trusted target checks. The controller replays the development starter first. The tool returns rejected fields and observed control outcomes so you can revise and resubmit. Use the saved draft if one is present.',
  'Answer the focused question for one retained actor correctness failure. Cite the original verification or primary diagnostic. Inspect successful projects and KDB memory for a passing reference. Preserve the verifier’s input placement and invocation or explain isolation edits using evidence. Attribute behaviour only where a check directly establishes it.',
  'Freeze one development task and two transfer tasks with meaningful, distinct layout, operation, dtype, or boundary adaptations. Use paths relative to draftDirectory and separate the public and private checkers. Keep private helpers out of editable files. Include a passing reference and failing negative controls for each task. Skill writing starts after the family is established.'+
   (focusedCalibration?' Include a distinct passing calibration for every task that exercises the same oracle without using the failing construct.':''),
  'The checker receives the candidate directory in argv1. Enforce a 90-second elapsed deadline per case. End with one JSON line containing correct and evidence.caseResults. Supply one {id,correct:boolean} row for every frozen case. Mark a case true only after it passes, and mark overall correct true only if all cases pass. Set evidence.checkedCases to the IDs actually executed, including any observed failed case. On every failing exit emit evidence.phase and evidence.type, and align the manifest mechanism with the values the checker actually reports for the reproduced failure. The trusted guard supplies the candidate hash.',
  'If the evidence cannot support a family, use submit_review with status needs-evidence. Pick the closest blocker, select verificationReference from sourceInventory, cite the original evidence, and explain the unknown. For an oracle or semantics gap, give one question that a small Phase 1 exercise with staged PyTorch outputs could answer. For other gaps, include a question when a specific probe can distinguish plausible explanations.',
 ].join('\n\n');
}
function verifiedChecks(checks=[]){
 const starter=checks.find(check=>check.variant==='starter');
 const replay=starter?.correct===false?(starter.mechanismMatch?'reproduced':'different-failure'):starter?.correct===true?'passed':'unverified';
 return {replay,checks,passedChecks:checks.filter(check=>check.correct===true)};
}
async function isolationDisposition({value,sources,groups,config,item,observations,signal}){
 const source=sources.find(source=>source.reference===value.verificationReference);
 if(!source)throw Error('verificationReference must select an original failed verification from sourceInventory');
 if(!missingEvidence.includes(value.missing))throw Error('Choose a missing evidence category from the submission schema');
 const witness=await failureWitness({config,item,provenance:source,signal});
 const eventAlias='project:'+witness.projectId+':'+source.reference;
 const artifactReferences=groups.flatMap(group=>[...group.occurrences,...(group.linkedDiagnostics??[])])
  .filter(observation=>observation.reference===source.reference).flatMap(observation=>(observation.artifacts??[]).map(artifact=>'artifact:'+artifact.sha256));
 const references=(value.claims??[]).flatMap(claim=>claim.proofs??[]).flatMap(proof=>[proof.reference,...(proof.sourceRefs??[])]);
 if(!references.some(reference=>reference===source.reference||reference===eventAlias||artifactReferences.includes(reference)))
  throw Error('Cite the selected verification or its primary diagnostic in the reason claim for '+source.reference);
 if(isOracleGap(value.missing)&&!text(value.nextQuestion))throw Error('An oracle or semantics gap needs one discriminating nextQuestion for Phase 1');
 const db=new DatabaseSync(join(config.project_evidence_root??config.root,'state.sqlite'),{readOnly:true});
 let verification;
 try{
  const {data}=failedRecord(db,item,source),correctness=data.correctness;
  verification={reference:witness.reference,attemptId:witness.attemptId,projectId:witness.projectId,sourceSha256:witness.sourceSha256,
   verificationSha256:witness.verificationSha256,check:'correctness-verification',passed:false,exitCode:correctness.exitCode??null,
   phase:correctness.receipt?.evidence?.phase??null,type:correctness.receipt?.evidence?.type??null,
   ...(correctness.evidence_sha256&&correctness.evidence?{diagnostic:{path:correctness.evidence,sha256:correctness.evidence_sha256}}:{})};
 }finally{db.close();}
 return {blocker:value.missing,route:isOracleGap(value.missing)?'phase1':value.nextQuestion?'probe':'hold',
  verification,...verifiedChecks(observations[source.reference]),nextQuestion:text(value.nextQuestion)||null};
}
async function freezeDraft(manifest,draft,directory){
 const names=new Set();for(const task of [manifest.development,...manifest.transfer])for(const name of [task.checker,task.publicChecker,...Object.values(task.files??{}),...Object.values(task.checkerFiles??{}),...Object.values(task.controls?.reference?.files??{}),...Object.values(task.controls?.calibration?.files??{}),...(task.controls?.negatives??[]).flatMap(n=>Object.values(n.files??{}))])names.add(name);
 if(names.size>64)throw Error('Bounded family file bundle required');let size=0;const bytes=[];for(const name of names){if(['family.json','isolation.json','draft','research','sources','evidence'].includes(relative(directory,resolve(directory,name)).split('/')[0]))throw Error('Reserved family file');const content=await utf8(await scoped(draft,name).catch(error=>{if(error.code==='ENOENT')throw Error('Missing draft file: '+name+'. Write it under draftDirectory, then resubmit the manifest.');throw error;}));size+=Buffer.byteLength(content);if(size>4194304)throw Error('Bounded family file bundle required');bytes.push([name,content]);}
 for(const [name,content]of bytes){const path=resolve(directory,name);if(!inside(directory,path))throw Error('Family output path escapes accepted directory');await mkdir(dirname(path),{recursive:true});await writeFile(path,content,{mode:0o600});}
}
async function draftFingerprint(draft){
 const files=(await readdir(draft,{recursive:true,withFileTypes:true})).filter(e=>e.isFile());
 const inputs=files.map(file=>join(file.parentPath,file.name)).sort();
 const hashes=[];let bytes=0;
 for(const path of inputs.slice(0,64)){
  const size=(await stat(path)).size;if(size>1048576||bytes+size>4194304)continue;
  const content=await readFile(path);bytes+=content.length;hashes.push([relative(draft,path),sha(content)]);
 }
 return sha(hashes);
}
/** A bounded native research session may produce fixtures, never skill instructions or curriculum jobs. */
export async function prepareFailureFamily({config,backlog,item,directory,seconds=failureTiming.isolationSeconds,signal,checkReferences,approvedInstruction},dependencies={}){
 if(!['a3','a5'].includes(config.target))throw Error('Failure isolation requires A3 or A5 target');
 if(!Number.isFinite(seconds)||seconds<=0||seconds>failureTiming.isolationSeconds)throw Error('Isolation must fit '+failureTiming.isolationSeconds+' elapsed seconds');await mkdir(directory,{recursive:true});
 const draft=join(directory,'draft');await mkdir(draft,{recursive:true});
 const file=join(directory,'family.json'),stateFile=join(directory,'isolation.json'),prior=await optional(stateFile),identity=sha([item.id,item.base,item.evidence,config.target,...(approvedInstruction?[approvedInstruction]:[])]);
 if(prior&&prior.identity!==identity)throw Error('Isolation inputs changed');
 if(prior?.status==='ready'){const family=await loadFailureFamily({file,config,item});if(prior.familyIdentity!==family.identity)throw Error('Frozen family changed');return {status:'ready',family,elapsedSeconds:prior.elapsedSeconds};}
 if(['needs-evidence','expired'].includes(prior?.status))return prior;
 if(prior?.running)throw Error('Interrupted isolation elapsed custody is unknown');
 const elapsed=prior?.elapsedSeconds??0;
 const saveIsolation=dependencies.saveIsolation??save;
 const observations=prior?.referenceChecks??{},completedChecks=prior?.completedReferenceChecks??{};
 let recovery=prior?.recovery??{contexts:0};
 let lastChecked=prior?.lastChecked??null;
 const now=dependencies.now??(()=>performance.now()),begun=now(),allocation=Math.min(seconds,failureTiming.isolationSeconds-elapsed);
 const left=()=>allocation-(now()-begun)/1000;
 const isolationTimeout=AbortSignal.timeout(Math.max(1,Math.ceil(allocation*1000))),bounded=signal?AbortSignal.any([signal,isolationTimeout]):isolationTimeout;
 const timings={focusedQuestion:0,verification:0,piTurnWait:0,piResponse:0,piRetry:0,...prior?.timings};
 let focus=new AbortController(),focusedSignal=AbortSignal.any([bounded,focus.signal]);
 let question=prior?.question??{number:1,kind:'isolate-failure',elapsedSeconds:timings.focusedQuestion};
 let ledger={identity,elapsedSeconds:elapsed,timings,question,referenceChecks:observations,completedReferenceChecks:completedChecks,recovery,lastChecked,running:true},focusStarted,focusTimer,verificationStarted;
 const focusDelta=()=>focusStarted===undefined?0:(now()-focusStarted)/1000;
 const focusedElapsed=()=>timings.focusedQuestion+focusDelta();
 const questionElapsed=()=>question.elapsedSeconds+focusDelta();
 const verificationElapsed=()=>timings.verification+(verificationStarted===undefined?0:(now()-verificationStarted)/1000);
 const stopFocus=()=>{clearTimeout(focusTimer);timings.focusedQuestion=focusedElapsed();question.elapsedSeconds=questionElapsed();focusStarted=undefined;};
 const checkFocus=()=>{if(questionElapsed()>=failureTiming.actorSeconds)focus.abort(failureStageElapsed('focused-question'));focusedSignal.throwIfAborted();};
 const startFocus=()=>{
  checkFocus();focusStarted=now();
  focusTimer=setTimeout(()=>focus.abort(failureStageElapsed('focused-question')),Math.max(1,Math.ceil((failureTiming.actorSeconds-questionElapsed())*1000)));
  focusTimer.unref();
 };
 const checkpoint=async()=>{
  ledger.elapsedSeconds=elapsed+(now()-begun)/1000;
  ledger.timings={...timings,focusedQuestion:focusedElapsed(),verification:verificationElapsed(),handoff:Math.max(0,ledger.elapsedSeconds-focusedElapsed()-verificationElapsed())};
  ledger.question={...question,elapsedSeconds:questionElapsed()};
  ledger.recovery=recovery;
  ledger.lastChecked=lastChecked;
  await saveIsolation(stateFile,ledger);
 };
 await saveIsolation(stateFile,ledger);
 let pending=Promise.resolve();const timer=setInterval(()=>{pending=pending.then(checkpoint);},1000);
 try{
  signal?.throwIfAborted();if(left()<=0)throw failureStageElapsed('isolate');
  const sources=await failureSourceInventory({config,item,directory:join(directory,'sources'),signal:bounded});
  bounded.throwIfAborted();if(!sources.length){
   ledger.status='needs-evidence';ledger.reason='No recoverable source bound to an actual actor correctness failure';
   ledger.disposition={blocker:'retained-source',route:'hold',verification:null,...verifiedChecks(),nextQuestion:null};
   return ledger;
  }
  const groups=await freezeGroups(config.root,item.evidence?.groups??[],join(directory,'evidence'));bounded.throwIfAborted();
  let disposition;
  const validateDisposition=value=>isolationDisposition({value,sources,groups,config,item,observations,signal:bounded});
  const focusedCalibration=config.target==='a5'||Boolean(approvedInstruction);
  const session=dependencies.session??reviewSession;
  question.draftFingerprint??=await draftFingerprint(draft);
  let result;
  while(!result){
   let rejection;
   try{
    if(left()<=0)throw failureStageElapsed('isolate');
    startFocus();
    const nextQuestion=question.kind==='revise-checked-draft'
     ? 'What correction addresses the first failed trusted check from the previous submission? Use the checker feedback in this session.'
     :question.kind==='finish-draft'
      ? 'What is the next missing piece needed to check the retained draft against the observed failure? Inspect the current files and focus on that piece.'
      :'What mechanism from this retained failure can be verified by replaying it?';
    const research=join(directory,recovery.contexts?'research-recovery-'+recovery.contexts:'research');
    const authorRecovery=recovery.contexts?{
     task:'Continue the current question from the saved draft and trusted checker feedback. Prior dialogue is unavailable.',
     trustedChecker:lastChecked?{reason:lastChecked.reason,diagnostics:observations[lastChecked.reference]??[]}:null
    }:null;
    result=await session({directory:research,config,seconds:Math.max(.001,left()),signal:focusedSignal,writableDirectory:draft,knowledge:true,evidenceGroups:groups,allowedPaths:[config.project_evidence_root??config.root,...sources.flatMap(source=>[source.path,...source.replayInputs.map(input=>input.path)])],proofRequired:true,selectorOnly:true,resultSchema:resultSchema(config.target),
   onModelTiming:(kind,seconds)=>{timings[kind]+=seconds;},
   prompt:{...(approvedInstruction?{approvedInstruction,testObjective:'Design or reuse a discriminating experiment for this immutable instruction and the original ranked failure. Keep the evaluated actor task blind to the instruction.'}:{}),...(authorRecovery?{recovery:authorRecovery}:{}),target:config.target,draftDirectory:draft,focusedQuestion:{number:question.number,prompt:nextQuestion},task:authorInstructions(focusedCalibration),contract:{format:2,target:config.target,objective:'correctness',id:'family-id',mechanism:{phase:'exact checker phase',type:'exact checker failure type'},provenance:{reference:'events:N',source:{path:'campaign-relative retained kernel OR snapshotReference OR iterationId, repository and file'},isolationEdits:'Only if adjusting orthogonal issues'},development:{id:'development',prompt:'fixed task contract',caseIds:['case'],files:{'kernel.py':'dev/starter.py'},checker:'dev/private.py',publicChecker:'dev/public.py',controls:{reference:{files:{'kernel.py':'dev/reference.py'}},negatives:[{files:{'kernel.py':'dev/starter.py'}}],...(focusedCalibration?{calibration:{files:{'kernel.py':'dev/calibration.py'}}}:{})}},transfer:'exactly two task records, each additionally changeType and relation'},groupIds:item.groupIds,sourceInventory:sources},
   validate:async value=>{
    if(value.status==='needs-evidence'){disposition=await validateDisposition(value);return value;}
    if(!value.family||value.files)throw Error('Submit a typed family manifest referencing draftDirectory files, without inline file contents');
    if(focusedCalibration){
     for(const task of [value.family.development,...(value.family.transfer??[])]){
      if(!text(task?.controls?.calibration?.files?.['kernel.py']))throw Error('A5 task '+(task?.id??'unknown')+' requires a calibration kernel before qualification');
     }
    }
    await freezeDraft(value.family,draft,directory);
    await save(file,value.family);const family=await loadFailureFamily({file,config,item,signal:bounded});
    if(focusedCalibration){
     for(const task of [family.development,...family.transfer]){
      const calibration=task.controls.calibration.files['kernel.py'];
      const controls=[['reference',task.controls.reference.files],...task.controls.negatives.map((negative,index)=>['negative '+(index+1),negative.files]),['starter',task.files]];
      for(const [label,files] of controls){
       if(calibration===files['kernel.py'])throw Error('A5 task '+task.id+' calibration kernel duplicates the '+label+'; supply a separate passing implementation');
      }
     }
    }
    let retained=completedChecks[family.identity];
    if(!retained)await checkPythonSyntax({config,family,manifest:value.family,directory,signal:bounded});
    if(checkReferences){
     checkFocus();
     const remaining=left();if(remaining<=0)throw failureStageElapsed('isolate');
     if(!retained){
      stopFocus();verificationStarted=now();
      try{
       const verified=await checkReferences({family,directory,seconds:remaining,signal:bounded});
       retained={diagnostics:verified?.diagnostics??[]};
      }catch(error){
       if(error.code!=='FAILURE_REFERENCE_REJECTED')throw Error('Family reference checks failed: '+error.message,{cause:error});
       retained={reason:error.message,diagnostics:error.diagnostics??[]};
      }finally{
       timings.verification=verificationElapsed();
       verificationStarted=undefined;
       if(!bounded.aborted)startFocus();
      }
      completedChecks[family.identity]=retained;
     }
     lastChecked={reference:family.provenance.reference,reason:retained.reason??null};
     observations[lastChecked.reference]=retained.diagnostics;
     pending=pending.then(checkpoint);await pending;
     if(retained.reason!==undefined){
      rejection=Object.assign(Error(retained.reason),{code:'FAILURE_REFERENCE_REJECTED',diagnostics:retained.diagnostics});
      throw Error('Family reference checks failed: '+rejection.message,{cause:rejection});
     }
    }
    bounded.throwIfAborted();return value;
    }});
    checkFocus();
   }catch(error){
    result=undefined;
    if(error.reviewSession?.retryableTimeout===true){
     if(bounded.aborted||focus.signal.aborted||left()<=0)throw error;
     // The question clock keeps running during recovery backoff. Another
     // session gets the same question, along with saved files and evidence.
     const backoff=Math.min(30,2**Math.min(recovery.contexts,5))*1000;
     await (dependencies.retryDelay??delay)(backoff,undefined,{signal:focusedSignal});
     checkFocus();
     recovery={contexts:recovery.contexts+1};
     pending=pending.then(checkpoint);await pending;
     continue;
    }
    if(bounded.aborted||left()<=0||!(focus.signal.aborted||['no-submit','rejected-submissions'].includes(error.reviewSession?.kind)))throw error;
    const fingerprint=await draftFingerprint(draft);
    if(fingerprint===question.draftFingerprint)throw error;
    stopFocus();
    question={number:question.number+1,kind:rejection?'revise-checked-draft':'finish-draft',elapsedSeconds:0,draftFingerprint:fingerprint};
    focus=new AbortController();focusedSignal=AbortSignal.any([bounded,focus.signal]);
    // Serialize with heartbeat because resuming this loop must retain which question owns the new grant.
    pending=pending.then(checkpoint);await pending;
   }finally{stopFocus();}
  }
  if(left()<=0)throw failureStageElapsed('isolate');
  const value=result.value??result,family=value.status==='ready'?await loadFailureFamily({file,config,item,signal:bounded}):null;
  bounded.throwIfAborted();
  if(!family)disposition??=await validateDisposition(value);
  Object.assign(ledger,{status:value.status,reason:value.reason,...(family?{familyIdentity:family.identity}:{disposition}),elapsedSeconds:elapsed+(now()-begun)/1000,running:false});
  return family?{status:'ready',family,elapsedSeconds:ledger.elapsedSeconds}:ledger;
 }catch(error){
  const exhausted=error.failureStage??signal?.reason?.failureStage??(!signal?.aborted&&(isolationTimeout.aborted||left()<=0?'isolate':focus.signal.aborted?'focused-question':null));
  if(!exhausted)throw error;
  return Object.assign(ledger,{status:'expired',stage:exhausted,reason:failureStageElapsed(exhausted).message});
 }finally{
  clearInterval(timer);
  stopFocus();
  await pending;
  ledger.running=false;
  await checkpoint();
 }
}
