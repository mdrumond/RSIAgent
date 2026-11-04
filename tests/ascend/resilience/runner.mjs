import {mkdir,writeFile,readdir,mkdtemp,access} from 'node:fs/promises';
import {join,dirname,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

export class Blocked extends Error {}
export async function runCases(cases,{output,tier='all',caseId,seed=1,profile='normal',
  deadlineSeconds=5400,cleanupSeconds=300}={}) {
  if(!['local','deepseek','remote','all'].includes(tier))throw Error('Unknown tier');
  if(!Number.isInteger(seed)||seed<0||seed>0xffffffff)throw Error('Seed must be an unsigned 32-bit integer');
  if(!['normal','slow','bursty'].includes(profile))throw Error('Unknown latency profile');
  if(!Number.isFinite(deadlineSeconds)||deadlineSeconds<=0)throw Error('Positive deadline required');
  const selected=cases.filter(c=>(tier==='all'||c.tier===tier)&&(!caseId||c.id===caseId));
  if(!selected.length)throw Error('No matching resilience cases');
  if(new Set(cases.map(c=>c.id)).size!==cases.length)throw Error('Duplicate resilience case id');
  output??=await mkdtemp(join(tmpdir(),'rsi-resilience-'));await mkdir(output,{recursive:true});
  try {await access(join(output,'report.json'));throw Error('Use a fresh report directory');}
  catch(error){if(error.code!=='ENOENT')throw error;}
  const started=performance.now(),cutoff=started+(deadlineSeconds-cleanupSeconds)*1000;
  let exhausted=false;
  const report={startedAt:new Date().toISOString(),deadlineSeconds,seed,profile,tier,caseId,results:[]};
  if(tier==='all'&&!caseId)for(const required of ['local','deepseek','remote'])
    if(!cases.some(c=>c.tier===required))report.results.push({id:`tier:${required}`,tier:required,
      status:'incomplete',reason:'Required tier has no registered cases'});
  try {report.revision=(await promisify(execFile)('git',['rev-parse','HEAD'])).stdout.trim();}catch{report.revision=null;}
  const save=()=>writeFile(join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
  for(const item of selected) {
    const remaining=cutoff-performance.now();
    if(exhausted||remaining<=0) {
      report.results.push({id:item.id,tier:item.tier,status:'incomplete',reason:'Validation deadline reserve reached'});
      continue;
    }
    const directory=join(output,item.id),abort=new AbortController(),cleanups=[];
    await mkdir(directory,{recursive:true});const before=performance.now();
    const timer=setTimeout(()=>{exhausted=true;abort.abort(new Error('Validation elapsed deadline reached'));},remaining);
    const result={id:item.id,tier:item.tier,status:'passed',seed,profile};
    try {
      result.evidence=await item.run({directory,signal:abort.signal,seed,profile,
        onCleanup:fn=>cleanups.push(fn),event:async value=>{
          const {appendFile}=await import('node:fs/promises');
          await appendFile(join(directory,'events.jsonl'),JSON.stringify(value)+'\n');
        }});
      if(abort.signal.aborted||performance.now()>=cutoff){exhausted=true;result.status='incomplete';}
    }catch(error){
      if(abort.signal.aborted||performance.now()>=cutoff)exhausted=true;
      result.status=exhausted?'incomplete':error instanceof Blocked?'blocked':'failed';
      result.error={name:error.name,message:error.message};}
    finally {
      clearTimeout(timer);
      for(const cleanup of cleanups.reverse())try{await cleanup();}catch(error){
        result.status='failed';(result.cleanupErrors??=[]).push(error.message);
      }
    }
    result.elapsedSeconds=(performance.now()-before)/1000;report.results.push(result);await save();
  }
  report.elapsedSeconds=(performance.now()-started)/1000;
  report.status=report.results.every(r=>r.status==='passed')?'passed':
    report.results.some(r=>r.status==='failed')?'failed':
    report.results.some(r=>r.status==='incomplete')?'incomplete':'blocked';
  if(report.elapsedSeconds>deadlineSeconds&&report.status!=='failed')report.status='incomplete';
  await save();return {...report,output};
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  try {
    const {values}=parseArgs({options:{tier:{type:'string',default:'all'},case:{type:'string'},
      seed:{type:'string',default:'1'},profile:{type:'string',default:'normal'},output:{type:'string'},
      'deadline-seconds':{type:'string',default:'5400'}}});
    const directory=dirname(fileURLToPath(import.meta.url)),cases=[];
    for(const file of (await readdir(directory)).filter(f=>f.endsWith('.case.mjs')).sort())
      cases.push(...(await import(pathToFileURL(join(directory,file)))).cases);
    const report=await runCases(cases,{tier:values.tier,caseId:values.case,seed:Number(values.seed),
      profile:values.profile,output:values.output,deadlineSeconds:Number(values['deadline-seconds'])});
    console.log(JSON.stringify(report));process.exitCode=report.status==='passed'?0:1;
  }catch(error){console.error(error.message);process.exitCode=2;}
}
