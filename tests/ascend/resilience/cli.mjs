import {readdir} from 'node:fs/promises';
import {dirname,join} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {runCases} from './runner.mjs';
import {requiredCases} from './manifest.mjs';

const abort=new AbortController(),stop=()=>abort.abort(new Error('Validation interrupted'));
process.on('SIGINT',stop);process.on('SIGTERM',stop);
try {
    const {values}=parseArgs({options:{tier:{type:'string',default:'all'},case:{type:'string'},
      seed:{type:'string',default:'1'},profile:{type:'string',default:'normal'},output:{type:'string'},
      'deadline-seconds':{type:'string',default:'5400'}}});
    const directory=dirname(fileURLToPath(import.meta.url)),cases=[];
    for(const file of (await readdir(directory)).filter(f=>f.endsWith('.case.mjs')).sort())
      cases.push(...(await import(pathToFileURL(join(directory,file)))).cases);
    cases.sort((a,b)=>{
      const rank=id=>{const n=requiredCases.findIndex(c=>c.id===id);return n<0?requiredCases.length:n;};
      return rank(a.id)-rank(b.id);
    });
    const report=await runCases(cases,{tier:values.tier,caseId:values.case,seed:Number(values.seed),
      profile:values.profile,output:values.output,deadlineSeconds:Number(values['deadline-seconds']),signal:abort.signal,
      requiredCases:values.case?[]:requiredCases.filter(c=>values.tier==='all'||c.tier===values.tier)});
    console.log(JSON.stringify(report));process.exitCode=report.status==='passed'?0:1;
}catch(error){console.error(error.message);process.exitCode=2;}
finally{process.off('SIGINT',stop);process.off('SIGTERM',stop);}
