import {readdir} from 'node:fs/promises';
import {dirname,join} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {runCases} from './runner.mjs';

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
