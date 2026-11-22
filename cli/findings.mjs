import { readFile,writeFile,mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { summarize } from './curriculum.mjs';
import {projectSummary} from './project-summary.mjs';

export async function findingsFor(options) {
  const {state,job,path,passed}=options,file=join(path,'findings.json');
  const current=state.attempt(job.attempt.id),summary=await projectSummary({state,projectId:current.project});
  await writeFile(join(path,'project-summary.json'),JSON.stringify(summary,null,2)+'\n',{mode:0o600});
  options={...options,projectSummary:summary};
  try{const cached=JSON.parse(await readFile(file,'utf8'));if(cached.projectSummary?.identity===summary.identity)return cached;await mkdir(join(path,'findings-history'),{recursive:true});const retained=join(path,'findings-history',createHash('sha256').update(JSON.stringify(cached)).digest('hex')+'.json');await writeFile(retained,JSON.stringify(cached,null,2)+'\n',{flag:'wx',mode:0o600}).catch(error=>{if(error.code!=='EEXIST')throw error;});}
  catch(error){if(error.code!=='ENOENT')throw error;}
  const retain=async findings=>{const result={...findings,projectSummary:summary};await writeFile(file,JSON.stringify(result,null,2)+'\n',{mode:0o600});return result;};
  const invalid=state.queryEvents(job.attempt.id,{types:['project-invalid'],order:'desc',limit:1})[0];
  if(invalid) {
    const findings={findings:'Generated project rejected before actor startup: target performance baseline failed. '+
      JSON.stringify(invalid.data).slice(0,3000),useful:false,evidence:[invalid.data.evidence]};
    return retain(findings);
  }
  const events=state.queryEvents(job.attempt.id,{types:['verification','rejected','attempt-error','failure-recorded','stage-completed','curriculum-session','curriculum-interrupted'],order:'desc',limit:12}).reverse().map(({type,data})=>({type,data}));
  const evidence=join(path,'diagnostics.json'),content=JSON.stringify(events,null,2)+'\n';
  await writeFile(evidence,content);
  const infrastructure=!state.attempt(job.attempt.id).session&&
    state.queryEvents(job.attempt.id,{types:['preactor-infrastructure'],order:'desc',limit:1})[0];
  if(infrastructure) {
    const findings={findings:'Execution infrastructure failed before actor startup; no kernel verdict. '+
      JSON.stringify(infrastructure.data).slice(0,3000),useful:false,
      evidence:[infrastructure.data.evidence??{path:evidence,sha256:createHash('sha256').update(content).digest('hex')}]};
    return retain(findings);
  }
  let findings;
  try {findings=await summarize(options);}
  catch(error) {
    if(options.signal?.aborted)throw error;
    state.event(job.attempt.id,'summary-error',{message:error.message});
    const verified=events.some(e=>e.type==='verification' && e.data.passed);
    findings={findings:`Observed ${job.kind} outcome: ${passed?'passed':'failed'}. `+
      JSON.stringify(events.filter(e=>e.type!=='stage-completed')).slice(0,3500),
      useful:job.kind==='investigation' && passed && verified,
      evidence:[{path:evidence,sha256:createHash('sha256').update(content).digest('hex')}]};
  }
  return retain(findings);
}
