import { readFile,writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { summarize } from './curriculum.mjs';

export async function findingsFor(options) {
  const {state,job,path,passed}=options,file=join(path,'findings.json');
  try{return JSON.parse(await readFile(file,'utf8'));}
  catch(error){if(error.code!=='ENOENT')throw error;}
  const events=state.queryEvents(job.attempt.id,{types:['verification','rejected','attempt-error','failure-recorded','stage-completed','curriculum-session','curriculum-interrupted'],order:'desc',limit:12}).reverse().map(({type,data})=>({type,data}));
  const evidence=join(path,'diagnostics.json'),content=JSON.stringify(events,null,2)+'\n';
  await writeFile(evidence,content);
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
  await writeFile(file,JSON.stringify(findings,null,2)+'\n',{mode:0o600});
  return findings;
}
