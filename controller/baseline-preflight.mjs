import {writeFile,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';

// Run through the attempt's existing resource admission, execution journal and
// frozen-file checks. CPU construction proves the math, not target placement.
export async function validateTargetBaseline({project,state,attempt,attemptPath,
  execute,checkVerifier,signal,remaining,register}) {
  signal?.throwIfAborted();await checkVerifier();
  const command=project.verifier.baseline_command;
  const previous=state.queryEvents(attempt.id,{types:['baseline-preflight-completed'],order:'desc',limit:1})[0];
  if(previous?.data.passed===true&&previous.data.command===command)return;
  state.event(attempt.id,'baseline-preflight-started',{command});
  let output='';
  const response=await execute(command,project.source,{signal,timeout:remaining(),purpose:'performance',
    onData:chunk=>{output=(output+chunk.toString()).slice(-64000);}});
  signal?.throwIfAborted();await checkVerifier();
  // An uncertain transport or interrupted execution is never a verifier verdict.
  if(!response.settled||!Number.isInteger(response.exitCode))throw new Error('Target baseline execution unresolved');
  if(response.exitCode===124||response.exitCode<0||response.exitCode>=128)
    throw new Error('Target baseline interrupted before a verifier verdict');
  const path=join(attemptPath,'baseline-preflight.log');await writeFile(path,output);
  await register(path,'baseline-preflight');
  const evidence={path,sha256:createHash('sha256').update(await readFile(path)).digest('hex')};
  const verdict={command,passed:response.exitCode===0,exitCode:response.exitCode,evidence};
  state.event(attempt.id,'baseline-preflight-completed',verdict);
  if(!verdict.passed) {
    state.event(attempt.id,'project-invalid',{reason:'target-baseline-failed',...verdict});
    const error=new Error('Generated project target baseline failed before actor startup');
    error.code='PROJECT_BASELINE_INVALID';throw error;
  }
}
