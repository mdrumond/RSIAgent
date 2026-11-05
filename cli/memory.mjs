import { spawn } from 'node:child_process';
import { resolve } from 'node:path';

export async function deliverLessons({config,queue,url,signal,invoke}) {
  for(const item of queue.lessons()) {
    signal?.throwIfAborted();
    if(config.storage) {
      const {publishStorageMemory}=await import('../controller/storage-knowledge.mjs');
      await publishStorageMemory(config,item,{url,signal,invoke});
      signal?.throwIfAborted();
      queue.acknowledgeLesson(item.id);
      continue;
    }
    const args={lesson:item.findings,outcome:item.outcome==='passed'?'good':'negative',
      applicability:config.target??'general',evidence:item.evidence,attempt_id:item.attempt_id};
    await new Promise((ok,no)=>{
      const child=spawn(config.python??'python',['-m','ascend_tools.kdb_client','--url',url,
        '--attempt',item.attempt_id,'--tool','record_lesson','--arguments',JSON.stringify(args)],{
        env:{...process.env,PYTHONPATH:resolve(new URL('..',import.meta.url).pathname)},signal,stdio:['ignore','ignore','pipe']});
      let error='',failure;child.stderr.on('data',d=>error+=d);child.on('error',value=>{failure=value;});
      child.on('close',code=>{if(failure)no(failure);else if(signal?.aborted)no(signal.reason??new Error('Memory delivery aborted'));else if(code)no(new Error(error));else ok();});
    });
    signal?.throwIfAborted();
    queue.acknowledgeLesson(item.id);
  }
}
