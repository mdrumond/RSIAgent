import { spawn } from 'node:child_process';
import { resolve } from 'node:path';

export async function deliverLessons({config,queue,url}) {
  for(const item of queue.lessons()) {
    const args={lesson:item.findings,outcome:item.outcome==='passed'?'good':'negative',
      applicability:config.target??'general',evidence:item.evidence,attempt_id:item.attempt_id};
    await new Promise((ok,no)=>{
      const child=spawn(config.python??'python',['-m','ascend_tools.kdb_client','--url',url,
        '--attempt',item.attempt_id,'--tool','record_lesson','--arguments',JSON.stringify(args)],{
        env:{...process.env,PYTHONPATH:resolve(new URL('..',import.meta.url).pathname)},stdio:['ignore','ignore','pipe']});
      let error='';child.stderr.on('data',d=>error+=d);child.on('error',no);
      child.on('close',code=>code?no(new Error(error)):ok());
    });
    queue.acknowledgeLesson(item.id);
  }
}
