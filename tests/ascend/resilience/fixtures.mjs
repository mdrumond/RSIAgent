import {mkdir,readFile,writeFile,cp} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {createServer} from 'node:http';
import {once} from 'node:events';

export function schedule({seed=1,profile='normal',faults=[]}={}) {
  if(!['normal','slow','bursty'].includes(profile))throw Error('Unknown latency profile');
  const events=[],counts=new Map();let random=seed>>>0;
  return {events,async at(point,details={},signal) {
    const occurrence=(counts.get(point)??0)+1;counts.set(point,occurrence);
    random=(Math.imul(random,1664525)+1013904223)>>>0;
    const milliseconds=profile==='slow'?250:profile==='bursty'&&random%5===0?1000:10;
    const fault=faults.find(f=>f.point===point&&f.occurrence===occurrence);
    const event={point,occurrence,milliseconds,...details,...(fault?{fault}: {})};
    events.push(event);const before=performance.now();
    try{await delay(milliseconds,undefined,{signal});}
    finally{event.observedMilliseconds=performance.now()-before;}
    if(fault?.disconnect)throw Error(`Injected disconnect at ${point}`);
    return fault?.state?{state:fault.state}:null;
  }};
}

// Test-only retained backend. Production environment/SDK perform staging and execution.
// Keep this backend alive while restarting an observer to model retained CPL jobs.
export async function target({root,scheduler=schedule()}={}) {
  const sdk=await import('@earendil-works/pi-coding-agent');
  const workspace=join(root,'target'),jobs=new Map(),calls=[];let sequence=0;
  await mkdir(workspace,{recursive:true});
  const operations=sdk.createLocalBashOperations();
  const save=()=>writeFile(join(root,'backend.json'),JSON.stringify([...jobs].map(([handle,j])=>
    ({handle,state:j.state,exit:j.exit})),null,2));
  const client=async(args,{signal}={})=>{
    calls.push(args);const override=await scheduler.at(args[0],{handle:args[1]},signal);
    if(override)return override;
    if(args[0]==='capabilities')return {state:'available'};
    if(args[0]==='preflight')return {state:'completed'};
    if(args[0]==='upload'||args[0]==='download') {
      const source=args[0]==='upload'?args[2]:join(workspace,args[2]);
      const destination=args[0]==='upload'?join(workspace,args[3]):args[3];
      const handle=`remote:fixture:transfer:${++sequence}`,abort=new AbortController();
      const job={state:'running',stdout:'',abort};jobs.set(handle,job);await save();
      job.done=(async()=>{await mkdir(dirname(destination),{recursive:true});
        await cp(source,destination,{recursive:true,verbatimSymlinks:true});job.state=abort.signal.aborted?'cancelled':'completed';})()
        .catch(error=>{job.state='failed';job.stdout=String(error);}).then(save);
      if(args.includes('--receipt-file'))await writeFile(args[args.indexOf('--receipt-file')+1],JSON.stringify({handle}));
      await scheduler.at(`${args[0]}.after-receipt`,{handle},signal);
      return {state:'queued',handle};
    }
    if(args[0]==='run') {
      const script=await readFile(args[args.indexOf('--file')+1],'utf8');
      const handle=`remote:fixture:job:${++sequence}`,abort=new AbortController();
      const job={state:'running',stdout:'',abort};jobs.set(handle,job);await save();
      job.done=operations.exec(script,workspace,{signal:abort.signal,onData:data=>{job.stdout+=data;}})
        .then(result=>{job.exit=result.exitCode;job.state=abort.signal.aborted?'cancelled':'completed';})
        .catch(error=>{job.state=abort.signal.aborted?'cancelled':'failed';job.stdout+=String(error);})
        .then(save);
      await scheduler.at('run.before-receipt',{handle},signal);
      const receipt=args[args.indexOf('--receipt-file')+1];
      if(args.includes('--receipt-file'))await writeFile(receipt,JSON.stringify({handle}));
      await scheduler.at('run.after-receipt',{handle},signal);
      return {handle,state:'queued'};
    }
    const job=jobs.get(args[1]);if(!job)throw Error('Unknown fixture handle');
    if(args[0]==='cancel'){job.abort.abort();await job.done;return {state:job.state};}
    if(args[0]==='logs')return {state:'completed',content:job.stdout};
    if(args[0]==='fetch'){await job.done;return {state:job.state};}
    if(args[0]==='observe')return {state:job.state,exit:job.exit};
    throw Error(`Unexpected fixture operation ${args[0]}`);
  };
  return {workspace,client,calls,jobs,scheduler,async close(){
    for(const job of jobs.values())if(job.state==='running')job.abort.abort();
    await Promise.all([...jobs.values()].map(job=>job.done));
    await save();
    await writeFile(join(root,'schedule.json'),JSON.stringify(scheduler.events,null,2));
  }};
}

// Separate observer processes use the same retained backend, not a replacement dispatcher.
export async function serve(backend) {
  const server=createServer(async(req,res)=>{
    try {
      let text='';for await(const chunk of req)text+=chunk;
      const result=await backend.client(JSON.parse(text));
      res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify(result));
    }catch(error){res.writeHead(503);res.end(JSON.stringify({error:error.message}));}
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  return {url:`http://127.0.0.1:${server.address().port}`,async close(){
    server.closeAllConnections();await new Promise(resolve=>server.close(resolve));await backend.close();
  }};
}
export const connect=url=>async(args,{signal}={})=>{
  const response=await fetch(url,{method:'POST',body:JSON.stringify(args),signal});
  const result=await response.json();if(!response.ok)throw Error(result.error);return result;
};
