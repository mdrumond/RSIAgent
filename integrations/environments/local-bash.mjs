import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {randomUUID} from 'node:crypto';
import {setTimeout as delay} from 'node:timers/promises';

const quote=value=>"'"+String(value).replaceAll("'","'\\''")+"'";
const notStarted=(error,stage)=>Object.assign(error,{localPrelaunch:{closed:true,payloadStarted:false,stage}});
const spawnFailure=error=>typeof error?.syscall==='string'&&error.syscall.startsWith('spawn ')
  ?notStarted(error,'shell-spawn'):error;
const beforeLaunch=options=>{if(options.signal?.aborted)throw notStarted(Object.assign(new Error('Local execution aborted before launch'),{code:'ABORT_ERR'}),'cancelled');};

const units=/^rsi-command-[a-f0-9-]+\.scope$/;
const counters=text=>Object.fromEntries(text.trim().split('\n').filter(Boolean).map(line=>{
  const [key,value]=line.split(/\s+/);return [key,Number(value)];
}));

// Pi owns command execution, output streaming, cancellation and timeouts. The
// operating system bounds its complete command cgroup, including detached children.
export function createLocalExecutionOperations(sdk,limits,{signal:cleanupSignal}={}) {
  const pi=sdk.createLocalBashOperations();
  if(!limits)return {...pi,async exec(command,cwd,options) {
    beforeLaunch(options);
    try{return await pi.exec(command,cwd,options);}catch(error){throw spawnFailure(error);}
  }};
  const {tasks=128,memory_bytes=4*1024**3}=limits;
  if(!Number.isSafeInteger(tasks)||tasks<4||!Number.isSafeInteger(memory_bytes)||memory_bytes<1024**2)
    throw Error('Positive local execution task and memory limits required');
  const control=async(command,{signal=cleanupSignal}={})=>{
    let output='';const result=await pi.exec(command,process.cwd(),{timeout:5,signal,onData:chunk=>{output=(output+chunk).slice(-16384);}});
    return {...result,output};
  };
  const observe=async(unit,{signal=cleanupSignal}={})=>{
    if(!units.test(unit))throw Error('Owned command scope required');
    const result=await control(`systemctl --user show ${quote(unit)} --property=LoadState --property=ActiveState --property=ControlGroup`,{signal});
    const properties=Object.fromEntries(result.output.trim().split('\n').filter(line=>line.includes('=')).map(line=>{
      const index=line.indexOf('=');return [line.slice(0,index),line.slice(index+1)];
    }));
    if(properties.LoadState==='not-found')return {state:'stopped'};
    if(result.exitCode!==0)return {state:'unknown'};
    return {state:['inactive','failed'].includes(properties.ActiveState)?'stopped':'running',cgroup:properties.ControlGroup};
  };
  const stop=async(unit,{signal=cleanupSignal}={})=>{
    if(!units.test(unit))throw Error('Owned command scope required');
    // A timeout/abort may have stopped systemd-run while its scope still has work.
    await control(`systemctl --user kill --kill-whom=all --signal=KILL ${quote(unit)}; systemctl --user stop ${quote(unit)}`,{signal});
    const result=await observe(unit,{signal});
    if(result.state!=='stopped')throw Error('Local command scope remains unresolved: '+unit);
    await control(`systemctl --user reset-failed ${quote(unit)}`,{signal});
    return result;
  };
  return {observe,stop,async exec(command,cwd,options) {
    beforeLaunch(options);
    const unit='rsi-command-'+randomUUID()+'.scope';
    let directory;try{directory=await mkdtemp(join(tmpdir(),'rsi-pi-command-'));}
    catch(error){throw notStarted(error,'containment-directory');}
    const ready=join(directory,'cgroup'),finished=join(directory,'limits.events');
    let done=false,violation,group,result,failure;
    try{await options.onContainment?.({unit,closed:false});beforeLaunch(options);}
    catch(error){await rm(directory,{recursive:true,force:true});throw notStarted(error,'containment-receipt');}
    // Bash builtins can retain the denial counter even when no further fork is possible.
    const script=`IFS=: read -r hierarchy controllers group < /proc/self/cgroup\nprintf '%s' "$group" > ${quote(ready)}\n`+
      `if ! read -r actual_tasks < "/sys/fs/cgroup$group/pids.max" || ! read -r actual_memory < "/sys/fs/cgroup$group/memory.max" || ! read -r actual_swap < "/sys/fs/cgroup$group/memory.swap.max" || [[ "$actual_tasks" != ${tasks} || "$actual_memory" != ${memory_bytes} || "$actual_swap" != 0 ]]; then printf 'Local execution resource limits are unavailable; command was not executed.\\n' >&2; exit 126; fi\n`+
      `trap ${quote(`{ while read -r key value; do printf 'pids.%s %s\\n' "$key" "$value"; done < "/sys/fs/cgroup$group/pids.events"; while read -r key value; do printf 'memory.%s %s\\n' "$key" "$value"; done < "/sys/fs/cgroup$group/memory.events"; } > ${quote(finished)}`)} EXIT\n`+
      `bash -c ${quote(command)}`;
    const wrapper=['systemd-run','--user','--scope','--quiet','--unit='+unit,
      '--property=TasksMax='+tasks,'--property=MemoryMax='+memory_bytes,'--property=MemorySwapMax=0',
      ...(options.timeout===undefined?[]:['--property=RuntimeMaxSec='+options.timeout+'s']),
      'bash','-c',script].map(quote).join(' ');
    const abort=new AbortController();
    const execution=Promise.resolve().then(()=>pi.exec(wrapper,cwd,{...options,signal:options.signal?AbortSignal.any([options.signal,abort.signal]):abort.signal}))
      .then(value=>{result=value;},error=>{failure=spawnFailure(error);}).finally(()=>{done=true;});
    try {
      while(!done) {
        try {
          group??=(await readFile(ready,'utf8')).trim();
          if(!group.startsWith('/')||!group.endsWith('/'+unit)||group.includes('..'))throw Error('Unexpected command cgroup');
          const denied=counters(await readFile('/sys/fs/cgroup'+group+'/pids.events','utf8')).max;
          const memory=counters(await readFile('/sys/fs/cgroup'+group+'/memory.events','utf8'));
          if(denied>0||memory.oom_kill>0){violation=denied>0?'process':'memory';await stop(unit,{signal:options.stopSignal});abort.abort();break;}
        }catch(error){if(!['ENOENT','ESRCH'].includes(error.code))throw error;}
        await delay(25);
      }
      await execution;
      try{const events=counters(await readFile(finished,'utf8'));
        if(events['pids.max']>0)violation='process';else if(events['memory.oom_kill']>0)violation='memory';
      }catch(error){if(error.code!=='ENOENT')throw error;}
      if(!violation&&result&&result.exitCode!==0) {
        const status=await control(`systemctl --user show ${quote(unit)} --property=Result --value`,{signal:options.stopSignal});
        if(status.output.trim()==='oom-kill')violation='memory';
      }
    }finally {
      try {
        try{if(!failure?.localPrelaunch)await stop(unit,{signal:options.stopSignal});}finally{abort.abort();await execution;}
        await options.onContainment?.({unit,closed:true});
      }finally{await rm(directory,{recursive:true,force:true});}
    }
    if(violation){options.onData(Buffer.from(`Command exceeded its ${violation} limit; all descendants were terminated. Fix the command and retry within the remaining allocation.\n`));
      return {exitCode:125,outcome:{kind:'infrastructure',reason:violation+'-limit'}};}
    if(failure)throw failure;
    if(result.exitCode!==0&&!group){options.onData(Buffer.from('Local command containment could not start; no uncontained fallback was executed.\n'));
      return {...result,outcome:{kind:'infrastructure',reason:'containment-unavailable'}};}
    return result;
  }};
}
