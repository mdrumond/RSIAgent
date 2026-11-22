/** Explicit A/B-only campaign entrypoint. Infrastructure failures pause, never retry. */
import {mkdir,appendFile,stat} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {State} from '../controller/state.mjs';
import {SkillBacklog} from '../controller/skill-backlog.mjs';
import {campaignBudget} from '../controller/campaign-budget.mjs';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';
import {deviceLeaseReceipt} from '../integrations/environments/device-leases.mjs';
import {readConfig,updateConfigImmediate} from './instance.mjs';
import {withSkillLock} from './evolution-lock.mjs';
import {failureFirstEnabled,runCampaignFailureStep,failureMemoryGeneration,failureCampaignSettled,failureCampaignStatus} from './skill-failure-campaign.mjs';
import {failureTiming} from './skill-failure-protocol.mjs';

const get=(state,key)=>JSON.parse(state.db.prepare('SELECT value FROM control WHERE key=?').get(key)?.value??'null');
const put=(state,key,value)=>state.db.prepare('INSERT OR REPLACE INTO control(key,value) VALUES(?,?)').run(key,JSON.stringify(value));
export function failureProtocol(config){
 const requested=config.skills?.failureFirst;
 if(!requested?.enabled||!['a3','a5'].includes(config.target))throw Error('Explicit A3 or A5 failure-first configuration required');
 if(!/^[a-f0-9]{40}$/.test(requested.sourceRevision??''))throw Error('Pin the installed machinery source revision before A/B startup');
 const protocol={workflow:'failure-first-v1',...failureTiming,sampleSeconds:failureTiming.actorSeconds+failureTiming.verificationSeconds,concurrency:4,repetitions:5,candidates:3};
 for(const [key,value]of Object.entries(protocol))if(requested[key]!==undefined&&requested[key]!==value)throw Error('Unsupported failure-first protocol field: '+key);
 if(requested.objectives&&JSON.stringify(requested.objectives)!==JSON.stringify(['correctness']))throw Error('Initial failure-first rollout supports correctness only');
 return protocol;
}

export async function selectFailureSlots(config,{directory,signal,retained},dependencies={}){
 signal=signal?AbortSignal.any([signal,AbortSignal.timeout(120000)]):AbortSignal.timeout(120000);
 await mkdir(directory,{recursive:true});const factory=dependencies.createEnvironment??(await import(pathToFileURL(resolve(config.root,config.environment.module)))).createEnvironment;
 const env=await factory({...config.environment.options,localExecution:config.local_execution},{instancePath:directory,campaignRoot:config.root,activateCleanup:false,signal,
  exclude:async(_name,fn)=>fn(),event:(type,data)=>appendFile(join(directory,'environment-events.jsonl'),JSON.stringify({type,data,created:new Date().toISOString()})+'\n')});
 const inventory=await env.inspectDevices({target:config.environment.options.target,purpose:'ordinary'}),resources=await env.resources({target:config.target});
 const state=new State(join(config.root,'state.sqlite'));let leased;
 try{leased=new Set(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='resource_leases'").get()?state.db.prepare('SELECT resource FROM resource_leases').all().map(r=>r.resource):[]);}finally{state.close();}
 const hosts=resources.filter(r=>r.available&&r.metadata?.kind==='host-slot'&&!leased.has(r.id)&&(!config.resource_ids||config.resource_ids.includes(r.id)))
  .sort((a,b)=>Number(!!b.metadata.backgroundSlot)-Number(!!a.metadata.backgroundSlot)||a.metadata.hostSlot-b.metadata.hostSlot);
 const devices=[];
 for(const row of inventory.filter(d=>d.available&&d.metadata?.measurements?.idle===true)){
  const physical=row.metadata.physicalDevice,pool=hosts[0]?.metadata.pool;if(!Number.isInteger(physical)||!pool)continue;
  try{await stat((dependencies.leaseReceipt??deviceLeaseReceipt)(pool,physical));}catch(e){if(e.code!=='ENOENT')throw e;devices.push(physical);}
 }
 devices.sort((a,b)=>Number(b===config.environment.options.backgroundDevice)-Number(a===config.environment.options.backgroundDevice)||b-a);
 if(retained){if(retained.length!==4||retained.some(s=>!devices.includes(s.physical)||!hosts.some(r=>r.id===s.resource.id)))throw Error('Retained A/B resources are unavailable; keep their exact bindings');return retained;}
 if(hosts.length<4||devices.length<4)throw Error('Four healthy idle unleased target devices and host slots required');
 return hosts.slice(0,4).map((resource,i)=>({resource,physical:devices[i]}));
}

export async function runFailureCampaign(config,{signal,once=false}={},dependencies={}){
 failureProtocol(config);const ops={read:readConfig,update:updateConfigImmediate,lock:withSkillLock,slots:selectFailureSlots,settled:failureCampaignSettled,
  memory:async args=>(await import('./skill-failure-memory.mjs')).prepareFailureMemory(args),step:runCampaignFailureStep,wait:delay,...dependencies};
 return ops.lock({...config,skill_queue_seconds:1},async()=>{
  const state=new State(join(config.root,'state.sqlite')),backlog=new SkillBacklog(state,{target:config.target}),owner=await processIdentity(),abort=new AbortController();
  const active=signal?AbortSignal.any([signal,abort.signal]):abort.signal;let clock,timer,memory,generation,slots,result;
  const status=(phase,extra={})=>backlog.meta('failure-first-status',{status:phase,at:Date.now(),owner,...extra});
  try{
   if(get(state,'pause')!==true)throw Error('Pause foreground work before starting four-device A/B');
   for(const key of ['owner','supervisor-owner','memory-writer']){const prior=get(state,key);if(prior&&!await ownerStopped(prior))throw Error('Campaign writer remains live: '+key);}
   if(!await ops.settled(config))throw Error('Previous A/B execution remains unresolved');
   const budget=get(state,'campaign-budget');if(!budget)throw Error('Retained campaign elapsed allocation required');
   clock=campaignBudget(state,{seconds:budget.seconds,cycle:budget.cycle,boot:owner.boot});
   put(state,'stop',false);put(state,'operator-stop',false);
   timer=setInterval(()=>{try{clock.checkpoint();if(clock.remaining<=0||get(state,'stop')||get(state,'pause')!==true)abort.abort(Error('Campaign stopped or elapsed allowance exhausted'));}catch(error){abort.abort(error);}},1000);
   const id=backlog.meta('failure-first-current'),retained=id&&backlog.run(id)?.checkpoint?.slots;
   status('device-admission');slots=await ops.slots(config,{directory:join(config.root,'skills','priority','failure-first-admission'),signal:active,retained});put(state,'skill-background-resources',slots);
   while(!active.aborted){
    config=await ops.read(config.root);failureProtocol(config);if(config.skills.tuningDisabled||!config.skills.background)throw Error('A/B disabled during execution');
    const next=failureMemoryGeneration(config,state);
    if(!memory||generation!==next){await memory?.close();memory=null;status('memory-readiness');memory=await ops.memory({config,directory:join(config.root,'skills','priority','failure-first-memory'),generation:next,signal:AbortSignal.any([active,AbortSignal.timeout(300000)])});generation=next;}
    status('running');result=await ops.step(config,{slots,memory,remainingSeconds:clock.remaining,signal:active});
    status(result.status,{reason:result.reason,...(result.stage?{stage:result.stage}:{}),
     ...(result.disposition?{disposition:result.disposition}:{}),...(result.learningRequest?{learningRequest:result.learningRequest}:{})});
    if(once)return result;
    if(result.status==='idle'){
     const pending=await failureCampaignStatus(config);
     if(pending.dirtyFailures>0&&!pending.hasWork)return result;
     await ops.wait(30000,undefined,{signal:active});
    }
   }
   active.throwIfAborted();
  }catch(error){
   const current=await ops.read(config.root);status('stopped',{reason:error.message,activeVersion:current.skills.activeVersion});await ops.update(config.root,(current,ledger)=>{put(ledger,'pause',true);put(ledger,'stop',true);return {...current,skills:{...current.skills,background:false,tuningDisabled:true}};});
   return {status:'stopped',reason:error.message,activeVersion:current.skills.activeVersion};
  }finally{
   clearInterval(timer);clock?.stop();
   try{
    await memory?.close();
    // Never erase resource identities before positive command closure.
    const id=backlog.meta('failure-first-current'),run=id&&backlog.run(id);
    if(!run?.checkpoint?.unresolved)put(state,'skill-background-resources',[]);
    else status('needs-settlement',{reason:'Owned commands remain unresolved',resources:slots});
   }catch(error){
    status('stopped',{reason:'Cleanup failed: '+error.message});await ops.update(config.root,(current,ledger)=>{put(ledger,'pause',true);put(ledger,'stop',true);return {...current,skills:{...current.skills,background:false,tuningDisabled:true}};});throw error;
   }finally{state.close();}
  }
 },{signal});
}
