/** Device admission before model construction, using the registered environment. */
import {appendFile,mkdir,readFile,stat} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {isDeepStrictEqual} from 'node:util';
import {deviceLeaseReceipt} from '../integrations/environments/device-leases.mjs';

export const deviceLearning=(config,request)=>request.learningTarget!=='cpu'&&config.target!=='cpu'&&
 (request.learningTarget==='device'||request.workKind==='suite');
export function refreshLearningRoute(child,config){
 const routing=['deviceAffinity','backgroundDevice'],without=options=>Object.fromEntries(Object.entries(options??{}).filter(([key])=>!routing.includes(key)));
 const modulePath=owner=>owner.environment.module?resolve(owner.root??config.root??'.',owner.environment.module):null;
 if(child.target!==config.target||modulePath(child)!==modulePath(config)||!isDeepStrictEqual(without(child.environment.options),without(config.environment.options)))return null;
 const options={...child.environment.options};for(const key of routing){delete options[key];if(Object.hasOwn(config.environment.options,key))options[key]=config.environment.options[key];}
 return {...child,environment:{...child.environment,options}};
}
async function events(directory){
 const text=await readFile(join(directory,'environment-events.jsonl'),'utf8').catch(e=>{if(e.code==='ENOENT')return '';throw e;});
 const lines=text.split('\n');lines.pop();return lines.filter(Boolean).map(line=>JSON.parse(line));
}
async function environment({config,directory,resource,signal},dependencies){
 await mkdir(directory,{recursive:true});
 const factory=dependencies.createEnvironment??(await import(pathToFileURL(resolve(config.root,config.environment.module)))).createEnvironment;
 return factory({...config.environment.options,localExecution:config.local_execution,asyncCleanup:false},{instancePath:directory,campaignRoot:config.root,resource,signal,activateCleanup:false,
  exclude:async(_label,fn)=>fn(),event:(type,data)=>appendFile(join(directory,'environment-events.jsonl'),JSON.stringify({type,data,created:new Date().toISOString()})+'\n')});
}
export async function learningAdmissionSettled(input,dependencies={}){
 const journal=await events(input.directory),submissions=journal.filter(e=>e.type==='execution-submitting');
 if(!submissions.length)return true;
 const closedHandles=new Set(journal.filter(e=>e.type==='execution-state'&&e.data.state==='stopped').map(e=>e.data.handle));
 const pending=submissions.filter(e=>{
  const response=journal.find(r=>['execution-submitted','execution-not-admitted'].includes(r.type)&&r.data.submission===e.data.submission);
  return !(response?.type==='execution-not-admitted'&&response.data.closed===true||response?.data.handle&&closedHandles.has(response.data.handle));
 });
 if(!pending.length)return true;
 const env=await environment({...input,signal:input.signal??AbortSignal.timeout(30000)},dependencies);
 for(const operation of new Set(pending.map(e=>e.data.operation??e.data.submission))){
  if(!env.reconcileOperation||!(await env.reconcileOperation({operation,events:journal})).closed)return false;
 }
 return true;
}
export async function admitLearning(input,dependencies={}){
 if(!deviceLearning(input.config,input.request))return {admitted:true,closed:true,kind:'cpu'};
 try{
  const env=await environment(input,dependencies);
  if(!env.inspectDevices)throw Error('Target environment does not support device admission');
  const rows=await env.inspectDevices({purpose:'ordinary',target:input.config.environment.options.target});
  const pool=input.resource.metadata?.pool;
  if(!pool)throw Error('Device learning requires a pool-bound scheduler resource');
  const blocked=[];
  for(const row of rows.filter(row=>row.available&&row.metadata?.admitted===true)){
   const physical=row.metadata.physicalDevice;
   if(!Number.isInteger(physical))continue;
   const receipt=deviceLeaseReceipt(pool,physical,dependencies.leaseDirectory);
   try{await stat(receipt);blocked.push({physicalDevice:physical,receipt,reason:'reservation-present'});}
   catch(error){if(error.code!=='ENOENT')throw error;return {admitted:true,closed:true,physicalDevice:physical};}
  }
  return {admitted:false,closed:true,reason:blocked.length?'device-reservation-unresolved':'no-healthy-idle-device',reservations:blocked};
 }catch(error){
  let closed=false;try{closed=await learningAdmissionSettled(input,dependencies);}catch{}
  return {admitted:false,closed,reason:'device-admission-unavailable',message:error.message};
 }
}
