import {createHash} from 'node:crypto';

const execution=type=>type.startsWith('execution-')||type.startsWith('native-bash-');
export function administrativeHealthHistory(events) {
  const handles=new Set(events.filter(e=>e.type==='execution-handle'&&e.data.purpose==='resource-health').map(e=>e.data.handle));
  const submissions=new Map(),responses=new Map();
  for(const {type,data} of events) {
    if(type.startsWith('native-bash-'))throw Error('Portable binding cannot retire actor shell execution');
    if(type==='execution-submitting') {
      if(data.kind!=='run'||data.accountingExcluded!==true)throw Error('Portable binding requires only administrative health execution');
      submissions.set(data.submission,data);
    }
    if(type==='execution-submitted')responses.set(data.submission,data.handle);
    if(type==='execution-handle'&&(!handles.has(data.handle)||!/^local:exec-[A-Za-z0-9]+$/.test(data.handle)))
      throw Error('Portable binding cannot retire payload or foreign execution');
  }
  for(const id of submissions.keys())if(!handles.has(responses.get(id)))throw Error('Unidentified administrative submission');
  for(const {type,data} of events)if(execution(type)&&data.handle&&!handles.has(data.handle))throw Error('Unidentified historical execution');
  for(const {data} of events.filter(e=>['execution-starting','execution-unresolved'].includes(e.type))) {
    if(typeof data.operation!=='string'||!data.operation)throw Error('Unidentified historical operation');
    const identified=events.some(e=>e.type==='execution-handle'&&e.data.operation===data.operation&&handles.has(e.data.handle));
    const closed=events.some(e=>e.type==='execution-settled'&&e.data.operation===data.operation&&
      e.data.setupClosure?.closed===true&&e.data.setupClosure.payloadStarted===false&&
      Array.isArray(e.data.setupClosure.handles)&&e.data.setupClosure.handles.every(h=>handles.has(h)));
    if(!identified&&!closed)throw Error('Unidentified historical operation');
  }
  return [...handles].sort();
}
export function executionDigest(events) {
  const hash=createHash('sha256');
  for(const {type,data} of events)if(execution(type))hash.update(JSON.stringify({type,data}));
  return hash.digest('hex');
}

// A binding acknowledges only its proven source-side administrative prefix.
// New destination executions and all actor payloads retain ordinary fences.
export function destinationExecutionEvents(events) {
  const index=events.findLastIndex(e=>e.type==='portable-bound');
  if(index<0)return events;
  const proof=events[index].data,prefix=events.slice(0,index);
  try {
    if(proof.eventCount!==prefix.length||executionDigest(prefix)!==proof.executionSha256||
        JSON.stringify(administrativeHealthHistory(prefix))!==JSON.stringify(proof.healthHandles)||
        !/^[a-f0-9]{64}$/.test(proof.sourceShutdown?.sha256??''))return events;
    return events.slice(index+1);
  }catch{return events;}
}
