import { createEnvironment as genericEnvironment, createCplClient, closedSetupError } from './index.mjs';
import { readFile, writeFile, mkdir, mkdtemp, rename, unlink } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { reserveDevice, openDeviceLease, hostLease, hostReconcile } from './device-leases.mjs';
import { join, dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const quote = value => "'" + String(value).replaceAll("'", "'\\''") + "'";
const digest = text => createHash('sha256').update(text).digest('hex');
const terminal = new Set(['completed', 'failed', 'cancelled']);
const timing = purpose => ['timing', 'profiling', 'performance'].includes(purpose);
const readJson = async path => JSON.parse(await readFile(path, 'utf8'));
async function save(path, value) {
  await writeFile(path + '.tmp', JSON.stringify(value), { mode: 0o600 });
  await rename(path + '.tmp', path);
}

// Health parsing and admission remain entirely in the target foundation helper.
const probe = `import hashlib,json,os,sys,time
c=json.loads(sys.argv[1]); source=c['source']
if hashlib.sha256(source.encode()).hexdigest()!=c['sha256']: raise ValueError('health helper identity mismatch')
ns={'__name__':'_ascend_health'}; exec(compile(source,'trusted-ascend-health.py','exec'),ns)
deadline=time.monotonic()+c['seconds']; rows=[]
for device in ([] if 'command' in c else c['devices']):
 rows.append(ns['local_probe'](physical_device=device,configuration=c['configuration'],remaining=lambda:max(0.,deadline-time.monotonic())))
 if time.monotonic()>=deadline: break
report=json.dumps({'observed_at':time.time(),'devices':rows},allow_nan=False)
if 'command' not in c: print(report,flush=True)
if 'command' in c:
 exec(c['hostLease'])
`;

/** Scheduler resources are host slots. Each execute selects an idle physical
 * device from fresh CPL permission/health evidence and holds one command lease.
 * inspectDevices is diagnostic inventory, never an attempt reservation API.
 */
export async function createEnvironment(options, context) {
  const local = options.mode === 'local';
  if ((!local && options.mode !== 'cpl') || !options.target || !options.runtime || !options.instructionFile)
    throw new Error('Ascend binding requires target, registered runtime, and instructionFile');
  if (!['a3', 'a5'].includes(options.health?.product)) throw new Error('Explicit A3/A5 health profile required');
  const seconds = options.probeSeconds ?? 60;
  const observationSeconds = options.probeObservationSeconds ?? 300;
  if (!Number.isFinite(observationSeconds) || observationSeconds <= 0 || observationSeconds > 600)
    throw new Error('probeObservationSeconds must be in (0,600]');
  if (!Number.isFinite(seconds) || seconds <= 0 || seconds > 120) throw new Error('probeSeconds must be in (0,120]');
  if (options.deviceAffinity !== undefined && (!Array.isArray(options.deviceAffinity) || options.deviceAffinity.some(id => !Number.isInteger(id) || id < 0)))
    throw new Error('deviceAffinity must contain explicit nonnegative physical IDs');
  const affinity = options.deviceAffinity?.slice();
  await context.event('resource-policy', { target: options.target, runtime: options.runtime, deviceAffinity: affinity ?? null });
  const nativeClient = context.client ?? createCplClient(options.clientPath);
  const client = (args, {signal = context.signal} = {}) => {
    signal?.throwIfAborted();
    return nativeClient(args, {signal});
  };
  let activeLease, activeExclude;
  const leasePath = join(context.instancePath, '.ascend-command-lease.json');
  const commandContext = {...context, client, exclude:(label,fn)=>activeExclude?activeExclude(label,fn):context.exclude(label,fn), event: async (type, data) => {
    if (activeLease && ['execution-submitting','execution-submitted','execution-not-admitted','execution-handle'].includes(type)) {
      await activeLease.event(type,data);
      if(type==='execution-handle'&&!data.purpose)await activeLease.update({[activeLease.record.controlStarted?'controlHandle':'payloadHandle']:data.handle});
    }
    await context.event(type, data);
  }};
  const idFor = physical => 'ascend-' + digest(JSON.stringify(local
    ? ['local', options.target, options.runtime, physical] : [options.target, options.runtime, physical]));
  const root = join(context.instancePath, '.ascend-resources', digest(JSON.stringify([options.target, options.runtime, 'inventory'])));
  await mkdir(root, { recursive: true, mode: 0o700 });
  const pendingPath = join(root, 'pending.json');
  const source = await readFile(new URL('../../ascend_tools/targets/health.py', import.meta.url), 'utf8');
  const healthSha256 = digest(source);
  const bindings = { ...options.env };
  const environment = await genericEnvironment({ ...options, env: bindings }, commandContext);
  let localPending;
  const localProbe = local ? await genericEnvironment(options, {...context,event:async(type,data)=>{
    if(type==='execution-handle'&&localPending){
      localPending.handle=data.handle;await save(pendingPath,localPending);
      await context.event('execution-submitted',{submission:localPending.submission,handle:data.handle,
        closed:false,accountingExcluded:true});
    }
    await context.event(type,{...data,purpose:'resource-health',
      ...(type==='execution-state'&&data.state==='stopped'?{executionSeconds:0,accountingExcluded:true}:{})});
  }}) : null;
  let queue = Promise.resolve();
  function serial(fn, wait = operation => operation()) {
    const previous = queue;
    const result = (async () => { await wait(() => previous); return fn(); })();
    queue = result.catch(() => {});
    return result;
  }
  async function preflight(fn) {
    return context.exclude('setup', async () => {
      let linked = false;
      const emit = async (type, data) => {
        if (type === 'execution-submitting') linked = true;
        await context.event(type, data);
      };
      try { return await fn(emit); }
      catch (error) {
        if (!linked) {
          const submission = randomUUID();
          await emit('execution-submitting', { submission, kind: 'resource-policy', accountingExcluded: true });
          await emit('execution-not-admitted', { submission, closed: true });
        }
        throw error;
      }
    });
  }
  let pool;
  async function enabled() {
    const policy = await client(['npu', 'list', options.target]);
    if (policy.state !== 'completed') throw new Error('NPU permission inventory unavailable');
    const profile = policy.profiles?.find(row => row.target === options.target);
    if (!profile) throw new Error('NPU permission profile missing');
    if (!local && (typeof profile.pool !== 'string' || !profile.pool.trim())) throw new Error('CPL NPU profile requires an actual alias-group pool');
    pool = profile.pool ?? 'local:' + options.target;
    return profile.devices.filter(row => row.enabled === true && Number.isInteger(row.id) && row.id >= 0).map(row => row.id).filter(id => !affinity || affinity.includes(id));
  }
  async function localSnapshot(devices,purpose,emit,budget,signal) {
    const bounded=AbortSignal.any([AbortSignal.timeout(Math.ceil(Math.min(observationSeconds,budget)*1000)),signal].filter(Boolean));
    const configuration={...options.health,require_idle:timing(purpose)||options.health.require_idle===true};
    try {for(;;){
      bounded.throwIfAborted();
      try{localPending=await readJson(pendingPath);}catch(error){if(error.code!=='ENOENT')throw error;localPending=null;}
      if(!localPending){
        const dir=await mkdtemp(join(root,'probe-')),submission=randomUUID();
        localPending={submission,startedMonotonic:process.hrtime.bigint().toString(),devices,configuration,
          seconds:Math.min(seconds,budget),output:join(dir,'stdout.json'),handle:null};
        const request={source,sha256:healthSha256,seconds:localPending.seconds,devices,configuration};
        await save(pendingPath,localPending);
        if(bounded.aborted){await unlink(pendingPath);bounded.throwIfAborted();}
        await emit('execution-submitting',{submission,receiptFile:pendingPath,kind:'run',accountingExcluded:true});
        const command=`${quote(options.python??'python3')} -c ${quote(probe)} ${quote(JSON.stringify(request))} > ${quote(localPending.output)}`;
        let result;
        try{result=await localProbe.execute(command,context.instancePath,{signal:bounded,
          timeout:Math.min(seconds,budget,observationSeconds),env:options.env});}
        catch(error){
          // Generic local execution records its handle before spawning native Bash.
          if(!localPending.handle){await unlink(pendingPath);await emit('execution-not-admitted',{submission,closed:true});}
          throw error;
        }
        if(!result.settled)throw new Error(`Local probe closure unknown; observe ${localPending.handle}`);
        if(result.exitCode!==0){await unlink(pendingPath);throw closedSetupError(`Health probe failed: ${localPending.handle}`,{handle:localPending.handle,reason:result});}
      }else{
        await emit('execution-submitting',{submission:localPending.submission,receiptFile:pendingPath,kind:'run',accountingExcluded:true});
        if(!localPending.handle)throw new Error('Local probe submission identity unknown; do not resubmit');
        await emit('execution-submitted',{submission:localPending.submission,handle:localPending.handle,closed:false,accountingExcluded:true});
        await context.event('execution-handle',{handle:localPending.handle,purpose:'resource-health'});
        let observed;
        do{
          bounded.throwIfAborted();observed=await localProbe.observe(localPending.handle);
          if(observed.state==='running')await delay(25,undefined,{signal:bounded});
        }while(observed.state==='running');
        if(observed.state!=='stopped')throw new Error(`Local probe unresolved; observe ${localPending.handle}`);
        if(observed.exitCode!==0){await unlink(pendingPath);throw closedSetupError(`Health probe failed: ${localPending.handle}`,{handle:localPending.handle,reason:observed});}
      }
      const result=await readJson(localPending.output),retained={...result,checked_at:Date.now()/1000,
        handle:localPending.handle,healthSha256};
      const observation=join(dirname(localPending.output),'observation.json');
      await save(observation,retained);await save(join(root,'last-observation.json'),retained);
      await context.event('resource-probe-completed',{handle:localPending.handle,receipt:pendingPath,observation,
        executionSeconds:0,accountingExcluded:true});
      const age=Number(process.hrtime.bigint()-BigInt(localPending.startedMonotonic))/1e9;
      await unlink(pendingPath);
      if(!Number.isFinite(age)||age<0||age>seconds||JSON.stringify(devices)!==JSON.stringify(localPending.devices)||
          JSON.stringify(configuration)!==JSON.stringify(localPending.configuration)){
        await context.event('resource-probe-stale',{handle:localPending.handle,age});continue;
      }
      return retained;
    }}finally{localPending=null;}
  }
  async function snapshot(devices, purpose, emit, budget = seconds, signal = context.signal) {
    if (!(budget > 0)) throw new Error('Resource probe elapsed allocation exhausted');
    if(local)return localSnapshot(devices,purpose,emit,budget,signal);
    const deadlineSignal=AbortSignal.timeout(Math.ceil(observationSeconds*1000));
    const observationSignal=signal?AbortSignal.any([signal,deadlineSignal]):deadlineSignal;
    const callDeadline=performance.now()+observationSeconds*1000;
    while(true) {
      signal?.throwIfAborted();
      if(observationSignal.aborted)throw new Error('Resource probe observation elapsed allocation exhausted');
      const submission = randomUUID();
      let pending;
      try { pending = await readJson(pendingPath); } catch (error) { if (error.code !== 'ENOENT') throw error; }
      const configuration = { ...options.health, require_idle: timing(purpose) || options.health.require_idle === true };
      if (!pending) {
        const dir = await mkdtemp(join(root, 'probe-'));
        const file = join(dir, 'probe.sh'), receipt = join(dir, 'native-submission.json');
        const request = {  source, sha256: healthSha256, seconds: Math.min(seconds, budget), devices, configuration };
        await writeFile(file, `${quote(options.python ?? 'python3')} -c ${quote(probe)} ${quote(JSON.stringify(request))}\n`, { mode: 0o600 });
        pending = { started: Date.now(), startedMonotonic: process.hrtime.bigint().toString(), seconds: request.seconds, devices, configuration, receipt, handle: null };
        // Persist ambiguous submission BEFORE calling transport; never infer no work from no ID.
        await save(pendingPath, pending);
        await emit('execution-submitting', { submission, receiptFile: receipt, kind: 'run', accountingExcluded: true });
        let submitted;
        try { submitted = await client(['run', options.target, '--runtime', options.runtime, '--file', file, '--receipt-file', receipt]); }
        catch { try { submitted = await readJson(receipt); } catch { /* unknown; retain pending */ } }
        if (submitted?.admitted === false || submitted?.submitted === false) {
          await emit('execution-not-admitted', { submission, closed: true });
          await unlink(pendingPath);
          throw closedSetupError('Health probe was not admitted',{notAdmitted:true,reason:submitted.reason});
        }
        if (submitted?.handle) { pending.handle = submitted.handle; await save(pendingPath, pending); }
      } else {
        await emit('execution-submitting', { submission, receiptFile: pending.receipt, kind: 'run', accountingExcluded: true });
      }
      if (!pending.handle) {
        try { pending.handle = (await readJson(pending.receipt)).handle; await save(pendingPath, pending); } catch { /* preserve ambiguity */ }
      }
      if (!pending.handle) {
        await emit('execution-unresolved', { submission, handle: null, message: 'Probe submission identity unknown; do not resubmit' });
        throw new Error(`Probe submission identity unknown; do not resubmit (${pendingPath})`);
      }
      await emit('execution-submitted', { submission, handle: pending.handle, closed: false, accountingExcluded: true });
      await context.event('execution-handle', { handle: pending.handle, purpose: 'resource-health' });
      const elapsed = () => Number(process.hrtime.bigint() - BigInt(pending.startedMonotonic)) / 1e9;
      let observed;
      do {
        signal?.throwIfAborted();
        const queryStarted=process.hrtime.bigint().toString();
        try { observed = await client(['observe',pending.handle],{signal:observationSignal}); }
        catch { observed = {state:'observation-unavailable'}; }
        await context.event('execution-state',{handle:pending.handle,purpose:'resource-health',
          ...(observed.reason == null ? {} : {reason:observed.reason}),
          state:terminal.has(observed.state)?'stopped':observed.state,
          ...(terminal.has(observed.state)?{executionSeconds:0,accountingExcluded:true}:{})});
        if(terminal.has(observed.state))break;
        signal?.throwIfAborted();
        if(!['queued','dispatching','running'].includes(observed.state) || observationSignal.aborted)
          throw new Error(`Probe unresolved; observe ${pending.handle}`);
        if(observed.state==='queued') {
          // A confirmed queued job has not begun measuring. This is a conservative
          // local lower bound on query start; never reset freshness on completion.
          pending.freshnessMonotonic=queryStarted;await save(pendingPath,pending);
        }
        if(observed.state==='running') {
          try { observed=await client(['observe',pending.handle,'--wait','--timeout',String(Math.max(1,Math.ceil((callDeadline-performance.now())/1000)))],{signal:observationSignal}); }
          catch { observed={state:'observation-unavailable'}; }
          await context.event('execution-state',{handle:pending.handle,purpose:'resource-health',
            ...(observed.reason == null ? {} : {reason:observed.reason}),
            state:terminal.has(observed.state)?'stopped':observed.state,
            ...(terminal.has(observed.state)?{executionSeconds:0,accountingExcluded:true}:{})});
          if(terminal.has(observed.state))break;
          signal?.throwIfAborted();
          throw new Error(`Probe unresolved; observe ${pending.handle}`);
        }
        await delay(Math.min(options.pollMs??250,Math.max(0,callDeadline-performance.now())),undefined,{signal:observationSignal});
      }while(true);
      if (observed.state !== 'completed' || observed.exit !== 0) {
        await unlink(pendingPath);
        throw closedSetupError(`Health probe failed: ${pending.handle}`,{handle:pending.handle,reason:observed});
      }
      // If log retrieval disconnects, retain pending handle and retry the SAME logs next call.
      const logs = await client(['logs', pending.handle, '--stream', 'stdout', '--tail', '1'],{signal:observationSignal});
      signal?.throwIfAborted();
      const result = JSON.parse(logs.content);
      const retained={...result,checked_at:Date.now()/1000,handle:pending.handle,healthSha256};
      await save(join(dirname(pending.receipt),'observation.json'),retained);
      await save(join(root,'last-observation.json'),retained);
      await context.event('resource-probe-completed',{handle:pending.handle,receipt:pending.receipt,observation:join(dirname(pending.receipt),'observation.json'),executionSeconds:0,accountingExcluded:true});
      await unlink(pendingPath);
      // Remote wall clocks may differ. Use submission or the last confirmed queued
      // observation as an earliest possible query start, never terminal receipt time.
      const age=Number(process.hrtime.bigint()-BigInt(pending.freshnessMonotonic??pending.startedMonotonic))/1e9;
      if(!Number.isFinite(age)||age<0||age>seconds||JSON.stringify(configuration)!==JSON.stringify(pending.configuration)||JSON.stringify(devices)!==JSON.stringify(pending.devices)) {
        await context.event('resource-probe-stale',{handle:pending.handle,age,submission_elapsed:elapsed()});
        continue; // Only a KNOWN terminal probe is replaced with fresh measurements.
      }
      return retained;
    }
  }
  async function inspectDevices({ purpose = 'ordinary', target } = {}) {
    if (target && target !== options.target && target !== options.profile) return [];
    return serial(() => preflight(async emit => {
      const devices = await enabled();
      if (!devices.length) return [];
      const result = await snapshot(devices, purpose, emit);
      const permitted = new Set(await enabled());
      return result.devices.filter(row => devices.includes(row.physical_device) && permitted.has(row.physical_device)).map(row => ({
        id: idFor(row.physical_device),
        available: row.admitted === true && row.measurements?.idle === true,
        ...(local?{env:{ASCEND_PHYSICAL_DEVICE:String(row.physical_device),
          ASCEND_RT_VISIBLE_DEVICES:String(row.physical_device),ASCEND_VISIBLE_DEVICES:String(row.physical_device)}}:{}),
        metadata: { target: options.target, runtime: options.runtime, product: options.health.product,
          physicalDevice: row.physical_device, logicalDevice: 0, deviceAffinity: affinity ?? null, observedAt: result.observed_at, checkedAt: result.checked_at,
          admitted: row.admitted === true, measurements: row.measurements, failure: row.failure,
          probeHandle: result.handle, healthSha256 },
      }));
    }));
  }
  // Scheduler leases represent host concurrency slots, never physical reservations.
  async function resources({target} = {}) {
    if (target && target !== options.target && target !== options.profile) return [];
    const devices = await enabled();
    if (!devices.length) return [];
    return devices.map((_, slot) => ({ id: 'ascend-host-' + digest(JSON.stringify([pool, slot])),
      available: true, metadata: {kind:'host-slot', pool, target:options.target} }));
  }
  let commandActive=false, controlEnvironment;
  async function control() {
    if(controlEnvironment)return controlEnvironment;
    const path=join(root,'lease-control'),sourcePath=join(path,'source');
    await mkdir(sourcePath,{recursive:true});
    controlEnvironment=await genericEnvironment(options,{...commandContext,signal:undefined,client:(args,settings)=>nativeClient(args,settings),event:async(type,data)=>{
      await commandContext.event(type,{...data,...(type==='execution-submitting'&&data.kind==='run'&&data.accountingExcluded===false?{controlPayload:true}:{}),purpose:'device-lease-control',accountingExcluded:true});
      if(activeLease&&type==='execution-handle'&&!data.purpose)await activeLease.update({controlHandle:data.handle});
    }});
    await controlEnvironment.prepare({project:{source:sourcePath},attempt:{id:'lease-control'}});
    return controlEnvironment;
  }
  async function releaseLease(lease) {
    await lease.release();
    await unlink(leasePath);
    await context.event('device-lease-released', {pool:lease.record.pool,
      physicalDevice:lease.record.physicalDevice, receipt:lease.receipt, token:lease.token});
  }
  let reconciling;
  function reconcileLease(){
    if(!reconciling)reconciling=reconcileCurrent().finally(()=>{reconciling=null;});
    return reconciling;
  }
  async function reconcileCurrent() {
    let retained;
    try{retained=await readJson(leasePath);}catch(error){if(error.code==='ENOENT')return true;throw error;}
    const lease=await openDeviceLease(retained.receipt,retained.token);
    if(lease.record.version!==2)return false;
    activeLease=lease;
    try {
      const handles=new Set(lease.record.handles);
      for(const row of lease.record.submissions){
        if(row.closed===true)continue;
        let handle=row.handle;
        if(!handle&&row.receiptFile)try{handle=(await readJson(row.receiptFile)).handle;}catch{}
        if(!handle)return false;
        if(row.controlPayload&&!lease.record.controlHandle)await lease.update({controlHandle:handle});
        handles.add(handle);
      }
      // Re-observe exact identities; neither PID disappearance nor a fresh job is proof.
      for(const handle of handles)if((await environment.observe(handle,{signal:AbortSignal.timeout(observationSeconds*1000)})).state!=='stopped')return false;
      const cleanup=await control();
      // A positively refused control submission may be retried. Unknown submissions
      // returned above; a recovered payload handle is always observed, never replayed.
      if(lease.record.controlStarted&&!lease.record.controlHandle)await lease.update({controlStarted:false});
      if(!lease.record.controlStarted){
        await lease.update({controlStarted:true});
        const request={devices:[retained.physicalDevice],lease:{pool:retained.pool,token:lease.token}};
        const code="import os,sys,json,hashlib,time\nc=json.loads(sys.argv[1])\n"+hostReconcile;
        const result=await context.exclude('setup',()=>cleanup.execute(`${quote(options.python??'python3')} -c ${quote(code)} ${quote(JSON.stringify(request))}`,
          join(root,'lease-control','source'),{timeout:seconds,leaseReceipt:true}));
        if(!result.settled)return false;
      }
      if(!lease.record.controlHandle)return false;
      if((await cleanup.observe(lease.record.controlHandle)).state!=='stopped')return false;
      const receipt=await cleanup.readAdmission(lease.record.controlHandle);
      if(receipt?.token===lease.token&&receipt.decision==='contended'){
        await lease.update({controlStarted:false,controlHandle:null});return false;
      }
      if(receipt?.token!==lease.token||receipt.decision!=='reconciled')return false;
      await releaseLease(lease);return true;
    } finally {activeLease=null;}
  }
  return {
    ...environment, resources, inspectDevices,
    reconcileLease:()=>commandActive?Promise.resolve(false):reconcileLease(), shellOnTarget:local,
    async observe(handle,settings){
      const result=await environment.observe(handle,settings);
      if(result.state==='stopped'&&!commandActive)await reconcileLease();
      return result;
    },
    async reconcileOperation(operation){
      const result=await environment.reconcileOperation(operation);
      return {...result,closed:result.closed && await reconcileLease()};
    },
    instructions:environment.instructions+(local?'\nBoth bash and rsi_target execute in the persistent local workspace through the same device admission, masks and execution supervision.':''),
    async execute(command, cwd, opts = {}) {
      if(commandActive)throw Object.assign(new Error('Concurrent device commands on one actor are unsupported'),{code:'CONCURRENT_DEVICE_COMMAND'});
      commandActive=true;
      try {
      const signal = context.signal && opts.signal ? AbortSignal.any([context.signal, opts.signal]) : opts.signal ?? context.signal;
      const started = performance.now();
      let excludedMs = 0;
      const remaining = () => opts.timeout === undefined ? undefined : Math.max(0, opts.timeout - (performance.now() - started - excludedMs) / 1000);
      const exclude = async (label, fn) => {
        const before = performance.now();
        try { return await context.exclude(label, fn); }
        finally { excludedMs += performance.now() - before; }
      };
      let waiting = false;
      async function waitForDevice() {
        if (waiting) return;
        waiting = true;
        await context.event('device-wait', {pool, waiting:true, reason:'idle-devices'});
      }
      activeExclude=exclude;
      return await serial(async () => {
        if(!await reconcileLease())throw new Error('Device lease unresolved; observe retained handles before retry');
        for(;;){
        let physical, lease;
        await preflight(async emit => {
          for (;;) {
            signal?.throwIfAborted();
            if (remaining() === 0) throw new Error('Execution elapsed budget exhausted');
            const devices = await enabled();
            if (!devices.length) throw closedSetupError('No enabled devices in pool');
            const result = await exclude('setup', () => snapshot(devices, opts.purpose ?? 'ordinary', emit, seconds, signal));
            const permitted = new Set(await enabled());
            for (const row of result.devices) {
              if (!devices.includes(row.physical_device) || !permitted.has(row.physical_device) || row.admitted !== true || row.measurements?.idle !== true) continue;
              lease = await reserveDevice(pool, row.physical_device, context.leaseDirectory);
              if (lease) { physical = row.physical_device; break; }
            }
            if (lease) break;
            await waitForDevice();
            await exclude('queue-wait', () => delay(options.leasePollMs ?? 250, undefined, {signal}));
          }
        });
        activeLease = lease;
        // Persist before any transfer or submission; a missing handle is not closure.
        await save(leasePath, {receipt:lease.receipt, token:lease.token, pool, physicalDevice:physical});
        await context.event('device-lease-acquired', {pool, physicalDevice:physical, receipt:lease.receipt, token:lease.token});
        waiting = false;
        await context.event('device-wait', {pool, waiting:false});
        const request = {  source, sha256: healthSha256, seconds, devices: [physical], command, hostLease,
          lease:{pool,token:lease.token,receipt:lease.receipt},
          configuration: { ...options.health, allow_busy:false, require_idle:timing(opts.purpose) || options.health.require_idle === true } };
        const guarded = `${quote(options.python ?? 'python3')} -c ${quote(probe)} ${quote(JSON.stringify(request))}`;
        try {
          const attemptStarted=performance.now(),attemptExcluded=excludedMs;
          const measured=await context.exclude('admission',async()=>{
            const result = await environment.execute(guarded, cwd, { ...opts, signal, timeout: remaining(), resourceId: physical, leaseReceipt:true,admissionToken:lease.token,
              env: { ...options.env, ...opts.env, ASCEND_PHYSICAL_DEVICE: String(physical), ASCEND_RT_VISIBLE_DEVICES: String(physical), ASCEND_VISIBLE_DEVICES: String(physical) } });
            const admission=result.settled?await environment.readAdmission(result.handle,{signal:AbortSignal.timeout(observationSeconds*1000)}):null;
            return {result,admission,queueWait:admission?.token===lease.token && admission.decision==='contended'};
          });
          const {result,admission}=measured;
          if(!result.settled)return result;
          if(admission?.token!==lease.token)return {...result,settled:false,admissionUnknown:true};
          if(measured.queueWait){
            excludedMs+=performance.now()-attemptStarted-(excludedMs-attemptExcluded);
            await context.event('execution-state',{handle:result.handle,state:'stopped',executionSeconds:0,accountingExcluded:true});
          }
          if(['contended','health-refused'].includes(admission.decision)){
            await releaseLease(lease);
          } else {
            activeLease=null;
            if(!await reconcileLease())return {...result,settled:false,admissionUnknown:true};
          }
          if(admission.decision==='contended'){
            await waitForDevice();
            await exclude('queue-wait',()=>delay(options.leasePollMs??250,undefined,{signal}));
            continue;
          }
          return {...result,admission};
        } catch (error) {
          // Even known native closure requires target-fence reconciliation.
          if(error.closure?.closed){activeLease=null;await reconcileLease();}
          throw error;
        } finally { activeLease = null; }
        }
      });
      } finally {commandActive=false;activeExclude=null;}
    },
  };
}
