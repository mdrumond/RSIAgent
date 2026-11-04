import { createEnvironment as genericEnvironment, createCplClient } from './index.mjs';
import { readFile, writeFile, mkdir, mkdtemp, rename, unlink } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
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
for device in c['devices']:
 rows.append(ns['local_probe'](physical_device=device,configuration=c['configuration'],remaining=lambda:max(0.,deadline-time.monotonic())))
 if time.monotonic()>=deadline: break
report=json.dumps({'observed_at':time.time(),'devices':rows},allow_nan=False)
print(report,file=sys.stderr if 'command' in c else sys.stdout,flush=True)
if 'command' in c:
 if len(rows)!=1 or rows[0]['admitted'] is not True: raise SystemExit(125)
 os.execvp('bash',['bash','-c',c['command']])
`;

/** CPL options plus health:{product,chip,allow_alarm?,allow_busy?}, instructionFile,
 * profile?, deviceAffinity?:physicalIDs[], probeSeconds? (query/freshness, default60),
 * probeObservationSeconds? (completion wait, default300, maximum600). context.resource/options.resource is the
 * leased row or opaque id returned by resources(); no caller-selected raw NPU ID.
 * Idle-only availability keeps the core's parallel leases independent of health policy.
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
  const assigned = context.resource ?? options.resource;
  const assignedId = typeof assigned === 'string' ? assigned : assigned?.id;
  const idFor = physical => 'ascend-' + digest(JSON.stringify(local
    ? ['local', options.target, options.runtime, physical] : [options.target, options.runtime, physical]));
  const root = join(context.instancePath, '.ascend-resources', digest(JSON.stringify([options.target, options.runtime, assignedId ?? 'inventory'])));
  await mkdir(root, { recursive: true, mode: 0o700 });
  const pendingPath = join(root, 'pending.json');
  const source = await readFile(new URL('../../ascend_tools/targets/health.py', import.meta.url), 'utf8');
  const healthSha256 = digest(source);
  const bindings = { ...options.env };
  const environment = await genericEnvironment({ ...options, env: bindings }, { ...context, client });
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
  function serial(fn) {
    const result = queue.then(fn);
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
  async function enabled() {
    const policy = await client(['npu', 'list', options.target]);
    if (policy.state !== 'completed') throw new Error('NPU permission inventory unavailable');
    const profile = policy.profiles?.find(row => row.target === options.target);
    if (!profile) throw new Error('NPU permission profile missing');
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
        if(result.exitCode!==0){await unlink(pendingPath);throw new Error(`Health probe failed: ${localPending.handle}`);}
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
        if(observed.exitCode!==0){await unlink(pendingPath);throw new Error(`Health probe failed: ${localPending.handle}`);}
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
        const request = { source, sha256: healthSha256, seconds: Math.min(seconds, budget), devices, configuration };
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
          throw new Error('Health probe was not admitted');
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
          signal?.throwIfAborted();
          await context.event('execution-state',{handle:pending.handle,purpose:'resource-health',
            state:terminal.has(observed.state)?'stopped':observed.state,
            ...(terminal.has(observed.state)?{executionSeconds:0,accountingExcluded:true}:{})});
          if(!terminal.has(observed.state))throw new Error(`Probe unresolved; observe ${pending.handle}`);
          break;
        }
        await delay(Math.min(options.pollMs??250,Math.max(0,callDeadline-performance.now())),undefined,{signal:observationSignal});
      }while(true);
      if (observed.state !== 'completed' || observed.exit !== 0) {
        await unlink(pendingPath);
        throw new Error(`Health probe failed: ${pending.handle}`);
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
  async function resources({ purpose = 'ordinary', target } = {}) {
    if (target && target !== options.target && target !== options.profile) return [];
    return serial(() => preflight(async emit => {
      const devices = await enabled();
      if (!devices.length) return [];
      const result = await snapshot(devices, purpose, emit);
      const permitted = new Set(await enabled());
      return result.devices.filter(row => devices.includes(row.physical_device) && permitted.has(row.physical_device)).map(row => ({
        id: idFor(row.physical_device),
        available: row.admitted === true && row.measurements?.idle === true,
        metadata: { target: options.target, runtime: options.runtime, product: options.health.product,
          physicalDevice: row.physical_device, logicalDevice: 0, deviceAffinity: affinity ?? null, observedAt: result.observed_at, checkedAt: result.checked_at,
          admitted: row.admitted === true, measurements: row.measurements, failure: row.failure,
          probeHandle: result.handle, healthSha256 },
      }));
    }));
  }
  return {
    ...environment, resources,
    instructions:environment.instructions+(local?'\nUse rsi_target for device execution, compilation and profiling: it applies the leased physical-device masks. Native local bash is for persistent file editing and inherits the worker launch environment; it does not select a leased device.':''),
    async execute(command, cwd, opts = {}) {
      const started = performance.now();
      let excludedMs = 0;
      const remaining = () => opts.timeout === undefined ? undefined : Math.max(0, opts.timeout - (performance.now() - started - excludedMs) / 1000);
      return serial(async () => {
        const setupStarted = performance.now();
        let physical;
        try { physical = await preflight(async emit => {
          if (opts.signal?.aborted || remaining() === 0) throw new Error('Execution cancelled or elapsed budget exhausted');
          const requested = opts.resourceId ?? assignedId;
          if (!requested || assignedId && requested !== assignedId) throw new Error('Assigned opaque Ascend resource required');
          const devices = await enabled();
          const physical = devices.find(device => idFor(device) === requested);
          if (physical === undefined) throw new Error('Assigned resource is not enabled for this target/runtime');
          const result = await snapshot([physical], opts.purpose ?? 'ordinary', emit, remaining() ?? seconds,opts.signal);
          const row = result.devices.find(item => item.physical_device === physical);
          if (row?.admitted !== true || !(await enabled()).includes(physical)) throw new Error('Fresh resource admission refused');
          return physical;
        }); } finally { excludedMs += performance.now() - setupStarted; }
        bindings.ASCEND_PHYSICAL_DEVICE = bindings.ASCEND_RT_VISIBLE_DEVICES = bindings.ASCEND_VISIBLE_DEVICES = String(physical);
        // Generic sync completes before this payload runs. Probe and exec share the
        // same declared native workload, masks, supervisor, receipt and elapsed budget.
        const request = { source, sha256: healthSha256, seconds, devices: [physical], command,
          configuration: { ...options.health, require_idle: timing(opts.purpose) || options.health.require_idle === true } };
        const guarded = `${quote(options.python ?? 'python3')} -c ${quote(probe)} ${quote(JSON.stringify(request))}`;
        // Native Bash supplies its full inherited env. Local runtime bindings win;
        // per-call-only values survive, and the leased masks always win last.
        const executionEnv = local ? {...opts.env,...options.env} : {...options.env,...opts.env};
        return environment.execute(guarded, cwd, { ...opts, timeout: remaining(), resourceId: physical,
          env: { ...executionEnv, ASCEND_PHYSICAL_DEVICE: String(physical), ASCEND_RT_VISIBLE_DEVICES: String(physical), ASCEND_VISIBLE_DEVICES: String(physical) } });
      });
    },
  };
}
