import { join, resolve, dirname } from 'node:path';
import { ownerStopped, operationsStopped } from '../controller/lifecycle.mjs';
import { bootstrap as createInstance, readConfig } from './instance.mjs';
import { spawn } from 'node:child_process';
import { open, readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { State } from '../controller/state.mjs';

// Every A/B sample gets its own state, source, verifier, memory and native evidence.
export async function bootstrap(request) {
  request.signal?.throwIfAborted();
  const original=await readConfig(request.sourceInstanceRoot);
  const ledger=new State(join(original.root,'state.sqlite'));
  try {
    if(!request.resource) {
      const row=ledger.db.prepare("SELECT name FROM sqlite_master WHERE name='control'").get();
      const owner=row && ledger.db.prepare("SELECT value FROM control WHERE key='owner'").get();
      if(owner && JSON.parse(owner.value) && !await ownerStopped(JSON.parse(owner.value)))
        throw new Error('Stop the campaign before manual isolated evaluation');
    }
    for(const event of ledger.events(null).filter(event=>event.type==='evaluation-started')) {
      try {
        await readConfig(event.data.instanceRoot);
        const previous=new State(join(event.data.instanceRoot,'state.sqlite'));
        try {
          if(previous.db.prepare('SELECT 1 FROM attempts WHERE outcome IS NULL LIMIT 1').get())
            throw new Error(`Resume the unresolved evaluation instance before another round: ${event.data.instanceRoot}`);
        }finally{previous.close();}
      }catch(error){if(error.code!=='ENOENT')throw error;}
    }
    ledger.event(null,'evaluation-started',{instanceRoot:request.instanceRoot});
  }finally{ledger.close();}
  // Freeze once per round so both arms see identical memory and starting source.
  const inputs=join(dirname(request.instanceRoot),'inputs');
  let frozen;
  try{frozen=await readConfig(inputs);}
  catch(error) {
    if(error.code!=='ENOENT')throw error;
    const ids=request.projectIds??[request.projectId];
    const projects=ids.map(id=>{
      const project=original.projects.find(project=>project.id===id);
      if(!project)throw new Error('Unknown frozen regression project');
      return project;
    });
    await createInstance(inputs,{python:request.config.python??original.python,baseline:join(original.root,'knowledge.sqlite'),
      skills:join(original.root,'skills','versions',request.skillVersion),
      environment:original.environment,projects,install:false,signal:request.signal,machinery:join(original.root,'machinery')});
    frozen=await readConfig(inputs);
  }
  const project=frozen.projects.find(project=>project.id===request.projectId);
  if(!project)throw new Error('Project not in the frozen evaluation round');
  const {root:_root,skills:_skills,projects:_projects,baselineKDB:_baseline,
    campaign_plan:_plan,phase_sequence:_sequence,phase_targets:_targets,completion_target:_target,...settings}=request.config;
  const environment=structuredClone(frozen.environment);
  request.signal?.throwIfAborted();
  await createInstance(request.instanceRoot,{...settings,curriculum:{...settings.curriculum,[settings.mode]:[project.id]},resource_ids:request.resource?[request.resource.id]:settings.resource_ids,baseline:join(inputs,'knowledge.sqlite'),
    skills:join(original.root,'skills','versions',request.skillVersion),
    skillSettings:{...request.config.skills,tuningDisabled:true},projects:[project],environment,
    signal:request.signal,machinery:join(original.root,'machinery')});
}
export async function wave({instanceRoot,config,signal,resuming=false}) {
  const log=await open(join(instanceRoot,'evaluation-wave.log'),'a',0o600);
  try {
    await new Promise((ok,no)=>{
      let failure;
      // Internal evaluations inherit the maintenance/actor service cap. Calling
      // public dispatch here would reserve a second aggregate outside that cap.
      const child=spawn(process.execPath,[join(instanceRoot,'machinery','cli','runtime.mjs'),resuming?'resume':'wave',instanceRoot,JSON.stringify({mode:config.mode})],{signal,stdio:['ignore',log.fd,log.fd]});
      child.once('error',error=>{failure=error;if(!child.pid)no(error);});
      child.once('close',code=>failure?no(failure):code?no(new Error('Evaluation wave failed; inspect evaluation-wave.log')):ok());
    });
  }finally{await log.close();}
  return result(instanceRoot);
}

function result(instanceRoot) {
  const state=new State(join(instanceRoot,'state.sqlite'));
  try {
    const attempts=state.db.prepare('SELECT * FROM attempts ORDER BY created').all();
    const complete=attempts.length===1 && attempts[0].outcome!==null;
    const events=complete?state.events(attempts[0].id):[];
    const actorVerdict=events.some(event=>event.type==='accepted'||event.type==='rejected');
    const correct=complete && actorVerdict ? attempts[0].outcome==='passed' : null;
    // Correctness dominates; lower charged elapsed is the common tie-break measurement.
    return {correct,reason:correct===null?'unavailable-or-incomplete':undefined,score:correct===true?-attempts[0].elapsed:0,attempts:attempts.map(a=>a.id),instanceRoot};
  }finally{state.close();}
}

// Existing samples are never bootstrapped again. Canonical resume owns session and
// external-operation reconciliation; completed results recover without another wave.
export async function resume(request) {
  request.signal?.throwIfAborted();
  const config = await readConfig(request.instanceRoot);
  const state = new State(join(request.instanceRoot, 'state.sqlite'));
  let attempts;
  try {
    attempts = await stoppedAttempts(config, state, request.signal);
  } finally { state.close(); }
  if (attempts.length && attempts[0].outcome !== null) return result(request.instanceRoot);
  return wave({ ...request, config, resuming: attempts.length > 0 });
}

async function stoppedAttempts(config, state, signal, allOperations = false) {
    const control = state.db.prepare("SELECT name FROM sqlite_master WHERE name='control'").get();
    const owner = control && state.db.prepare("SELECT value FROM control WHERE key='owner'").get();
    if (owner && JSON.parse(owner.value) && !await ownerStopped(JSON.parse(owner.value)))
      throw new Error('Evaluation controller remains live; observe it before resuming');
    const attempts = state.db.prepare('SELECT * FROM attempts').all();
    if (attempts.length > 1) throw new Error('Evaluation instance has multiple attempts');
    for (const attempt of attempts) {
      if (!await ownerStopped(attempt.owner)) throw new Error('Evaluation worker remains live; observe it before resuming');
      if (allOperations || attempt.outcome !== null) {
        const path = join(config.root, 'attempts', attempt.id);
        let environment = {};
        if (state.events(attempt.id).some(event => event.type.startsWith('execution-'))) {
          const job = JSON.parse(await readFile(join(path, 'job.json'), 'utf8'));
          const module = await import(pathToFileURL(resolve(config.root, config.environment.module)));
          environment = await module.createEnvironment(config.environment.options, { instancePath: path,
            event: (type, data) => state.event(attempt.id, type, data), exclude: async (_label, fn) => fn(),
            resource: job.resource, signal });
        }
        if (!await operationsStopped(state, attempt, environment)) throw new Error('Evaluation execution remains unresolved; observe it before resuming');
      }
    }
    return attempts;
}

export async function settled(request) {
  try {
    const config = await readConfig(request.instanceRoot);
    const state = new State(join(config.root, 'state.sqlite'));
    try { await stoppedAttempts(config, state, AbortSignal.timeout(5000), true); return true; }
    finally { state.close(); }
  } catch { return false; }
}
