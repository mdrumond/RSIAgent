import { join, resolve, dirname } from 'node:path';
import { ownerStopped } from '../controller/lifecycle.mjs';
import { bootstrap as createInstance, readConfig } from './instance.mjs';
import { spawn } from 'node:child_process';
import { open } from 'node:fs/promises';
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
      if(owner && !await ownerStopped(JSON.parse(owner.value)))
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
    await createInstance(inputs,{baseline:join(original.root,'knowledge.sqlite'),
      skills:join(original.root,'skills','versions',request.skillVersion),
      environment:original.environment,projects,install:false,machinery:join(original.root,'machinery')});
    frozen=await readConfig(inputs);
  }
  const project=frozen.projects.find(project=>project.id===request.projectId);
  if(!project)throw new Error('Project not in the frozen evaluation round');
  const {root:_root,skills:_skills,projects:_projects,baselineKDB:_baseline,...settings}=request.config;
  const environment=structuredClone(original.environment);
  request.signal?.throwIfAborted();
  await createInstance(request.instanceRoot,{...settings,curriculum:{...settings.curriculum,[settings.mode]:[project.id]},resource_ids:request.resource?[request.resource.id]:settings.resource_ids,baseline:join(inputs,'knowledge.sqlite'),
    skills:join(original.root,'skills','versions',request.skillVersion),
    skillSettings:{...request.config.skills,tuningDisabled:true},projects:[project],environment,
    machinery:join(original.root,'machinery')});
}
export async function wave({instanceRoot,config,signal}) {
  const log=await open(join(instanceRoot,'evaluation-wave.log'),'a',0o600);
  try {
    await new Promise((ok,no)=>{
      let failure;
      const child=spawn(process.execPath,[join(instanceRoot,'machinery','cli','main.mjs'),'wave',
        '--instance',instanceRoot,'--mode',config.mode],{signal,stdio:['ignore',log.fd,log.fd]});
      child.once('error',error=>{failure=error;if(!child.pid)no(error);});
      child.once('close',code=>failure?no(failure):code?no(new Error('Evaluation wave failed; inspect evaluation-wave.log')):ok());
    });
  }finally{await log.close();}
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
