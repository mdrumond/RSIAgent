import { Type } from 'typebox';
import { randomUUID } from 'node:crypto';
import { mkdir, open, realpath, stat } from 'node:fs/promises';
import { writeSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';
import { validateAcceptance } from '../controller/state.mjs';
import { currentCampaignEntry, generationSettings } from '../controller/progress.mjs';
import { curriculumSession } from './curriculum.mjs';
import {claimLearningRequest} from '../controller/skill-review.mjs';
import { createEnvironment } from '../integrations/environments/index.mjs';
import {campaignEntryConfig} from './campaign-manifest.mjs';
import {projectBundle,bundleHashes} from './project-bundle.mjs';
import {compileVerifierEntries,entrypointCommand} from '../controller/project-commands.mjs';

const reply=text=>({content:[{type:'text',text}],details:{}});
const inside=(root,file)=>file===root || file.startsWith(root+sep);
const entrypoint=Type.Object({entrypoint:Type.String({minLength:1}),args:Type.Array(Type.String())});
const schema=Type.Object({goal:Type.String({minLength:1}),performance_goal:Type.Optional(Type.Unknown()),source:Type.String({minLength:1,description:'Relative DIRECTORY containing the editable PyTorch starter; not a Python file path.'}),
  verifier:Type.Object({files:Type.Array(Type.String({description:'Trusted verifier entrypoint/helper/data FILE outside every source, reference and negative-control directory. Fixture contents are bundled automatically; do not list them here.'}),{minItems:1}),command:entrypoint,
    performance_command:Type.Optional(entrypoint),baseline_command:Type.Optional(entrypoint)}),
  construction_command:Type.Optional(entrypoint),baseline_construction_command:Type.Optional(entrypoint),
  reference:Type.String({minLength:1,description:'Relative DIRECTORY containing the independent CPU reference; not a Python file path.'}),negative_controls:Type.Array(Type.String({minLength:1,description:'Relative DIRECTORY containing an incorrect implementation with the same candidate interface.'}),{minItems:1,maxItems:8})});

// Same bounded event contract as State, scoped exclusively to generation telemetry.
export function generationEventState(db) {
  return {
    queryEvents(id,{types,order='asc',limit=100,before,after}={}) {
      if(!Number.isSafeInteger(limit)||limit<1||limit>1000)throw new Error('Invalid event limit');
      if(!['asc','desc'].includes(order))throw new Error('Invalid event order');
      const clauses=['draft=?'],args=[id];
      if(types!==undefined) {
        if(!Array.isArray(types)||types.some(t=>typeof t!=='string'||!t))throw new Error('Invalid event types');
        clauses.push(types.length?`type IN (${types.map(()=>'?').join(',')})`:'0');args.push(...types);
      }
      for(const [cursor,operator] of [[before,'<'],[after,'>']]) {
        if(cursor===undefined)continue;
        if(!Number.isSafeInteger(cursor)||cursor<0)throw new Error('Invalid event cursor');
        clauses.push(`sequence ${operator} ?`);args.push(cursor);
      }
      return db.prepare(`SELECT sequence,type,data FROM generation_events WHERE ${clauses.join(' AND ')}
        ORDER BY sequence ${order} LIMIT ?`).all(...args,limit).map(e=>({...e,data:JSON.parse(e.data)}));
    },
    event:(id,type,data)=>db.prepare('INSERT INTO generation_events(draft,type,data) VALUES (?,?,?)')
      .run(id,type,JSON.stringify(data))
  };
}

function drafts(state, adoptPhase1=false) {
  state.db.exec(`CREATE TABLE IF NOT EXISTS generation_drafts (
    id TEXT PRIMARY KEY, data TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS generation_events (
    sequence INTEGER PRIMARY KEY, draft TEXT NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS generation_role_events ON generation_events(draft,type,sequence)`);
  // Old generation was phase1-only and had no campaign entry. Annotate only
  // pending non-plan rows; JSON updates preserve sessions, specs and allocations.
  if(adoptPhase1)state.db.prepare(`UPDATE generation_drafts
    SET data=json_set(data,'$.phase','phase1','$.campaign_entry',NULL)
    WHERE json_type(data,'$.phase') IS NULL AND json_extract(data,'$.campaign_entry') IS NULL
    AND json_extract(data,'$.status') IN ('draft','ready')
    AND (json_extract(data,'$.spec.phase') IS NULL OR json_extract(data,'$.spec.phase')='phase1')
    AND json_extract(data,'$.spec.campaign_entry') IS NULL`).run();
  const save=d=>state.db.prepare('INSERT OR REPLACE INTO generation_drafts VALUES (?,?)').run(d.id,JSON.stringify(d));
  return {save,count:()=>state.db.prepare('SELECT COUNT(*) AS n FROM generation_drafts').get().n,
    reserved:(phase,entry)=>state.db.prepare(`SELECT data FROM generation_drafts
      WHERE json_extract(data,'$.status') IN ('draft','ready') AND json_extract(data,'$.phase')=?
      AND json_extract(data,'$.campaign_entry') IS ?`).all(phase,entry).map(row=>JSON.parse(row.data)),
    pending:(phase,entry,limit)=>state.db.prepare(`SELECT data FROM generation_drafts
      WHERE json_extract(data,'$.status') IN ('draft','ready') AND json_extract(data,'$.phase')=?
      AND json_extract(data,'$.campaign_entry') IS ? ORDER BY rowid LIMIT ?`).all(phase,entry,limit).map(r=>JSON.parse(r.data)),
    reconcile:()=>state.db.prepare(`UPDATE generation_drafts SET data=json_object('id',id,'status','imported')
      WHERE json_extract(data,'$.status') IN ('draft','ready') AND EXISTS (SELECT 1 FROM projects WHERE projects.id=generation_drafts.id)`).run(),
    adapter:generationEventState(state.db)};
}

async function artifact(root,name,directory=false) {
  const file=await realpath(resolve(root,name));
  if(!inside(await realpath(root),file))throw new Error('Draft artifact escapes its directory');
  const info=await stat(file);
  if(directory?!info.isDirectory():!info.isFile())throw new Error(`Draft artifact ${JSON.stringify(name)} must be a ${directory?'directory':'file'}, but ${JSON.stringify(file)} is ${info.isDirectory()?'a directory':info.isFile()?'a file':'another artifact type'}`);
  return file;
}
async function checkProposal(root,p) {
  const source=await artifact(root,p.source,true),reference=await artifact(root,p.reference,true);
  const negatives=await Promise.all(p.negative_controls.map(n=>artifact(root,n,true)));
  const files=await Promise.all(p.verifier.files.map(n=>artifact(root,n)));
  const trees=[source,reference,...negatives];
  if(trees.some((a,i)=>trees.some((b,j)=>i!==j&&(inside(a,b)||inside(b,a)))))
    throw new Error('Source, reference and controls must be separate trees');
  const fixtureFiles=files.flatMap(file=>trees.filter(tree=>inside(tree,file)).map(tree=>({file,fixtureDirectory:tree})));
  if(fixtureFiles.length)throw new Error('Remove fixture-owned files from verifier.files; fixture directories are bundled automatically. Only trusted verifier entrypoints/helpers/data outside every fixture directory belong in verifier.files. Offending files: '+JSON.stringify(fixtureFiles));
  return {source,reference,negatives,files};
}

// Returns staged specs. The caller imports artifacts, admits to State, and updates
// config/selection; draft rows never count as original attempts or completions.
export async function refillProjects({config,state,bridge,concurrency,path,signal}) {
  const entry=currentCampaignEntry(state,config);
  if(config.campaign_plan&&!entry)return [];
  if(entry)config=campaignEntryConfig(config,entry.id);
  const phase=entry?.phase??config.mode,campaign_entry=entry?.id??null;
  const generation=generationSettings(config,phase);
  if(!['phase1','phase2'].includes(phase)||generation.enabled!==true)return [];
  if(!Number.isSafeInteger(concurrency)||concurrency<1)throw new Error('Invalid generation concurrency');
  if(!Array.isArray(generation.topics)||!generation.topics.length||generation.topics.some(t=>typeof t!=='string'||!t.trim()))
    throw new Error('Generation requires curriculum topics');
  if(!(config.actor_seconds>0))throw new Error('Generation requires actor elapsed allocation');
  const seconds=config.curriculum_seconds??240;
  if(!(seconds>0)||!Number.isFinite(seconds))throw new Error('Invalid curriculum elapsed allocation');
  const store=drafts(state,config.campaign_plan===undefined&&phase==='phase1');
  const selected=config.curriculum?.[phase];
  const queued=state.db.prepare("SELECT id,spec FROM projects WHERE status IN ('queued','running') AND frozen=0").all()
    .filter(r=>{const spec=JSON.parse(r.spec);return entry?spec.campaign_entry===entry.id:
      (!spec.phase||spec.phase===phase)&&(!Array.isArray(selected)||selected.includes(r.id)||spec.generation);});
  // The contract already subtracts admitted originals. Subtract only staged,
  // unattempted projects here, otherwise active originals would count twice.
  const unadmitted=entry?queued.filter(r=>!state.db.prepare('SELECT 1 FROM attempts WHERE project=? LIMIT 1').get(r.id)).length:0;
  const wanted=Math.min(8,Math.max(0,2*concurrency-queued.length),
    entry?Math.max(0,entry.remainingAdmissions-unadmitted):Infinity);
  store.reconcile();
  if(!wanted)return [];
  const result=[],batch=randomUUID();
  let processed=0;
  const pending=store.pending(phase,campaign_entry,wanted);
  const quota=entry?.generation?.topics;
  const reserved=new Map();
  const reserve=(topic,delta)=>reserved.set(topic,(reserved.get(topic)??0)+delta);
  if(quota) {
    for(const row of state.db.prepare('SELECT spec FROM projects').all()) {
      const spec=JSON.parse(row.spec);
      if(spec.campaign_entry===entry.id)reserve(spec.generation?.topic,1);
    }
    for(const draft of store.reserved(phase,campaign_entry))reserve(draft.topic,1);
  }
  const nextTopic=()=>quota?.find(topic=>(reserved.get(topic)??0)<quota.filter(item=>item===topic).length)
    ??(quota?undefined:generation.topics[topicIndex++%generation.topics.length]);
  let topicIndex=store.count();
  while(result.length<wanted && processed<wanted) {
    processed++;
    signal?.throwIfAborted();
    let d=pending.shift();
    if(!d){
      const topic=nextTopic();if(topic===undefined)break;
      // One bounded batch per call; rejected proposals cannot spin a provider loop.
      d={id:'generated-'+randomUUID(),batch,status:'draft',phase,campaign_entry,seconds,elapsed:0,
        topic};
      const atomic=!state.db.isTransaction;if(atomic)state.db.exec('BEGIN IMMEDIATE');
      try{
        if(phase==='phase1')d.learningRequest=claimLearningRequest(state,config,{draft:d.id,topic,target:config.target});
        store.save(d);if(atomic)state.db.exec('COMMIT');
      }catch(error){if(atomic)state.db.exec('ROLLBACK');throw error;}
      if(quota)reserve(topic,1);
    }
    if(d.status==='ready'){result.push(d.spec);continue;}
    const root=join(resolve(path),d.id);await mkdir(root,{recursive:true});
    const started=performance.now(),prior=d.elapsed??0;
    const spent=()=>prior+(performance.now()-started)/1000;
    const checkpoint=()=>{
      d.elapsed=spent();
      if(d.status==='ready'&&d.spec.bundle)d.spec.bundle.seconds=Math.max(0,d.seconds-d.elapsed);
      store.save(d);
    };
    if(prior>=d.seconds){d.status='exhausted';store.save(d);if(quota)reserve(d.topic,-1);continue;}
    const bounded=AbortSignal.timeout(Math.max(1,Math.ceil((d.seconds-prior)*1000)));
    const active=signal?AbortSignal.any([signal,bounded]):bounded;
    const timer=setInterval(checkpoint,250);
    const requirements=generation.requirements?.[d.topic]??{};
    const deviceGoal=phase==='phase1'&&config.target!=='cpu';
    try {
      await curriculumSession({config:{...config,curriculum_seconds:d.seconds},state:store.adapter,
        attemptId:d.id,role:'generation',path:root,bridge,signal:active,resume:true,accepted:()=>d.status==='ready',
        prompt:`Construct ONE fresh original ${phase==='phase1'?'microproject':'end-to-end project'} in Python for topic ${JSON.stringify(d.topic)}.\n`+
          (phase==='phase1'?`Issue only a runnable PyTorch workload specification, CPU reference, fixtures, negative controls and verifier interface. Do not generate any device kernel code in source, reference, controls or verifier: no Triton, Catlass DSL or Ascend C implementations, including SIMT/non-SIMT benchmark baselines. Define the callable interface, shapes, dtypes, mathematical behavior and tolerances; use small deterministic CPU fixtures. Editable source is a straightforward PyTorch baseline. ${deviceGoal?`The actor will write all ${config.kernel_language??'target'} kernel code and any device benchmark baselines after admission.`:'This is a CPU-only Python project. The PyTorch starter may satisfy final correctness; no device kernel or separate target acceptance is required.'}\n`:'')+
          `Retain performance goals requested by the curriculum as performance_goal with verifier.performance_command. For every performance command provide verifier.baseline_command: a target-only self-test that executes the SAME PyTorch baseline callable and placement path used by performance_command over ALL frozen timing cases, including warmup and timing calls. It must compare baseline results to the independent reference, run without an optimized candidate, and exit 0 only when all baseline cases execute correctly. The controller runs this gate on the target before starting the actor; do not run it during curriculum construction. For every performance command also submit baseline_construction_command as {entrypoint,args}, using the SAME frozen entrypoint as baseline_command and SAME baseline callable/dependency path with a tiny CPU fixture. This is only a small dependency/numerical selftest: no timing, warmups, repetitions or replay of expensive target timing cases. Do not invent performance requirements or claim CPU construction proves timing.\n`+
          (requirements.performance?`This topic requires a real performance_goal, verifier.performance_command and a separate CPU construction_command.\n`:'')+
          (requirements.capabilityEvidence?`SIMT is the actor's optimization objective. Specify deferred target acceptance that requires actual supported ${config.kernel_language??'target'} SIMT APIs/compiler mapping and comparison against an equivalent non-SIMT/vector baseline on an isolated device. Do not implement or benchmark either device kernel during curriculum construction. CPU construction cannot establish SIMT or timing evidence.\n`:'')+
          `${generation.preprompt??''}\n${d.learningRequest?'Learning request (preserve this topic requirements): '+JSON.stringify(d.learningRequest):''}\nTarget: ${JSON.stringify(config.target??null)}. Actor elapsed allocation: ${config.actor_seconds}.\n`+
          `Use the existing KDB tools and baseline skills at ${join(config.root,'skills','versions',config.skills?.activeVersion??'baseline')}.\n`+
          `Keep the complete project in one dedicated subdirectory, including all verifier helpers, reference data and negative controls, so it can be imported and staged independently of this generation session. List any dependencies outside that project subdirectory explicitly in verifier.files. `+
          `Create separate directories for editable source, an independently implemented trusted CPU reference, and incorrect negative controls. `+
          `Create an independent verifier outside those trees. Declare each target command as {entrypoint, args}: entrypoint is a frozen Python file path relative to this draft directory and args is an array of literal arguments. RSI derives its command from editable source; do not supply shell commands or cd prefixes. `+
          (deviceGoal?`Keep target acceptance separate from CPU specification validation: an unchanged PyTorch starter must not satisfy the target kernel goal. Provide the target correctness/performance commands for the actor to run later; do not probe target APIs, compile kernels, submit remote jobs or collect device timings to construct this specification.\n`:'')+
          `When target acceptance needs a device, provide a separate construction_command as {entrypoint, args}, with the same frozen-file and literal-argument contract, for CPU proof against the reference and controls. Omit construction_command when verifier.command already runs on CPU; the controller uses that entrypoint automatically. `+
          `The construction command (or verifier command when omitted) must accept the independent reference and unchanged PyTorch starter from their respective cwd with exit 0, and reject every negative with exit 1. CPU starter validation does not replace separate target-kernel acceptance. `+
          `CPU construction checks must require no device. Each check executes the SAME command with cwd set to the reference directory, then each negative-control directory. Load the candidate from cwd with the same filename in every tree; never hardcode the reference path in command arguments. Resolve trusted data relative to the frozen verifier file. The source is the actor starting point, not the trusted solution. `+
          `Submit relative artifact paths and the project goal via submit_project. Do not reuse completed implementations. Do not assign project IDs or lineage.`,
        tool:{name:'submit_project',label:'Submit project',description:'Submit constructed source, independent verifier and CPU controls.',parameters:schema,
          execute:async(_id,p)=>{
            const checks=[];
            try {
              if(requirements.performance&&(p.performance_goal==null||!p.verifier?.performance_command||!p.construction_command))
                throw new Error('Topic requires a performance goal, target performance command and separate CPU construction command');
              if(p.verifier?.performance_command&&!p.verifier?.baseline_command)
                throw new Error('Generated performance project requires verifier.baseline_command');
              const artifacts=await checkProposal(root,p);
              const verifier=compileVerifierEntries(p.verifier,{root,source:artifacts.source,files:artifacts.files,required:true});
              validateAcceptance({...p,verifier});
              if(p.verifier?.performance_command&&!p.baseline_construction_command)
                throw new Error('Generated performance project requires baseline_construction_command');
              const baseline=p.baseline_construction_command===undefined?undefined:
                compileVerifierEntries({command:p.baseline_construction_command},{root,files:artifacts.files,required:true});
              if(baseline&&baseline.entrypoints.command.entrypoint!==verifier.entrypoints.baseline_command?.entrypoint)
                throw new Error('CPU baseline selftest must use the frozen baseline_command entrypoint');
              const construction=p.construction_command===undefined?verifier:
                compileVerifierEntries({command:p.construction_command},{root,files:artifacts.files,required:true});
              const constructionCommand=entrypointCommand(construction.entrypoints.command);
              const packaged=await projectBundle(artifacts,constructionCommand,d.seconds,await realpath(root));
              artifacts.files=packaged.files;
              const env=await createEnvironment({mode:'local',localExecution:config.local_execution},{instancePath:root,signal:active,
                exclude:async(_label,fn)=>fn(),event:(type,data)=>store.adapter.event(d.id,type,data)});
              const hashFiles=()=>bundleHashes(artifacts.files);
              const hashes=await hashFiles();
              await mkdir(join(root,'construction'),{recursive:true});
              const cases=[{kind:'reference',cwd:artifacts.reference,expected:0},
                ...artifacts.negatives.map(cwd=>({kind:'negative',cwd,expected:1})),
                {kind:'starter',cwd:artifacts.source,expected:0},
                ...(baseline?[{kind:'baseline',cwd:artifacts.source,expected:0,command:entrypointCommand(baseline.entrypoints.command)}]:[])];
              for(const {kind,cwd,expected,command=constructionCommand} of cases) {
                active.throwIfAborted();
                // An omitted CPU command uses the frozen absolute entrypoint;
                // fixture cwd depths need not match the editable source.
                let output='',execution;
                const logPath=join(root,'construction',randomUUID()+'.log'),log=await open(logPath,'wx',0o600);
                try {
                  execution=await env.execute(command,cwd,{signal:active,timeout:Math.max(.001,d.seconds-spent()),
                    onData:chunk=>{writeSync(log.fd,chunk);output=(output+chunk.toString()).slice(-4000);}});
                }catch(error){execution={settled:false,exitCode:null,error:error.message};}
                finally{await log.close();}
                const check={kind,cwd,command,expected_exit:expected,...execution,log:logPath,output};
                checks.push(check);store.adapter.event(d.id,'generation-construction-check',check);
                if(JSON.stringify(await hashFiles())!==JSON.stringify(hashes))throw new Error('Verifier changed during CPU checks');
                if(!execution.settled||execution.exitCode!==expected)throw new Error('CPU construction check failed: '+JSON.stringify(check));
              }
              active.throwIfAborted();
              d.spec={id:d.id,phase:d.phase,...(d.campaign_entry?{campaign_entry:d.campaign_entry}:{}),goal:p.goal,source:artifacts.source,actor_seconds:config.actor_seconds,
                target:config.target,...(p.performance_goal!=null?{performance_goal:p.performance_goal}:{}),
                bundle:{...packaged.bundle,validate_starter:true,...(baseline?{baseline_construction_command:entrypointCommand(baseline.entrypoints.command)}:{}),seconds:Math.max(.001,d.seconds-spent())},
                verifier:{...verifier,files:artifacts.files,hashes},
                generation:{draft:d.id,topic:d.topic,requirements,checks,...(d.learningRequest?{learningRequest:d.learningRequest}: {})}};
              d.proposal=p;d.status='ready';delete d.error;checkpoint();
              return reply('Project validated and accepted; finish without further tool calls.');
            }catch(error){
              d.error=error.message;
              store.adapter.event(d.id,'generation-rejected',{proposal:p,checks,error:error.message,elapsed_seconds:spent()});
              checkpoint();throw error;
            }
          }}});
      if(d.status!=='ready')throw new Error(d.error??'No project proposal submitted');
      result.push(d.spec);
    }catch(error){
      if(signal?.aborted){checkpoint();throw error;}
      d.status=bounded.aborted?'exhausted':'invalid';d.error=error.message;checkpoint();
      if(quota)reserve(d.topic,-1);
    }finally{clearInterval(timer);checkpoint();}
  }
  return result;
}

// Retain the existing entry point for callers; phase selection is shared.
export const refillPhase1=refillProjects;

// An incomplete imported closure is a rejected specification, not permission to
// retry the same ready draft with a fresh validation allocation every refill.
export function rejectGeneratedProject(state,spec,error) {
  const row=state.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(spec.generation?.draft);
  if(!row)return;
  const draft=JSON.parse(row.data);draft.status='invalid';draft.error=error.message;
  state.db.prepare('UPDATE generation_drafts SET data=? WHERE id=?').run(JSON.stringify(draft),draft.id);
}

export function checkpointGeneratedBundle(state,spec,seconds) {
  const row=state.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(spec.generation?.draft);
  if(!row)return;
  const draft=JSON.parse(row.data);
  draft.spec.bundle.seconds=seconds;draft.elapsed=Math.max(draft.elapsed,draft.seconds-seconds);
  state.db.prepare('UPDATE generation_drafts SET data=? WHERE id=?').run(JSON.stringify(draft),draft.id);
}
