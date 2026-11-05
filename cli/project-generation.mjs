import { Type } from 'typebox';
import { randomUUID, createHash } from 'node:crypto';
import { mkdir, readFile, realpath, stat } from 'node:fs/promises';
import { join, resolve, sep } from 'node:path';
import { curriculumSession } from './curriculum.mjs';
import { createEnvironment } from '../integrations/environments/index.mjs';

const reply=text=>({content:[{type:'text',text}],details:{}});
const inside=(root,file)=>file===root || file.startsWith(root+sep);
const schema=Type.Object({goal:Type.String({minLength:1}),source:Type.String({minLength:1}),
  verifier:Type.Object({files:Type.Array(Type.String(),{minItems:1}),command:Type.String({minLength:1})}),
  reference:Type.String({minLength:1}),negative_controls:Type.Array(Type.String({minLength:1}),{minItems:1,maxItems:8})});

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

function drafts(state) {
  state.db.exec(`CREATE TABLE IF NOT EXISTS generation_drafts (
    id TEXT PRIMARY KEY, data TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS generation_events (
    sequence INTEGER PRIMARY KEY, draft TEXT NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS generation_role_events ON generation_events(draft,type,sequence)`);
  const save=d=>state.db.prepare('INSERT OR REPLACE INTO generation_drafts VALUES (?,?)').run(d.id,JSON.stringify(d));
  return {save,count:()=>state.db.prepare('SELECT COUNT(*) AS n FROM generation_drafts').get().n,
    pending:limit=>state.db.prepare(`SELECT data FROM generation_drafts
      WHERE json_extract(data,'$.status') IN ('draft','ready') ORDER BY rowid LIMIT ?`).all(limit).map(r=>JSON.parse(r.data)),
    reconcile:()=>state.db.prepare(`UPDATE generation_drafts SET data=json_object('id',id,'status','imported')
      WHERE json_extract(data,'$.status') IN ('draft','ready') AND EXISTS (SELECT 1 FROM projects WHERE projects.id=generation_drafts.id)`).run(),
    adapter:generationEventState(state.db)};
}

async function artifact(root,name,directory=false) {
  const file=await realpath(resolve(root,name));
  if(!inside(await realpath(root),file))throw new Error('Draft artifact escapes its directory');
  const info=await stat(file);
  if(directory?!info.isDirectory():!info.isFile())throw new Error('Invalid draft artifact: '+name);
  return file;
}
async function checkProposal(root,p) {
  const source=await artifact(root,p.source,true),reference=await artifact(root,p.reference,true);
  const negatives=await Promise.all(p.negative_controls.map(n=>artifact(root,n,true)));
  const files=await Promise.all(p.verifier.files.map(n=>artifact(root,n)));
  const trees=[source,reference,...negatives];
  if(trees.some((a,i)=>trees.some((b,j)=>i!==j&&(inside(a,b)||inside(b,a)))))
    throw new Error('Source, reference and controls must be separate trees');
  if(files.some(f=>trees.some(tree=>inside(tree,f))))throw new Error('Verifier must be outside editable fixtures');
  return {source,reference,negatives,files};
}

// Returns staged specs. The caller imports artifacts, admits to State, and updates
// config/selection; draft rows never count as original attempts or completions.
export async function refillPhase1({config,state,bridge,concurrency,path,signal}) {
  const generation=config.curriculum?.generation;
  if(config.mode!=='phase1'||generation?.enabled!==true)return [];
  if(!Number.isSafeInteger(concurrency)||concurrency<1)throw new Error('Invalid generation concurrency');
  if(!Array.isArray(generation.topics)||!generation.topics.length||generation.topics.some(t=>typeof t!=='string'||!t.trim()))
    throw new Error('Generation requires curriculum topics');
  if(!(config.actor_seconds>0))throw new Error('Generation requires actor elapsed allocation');
  const seconds=config.curriculum_seconds??240;
  if(!(seconds>0)||!Number.isFinite(seconds))throw new Error('Invalid curriculum elapsed allocation');
  const store=drafts(state);
  const selected=config.curriculum?.phase1;
  const backlog=state.db.prepare("SELECT id,spec FROM projects WHERE status IN ('queued','running') AND frozen=0").all()
    .filter(r=>{const spec=JSON.parse(r.spec);return (!spec.phase||spec.phase==='phase1')&&
      (!Array.isArray(selected)||selected.includes(r.id)||spec.generation);}).length;
  const wanted=Math.min(8,Math.max(0,2*concurrency-backlog));
  store.reconcile();
  if(!wanted)return [];
  const result=[],batch=randomUUID();
  let processed=0;
  const pending=store.pending(wanted);
  let topicIndex=store.count();
  while(result.length<wanted && processed<wanted) {
    processed++;
    signal?.throwIfAborted();
    let d=pending.shift();
    if(!d){
      // One bounded batch per call; rejected proposals cannot spin a provider loop.
      d={id:'generated-'+randomUUID(),batch,status:'draft',seconds,elapsed:0,
        topic:generation.topics[topicIndex++%generation.topics.length]};
      store.save(d);
    }
    if(d.status==='ready'){result.push(d.spec);continue;}
    const root=join(resolve(path),d.id);await mkdir(root,{recursive:true});
    const started=performance.now(),prior=d.elapsed??0;
    const spent=()=>prior+(performance.now()-started)/1000;
    const checkpoint=()=>{d.elapsed=spent();store.save(d);};
    if(prior>=d.seconds){d.status='exhausted';store.save(d);continue;}
    const bounded=AbortSignal.timeout(Math.max(1,Math.ceil((d.seconds-prior)*1000)));
    const active=signal?AbortSignal.any([signal,bounded]):bounded;
    const timer=setInterval(checkpoint,250);
    try {
      if(!d.proposal)await curriculumSession({config:{...config,curriculum_seconds:d.seconds},state:store.adapter,
        attemptId:d.id,role:'generation',path:root,bridge,signal:active,resume:true,accepted:()=>Boolean(d.proposal),
        prompt:`Construct ONE fresh original Python project for topic ${JSON.stringify(d.topic)}.\n`+
          `${generation.preprompt??''}\nTarget: ${JSON.stringify(config.target??null)}. Actor elapsed allocation: ${config.actor_seconds}.\n`+
          `Use the existing KDB tools and baseline skills at ${join(config.root,'skills','versions',config.skills?.activeVersion??'baseline')}.\n`+
          `Create separate directories for editable source, an independently implemented trusted CPU reference, and incorrect negative controls. `+
          `Create an independent verifier outside those trees. Its command runs from each tree, accepting reference with exit 0 and rejecting every negative with exit 1. `+
          `CPU construction checks must require no device. The source is the actor starting point, not the trusted solution. `+
          `Submit relative artifact paths and the project goal via submit_project. Do not assign project IDs or lineage.`,
        tool:{name:'submit_project',label:'Submit project',description:'Submit constructed source, independent verifier and CPU controls.',parameters:schema,
          execute:async(_id,p)=>{await checkProposal(root,p);d.proposal=p;store.save(d);return reply('Draft recorded for controller validation');}}});
      active.throwIfAborted();
      if(!d.proposal)throw new Error('No project proposal submitted');
      const p=d.proposal,artifacts=await checkProposal(root,p);
      const env=await createEnvironment({mode:'local'},{instancePath:root,signal:active,
        exclude:async(_label,fn)=>fn(),event:(type,data)=>store.adapter.event(d.id,type,data)});
      const hashFiles=async()=>Object.fromEntries(await Promise.all(artifacts.files.map(async f=>[f,createHash('sha256').update(await readFile(f)).digest('hex')])));
      const hashes=await hashFiles(),checks=[];
      for(const [index,cwd] of [artifacts.reference,...artifacts.negatives].entries()) {
        active.throwIfAborted();
        // Preserve submitted source-relative verifier commands across sibling fixtures.
        const command=p.verifier.command;
        const execution=await env.execute(command,cwd,{signal:active,timeout:Math.max(.001,d.seconds-spent())});
        checks.push({kind:index?'negative':'reference',cwd,...execution});
        if(JSON.stringify(await hashFiles())!==JSON.stringify(hashes))throw new Error('Verifier changed during CPU checks');
        if(!execution.settled||execution.exitCode!==(index?1:0))throw new Error('CPU construction check failed: '+checks.at(-1).kind);
      }
      active.throwIfAborted();
      d.spec={id:d.id,phase:'phase1',goal:p.goal,source:artifacts.source,actor_seconds:config.actor_seconds,
        target:config.target,verifier:{files:artifacts.files,command:p.verifier.command,hashes},
        generation:{draft:d.id,topic:d.topic,checks}};
      d.status='ready';checkpoint();
      result.push(d.spec);
    }catch(error){
      if(signal?.aborted){checkpoint();throw error;}
      d.status=bounded.aborted?'exhausted':'invalid';d.error=error.message;checkpoint();
    }finally{clearInterval(timer);checkpoint();}
  }
  return result;
}
