import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { KnowledgeHistory } from './knowledge-history.mjs';
import { CampaignStorage } from './storage.mjs';
import { State } from './state.mjs';
import { withSkillLock } from '../cli/evolution-lock.mjs';

const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const reference = value => typeof value === 'string' ? value :
  value.collection && value.entry_id ? `${value.collection}:${value.entry_id}` : `sha256:${hash(value)}`;
export function lessonRecord(config, item) {
  const delivery = String(item.delivery_id ?? `outbox-${config.storage?.campaign_id ?? item.attempt_id ?? 'local'}-${item.id}`);
  return { id: hash(delivery), delivery_id: delivery, lesson: item.findings ?? item.lesson,
    kind: ['good','negative'].includes(item.kind)?item.kind:
      (item.outcome === 'passed' || item.outcome === 'good' ? 'good' : 'negative'),
    target: item.target ?? item.applicability ?? config.target ?? 'general',
    evidence_refs: (item.evidence_refs ?? item.evidence ?? []).map(reference),
    evidence: item.evidence ?? [], attempt_id: item.attempt_id ?? null };
}
function python(config, args, input, signal) {
  return new Promise((ok, fail) => {
    const child = spawn(config.python ?? 'python', args, { signal,
      env: { ...process.env, PYTHONPATH: resolve(new URL('..', import.meta.url).pathname) },
      stdio: ['pipe', 'pipe', 'pipe'] });
    let output = '', error = '', failure;
    child.stdout.on('data', data => { output += data; });
    child.stderr.on('data', data => { error = (error + data).slice(-65536); });
    child.on('error', e => { failure=e;if(!child.pid)fail(e); });
    child.stdin.on('error', e => { if(e.code!=='EPIPE'){failure=e;child.kill('SIGTERM');} });
    child.on('close', code => {
      if(failure)fail(failure);
      else if (code) fail(new Error(error || `Python exited ${code}`));
      else { try { ok(JSON.parse(output)); } catch (e) { fail(e); } }
    });
    child.stdin.end(input);
  });
}
export async function invokeStorageKdb(config, url, tool, args, { signal, attempt } = {}) {
  // Use the existing Python MCP client directly, without extending its CLI choices.
  const result = await python(config, ['-c',
    'import asyncio,json,sys; from ascend_tools.kdb_client import invoke; p=json.load(sys.stdin); print(json.dumps(asyncio.run(invoke(**p))))'],
  JSON.stringify({ url, tool, arguments: args, attempt }), signal);
  if (result.isError) throw new Error('KDB publication failed');
  return result.structuredContent ?? JSON.parse(result.content.find(c => c.type === 'text').text);
}
async function access(config, options, operation) {
  if (!config.storage) return null;
  const run = async () => {
    const owned = !options.state, state = options.state ?? new State(join(config.root, 'state.sqlite'));
    try {
      const ledger = CampaignStorage.open(state, config.root);
      if (!ledger) throw new Error('Storage campaign is not initialized');
      const history = await new KnowledgeHistory(join(config.root, 'knowledge.git')).init();
      return await operation({ state, ledger, history });
    } finally { if (owned) state.close(); }
  };
  return options.serialized ? run() : withSkillLock(config, run, { signal: options.signal, key: options.lockKey??'skill-writer' });
}

// Export only lesson text, never repository corpora or vectors. Overlapping chunks
// reconstruct the original JSON by source line rather than duplicating evidence.
async function baselineLessons(config, database=join(config.root,'knowledge.sqlite'), signal) {
  return python(config, ['-c', String.raw`
import sqlite3,json,sys,pathlib
p=pathlib.Path(sys.argv[1])
if not p.exists(): print('{"records":[],"collections":[]}'); sys.exit()
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True)
if not c.execute("SELECT 1 FROM sqlite_master WHERE name='collections'").fetchone(): print('{"records":[],"collections":[]}'); sys.exit()
records={}
collections=[]
tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
query="SELECT name FROM collections WHERE language='lesson' ORDER BY name"
if {'memory_collections','active_generation','generation_collections'} <= tables:
    query="""SELECT c.name FROM collections c
    LEFT JOIN memory_collections m ON m.collection=c.name
    LEFT JOIN generation_collections g ON g.collection=c.name
      AND g.generation=(SELECT revision FROM active_generation WHERE singleton=1)
    WHERE c.language='lesson' AND (m.collection IS NULL OR g.collection IS NOT NULL)
    ORDER BY c.name"""
for (name,) in c.execute(query):
    collections.append(name)
    files={}
    for path,start,end,text in c.execute('SELECT path,start_line,end_line,text FROM chunks WHERE collection=? ORDER BY path,start_line,chunk_id',(name,)):
        lines=files.setdefault(path,{})
        for i,line in enumerate(text.splitlines(keepends=True),start): lines[i]=line
    for path,lines in files.items():
        record=json.loads(''.join(lines[i] for i in sorted(lines)))
        key=record.get('delivery_id',name+':'+path)
        if key in records and records[key]!=record: raise ValueError('Conflicting baseline delivery')
        records[key]=record
print(json.dumps(dict(records=list(records.values()),collections=collections)))
`, database], '', signal);
}

// Prepare immutable objects before the database swap; activate and roll back the
// three mutable refs together with the caller's configuration transaction.
export async function prepareStorageBaseline(config, database, state) {
  return access(config,{state,serialized:true},async({history})=>{
    const baseline=await baselineLessons(config,database);
    const records=baseline.records.map(item=>lessonRecord(config,
      {...item,delivery_id:item.delivery_id??`baseline-${hash(item)}`}));
    const files=Object.fromEntries(records.map(record=>[`memory/${record.id}.json`,JSON.stringify(record)+'\n']));
    const revision=await history.commit(null,files,{type:'baseline',deliveries:records.map(record=>record.delivery_id)});
    const blob=await history.git(['hash-object','-w','--stdin'],{input:JSON.stringify({revision,collections:baseline.collections})});
    const names=['refs/knowledge/baseline','refs/heads/memory','refs/knowledge/active-memory'];
    const previous=await Promise.all(names.map(name=>history.ref(name)));
    const swap=async values=>{
      let input='start\n';
      for(let i=0;i<names.length;i++) {
        if(values[i])input+=`update ${names[i]} ${values[i]}\n`;
        else if(await history.ref(names[i]))input+=`delete ${names[i]}\n`;
      }
      await history.git(['update-ref','--stdin'],{input:input+'prepare\ncommit\n'});
    };
    return {activate:async()=>{
      if(previous[0])await history.git(['update-ref',`refs/knowledge/baselines/${previous[0]}`,previous[0]]);
      if(previous[1])await history.git(['update-ref',`refs/knowledge/previous-memory/${previous[1]}`,previous[1]]);
      await swap([blob,revision,null]);
      state.db.prepare("DELETE FROM storage_revisions WHERE name='memory'").run();
    },rollback:()=>swap(previous)};
  });
}
export async function initializeStorageKnowledge(config, state, options = {}) {
  return access(config, { ...options, state }, async context => {
    const { history } = context;
    if (!(await history.ref('refs/knowledge/baseline'))) {
      const baseline=await baselineLessons(config,undefined,options.signal);
      const records=baseline.records.map(item=>lessonRecord(config,
        {...item,delivery_id:item.delivery_id??`baseline-${hash(item)}`}));
      let revision=await history.ref('refs/heads/memory');
      if(records.length) ({revision}=await history.commitMemory(records,{
        indexRevision:async revision=>revision,activate:async()=>{},
      }));
      const blob=await history.git(['hash-object','-w','--stdin'],{input:JSON.stringify({revision,collections:baseline.collections})});
      await history.git(['update-ref','refs/knowledge/baseline',blob]);
    }
    if (config.skills?.activeVersion) {
      const active=await history.state('skills');
      const mapped=context.state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='storage_skill_versions'").get()
        ?context.state.db.prepare('SELECT * FROM storage_skill_versions WHERE version=?').get(config.skills.activeVersion):null;
      if(!mapped||mapped.decision!=='promoted'||mapped.revision!==active?.revision)
        await recordSkill(context,config,config.skills.activeVersion,{promote:true,reason:active?'configuration-reconciliation':'baseline'});
    }
    return { baseline:await readBaseline(history),
      memory:context.state.db.prepare("SELECT * FROM storage_revisions WHERE name='memory'").get()??null,
      skills:await history.state('skills') };
  });
}
async function readBaseline(history) {
  const ref=await history.ref('refs/knowledge/baseline');
  return ref?JSON.parse(await history.git(['cat-file','blob',ref])):null;
}
async function activateBaseline({state,ledger,history},invoke) {
  if(state.db.prepare("SELECT 1 FROM storage_revisions WHERE name='memory'").get())return;
  const baseline=await readBaseline(history);
  if(!baseline)return;
  const result=await invoke('adopt_memory_revision',baseline);
  if(result.generation!==baseline.revision)throw new Error('KDB adopted a different baseline');
  ledger.activate('memory',baseline.revision,result.generation);
  await history.setState('memory',{revision:baseline.revision,generation:result.generation});
}
export async function activateStorageBaseline(config,options={}) {
  return access(config,options,context=>activateBaseline(context,options.invoke??
    ((tool,args)=>invokeStorageKdb(config,options.url,tool,args,{signal:options.signal}))));
}
export async function publishStorageMemory(config, item, options = {}) {
  return access(config, {...options,lockKey:'memory-writer'}, async context => {
    const {history,ledger}=context;
    const invoke = options.invoke ?? ((tool, args) => invokeStorageKdb(config, options.url, tool, args,
      { signal: options.signal, attempt: item.attempt_id }));
    await activateBaseline(context,invoke);
    const legacy=`outbox-${item.id}`;
    const prior=await history.ref('refs/knowledge/deliveries/'+createHash('sha256').update(legacy).digest('hex'));
    const recorded=prior&&(await history.readRevision(prior)).memory.find(record=>
      record.delivery_id===legacy && record.attempt_id===item.attempt_id);
    const record=lessonRecord(config,recorded?{...item,delivery_id:legacy}:item);
    return history.commitMemory([record], {
      indexRevision: async (revision, snapshot) => {
        const result = await invoke('stage_memory_revision', { revision, records: snapshot.memory });
        if (result.generation !== revision) throw new Error('KDB staged a different revision');
        return result.generation;
      },
      activate: async ({ revision, generation }) => {
        const result = await invoke('activate_memory_revision', { revision });
        if (result.generation !== generation) throw new Error('KDB activated a different generation');
        ledger.activate('memory', revision, generation);
      },
    });
  });
}
async function recordSkill({ state, ledger, history }, config, version, options) {
  state.db.exec(`CREATE TABLE IF NOT EXISTS storage_skill_versions (
    version TEXT PRIMARY KEY,revision TEXT NOT NULL,proposal_revision TEXT NOT NULL,
    base_version TEXT,decision TEXT NOT NULL,reason TEXT,map_json TEXT NOT NULL)`);
  const existing=state.db.prepare('SELECT * FROM storage_skill_versions WHERE version=?').get(version);
  if(existing && options.promote===undefined)return existing;
  const directory = join(config.root, 'skills', 'versions', version);
  const { validateSkillMap } = await import('../cli/skill-input.mjs');
  const map = await validateSkillMap(directory, JSON.parse(await readFile(join(directory, 'skill-map.json'), 'utf8')));
  const skills = Object.fromEntries(await Promise.all(Object.entries(map).map(async ([scope, path]) =>
    [scope, await readFile(join(directory, path), 'utf8')])));
  const active=await history.state('skills');
  if(!active&&options.promote===true&&existing?.decision==='promoted') {
    // A copied/recovered checkpoint can retain its ledger while losing the
    // materialized active ref. Restore the recorded decision, including any
    // later rollback promotion, without inventing a replacement lineage.
    const snapshot=await history.readRevision(existing.revision);
    const expected=Object.fromEntries(Object.entries(skills).map(([scope,text])=>[scope+'.md',text]));
    if(snapshot.event.type!=='skill-promotion'||snapshot.event.metadata?.version!==version||
        !isDeepStrictEqual(snapshot.skills,expected))throw new Error('Stored skill promotion does not match version');
    ledger.activate('skills',existing.revision);
    await history.setState('skills',{revision:existing.revision});
    return {revision:existing.revision,version,decision:'promoted'};
  }
  const reactivation=options.promote===true&&existing?.decision==='promoted'&&existing.revision!==active?.revision;
  const proposal = `version-${version}`+(reactivation?`-reactivate-${active.revision.slice(0,12)}`:'');
  let candidate = await history.ref(`refs/heads/skills/${proposal}`);
  if (!candidate) ({ revision: candidate } = await history.proposeSkill({ id: proposal, skills,
    metadata: { version, baseVersion: options.baseVersion ?? null, map } }));
  const snapshot = await history.readRevision(candidate);
  const proposalRevision = snapshot.event.candidate ?? candidate;
  let revision = candidate;
  const decision = options.promote === true ? 'promoted' : options.promote === false ? 'rejected' : 'proposed';
  if (options.promote !== undefined) ({ revision } = await history.decideSkill(proposal, {
    promote: options.promote, metadata: { version, baseVersion: options.baseVersion ?? null, reason: options.reason },
    activate: async ({ revision }) => ledger.activate('skills', revision),
  }));
  state.db.prepare('INSERT OR REPLACE INTO storage_skill_versions VALUES (?,?,?,?,?,?,?)')
    .run(version, revision, proposalRevision, options.baseVersion ?? null, decision, options.reason ?? null, JSON.stringify(map));
  return { revision, version, decision };
}
export async function recordStorageSkill(config, version, options = {}) {
  return access(config, options, context => recordSkill(context, config, version, options));
}
