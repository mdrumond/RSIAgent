#!/usr/bin/env node
import {DatabaseSync} from 'node:sqlite';
import {readFile,realpath} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve,join,relative,isAbsolute,sep} from 'node:path';
import {parseArgs} from 'node:util';
import {stages} from '../controller/workflow.mjs';

const protectedScopes=new Set(['global-kdb',...stages]);
const json=async path=>JSON.parse(await readFile(path,'utf8'));
const integer=(n,name,max)=>{if(!Number.isSafeInteger(n)||n<1||n>max)throw Error('Invalid '+name);return n;};
const has=(db,table)=>Boolean(db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?").get(table));
const key=e=>e.collection&&e.entry_id?JSON.stringify([e.collection,e.entry_id]):null;

// A bounded read transaction: no State constructor, schema creation, source edits,
// or inference that unobserved content is useless. Recommendations are proposals.
export async function auditSkillUsage({instance,version,maxAttempts=100,maxEventsPerAttempt=500,maxRetrievals=1000,minCompletedAttempts=8}={}) {
  integer(maxAttempts,'attempt limit',1000);integer(maxEventsPerAttempt,'event limit',1000);
  integer(maxRetrievals,'retrieval limit',10000);integer(minCompletedAttempts,'minimum exposure',1000);
  const root=await realpath(instance),config=await json(join(root,'instance.json'));
  version??=config.skills?.activeVersion;
  if(typeof version!=='string'||!version||version==='.'||version==='..'||/[\\/]/.test(version))throw Error('Invalid skill version');
  const directory=await realpath(join(root,'skills','versions',version)),map=await json(join(directory,'skill-map.json'));
  let guidance=[];try{guidance=await json(join(directory,'guidance.json'));}catch(e){if(e.code!=='ENOENT')throw e;}
  const scopes=[];
  for(const [scope,path] of Object.entries(map)) {
    const file=await realpath(resolve(directory,path)),rel=relative(directory,file);
    if(rel==='..'||rel.startsWith('..'+sep)||isAbsolute(rel))throw Error('Skill escapes immutable version');
    const bytes=await readFile(file),sha256=createHash('sha256').update(bytes).digest('hex');
    scopes.push({scope,path:rel,absolute:file,sha256,protected:protectedScopes.has(scope)||guidance.some(g=>g.scope===scope&&g.critical===true),
      exposures:0,exposedAttempts:new Set(),retrievalHits:0});
  }
  const protectedFiles=new Set(scopes.filter(s=>s.protected).map(s=>s.absolute));
  for(const scope of scopes)if(protectedFiles.has(scope.absolute))scope.protected=true;
  const db=new DatabaseSync(join(root,'state.sqlite'),{readOnly:true});
  const knowledge=new Map(),warn=new Set();let truncatedEvents=false;
  const record=(entry,source)=>{
    const id=key(entry);if(!id)return;
    const value=knowledge.get(id)??{collection:entry.collection,entryId:entry.entry_id,workflowHits:0,searchHits:0,readHits:0};
    value[source]++;knowledge.set(id,value);
  };
  try {
    db.exec('BEGIN');
    const attempts=db.prepare('SELECT id,outcome FROM attempts ORDER BY rowid DESC LIMIT ?').all(maxAttempts+1);
    const truncatedAttempts=attempts.length>maxAttempts;attempts.length=Math.min(attempts.length,maxAttempts);
    const selected=new Set(attempts.map(a=>a.id)),versionAttempts=new Set();let workflowSearches=0,workflowHits=0,unavailable=0,instrumentedRetrievals=0;
    const events=db.prepare(`SELECT type,data FROM events WHERE attempt=? AND type IN
      ('skill-read','knowledge-retrieved','knowledge-retrieval-unavailable','knowledge-retrieval-recorded') ORDER BY sequence DESC LIMIT ?`);
    for(const attempt of attempts) {
      const rows=events.all(attempt.id,maxEventsPerAttempt+1);
      if(rows.length>maxEventsPerAttempt)truncatedEvents=true;
      for(const row of rows.slice(0,maxEventsPerAttempt)) {
        const data=JSON.parse(row.data);
        if(row.type==='skill-read') {
          for(const scope of scopes.filter(s=>s.absolute===data.path)){
            if(scope.sha256===data.sha256){scope.exposures++;scope.exposedAttempts.add(attempt.id);versionAttempts.add(attempt.id);}
            else warn.add('skill-content-hash-mismatch');
          }
        } else if(row.type==='knowledge-retrieved') {
          workflowSearches++;
          for(const entry of data.evidence??[]){record(entry,'workflowHits');workflowHits++;}
        } else if(row.type==='knowledge-retrieval-recorded')instrumentedRetrievals++;
        else unavailable++;
      }
    }
    let searches=0,reads=0,citationHits=0,emptySearches=0,truncatedRetrievals=false;
    const retrievalTable=has(db,'storage_retrievals');
    if(retrievalTable) {
      const rows=db.prepare('SELECT attempt,operation,citations FROM storage_retrievals ORDER BY id DESC LIMIT ?').all(maxRetrievals+1);
      truncatedRetrievals=rows.length>maxRetrievals;
      for(const row of rows.slice(0,maxRetrievals)) {
        if(!selected.has(row.attempt))continue;
        const citations=JSON.parse(row.citations),search=row.operation==='mcp__kdb__search',read=row.operation==='mcp__kdb__read_entry';
        if(search){searches++;if(!citations.length)emptySearches++;}if(read)reads++;
        for(const c of citations) {
          citationHits++;const entries=c.evidence??[{...c,entry_id:c.entry_id??c.chunk_id}];
          for(const entry of entries){
            record(entry,read?'readHits':'searchHits');
            for(const scope of scopes)if([scope.absolute,'skills/versions/'+version+'/'+scope.path].includes(entry.path)&&
              (entry.source_sha256??entry.sha256)===scope.sha256)scope.retrievalHits++;
          }
        }
      }
    }
    const historicalCitationGap=workflowHits>0&&searches+reads>0&&citationHits===0;
    if(historicalCitationGap)warn.add('workflow-hits-missing-from-storage-citations');
    if(!retrievalTable||instrumentedRetrievals<searches+reads||!instrumentedRetrievals)warn.add('retrieval-instrumentation-unverified');
    if(unavailable)warn.add('retrieval-results-unavailable');
    const completedOpportunities=attempts.filter(a=>a.outcome!==null&&versionAttempts.has(a.id)).length;
    const coverageComplete=!truncatedAttempts&&!truncatedEvents&&!truncatedRetrievals&&retrievalTable&&!historicalCitationGap&&!unavailable&&!warn.has('skill-content-hash-mismatch')&&!warn.has('retrieval-instrumentation-unverified');
    const suggestions=[];
    for(const item of scopes) {
      if(!item.protected&&!item.exposures&&!item.retrievalHits&&coverageComplete&&completedOpportunities>=minCompletedAttempts)
        suggestions.push({scope:item.scope,path:item.path,sha256:item.sha256,action:'move-to-searchable',reason:'unobserved-in-complete-exposure-window',completedOpportunities});
    }
    const report={format:1,readOnly:true,version,coverage:{attempts:attempts.length,completedOpportunities,truncatedAttempts,truncatedEvents,truncatedRetrievals,
      citationLogging:historicalCitationGap?'incomplete-historical':retrievalTable?'observed':'unknown',complete:coverageComplete},
      scopes:scopes.map(({absolute,exposedAttempts,...s})=>({...s,exposedAttempts:exposedAttempts.size,interpretation:s.exposures?'exposed-not-proven-useful':'unobserved'})),
      retrieval:{workflowSearches,workflowHits,recordedSearches:searches,recordedReads:reads,recordedCitationHits:citationHits,emptySearches,
        instrumentedRetrievals,observedEntries:[...knowledge.values()],unseenInventory:'unknown'},
      warnings:[...warn],demotionSuggestions:suggestions};
    db.exec('COMMIT');return report;
  }finally {db.close();}
}

if(import.meta.main) {
  const {values}=parseArgs({options:{instance:{type:'string'},version:{type:'string'},'max-attempts':{type:'string'},'max-events':{type:'string'},'max-retrievals':{type:'string'},'minimum-opportunities':{type:'string'}}});
  auditSkillUsage({instance:values.instance,version:values.version,...Object.fromEntries([
    ['max-attempts','maxAttempts'],['max-events','maxEventsPerAttempt'],['max-retrievals','maxRetrievals'],['minimum-opportunities','minCompletedAttempts']]
    .filter(([flag])=>values[flag]!==undefined).map(([flag,key])=>[key,Number(values[flag])]))})
    .then(report=>console.log(JSON.stringify(report,null,2))).catch(error=>{console.error(error.message);process.exitCode=1;});
}
