import {DatabaseSync} from 'node:sqlite';
import {readFile,realpath} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {Type} from 'typebox';

const exec=promisify(execFile);
const inside=(root,path)=>path===root||path.startsWith(root+'/');

// Only explicitly named members of this fresh cohort are readable; never scan old instances.
export async function buildProjectCatalog({cohort,campaigns}) {
  if(!cohort||!Array.isArray(campaigns)||!campaigns.length)throw new Error('Fresh campaign cohort required');
  const entries=[];
  for(const campaign of campaigns) {
    const root=await realpath(campaign.root),config=JSON.parse(await readFile(join(root,'instance.json'),'utf8'));
    if(config.campaign_cohort!==cohort||config.target!==campaign.target||
      config.storage?.campaign_id!==campaign.campaignId)throw new Error('Campaign is outside the requested cohort');
    const db=new DatabaseSync(join(root,'state.sqlite'),{readOnly:true});
    try {
      db.exec('PRAGMA busy_timeout=5000');
      if(db.prepare('SELECT id FROM storage_campaign').get()?.id!==campaign.campaignId)
        throw new Error('Campaign storage identity mismatch');
      const rows=db.prepare(`SELECT a.id,a.project,a.outcome,p.spec,s.phase FROM attempts a
        JOIN projects p ON p.id=a.project JOIN storage_attempts s ON s.attempt=a.id
        WHERE a.outcome IN ('passed','failed') AND s.kind='original'
        AND EXISTS(SELECT 1 FROM recovery_lessons l WHERE l.attempt=a.id AND l.recorded=1)
        AND EXISTS(SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed')
        AND NOT EXISTS(SELECT 1 FROM events e WHERE e.sequence=(SELECT MAX(sequence) FROM events
          WHERE attempt=a.id AND type='failure-settlement') AND json_extract(e.data,'$.closed') IS NOT 1)
        ORDER BY a.created,a.id`).all();
      for(const row of rows) {
        const artifact=db.prepare("SELECT id,path FROM storage_artifacts WHERE project=? AND kind='source-git' AND external=0").get(row.project);
        const snapshot=db.prepare('SELECT source_hash,source_commit FROM storage_iterations WHERE attempt=? AND source_hash IS NOT NULL ORDER BY id DESC LIMIT 1').get(row.id);
        if(!artifact||!snapshot)continue; // Early failures may have no captured source.
        const gitDir=resolve(root,artifact.path);if(!inside(root,gitDir))throw new Error('Source history escaped campaign');
        const spec=JSON.parse(row.spec);
        const job=JSON.parse(await readFile(join(root,'attempts',row.id,'job.json'),'utf8'));
        const metadata=job.resource?.metadata??{};
        entries.push({id:campaign.campaignId+':'+row.id,campaignId:campaign.campaignId,projectId:row.project,
          attemptId:row.id,target:config.target,runtime:metadata.runtime??null,product:metadata.product??null,
          language:config.kernel_language??null,phase:row.phase,outcome:row.outcome,goal:spec.goal,
          source:{gitDir,tree:snapshot.source_hash,commit:snapshot.source_commit,artifactId:artifact.id}});
      }
    }finally{db.close();}
  }
  return entries;
}

async function git(entry,args) {
  if(!/^[a-f0-9]{40,64}$/.test(entry.source?.tree??''))throw new Error('Immutable source tree required');
  return exec('git',['--git-dir='+entry.source.gitDir,'-c','core.hooksPath=/dev/null',
    '-c','core.fsmonitor=false','-c','core.attributesFile=/dev/null',...args],
  {timeout:10000,maxBuffer:1024*1024,env:{...Object.fromEntries(Object.entries(process.env).filter(([key])=>!key.startsWith('GIT_'))),
    GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:'/dev/null',GIT_OPTIONAL_LOCKS:'0'}});
}
export async function listProjectFiles(entry) {
  const {stdout}=await git(entry,['ls-tree','-rz','--name-only',entry.source.tree]);
  return stdout.split('\0').filter(Boolean);
}
export async function readProjectSource(entry,path) {
  if(!(await listProjectFiles(entry)).includes(path))throw new Error('File is outside the retained project snapshot');
  return (await git(entry,['show',entry.source.tree+':'+path])).stdout;
}

export function projectCatalogExtension(options) {
  return async pi=>pi.registerTool({name:'rsi_previous_projects',label:'Previous campaign projects',
    description:'Inspect immutable sources from earlier settled projects in these fresh campaigns. Choose any useful starting point; results retain target provenance.',
    parameters:Type.Object({action:Type.Union(['list','files','read','select'].map(value=>Type.Literal(value))),
      id:Type.Optional(Type.String()),path:Type.Optional(Type.String()),
      starter:Type.Optional(Type.Union(['baseline','scratch','prior'].map(value=>Type.Literal(value)))),
      offset:Type.Optional(Type.Integer({minimum:0})),limit:Type.Optional(Type.Integer({minimum:1,maximum:100}))}),
    execute:async(_id,{action,id,path,starter='prior',offset=0,limit=20})=>{
      if(action==='select'&&['baseline','scratch'].includes(starter)) {
        options.event?.('starter-selected',{starter});
        return {content:[{type:'text',text:JSON.stringify({starter})}],details:{starter}};
      }
      const entries=await buildProjectCatalog(options);
      const entry=entries.find(item=>item.id===id);
      if(action!=='list'&&!entry)throw new Error('Unknown settled campaign project');
      const provenance=entry&&{campaignId:entry.campaignId,projectId:entry.projectId,attemptId:entry.attemptId,
        target:entry.target,runtime:entry.runtime,tree:entry.source.tree,commit:entry.source.commit};
      const result=action==='list'?{projects:entries.slice(offset,offset+limit),total:entries.length,
        nextOffset:offset+limit<entries.length?offset+limit:null}:
        action==='select'?{starter:'prior',...provenance}:
          action==='files'?await listProjectFiles(entry):await readProjectSource(entry,path);
      if(action==='select')options.event?.('starter-selected',{starter:'prior',...provenance});
      if(action==='read'||action==='files')options.event?.('prior-project-read',{action,...provenance,...(path?{path}:{})});
      return {content:[{type:'text',text:typeof result==='string'?result:JSON.stringify(result)}],details:result};
    }});
}
