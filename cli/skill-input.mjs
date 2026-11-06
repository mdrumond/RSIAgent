import { readFile, writeFile, cp } from 'node:fs/promises';
import { join, resolve, relative, sep } from 'node:path';
import { randomUUID } from 'node:crypto';
import { stages } from '../controller/workflow.mjs';

export async function validateSkillMap(directory,map) {
  for(const scope of ['global-kdb',...stages])
    if(typeof map[scope]!=='string')throw new Error(`Map a skill file for ${scope}`);
  const normalized={};
  for(const [scope,value] of Object.entries(map)) {
    if(typeof value!=='string')throw new Error(`Map a skill file for ${scope}`);
    const path=resolve(directory,value),rel=relative(directory,path);
    if(rel==='..'||rel.startsWith('..'+sep))throw new Error('Skill files must belong to their immutable version');
    if(!(await readFile(path,'utf8')).trim())throw new Error(`Empty skill for ${scope}`);
    normalized[scope]=rel;
  }
  return normalized;
}
export async function copyVersion(config,map,edit) {
  const source=join(config.root,'skills','versions',config.skills.activeVersion);
  map=await validateSkillMap(source,map);
  const version='user-'+randomUUID(),directory=join(config.root,'skills','versions',version);
  await cp(source,directory,{recursive:true});
  await writeFile(join(directory,'skill-map.json'),JSON.stringify(map,null,2)+'\n');
  if(edit) {
    const path=resolve(directory,map[edit.scope]);
    await writeFile(path,edit.content);
    try {
      const file=join(directory,'guidance.json');
      const entries=JSON.parse(await readFile(file,'utf8')).filter(e=>e.scope!==edit.scope);
      entries.push({id:'user-'+edit.scope,scope:edit.scope,text:edit.content,critical:true,negative:false});
      await writeFile(file,JSON.stringify(entries,null,2)+'\n');
    }catch(error){if(error.code!=='ENOENT')throw error;}
  }
  return version;
}
export async function run(config,action,options={}) {
  if(config.storage&&!options.serialized) {
    const {withSkillLock}=await import('./evolution-lock.mjs');
    return withSkillLock(config,()=>run(config,action,{serialized:true}));
  }
  const directory=join(config.root,'skills','versions',config.skills.activeVersion);
  const map=JSON.parse(await readFile(join(directory,'skill-map.json'),'utf8'));
  if(!['global-kdb',...stages].includes(action.scope))throw new Error('Choose a mapped workflow scope');
  await validateSkillMap(directory,map);
  if(action.content===undefined)return {scope:action.scope,content:await readFile(resolve(directory,map[action.scope]),'utf8')};
  if(typeof action.content!=='string'||!action.content.trim())throw new Error('Skill content is required');
  const storage=config.storage?await import('../controller/storage-knowledge.mjs'):null;
  if(storage) await storage.initializeStorageKnowledge(config,undefined,{serialized:true});
  const version=await copyVersion(config,map,action);
  if(storage) await storage.recordStorageSkill(config,version,{serialized:true,
    baseVersion:config.skills.activeVersion,reason:'user-edit'});
  const {updateConfig}=await import('./instance.mjs');
  await updateConfig(config.root,current=>{
    if(current.skills.activeVersion!==config.skills.activeVersion)throw new Error('Skills changed; refresh the editor');
    return {...current,skills:{...current.skills,activeVersion:version}};
  });
  if(storage) await storage.recordStorageSkill(config,version,{serialized:true,
    baseVersion:config.skills.activeVersion,promote:true,reason:'user-edit'});
  return {scope:action.scope,content:action.content,version};
}
