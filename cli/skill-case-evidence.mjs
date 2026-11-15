/** Read-only case-review snapshots; these do not qualify or execute a target. */
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {join,resolve,dirname} from 'node:path';
import {createHash} from 'node:crypto';
import {curatedPacks,prepareCuratedSuites} from './skill-curated-suites.mjs';
const sha=text=>createHash('sha256').update(text).digest('hex');
const optional=async path=>{try{return await readFile(path,'utf8');}catch(error){if(error.code!=='ENOENT')throw error;}};
async function qualificationSnapshot(suite){
 const provenance=suite?.qualification?.provenance??{};
 let directory=provenance.qualificationDirectory,path=provenance.qualificationReport;
 const seen=new Set();
 while(directory){
  if(seen.has(directory))throw Error('Cyclic qualification evidence');seen.add(directory);
  const index=await optional(join(directory,'qualification-continuation.json'));
  if(!index){path??=join(directory,'micro','results.json');break;}
  directory=JSON.parse(index).directory;path=join(directory,'micro','results.json');
 }
 const text=path?await optional(path):null;if(!text)return {summary:{status:'unavailable',reason:'No stored qualification report supplied'}};
 const report=JSON.parse(text);
 return {text,summary:{status:report.status,identity:report.identity,elapsedSeconds:report.elapsedSeconds,
  controls:(report.results??[]).map(row=>({key:row.key,expectedExit:row.expectedExit,passed:row.passed,exitCode:row.verification?.exitCode,
   kernelSha256:row.verification?.receipt?.kernelSha256,device:row.verification?.receipt?.evidence?.device})),
  claim:'Stored reference/negative-control observations only; no skill improvement or speedup is established.'}};
}
export async function caseSelectionEvidence({config,item,suites,cachedSuites=[],directory}){
 const packs=await curatedPacks(),groups=[],catalogues=[],identities=[],allowedPaths=[];
 for(const suite of suites){
  const files={},put=text=>{const path='files/'+sha(text)+'.txt';files[path]=text;return path;};let tasks=suite.tasks,pack,qualified=suite;
  if(suite.curated){
   pack=packs.find(candidate=>candidate.id===suite.mechanism);if(!pack)throw Error('Missing curated case inventory');
   qualified=cachedSuites.find(candidate=>candidate.curated&&candidate.familyKey===suite.familyKey);
   if(qualified?.binding?.runtimeIdentity){
    const generated=(await prepareCuratedSuites({config,item,directory:join(config.root,'skills','priority','curated'),runtimeIdentity:qualified.binding.runtimeIdentity})).find(candidate=>candidate.id===suite.id);
    if(generated?.packVersion===qualified.packVersion)tasks=generated.tasks;else qualified=null;
   }else qualified=null;
  }
  const snapshot=await qualificationSnapshot(qualified),taskSources=async task=>({
   prompt:task.prompt,files:Object.fromEntries(Object.entries(task.files??{}).map(([name,text])=>[name,put(text)])),
   checker:task.checker?put(await readFile(resolve(config.root,task.checker),'utf8')):undefined,
   pairedChecker:task.pairedChecker?put(await readFile(resolve(config.root,task.pairedChecker),'utf8')):undefined,
   controls:task.controls?{reference:Object.fromEntries(Object.entries(task.controls.reference.files).map(([name,text])=>[name,put(text)])),
    negatives:task.controls.negatives.map(control=>Object.fromEntries(Object.entries(control.files).map(([name,text])=>[name,put(text)])))}:undefined});
  const cases=pack?await Promise.all(pack.cases.map(async(entry,index)=>({id:pack.id+'-'+index,publicCase:entry.public,boundaryCases:entry.variants,
   ...(tasks?.[index]?await taskSources(tasks[index]):{starter:put(pack.starter),reference:put(pack.reference)})}))):await Promise.all((tasks??[]).map(async task=>({id:task.id,...await taskSources(task)})));
  const trusted=pack?Object.fromEntries(Object.entries(pack.trusted).map(([name,text])=>[name,put(text)])):undefined;
  const manifest={catalogueId:suite.id,target:suite.target,objective:suite.objective,description:suite.description,cases,trusted,
   workload:pack?put(pack.workload):undefined,checkerCore:pack?put(await readFile(new URL('./skill-suites/kernel-check-core.py',import.meta.url),'utf8')):undefined,
   qualification:{...snapshot.summary,...(snapshot.text?{report:'qualification.json'}:{})},
   claim:'Inspect these case specifications and protected sources to assess the mechanism and boundaries. Current target qualification follows selection; these artifacts do not establish skill improvement or speedup.'};
  if(snapshot.text)files['qualification.json']=snapshot.text;
  files['manifest.json']=JSON.stringify(manifest,null,2)+'\n';
  const fingerprint=sha(JSON.stringify(files)),home=join(directory,fingerprint);await mkdir(home,{recursive:true});
  for(const [name,text]of Object.entries(files)){const path=join(home,name);await mkdir(dirname(path),{recursive:true});try{await writeFile(path,text,{flag:'wx'});}catch(error){if(error.code!=='EEXIST')throw error;if(await readFile(path,'utf8')!==text)throw Error('Frozen case evidence changed');}}
  const artifactNames=['manifest.json',...(snapshot.text?['qualification.json']:[])];
  groups.push({id:'catalogue-'+fingerprint,message:'Protected evaluation case inventory',occurrences:[{reference:'catalogue:'+suite.id+':'+fingerprint,type:'case-catalogue',
   data:{catalogueId:suite.id,target:suite.target,objective:suite.objective,caseCount:cases.length,qualification:snapshot.summary.status},artifacts:artifactNames.map(name=>({path:join(home,name),sha256:sha(files[name])}))}]});
  allowedPaths.push(home);identities.push([suite.id,fingerprint]);
  catalogues.push({id:suite.id,description:suite.description,target:suite.target,objective:suite.objective,cases:cases.map(({id,publicCase})=>({id,...(publicCase?{publicCase}:{})})),
   evidence:{directory:home,manifest:join(home,'manifest.json'),qualification:snapshot.summary.status}});
 }
 return {catalogues,groups,allowedPaths,identity:sha(JSON.stringify(identities))};
}
