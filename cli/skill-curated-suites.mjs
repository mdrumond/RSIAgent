/** Curated cases are templates, not evidence: target controls qualify each runtime. */
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
const sha=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const asset=name=>readFile(new URL('./skill-suites/'+name,import.meta.url),'utf8');
const generatorFingerprint=async()=>sha(await readFile(new URL(import.meta.url),'utf8'));
export async function curatedPacks(){return JSON.parse(await asset('a3-packs.json')).packs;}
const mechanisms={
 'reduction-broadcast': /broadcast|reduc(?:ed|tion).*axis|row.*column.*(?:rank|shape)/i,
 'splitk-empty-slots': /split.?k|scratch|partial.*(?:initial|unwrit|empty)|empty.*split/i,
 'first-call-launch': /first.?call|first.*launch|compil.*(?:launch|execut)|launch.*cache/i,
 'splitk-joint-tiles': /split.?k|(?:producer|reduction).*til(?:e|ing)/i,
 'persistent-softmax': /persistent|softmax.*(?:schedul|dispatch|row|wide)/i
};
export function applicablePack(pack,item){
 const proposal=item.proposal??{},text=JSON.stringify(proposal.changes??proposal);
 return pack.objective===item.objective&&mechanisms[pack.id]?.test(text);
}
export async function curatedCatalogue({config,item}){
 if(config.target!=='a3')return [];
 const core=await asset('kernel-check-core.py'),identity=await asset('runtime-identity.py'),generator=await generatorFingerprint();
 return (await curatedPacks()).filter(pack=>applicablePack(pack,item)).map(pack=>({id:'curated-'+pack.id,familyKey:'curated-'+pack.id,
  target:'a3',objective:pack.objective,description:pack.description,mechanism:pack.id,tags:pack.tags,curated:true,descriptorFingerprint:sha({pack,core,identity,generator})}));
}
function wrapper(bundle,config,paired=false){
 const action=paired?'m.pair(sys.argv[1],sys.argv[2],CONFIG)':'m.main(sys.argv[1],CONFIG)';
 return 'import importlib.util,json,pathlib,sys,tempfile\nBUNDLE=json.loads('+JSON.stringify(JSON.stringify(bundle))+')\nCONFIG=json.loads('+JSON.stringify(JSON.stringify(config))+')\n'+
  'with tempfile.TemporaryDirectory(prefix="rsi-suite-") as tmp:\n root=pathlib.Path(tmp)\n for name,text in BUNDLE.items():\n  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)\n s=importlib.util.spec_from_file_location("trusted_suite",root/"kernel-check-core.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n '+action+'\n';
}
export async function prepareCuratedSuites({config,directory,runtimeIdentity,item}){
 if(config.target!=='a3')return [];
 if(!runtimeIdentity?.setup?.packages||!runtimeIdentity.product)throw Error('Actual target runtime identity required');
 const core=await asset('kernel-check-core.py'),identity=await asset('runtime-identity.py'),generator=await generatorFingerprint(),packs=await curatedPacks(),suites=[];
 for(const pack of packs.filter(pack=>!item||applicablePack(pack,item))){
  const bundle={'kernel-check-core.py':core,'runtime-identity.py':identity,...pack.trusted},familyKey='curated-'+pack.id;
  const version=sha({pack,bundle,runtimeIdentity,generator}),home=join(directory,familyKey,version);await mkdir(home,{recursive:true});
  const tasks=[];
  for(const [i,entry]of pack.cases.entries()){
   const id=pack.id+'-'+i,settings={family:pack.family,cases:entry.variants,hypothesis:pack.id,runtimeIdentity};
   const checker=join(home,id+'.py'),pairedChecker=join(home,id+'-paired.py');
   await writeFile(checker,wrapper(bundle,settings));if(pack.objective==='optimization')await writeFile(pairedChecker,wrapper(bundle,settings,true));
   const files={'kernel.py':pack.starter,'case.json':JSON.stringify(entry.public,null,2),'WORKLOAD.md':pack.workload,'check.py':wrapper(bundle,{...settings,cases:[entry.public]})};
   // Optimization's slower correct starter is not a negative correctness control.
   const negative=pack.objective==='correctness'?pack.starter:pack.reference+'\n_reference_run=run\ndef run(*args):\n    result=_reference_run(*args)\n    return tuple(x+1000 for x in result) if isinstance(result,(tuple,list)) else result+1000\n';
   tasks.push({id,kind:'target-kernel',target:'a3',bucket:pack.objective,prompt:(pack.objective==='correctness'?'Repair correctness; speed is not scored.':'Optimize callable latency while preserving correctness.')+' Edit only kernel.py; preserve run signature. Read case.json and WORKLOAD.md. The configured runtime is prepared. python check.py . runs the public case; protected checks also use seed variants and boundary cases.',
    files,checker,...(pack.objective==='optimization'?{pairedChecker}:{}),protectedFiles:{'case.json':'json','WORKLOAD.md':'bytes','check.py':'bytes'},controls:{reference:{files:{'kernel.py':pack.reference}},negatives:[{files:{'kernel.py':negative}}]}});
  }
  suites.push({id:familyKey,familyKey,packVersion:version,tags:pack.tags,description:pack.description,target:'a3',objective:pack.objective,mechanism:pack.id,tasks,curated:true,
   binding:{target:'a3',environment:config.environment.options,runtimeIdentity}});
 }
 return suites;
}
