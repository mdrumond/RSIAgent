import {writeFile,readFile} from 'node:fs/promises';
import {join,dirname,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

// Test-only OS supervision substitute. The actual public CLI is covered by the
// opt-in managed gate; portable scenarios still run the canonical actor runtime.
export async function portableCli(mainPath,directory) {
  const entry=join(directory,'portable-runtime.mjs');
  const root=resolve(dirname(mainPath),'../..'),configPath=join(root,'instance.json');
  const config=JSON.parse(await readFile(configPath,'utf8'));
  if(config.environment.options?.mode==='local') {
    const adapter=join(directory,'portable-supervision.mjs');
    const original=pathToFileURL(resolve(root,config.environment.module)).href;
    await writeFile(adapter,`// TEST ONLY: replace OS inventory, keep native execution and verification.
import {createEnvironment as base} from ${JSON.stringify(original)};
export async function createEnvironment(options,context){const environment=await base(options,context);
 return {...environment,resources:async()=>[{id:'local',available:true,metadata:{supervision:'fixture'}}]};}
`);
    config.environment={...config.environment,module:adapter};
    await writeFile(configPath,JSON.stringify(config),{mode:0o600});
  }
  await writeFile(entry,`
import {main} from ${JSON.stringify(pathToFileURL(mainPath).href)};
import {operate,campaign} from ${JSON.stringify(new URL('./runtime.mjs',pathToFileURL(mainPath)).href)};
import {readConfig} from ${JSON.stringify(new URL('./instance.mjs',pathToFileURL(mainPath)).href)};
import {parseArgs} from 'node:util';
const argv=process.argv.slice(2),command=argv[0],abort=new AbortController();
const stop=()=>abort.abort();process.on('SIGTERM',stop);process.on('SIGINT',stop);
async function run(){
 if(!['wave','campaign','resume','pause','stop','kdb','tune'].includes(command))return main(argv,{signal:abort.signal});
 const {values}=parseArgs({args:argv.slice(1),options:{instance:{type:'string'},mode:{type:'string'},
 config:{type:'string'},concurrency:{type:'string'},'no-recovery':{type:'boolean'}}});
 const config=await readConfig(values.instance),options={...values,signal:abort.signal};
 if(command==='kdb'||command==='tune')return (await import(command==='kdb'?${JSON.stringify(new URL('./knowledge.mjs',pathToFileURL(mainPath)).href)}:${JSON.stringify(new URL('./skills.mjs',pathToFileURL(mainPath)).href)})).run(config,options);
 return command==='campaign'?campaign(config,options):operate(command,config,options);
}
run().then(result=>console.log(JSON.stringify(result))).catch(error=>{
 console.error(error.stack);process.exitCode=1;
}).finally(()=>{process.off('SIGTERM',stop);process.off('SIGINT',stop);});
`,{mode:0o600});
  return entry;
}
